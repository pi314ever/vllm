#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# =============================================================================
# PBS Pro wrapper: TWO concurrent DeepSeek-V3 offline latency benchmarks
# in a single 12-node allocation (6 nodes per instance).
# =============================================================================
#
# Each instance runs the same offline-latency-only flow as
# qsub_polaris_deepseek_v3_latency.sh (RUN_LATENCY=1, RUN_SERVING=0,
# RUN_THROUGHPUT=0). The only per-instance difference exposed here is
# LATENCY_IO_CONFIGS_CSV (input:output shape sweep). Everything else
# (model, batch sizes, parallelism, profiling toggles) is shared and
# identical across the two instances.
#
# Layout inside the 12-node allocation:
#   Instance A : NODE_OFFSET=0  -> head=ALL_NODES[0],  workers=ALL_NODES[1..5]
#   Instance B : NODE_OFFSET=6  -> head=ALL_NODES[6],  workers=ALL_NODES[7..11]
#
# Each instance is dispatched onto its own head node via mpiexec because
# qsub_polaris_body.sh does `ray start --head` and `local_ifname_ip`
# locally, so the body must run ON HEAD_HOST. Both mpiexec processes are
# backgrounded; the wrapper waits on both and exits non-zero if either
# instance failed.
#
# Submit:
#   qsub qsub_polaris_deepseek_v3_latency_2x.sh
#
# Submit-time overrides (qsub -v):
#   # Per-instance IO shape sweeps (comma-separated input:output pairs):
#   qsub -v LATENCY_IO_CONFIGS_CSV_A=512:128,1024:1024 \
#        -v LATENCY_IO_CONFIGS_CSV_B=4096:1024,2048:2048 \
#        qsub_polaris_deepseek_v3_latency_2x.sh
#
#   # Different shared backend (ray | mp):
#   qsub -v BACKEND=mp qsub_polaris_deepseek_v3_latency_2x.sh
#
#   # Different shared model:
#   qsub -v MODEL=/grand/Intel/dhuang/DeepSeek-V3/ qsub_polaris_deepseek_v3_latency_2x.sh
#
# Output layout (PBS_JOBNAME=ds-v3-lat-2x, jobid=12345):
#   logs/ds-v3-lat-2x.o12345_A/run.log    <- instance A driver log
#   logs/ds-v3-lat-2x.o12345_A/results/   <- instance A latency JSONs
#   logs/ds-v3-lat-2x.o12345_B/run.log    <- instance B driver log
#   logs/ds-v3-lat-2x.o12345_B/results/   <- instance B latency JSONs
# Both instances share TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR /
# VLLM_CACHE_ROOT (flock-protected; concurrent access is safe).
#
# =============================================================================

#PBS -l select=12:system=polaris:ncpus=32:ngpus=4
#PBS -l walltime=03:00:00
#PBS -l filesystems=home:grand
#PBS -q prod
#PBS -A Intel
#PBS -N ds-v3-lat-2x
#PBS -j oe
#PBS -V

set -euo pipefail

# -----------------------------------------------------------------------------
# Shared configuration (identical across both instances).
# -----------------------------------------------------------------------------
# These are the same defaults as qsub_polaris_deepseek_v3_latency.sh,
# repeated here so the 2x wrapper is self-contained. All honor any
# qsub -v overrides the user supplied.
export MODEL="${MODEL:-/grand/Intel/dhuang/DeepSeek-V3/}"
export BATCH_SIZES_CSV="${BATCH_SIZES_CSV:-1}"
export RUN_LATENCY="${RUN_LATENCY:-1}"
export RUN_SERVING="${RUN_SERVING:-0}"
export RUN_THROUGHPUT="${RUN_THROUGHPUT:-0}"

# -----------------------------------------------------------------------------
# Per-instance configuration. Only LATENCY_IO_CONFIGS_CSV differs.
# -----------------------------------------------------------------------------
LATENCY_IO_CONFIGS_CSV_A="${LATENCY_IO_CONFIGS_CSV_A:-1024:4096}"
LATENCY_IO_CONFIGS_CSV_B="${LATENCY_IO_CONFIGS_CSV_B:-4096:1024}"

# -----------------------------------------------------------------------------
# Backend selection (must match the single-instance wrapper's contract).
# Both instances use the same backend.
# -----------------------------------------------------------------------------
BACKEND="${BACKEND:-ray}"
case "${BACKEND}" in
	ray) BODY_SCRIPT="qsub_polaris_body.sh" ;;
	mp)  BODY_SCRIPT="qsub_polaris_body_mp.sh" ;;
	*)
		echo "ERROR: Unknown BACKEND=${BACKEND}. Expected 'ray' or 'mp'."
		exit 1
		;;
esac

# -----------------------------------------------------------------------------
# Sanity checks on the PBS allocation.
# -----------------------------------------------------------------------------
if [[ -z "${PBS_NODEFILE:-}" ]]; then
	echo "ERROR: PBS_NODEFILE is not set. This script must be submitted via qsub."
	exit 1
fi
if [[ -z "${PBS_O_WORKDIR:-}" ]]; then
	echo "ERROR: PBS_O_WORKDIR is not set."
	exit 1
fi

mapfile -t ALL_NODES < <(awk '!seen[$0]++' "${PBS_NODEFILE}")

# This wrapper hard-codes 2x6=12. If you want a different split, copy
# this file and adjust select=, the offsets, and the per-instance
# NUM_NODES exports below.
INSTANCE_SIZE=6
EXPECTED_TOTAL=$((INSTANCE_SIZE * 2))
if [[ "${#ALL_NODES[@]}" -lt "${EXPECTED_TOTAL}" ]]; then
	echo "ERROR: Expected ${EXPECTED_TOTAL} unique nodes, but PBS allocated ${#ALL_NODES[@]}:"
	printf "  %s\n" "${ALL_NODES[@]}"
	exit 1
fi

HEAD_A="${ALL_NODES[0]}"
HEAD_B="${ALL_NODES[INSTANCE_SIZE]}"

echo "============================================="
echo "  PBS 12-Node 2x DeepSeek-V3 Latency Wrapper"
echo "============================================="
echo "  Job ID:           ${PBS_JOBID:-N/A}"
echo "  Workdir:          ${PBS_O_WORKDIR}"
echo "  Body script:      ${BODY_SCRIPT}"
echo "  Model:            ${MODEL}"
echo "  Batch sizes:      ${BATCH_SIZES_CSV}"
echo "  RUN_LATENCY:      ${RUN_LATENCY}"
echo "  RUN_SERVING:      ${RUN_SERVING}"
echo "  RUN_THROUGHPUT:   ${RUN_THROUGHPUT}"
echo "  Instance A head:  ${HEAD_A} (offset=0, io=${LATENCY_IO_CONFIGS_CSV_A})"
echo "  Instance B head:  ${HEAD_B} (offset=${INSTANCE_SIZE}, io=${LATENCY_IO_CONFIGS_CSV_B})"
echo "============================================="

# -----------------------------------------------------------------------------
# dispatch_instance <head_host> <node_offset> <instance_suffix> <io_csv>
# -----------------------------------------------------------------------------
# Ships the body's execution onto <head_host> via a single-rank mpiexec
# so that the body's `ray start --head` and `local_ifname_ip` calls run
# on the right node. PBS_NODEFILE is forwarded explicitly: PALS does
# propagate the launch environment, but pinning the path here makes
# this wrapper robust to env-forwarding policy changes.
#
# Per-instance vars (NODE_OFFSET, INSTANCE_SUFFIX, LATENCY_IO_CONFIGS_CSV)
# are exported INSIDE the remote bash so each instance gets a clean
# private copy and does not pollute the wrapper's environment.
dispatch_instance() {
	local head_host="$1"
	local node_offset="$2"
	local suffix="$3"
	local io_csv="$4"

	mpiexec -n 1 --ppn 1 --hosts "${head_host}" -- bash -c "
		set -euo pipefail
		cd '${PBS_O_WORKDIR}'

		# Per-instance partitioning knobs (consumed by the body).
		export NUM_NODES=${INSTANCE_SIZE}
		export NODE_OFFSET=${node_offset}
		export INSTANCE_SUFFIX='${suffix}'
		export LATENCY_IO_CONFIGS_CSV='${io_csv}'

		# Shared knobs (re-exported because mpiexec env forwarding is
		# not 100% guaranteed to land every var).
		export MODEL='${MODEL}'
		export BATCH_SIZES_CSV='${BATCH_SIZES_CSV}'
		export RUN_LATENCY='${RUN_LATENCY}'
		export RUN_SERVING='${RUN_SERVING}'
		export RUN_THROUGHPUT='${RUN_THROUGHPUT}'

		# PBS context the body reads. PBS_O_WORKDIR is needed for the
		# 'source \"\${PBS_O_WORKDIR}/\${BODY_SCRIPT}\"' pattern used
		# by single-instance wrappers; we re-export it here in case
		# this body script (or a future one) consults it. PBS_NODEFILE
		# is pinned for the same robustness reason as above.
		export PBS_NODEFILE='${PBS_NODEFILE}'
		export PBS_O_WORKDIR='${PBS_O_WORKDIR}'
		export PBS_JOBID='${PBS_JOBID:-}'
		export PBS_JOBNAME='${PBS_JOBNAME:-vllm-bench}'

		bash '${PBS_O_WORKDIR}/${BODY_SCRIPT}'
	"
}

# -----------------------------------------------------------------------------
# Background both dispatches; wait on each independently so we always
# learn about both rcs even if one fails first.
# -----------------------------------------------------------------------------
echo "Launching instance A on ${HEAD_A}..."
dispatch_instance "${HEAD_A}" 0 "_A" "${LATENCY_IO_CONFIGS_CSV_A}" &
PID_A=$!
echo "  -> mpiexec PID ${PID_A}"

echo "Launching instance B on ${HEAD_B}..."
dispatch_instance "${HEAD_B}" "${INSTANCE_SIZE}" "_B" "${LATENCY_IO_CONFIGS_CSV_B}" &
PID_B=$!
echo "  -> mpiexec PID ${PID_B}"

# wait+collect each rc independently. `set -e` does not fire on
# background-job failures; we explicitly check rcs after both finish.
RC_A=0
RC_B=0
wait "${PID_A}" || RC_A=$?
echo "Instance A finished (rc=${RC_A})."
wait "${PID_B}" || RC_B=$?
echo "Instance B finished (rc=${RC_B})."

echo "============================================="
echo "  Summary"
echo "============================================="
echo "  Instance A (offset=0, io=${LATENCY_IO_CONFIGS_CSV_A}):  rc=${RC_A}"
echo "  Instance B (offset=${INSTANCE_SIZE}, io=${LATENCY_IO_CONFIGS_CSV_B}):  rc=${RC_B}"
echo "============================================="

if (( RC_A != 0 || RC_B != 0 )); then
	exit 1
fi
exit 0

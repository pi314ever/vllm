#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# =============================================================================
# PBS Pro wrapper: 6-node DeepSeek-V3 offline generate.py example
# =============================================================================
#
# Self-contained wrapper (no shared body) that runs the simple offline
# inference example
#   examples/basic/offline_inference/generate.py
# on ALCF Polaris across 6 nodes with the multiprocessing executor
# backend (no Ray).
#
# Based on the user-supplied reference command:
#
#   export VLLM_WORKER_MULTIPROC_METHOD=spawn
#   export VLLM_PP_LAYER_PARTITION=21,20,20
#
#   python -u examples/basic/offline_inference/generate.py \
#     --model /workspace/models/DeepSeek-V3 \
#     --trust-remote-code \
#     --enforce-eager \
#     --dtype bfloat16 \
#     --max-tokens 100 \
#     --max-model-len 1024 \
#     --pipeline-parallel-size 3 \
#     --tensor-parallel-size 4 \
#     --enable-expert-parallel \
#     --kernel-config '{"moe_backend": "triton"}' \
#     --gpu-memory-utilization 0.95 \
#     --max-num-seqs 16 \
#     --temperature 0 \
#     --ignore-eos
#
# Adaptations vs. the reference command:
#   * Scaled from PP=3/3 nodes -> PP=6/6 nodes (Polaris has 4 GPUs per
#     node, so PP=6 TP=4 = 24 GPUs = 6 nodes fully utilized). The layer
#     partition is rewritten to 6 entries summing to 61 (DeepSeek-V3
#     hidden-layer count, matching the 21+20+20=61 of the reference).
#   * Dropped `ZE_FLAT_DEVICE_HIERARCHY=FLAT` (Intel Level Zero / XPU
#     knob; no-op on Polaris A100 / CUDA). Kept `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
#   * Dropped `--ignore-eos` (examples/basic/offline_inference/generate.py
#     does not declare it as a sampling param; behavior difference is
#     minor with `--max-tokens 100 --temperature 0`).
#   * Dropped `--kernel-config '{"moe_backend": "triton"}'`. On Polaris
#     (CUDA / A100) vLLM's CUDA MoE backend outperforms the Triton one,
#     so we let vLLM pick the platform default (see
#     vllm/config/vllm.py's per-platform `kernel_config` and
#     vllm/model_executor/layers/fused_moe/layer.py:496, which reads
#     `vllm_config.kernel_config.moe_backend` at build time). Set
#     KERNEL_CONFIG=<json> at submit time to force a specific backend.
#   * Runs under `--distributed-executor-backend mp` with multi-node
#     rendezvous (`--nnodes 6 --node-rank N --master-addr --master-port`).
#     Head node invokes generate.py; workers 1..5 run `vllm serve
#     --headless` with matching engine args.
#   * Loads the same Polaris tuning (thread limits, XALT, NCCL/gloo on
#     hsn0, persistent Triton/Inductor caches) as qsub_polaris_body_mp.sh.
#
# Hardware:      6 x Polaris nodes x 4 x A100-80GB = 24 GPUs
# Default model: /grand/Intel/dhuang/DeepSeek-V3/
#
# Submit:
#   qsub qsub_polaris_deepseek_v3_generate.sh
#
# Submit-time overrides (qsub -v):
#   qsub -v MAX_TOKENS=200 qsub_polaris_deepseek_v3_generate.sh
#   qsub -v MODEL=/grand/path/to/other-model qsub_polaris_deepseek_v3_generate.sh
#   qsub -v MAX_MODEL_LEN=2048,MAX_NUM_SEQS=32 qsub_polaris_deepseek_v3_generate.sh
#
# =============================================================================

# =============================================================================
# PBS DIRECTIVES
# =============================================================================
# 6 nodes on Polaris debug-scaling queue. 1 hour walltime is ample for
# one cold model load + one `llm.generate()` call with max-tokens=100
# and a warm compile cache on /grand.

#PBS -l select=6:system=polaris:ncpus=32:ngpus=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -q debug-scaling
#PBS -A Intel
#PBS -N ds-v3-generate
#PBS -j oe
#PBS -V

set -euo pipefail

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"

# Load CUDA runtime libraries (required for libcudart.so on compute nodes)
module load cuda/12.9

# =============================================================================
# PYTHON ENVIRONMENT SETUP
# =============================================================================
# Shares the .venv at ${PBS_O_WORKDIR}/.venv with the rest of the
# qsub_polaris_*.sh wrappers so the compile caches and installed vllm
# stay consistent across jobs.

VENV_DIR="${VENV_DIR:-${PBS_O_WORKDIR:-.}/.venv}"

if ! command -v uv &>/dev/null; then
	echo "Installing uv..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
	echo "Creating virtual environment at ${VENV_DIR}..."
	uv venv --python 3.12 "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "Activated venv: ${VENV_DIR}"
echo "Environment: $(python --version), vllm $(vllm --version 2>/dev/null || echo 'unknown')"

# =============================================================================
# PERSISTENT COMPILE CACHES (Triton / Inductor / vLLM)
# =============================================================================
# See qsub_polaris_body_mp.sh's "PERSISTENT COMPILE CACHES" block for full
# rationale. TL;DR: parks JIT caches on /grand so second and later jobs
# skip gcc/ld forks that would otherwise blow through the Polaris cgroup
# pids.max on first decode.
CACHE_ROOT="/grand/Intel/dhuang/vllm_polaris_cache"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"

mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}"

# Serialize concurrent gcc stubs across the 4 TP worker ranks on each
# node via Triton's per-key flock cache manager.
export TRITON_CACHE_MANAGER="${TRITON_CACHE_MANAGER:-triton.runtime.cache:FileCacheManager}"

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Model (local HF checkpoint on Polaris /grand). Override with qsub -v MODEL=...
MODEL="${MODEL:-/grand/Intel/dhuang/DeepSeek-V3/}"
DTYPE="${DTYPE:-bfloat16}"

# Engine knobs from the reference command.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
# --kernel-config is OMITTED. The reference command passed
# '{"moe_backend": "triton"}', but on Polaris (CUDA / A100) vLLM's CUDA
# MoE backends outperform the Triton one, so we let vLLM pick the
# default (see vllm/config/vllm.py "kernel_config" platform defaults and
# vllm/model_executor/layers/fused_moe/layer.py:496, which reads
# vllm_config.kernel_config.moe_backend at build time). Set KERNEL_CONFIG
# at submit time if you need to override for A/B testing.
KERNEL_CONFIG="${KERNEL_CONFIG-}"

# Sampling knobs (forwarded to generate.py).
MAX_TOKENS="${MAX_TOKENS:-100}"
TEMPERATURE="${TEMPERATURE:-0}"

# Parallelism / cluster sizing. Invariant: PP_SIZE * TP_SIZE ==
# NUM_NODES * GPUS_PER_NODE. Defaults scale the reference PP=3/TP=4 (12
# GPUs / 3 nodes) up to PP=6/TP=4 (24 GPUs / 6 nodes) so this wrapper
# actually uses the whole `select=6` allocation.
NUM_NODES="${NUM_NODES:-6}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-6}"

EXPECTED_GPUS=$((NUM_NODES * GPUS_PER_NODE))

if (( PP_SIZE * TP_SIZE != EXPECTED_GPUS )); then
	echo "ERROR: Invalid parallelism configuration."
	echo "       PP_SIZE (${PP_SIZE}) * TP_SIZE (${TP_SIZE}) = $((PP_SIZE * TP_SIZE))"
	echo "       NUM_NODES (${NUM_NODES}) * GPUS_PER_NODE (${GPUS_PER_NODE}) = ${EXPECTED_GPUS}"
	echo "       These must be equal."
	exit 1
fi

# VLLM_WORKER_MULTIPROC_METHOD: inherited verbatim from the reference
# command. `spawn` (vs. the torch default `fork`) sidesteps a CUDA-fork
# corruption class on some drivers and is recommended for MultiprocExecutor.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# Pipeline-parallel layer partition. The reference command used
# "21,20,20" for PP=3 on a 61-hidden-layer DeepSeek-V3. Rewritten for
# PP=6: 11,10,10,10,10,10 sums to 61 and keeps the head stage only one
# layer heavier than the rest (mirrors the reference's "head-heavy"
# intent). Override by setting VLLM_PP_LAYER_PARTITION at submit time.
# Leave empty ("") to let vLLM pick the partition automatically.
export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-11,10,10,10,10,10}"

# torch.distributed rendezvous port for the MultiprocExecutor.
# 29501 matches vllm/config/parallel.py ParallelConfig.master_port default.
MP_MASTER_PORT="${MP_MASTER_PORT:-29501}"

# How long (seconds) to wait for each headless worker to print its
# rendezvous-ready log line before aborting.
MP_WORKER_READY_TIMEOUT="${MP_WORKER_READY_TIMEOUT:-600}"

# How long (seconds) to give each headless worker to exit on its own
# after the head-side generate.py returns before we SIGTERM/SIGKILL them.
MP_WORKER_SHUTDOWN_TIMEOUT="${MP_WORKER_SHUTDOWN_TIMEOUT:-60}"

# GPU visibility: expose GPUS_PER_NODE devices per node (0,1,...,N-1)
_default_cvd="$(seq -s, 0 $((GPUS_PER_NODE - 1)))"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${_default_cvd}}"
unset _default_cvd

# Short, filesystem-friendly PBS job id (no host suffix) for log paths.
PBS_SHORT_JOBID="${PBS_JOBID%%.*}"
PBS_SHORT_JOBID="${PBS_SHORT_JOBID:-manual}"
export PBS_SHORT_JOBID

JOB_NAME="${JOB_NAME:-${PBS_JOBNAME:-ds-v3-generate}.o${PBS_SHORT_JOBID}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${PBS_O_WORKDIR:-.}/logs/${JOB_NAME}}"
mkdir -p "${RUN_LOG_DIR}"

# Tee every byte written from this point forward to run.log. PBS still
# maintains its own <jobname>.o<jobid> spool file.
RUN_LOG_FILE="${RUN_LOG_DIR}/run.log"
exec > >(tee -a "${RUN_LOG_FILE}") 2>&1
echo "Run log teeing to: ${RUN_LOG_FILE}"

# Per-node `vllm serve --headless` worker logs.
MP_WORKER_LOG_DIR="${RUN_LOG_DIR}/workers"
mkdir -p "${MP_WORKER_LOG_DIR}"

# =============================================================================
# NETWORK CONFIGURATION (Polaris Slingshot)
# =============================================================================
# All inter-node traffic (NCCL, Gloo, torch.distributed TCPStore) goes
# over hsn0. Mismatching here leaves workers stranded in rendezvous.
RAY_IFNAME="${RAY_IFNAME:-hsn0}"

export NCCL_SOCKET_IFNAME="${RAY_IFNAME}"
export GLOO_SOCKET_IFNAME="${RAY_IFNAME}"
export NCCL_DEBUG=WARN

# =============================================================================
# THREAD LIMITS
# =============================================================================
# Same rationale as qsub_polaris_body_mp.sh: uncapped OMP/BLAS defaults
# blow past the per-job cgroup pids.max (~4096) on Polaris nodes, tripping
# pthread_create EAGAIN during model init / CUDA graph capture. Cap at
# 4 threads per worker.
NUM_THREADS_PER_WORKER="${NUM_THREADS_PER_WORKER:-4}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"

# =============================================================================
# SUBPROCESS FAN-OUT LIMITS (Inductor + ALCF XALT)
# =============================================================================
# Force Inductor to compile serially (cold compile slower; warm cache
# unaffected). Disable XALT's ld wrapper that re-sources Lmod on every
# link. Both are copied verbatim from qsub_polaris_body_mp.sh.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export XALT_EXECUTABLE_TRACKING="${XALT_EXECUTABLE_TRACKING:-no}"

# Tokenizers / telemetry / NCCL thread-budget caps (same rationale as
# qsub_polaris_body_mp.sh's "ADDITIONAL THREAD-COUNT REDUCTION" block).
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-1}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"

# PyTorch NCCL per-PG thread minimization (same rationale as body_mp).
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-0}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-0}"

# =============================================================================
# NODE DISCOVERY FROM PBS
# =============================================================================

if [[ -z "${PBS_NODEFILE:-}" ]]; then
	echo "ERROR: PBS_NODEFILE is not set. This script must be submitted via qsub."
	exit 1
fi

# Unique hostnames preserving PBS allocation order. PBS runs the script
# on the first node, so ALL_NODES[0] is the head.
mapfile -t ALL_NODES < <(awk '!seen[$0]++' "${PBS_NODEFILE}")

if [[ "${#ALL_NODES[@]}" -lt "${NUM_NODES}" ]]; then
	echo "ERROR: Expected ${NUM_NODES} nodes, but PBS allocated ${#ALL_NODES[@]}:"
	printf "  %s\n" "${ALL_NODES[@]}"
	exit 1
fi

HEAD_HOST="${ALL_NODES[0]}"
# Take exactly NUM_NODES-1 workers (ignore any extra nodes PBS allocated).
WORKER_HOSTS=("${ALL_NODES[@]:1:$((NUM_NODES - 1))}")

# -----------------------------------------------------------------------------
# IP resolution
# -----------------------------------------------------------------------------
# Force torch.distributed TCPStore and NCCL/Gloo to bind to the Slingshot
# (hsn0) address on every node. `getent hosts` returns a management
# network IP on Polaris; using that here would strand workers in
# rendezvous against a head listening on hsn0.

local_ifname_ip() {
	local ifname="${1:-${RAY_IFNAME}}"
	if command -v ip &>/dev/null; then
		ip -4 -o addr show dev "${ifname}" 2>/dev/null |
			awk '{print $4}' | cut -d'/' -f1 | head -n1
	else
		awk '{print $1}' "/sys/class/net/${ifname}/address" 2>/dev/null || true
	fi
}

remote_ifname_ip() {
	local host="$1"
	local ifname="${2:-${RAY_IFNAME}}"
	mpiexec -n 1 --ppn 1 --hosts "${host}" -- bash -c \
		"ip -4 -o addr show dev '${ifname}' | awk '{print \$4}' | cut -d'/' -f1 | head -n1" \
		2>/dev/null | tr -d '[:space:]'
}

HEAD_IP="$(local_ifname_ip "${RAY_IFNAME}")"
if [[ -z "${HEAD_IP}" ]]; then
	echo "ERROR: Could not determine IP for interface ${RAY_IFNAME} on head node ${HEAD_HOST}."
	echo "       Set RAY_IFNAME to the correct interface (e.g. RAY_IFNAME=hsn1) and retry."
	exit 1
fi

WORKER_IPS=()
for WHOST in "${WORKER_HOSTS[@]}"; do
	WIP="$(remote_ifname_ip "${WHOST}" "${RAY_IFNAME}")"
	if [[ -z "${WIP}" ]]; then
		echo "ERROR: Could not resolve ${RAY_IFNAME} IP on worker ${WHOST}."
		exit 1
	fi
	WORKER_IPS+=("${WIP}")
done

# VLLM_HOST_IP identifies this process's local interface IP for vLLM
# IPC. Head-side value is set here; per-worker mpiexec commands export
# their own local IP.
export VLLM_HOST_IP="${HEAD_IP}"

echo "============================================="
echo "  PBS 6-Node DeepSeek-V3 Offline Generate Example"
echo "============================================="
echo "  Job ID:         ${PBS_JOBID:-N/A}"
echo "  Backend:        mp (MultiprocExecutor + torch.distributed)"
echo "  Head node:      ${HEAD_HOST} (${HEAD_IP})"
for i in "${!WORKER_HOSTS[@]}"; do
	printf "  Worker %-9s %s (%s)\n" "$((i + 1)):" "${WORKER_HOSTS[$i]}" "${WORKER_IPS[$i]}"
done
echo "  Model:          ${MODEL}"
echo "  DType:          ${DTYPE}"
echo "  TP size:        ${TP_SIZE}"
echo "  PP size:        ${PP_SIZE}"
echo "  PP partition:   ${VLLM_PP_LAYER_PARTITION:-auto}"
echo "  Expected GPUs:  ${EXPECTED_GPUS}"
echo "  Max model len:  ${MAX_MODEL_LEN}"
echo "  Max num seqs:   ${MAX_NUM_SEQS}"
echo "  GPU mem util:   ${GPU_MEMORY_UTILIZATION}"
echo "  Kernel config:  ${KERNEL_CONFIG:-default (CUDA MoE backend)}"
echo "  Max tokens:     ${MAX_TOKENS}"
echo "  Temperature:    ${TEMPERATURE}"
echo "  Threads/worker: ${NUM_THREADS_PER_WORKER}"
echo "  MP master:      ${HEAD_IP}:${MP_MASTER_PORT}"
echo "  Cache root:     ${CACHE_ROOT}"
echo "  Run log:        ${RUN_LOG_FILE}"
echo "  Worker logs:    ${MP_WORKER_LOG_DIR}"
echo "============================================="

# =============================================================================
# CLEANUP TRAP
# =============================================================================
# Ensure headless workers and any stray vllm serve children are killed
# on exit. Under mp there is no persistent cluster daemon; teardown is
# strictly:
#   1. Kill tracked mpiexec-launched headless worker PIDs.
#   2. Belt-and-suspenders: `pkill -f "vllm serve.*--headless"` on each
#      worker host, in case an mpiexec session exited cleanly but left
#      its vllm child behind.

MP_WORKER_PIDS=()

cleanup() {
	echo ""
	echo "Cleaning up..."

	local pid
	for pid in "${MP_WORKER_PIDS[@]:-}"; do
		if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
			kill "${pid}" 2>/dev/null || true
		fi
	done
	for pid in "${MP_WORKER_PIDS[@]:-}"; do
		if [[ -n "${pid}" ]]; then
			wait "${pid}" 2>/dev/null || true
		fi
	done

	local whost
	for whost in "${WORKER_HOSTS[@]}"; do
		mpiexec -n 1 --ppn 1 --hosts "${whost}" -- bash -c \
			"pkill -f 'vllm serve.*--headless' 2>/dev/null || true" \
			2>/dev/null || true
	done

	echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

# =============================================================================
# SHARED ENGINE ARGS (head + workers)
# =============================================================================
# IMPORTANT: every value in this array must appear IDENTICALLY on the
# head (generate.py) and every worker (vllm serve --headless) or
# MultiprocExecutor's VllmConfig consistency check will reject the
# rendezvous. --node-rank and --master-addr/--master-port differ per
# node and are appended separately by the head-side and worker-side
# invocations below.
#
# `--kernel-config` is injected conditionally: omitted by default
# (letting vLLM pick the platform-optimal CUDA MoE backend on A100),
# and included only when KERNEL_CONFIG is set at submit time. The JSON
# value is kept in one shell-word using array element quoting so it
# survives `printf %q` below.
KERNEL_CONFIG_ARGS=()
if [[ -n "${KERNEL_CONFIG}" ]]; then
	KERNEL_CONFIG_ARGS=(--kernel-config "${KERNEL_CONFIG}")
fi

MP_COMMON_ENGINE_ARGS=(
	--model "${MODEL}"
	--trust-remote-code
	--enforce-eager
	--dtype "${DTYPE}"
	--max-model-len "${MAX_MODEL_LEN}"
	--pipeline-parallel-size "${PP_SIZE}"
	--tensor-parallel-size "${TP_SIZE}"
	--enable-expert-parallel
	${KERNEL_CONFIG_ARGS[@]+"${KERNEL_CONFIG_ARGS[@]}"}
	--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
	--max-num-seqs "${MAX_NUM_SEQS}"
	--distributed-executor-backend mp
)

# Head-side rendezvous args. generate.py's FlexibleArgumentParser gets
# --nnodes / --node-rank / --master-addr / --master-port from
# EngineArgs.add_cli_args (vllm/engine/arg_utils.py:909-912).
MP_HEAD_DIST_ARGS=(
	--nnodes "${NUM_NODES}"
	--node-rank 0
	--master-addr "${HEAD_IP}"
	--master-port "${MP_MASTER_PORT}"
)

# =============================================================================
# LAUNCH HEADLESS WORKERS ON NODES 1..N-1
# =============================================================================
# One mpiexec-scoped bash per worker host, each running `vllm serve
# --headless` with the shared engine args plus a per-node --node-rank
# and the head's --master-addr/--master-port. Workers block in
# torch.distributed rendezvous until the head-side generate.py completes
# init; when generate.py exits, torch.distributed tears down and each
# worker's MultiprocExecutor monitor loop detects the broken process
# group and calls sys.exit().

# Pre-quote the shared engine args so they survive the outer->inner
# bash -c expansion. printf %q produces output that is safe to pass
# through a single round of word-splitting and preserves JSON braces
# / internal spaces in --kernel-config.
WORKER_ARGS_STR="$(printf '%q ' "${MP_COMMON_ENGINE_ARGS[@]}")"

echo ""
echo "Launching headless workers on $((NUM_NODES - 1)) worker nodes..."

for i in "${!WORKER_HOSTS[@]}"; do
	whost="${WORKER_HOSTS[$i]}"
	wip="${WORKER_IPS[$i]}"
	wrank="$((i + 1))"
	worker_log="${MP_WORKER_LOG_DIR}/generate_rank${wrank}_${whost}.log"

	echo "  Launching mp worker rank ${wrank} on ${whost} (${wip}); log: ${worker_log}"

	# The remote command re-exports every env var the head has that
	# materially affects engine init, activates the venv, then execs
	# `vllm serve --headless`. Using `exec` so signals from mpiexec's
	# process-manager flow to vllm directly.
	mpiexec -n 1 --ppn 1 --hosts "${whost}" -- bash -c "
		set -euo pipefail

		export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'
		export VLLM_HOST_IP='${wip}'

		# Persistent compile caches (shared filesystem; must match the
		# head node so Triton's FileCacheManager flock serializes
		# correctly across all ranks in the cluster).
		export TRITON_CACHE_DIR='${TRITON_CACHE_DIR}'
		export TORCHINDUCTOR_CACHE_DIR='${TORCHINDUCTOR_CACHE_DIR}'
		export VLLM_CACHE_ROOT='${VLLM_CACHE_ROOT}'
		export TRITON_CACHE_MANAGER='${TRITON_CACHE_MANAGER}'

		# vLLM / torch worker method.
		export VLLM_WORKER_MULTIPROC_METHOD='${VLLM_WORKER_MULTIPROC_METHOD}'
		export VLLM_PP_LAYER_PARTITION='${VLLM_PP_LAYER_PARTITION}'

		# Thread limits (see THREAD LIMITS block).
		export OMP_NUM_THREADS='${OMP_NUM_THREADS}'
		export OPENBLAS_NUM_THREADS='${OPENBLAS_NUM_THREADS}'
		export MKL_NUM_THREADS='${MKL_NUM_THREADS}'
		export NUMEXPR_NUM_THREADS='${NUMEXPR_NUM_THREADS}'
		export VECLIB_MAXIMUM_THREADS='${VECLIB_MAXIMUM_THREADS}'
		export RAYON_NUM_THREADS='${RAYON_NUM_THREADS}'
		export TORCH_NUM_THREADS='${TORCH_NUM_THREADS}'

		# Subprocess fan-out caps.
		export TORCHINDUCTOR_COMPILE_THREADS='${TORCHINDUCTOR_COMPILE_THREADS}'
		export XALT_EXECUTABLE_TRACKING='${XALT_EXECUTABLE_TRACKING}'

		# Additional thread-count caps.
		export NCCL_SOCKET_NTHREADS='${NCCL_SOCKET_NTHREADS}'
		export NCCL_NSOCKS_PERTHREAD='${NCCL_NSOCKS_PERTHREAD}'
		export TOKENIZERS_PARALLELISM='${TOKENIZERS_PARALLELISM}'
		export VLLM_NO_USAGE_STATS='${VLLM_NO_USAGE_STATS}'
		export DO_NOT_TRACK='${DO_NOT_TRACK}'

		# PyTorch NCCL per-PG thread minimization.
		export TORCH_NCCL_ENABLE_MONITORING='${TORCH_NCCL_ENABLE_MONITORING}'
		export TORCH_NCCL_TRACE_BUFFER_SIZE='${TORCH_NCCL_TRACE_BUFFER_SIZE}'
		export TORCH_NCCL_DUMP_ON_TIMEOUT='${TORCH_NCCL_DUMP_ON_TIMEOUT}'

		# NCCL/Gloo must use the same interface as on the head node.
		export NCCL_SOCKET_IFNAME='${RAY_IFNAME}'
		export GLOO_SOCKET_IFNAME='${RAY_IFNAME}'
		export NCCL_DEBUG='${NCCL_DEBUG}'

		# Load CUDA runtime libraries on the remote node.
		module load cuda/12.9

		# Activate the venv so vllm is on PATH.
		source '${VENV_DIR}/bin/activate'

		echo '[mp-worker rank=${wrank} host=${whost}] Starting vllm serve --headless'
		echo '[mp-worker rank=${wrank} host=${whost}] master=${HEAD_IP}:${MP_MASTER_PORT}'

		exec vllm serve ${WORKER_ARGS_STR} \\
			--nnodes ${NUM_NODES} \\
			--node-rank ${wrank} \\
			--master-addr '${HEAD_IP}' \\
			--master-port '${MP_MASTER_PORT}' \\
			--headless
	" >"${worker_log}" 2>&1 &

	MP_WORKER_PIDS+=($!)
done

echo "Launched ${#MP_WORKER_PIDS[@]} mp worker(s): ${MP_WORKER_PIDS[*]}"

# =============================================================================
# WAIT FOR WORKERS TO BE RENDEZVOUS-READY
# =============================================================================
# Poll each worker log for the
#   "Launching vLLM ... headless multiproc executor"
# line (vllm/entrypoints/cli/serve.py:186) that workers print right
# before entering torch.distributed init. Returning from this loop means
# workers are alive and blocking on the rendezvous; the head-side
# generate.py can then start and complete the handshake.

ready_pattern="headless multiproc executor"
deadline=$(( $(date +%s) + MP_WORKER_READY_TIMEOUT ))

while true; do
	ready_count=0
	for i in "${!WORKER_HOSTS[@]}"; do
		whost="${WORKER_HOSTS[$i]}"
		wrank="$((i + 1))"
		worker_log="${MP_WORKER_LOG_DIR}/generate_rank${wrank}_${whost}.log"
		if [[ -f "${worker_log}" ]] && grep -Fq "${ready_pattern}" "${worker_log}" 2>/dev/null; then
			ready_count=$((ready_count + 1))
		fi
	done

	if (( ready_count >= ${#WORKER_HOSTS[@]} )); then
		echo "All ${#WORKER_HOSTS[@]} mp workers are rendezvous-ready."
		break
	fi

	# Detect premature worker death.
	for pid in "${MP_WORKER_PIDS[@]}"; do
		if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
			echo "ERROR: mp worker PID ${pid} died during startup."
			echo "       Check ${MP_WORKER_LOG_DIR}/generate_rank*_*.log for tracebacks."
			exit 1
		fi
	done

	if (( $(date +%s) >= deadline )); then
		echo "ERROR: Only ${ready_count}/${#WORKER_HOSTS[@]} mp workers ready after ${MP_WORKER_READY_TIMEOUT}s."
		echo "       See ${MP_WORKER_LOG_DIR}/generate_rank*_*.log for slow starters."
		exit 1
	fi

	echo "  Waiting for mp workers... ${ready_count}/${#WORKER_HOSTS[@]} ready"
	sleep 5
done

# =============================================================================
# RUN THE HEAD-SIDE GENERATE EXAMPLE
# =============================================================================
# generate.py builds an LLM(...) in-process on the head and rendezvouses
# with the headless workers via torch.distributed. On completion, the
# Python process exits; torch.distributed tears down; MultiprocExecutor
# monitor loops on the workers detect the broken group and sys.exit().

EXAMPLE_SCRIPT="${EXAMPLE_SCRIPT:-${PBS_O_WORKDIR:-.}/examples/basic/offline_inference/generate.py}"
if [[ ! -f "${EXAMPLE_SCRIPT}" ]]; then
	echo "ERROR: Example script not found at ${EXAMPLE_SCRIPT}"
	echo "       Set EXAMPLE_SCRIPT=/path/to/generate.py and retry."
	exit 1
fi

echo ""
echo "Running head-side generate example: ${EXAMPLE_SCRIPT}"
echo ""

python -u "${EXAMPLE_SCRIPT}" \
	"${MP_COMMON_ENGINE_ARGS[@]}" \
	"${MP_HEAD_DIST_ARGS[@]}" \
	--max-tokens "${MAX_TOKENS}" \
	--temperature "${TEMPERATURE}"

GENERATE_RC=$?

echo ""
echo "generate.py exited with rc=${GENERATE_RC}"

# =============================================================================
# WAIT FOR WORKERS TO EXIT NATURALLY
# =============================================================================
# Once generate.py returns, workers should see their torch.distributed
# group collapse and exit on their own within MP_WORKER_SHUTDOWN_TIMEOUT.
# If any are still alive after the deadline, the cleanup trap will
# SIGTERM/SIGKILL and then `pkill -f "vllm serve.*--headless"` on each
# worker host as belt-and-suspenders.

echo ""
echo "Waiting up to ${MP_WORKER_SHUTDOWN_TIMEOUT}s for workers to exit naturally..."

deadline=$(( $(date +%s) + MP_WORKER_SHUTDOWN_TIMEOUT ))
while (( $(date +%s) < deadline )); do
	still_alive=0
	for pid in "${MP_WORKER_PIDS[@]}"; do
		if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
			still_alive=$((still_alive + 1))
		fi
	done
	if (( still_alive == 0 )); then
		echo "All workers exited."
		break
	fi
	sleep 2
done

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "============================================="
echo "  generate.py job complete"
echo "============================================="
echo "  Job ID:      ${PBS_JOBID:-N/A}"
echo "  rc:          ${GENERATE_RC}"
echo "  Run log:     ${RUN_LOG_FILE}"
echo "  Worker logs: ${MP_WORKER_LOG_DIR}/generate_rank*_*.log"
echo "============================================="

# Cleanup trap handles any remaining teardown on exit.
exit "${GENERATE_RC}"

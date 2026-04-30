#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# =============================================================================
# PBS Pro Job Script - N-Node DeepSeek-R1 Benchmark on ALCF Polaris
# =============================================================================
#
# Single qsub-submittable script that orchestrates the full N-node benchmark:
#   1. Discovers allocated nodes from PBS_NODEFILE
#   2. Launches Ray head on node 0, Ray workers on nodes 1..N-1 (via mpiexec)
#   3. Runs offline latency benchmarks (vllm bench latency)
#   4. Starts vLLM API server and runs online serving benchmarks
#   5. Cleans up Ray on all nodes on exit
#
# Parallelism model (defaults):
#   NUM_NODES=4, GPUS_PER_NODE=4  -> 16 GPUs total
#   PP_SIZE=4, TP_SIZE=4          -> PP across nodes, TP within each node
#   --enable-expert-parallel      -> EP size = TP_SIZE * DP_SIZE = 4 (DP=1)
#
# NOTE on EP: vLLM exposes expert parallelism as a boolean flag
# (--enable-expert-parallel). The effective EP group size is TP * DP. To get
# EP=M with DP=1, set TP_SIZE=M. To get EP=M with DP>1, size TP and DP so
# that TP*DP=M and PP*TP*DP = NUM_NODES*GPUS_PER_NODE (DP is not configured
# by this script; extend QUANT_ARGS / ENGINE_ARGS if you need it).
#
# Hardware:      Polaris nodes x 4x A100-80GB
# Model:         deepseek-ai/DeepSeek-R1 (671B MoE, FP8)
#
# Usage:
#   # Edit the -A project allocation below, then:
#   qsub qsub_polaris.sh
#
#   # Override settings at submission time (note: the PBS `select=` directive
#   # is baked into the file, so if you change NUM_NODES you must either edit
#   # the -l select=... line below OR override it on the qsub command line):
#   qsub -l select=6:system=polaris:ncpus=32:ngpus=4 \
#        -v NUM_NODES=6,PP_SIZE=6,TP_SIZE=4 qsub_polaris.sh
#
# =============================================================================

# =============================================================================
# PBS DIRECTIVES
# =============================================================================
# NOTE: The `select=N` count here MUST match NUM_NODES in the user config
# section below, or be overridden at submission time with `qsub -l select=...`.

#PBS -l select=4:system=polaris:ncpus=32:ngpus=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -q debug-scaling
#PBS -A Intel
#PBS -N vllm-deepseek-r1-bench
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
# Ensure uv is available, create a venv if needed, and install vllm (+ ray).

VENV_DIR="${VENV_DIR:-${PBS_O_WORKDIR:-.}/.venv}"

# Ensure uv is on PATH (install if missing)
if ! command -v uv &>/dev/null; then
	echo "Installing uv..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="${HOME}/.local/bin:${PATH}"
fi

# Create the venv if it doesn't exist
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
	echo "Creating virtual environment at ${VENV_DIR}..."
	uv venv --python 3.12 "${VENV_DIR}"
fi

# Activate the venv
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "Activated venv: ${VENV_DIR}"

echo "Environment ready: $(python --version), ray $(ray --version 2>/dev/null || echo 'unknown'), vllm $(vllm --version 2>/dev/null || echo 'unknown')"

# =============================================================================
# STEP 0: Patch vLLM for PP>1 Ray KeyError (vllm-project/vllm#41287)
# =============================================================================
# vllm==0.20.0 has a bug where RayDistributedExecutor reorders workers and
# calls WorkerWrapperBase.adjust_rank, but adjust_rank only updates rpc_rank
# and not global_rank. That desync causes each worker to pick the wrong
# entry from the per-worker kv_cache_configs list, and PP>1 jobs crash with
# e.g. `KeyError: 'model.layers.30.self_attn.attn'` during
# initialize_attn_backend. See:
#   - Issue: https://github.com/vllm-project/vllm/issues/41287
#   - Fix PR (not yet merged): https://github.com/vllm-project/vllm/pull/41298
#
# The patch below is PR #41298's one-liner: also assign self.global_rank.
# It is idempotent (skips if already patched or if upstream has fixed it) and
# aborts the job if it cannot recognize the source shape.
#
# DELETE this block once your installed vLLM version contains PR #41298.

python - <<'PY' || { echo "ERROR: vLLM adjust_rank patch failed. Aborting."; exit 1; }
import sys, pathlib, shutil, tempfile, inspect, importlib

# In vllm 0.20.0 adjust_rank lives on RayWorkerWrapper in
# vllm/v1/executor/ray_utils.py. Older/newer layouts may place it on
# WorkerWrapperBase in vllm/v1/worker/worker_base.py. We look for whichever
# class actually defines (not inherits) adjust_rank.
CANDIDATES = [
    ("vllm.v1.executor.ray_utils", "RayWorkerWrapper"),
    ("vllm.v1.worker.worker_base", "WorkerWrapperBase"),
]

target_module = None
target_class = None
for module_name, class_name in CANDIDATES:
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        continue
    cls = getattr(mod, class_name, None)
    if cls is None:
        continue
    # `vars(cls)` gives only attributes defined directly on cls, so this
    # excludes methods inherited from a parent.
    if "adjust_rank" in vars(cls):
        target_module = mod
        target_class = cls
        break

if target_module is None:
    print(
        "[vllm-patch] ERROR: adjust_rank not found on RayWorkerWrapper or "
        "WorkerWrapperBase; refusing to patch.",
        file=sys.stderr,
    )
    sys.exit(1)

src_path = pathlib.Path(inspect.getsourcefile(target_module))
src = src_path.read_text()

# Match on the two-line body only. The indentation is dynamic because in
# v0.20.0 `class RayWorkerWrapper` is defined inside a `try:` block, so the
# `if` body is 12/16-space indented there, vs. 8/12-space if the method
# lives on WorkerWrapperBase (top-level class) in other vLLM versions. We
# synthesize both candidates and try each.
BODY_ORIGINALS = [
    (
        "        if self.rpc_rank in rank_mapping:\n"
        "            self.rpc_rank = rank_mapping[self.rpc_rank]\n"
    ),
    (
        "            if self.rpc_rank in rank_mapping:\n"
        "                self.rpc_rank = rank_mapping[self.rpc_rank]\n"
    ),
]


def _patched_for(original: str) -> str:
    # Preserve the existing indentation of the `if` line, then add an
    # extra `self.global_rank = self.rpc_rank` line at the same indent as
    # the existing `self.rpc_rank = ...` assignment.
    if_line, assign_line, _ = original.split("\n", 2)
    inner_indent = assign_line[: len(assign_line) - len(assign_line.lstrip())]
    return (
        f"{if_line}\n"
        f"{assign_line}\n"
        f"{inner_indent}self.global_rank = self.rpc_rank\n"
    )


matched_original = None
patched = None
for candidate in BODY_ORIGINALS:
    if _patched_for(candidate) in src:
        print(f"[vllm-patch] already patched: {src_path}")
        sys.exit(0)
    if candidate in src:
        matched_original = candidate
        patched = _patched_for(candidate)
        break

if matched_original is None:
    # Accept any source where the fix line is already present as "fixed upstream".
    if "self.global_rank = self.rpc_rank" in inspect.getsource(target_class.adjust_rank):
        print(f"[vllm-patch] upstream fix detected, skipping: {src_path}")
        sys.exit(0)
    print(f"[vllm-patch] ERROR: adjust_rank source shape not recognized in {src_path}.", file=sys.stderr)
    print("[vllm-patch] Refusing to run the benchmark; it would crash during profiling.", file=sys.stderr)
    print("[vllm-patch] Inspect the file manually and update this patch block.", file=sys.stderr)
    sys.exit(1)

if src.count(matched_original) != 1:
    print(
        f"[vllm-patch] ERROR: expected exactly 1 match for adjust_rank body, "
        f"found {src.count(matched_original)}.",
        file=sys.stderr,
    )
    sys.exit(1)

new_src = src.replace(matched_original, patched, 1)
# Atomic replace via a sibling temp file so readers never see a partial file.
with tempfile.NamedTemporaryFile("w", dir=str(src_path.parent), delete=False) as tmp:
    tmp.write(new_src)
    tmp_path = pathlib.Path(tmp.name)
shutil.copymode(src_path, tmp_path)
tmp_path.replace(src_path)
print(f"[vllm-patch] patch applied: {src_path}")
PY

# Post-patch sanity check: ensure the fix line is actually present in the
# class we will use at runtime. Guards against import-caching oddities and
# anything else that could silently leave the file unpatched.
python - <<'PY' || { echo "ERROR: post-patch verification failed. Aborting."; exit 1; }
import importlib, inspect, sys
CANDIDATES = [
    ("vllm.v1.executor.ray_utils", "RayWorkerWrapper"),
    ("vllm.v1.worker.worker_base", "WorkerWrapperBase"),
]
for module_name, class_name in CANDIDATES:
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        continue
    cls = getattr(mod, class_name, None)
    if cls and "adjust_rank" in vars(cls):
        assert "self.global_rank = self.rpc_rank" in inspect.getsource(cls.adjust_rank), (
            f"vLLM patch did not stick in {module_name}.{class_name}.adjust_rank"
        )
        break
else:
    print("could not locate adjust_rank for verification", file=sys.stderr)
    sys.exit(1)
PY

# =============================================================================
# PERSISTENT COMPILE CACHES (Triton / Inductor / vLLM)
# =============================================================================
# First-run JIT compilation is the main source of process pressure during
# the initial decode step (each of the 4 workers per node shells out to
# /usr/bin/gcc to build the Triton launcher stub). On a cold cache, 4
# concurrent gcc invocations per node can trip `vfork: Resource
# temporarily unavailable` from Triton's `_build`, crashing the first
# kernel launch in `_compute_slot_mapping_kernel` (see worker_base.py ->
# gpu_model_runner.py -> block_table.compute_slot_mapping).
#
# Parking these caches on a shared, persistent filesystem means SECOND
# and later jobs hit the cache and skip gcc entirely. /grand is Polaris's
# project filesystem and is readable/writable from every compute node.
#
# Hardcoded to /grand/Intel/dhuang because this script is expected to be
# used by dhuang. Change if another user adopts it.
CACHE_ROOT="/grand/Intel/dhuang/vllm_polaris_cache"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"

mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}"

# Serialize concurrent compilation through Triton's file-lock cache
# manager. With 4 workers per node all JIT-compiling the same kernel
# simultaneously, concurrent gcc invocations can transiently spike the
# per-user/per-cgroup process count even with a shared cache.
# FileCacheManager takes a per-cache-key flock so only one rank actually
# runs gcc; the other ranks block until the .so lands, then mmap it.
export TRITON_CACHE_MANAGER="${TRITON_CACHE_MANAGER:-triton.runtime.cache:FileCacheManager}"

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Model configuration
MODEL="${MODEL:-/grand/Intel/dhuang/DeepSeek-V3-0324/}"
QUANTIZATION="${QUANTIZATION:-}" # e.g. "fp8"; leave empty for no quantization
DTYPE="${DTYPE:-auto}"

# Build conditional quantization args (used by ENGINE_ARGS and vllm serve)
QUANT_ARGS=()
if [[ -n "${QUANTIZATION}" ]]; then
	QUANT_ARGS=(--quantization "${QUANTIZATION}")
fi

# Parallelism / cluster sizing
#
# NUM_NODES     : Number of Polaris nodes (must match the PBS `select=` count).
# GPUS_PER_NODE : GPUs per node (4 on Polaris A100 nodes).
# TP_SIZE       : Tensor-parallel size. Must evenly divide GPUS_PER_NODE so
#                 TP groups stay inside a node (NVLink).
# PP_SIZE       : Pipeline-parallel size. PP groups span nodes.
# EP_SIZE       : Expert-parallel group size (DERIVED, not set directly).
#                 With --enable-expert-parallel and DP=1, EP_SIZE == TP_SIZE.
#
# Invariant: PP_SIZE * TP_SIZE == NUM_NODES * GPUS_PER_NODE
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-4}"

# Ray configuration
RAY_PORT="${RAY_PORT:-6379}"
EXPECTED_NODES="${NUM_NODES}"
EXPECTED_GPUS=$((NUM_NODES * GPUS_PER_NODE))
EP_SIZE="${TP_SIZE}" # derived; only valid while DP=1
RAY_CLUSTER_TIMEOUT="${RAY_CLUSTER_TIMEOUT:-600}" # seconds

# Sanity check: total GPU count must match PP * TP.
if (( PP_SIZE * TP_SIZE != EXPECTED_GPUS )); then
	echo "ERROR: Invalid parallelism configuration."
	echo "       PP_SIZE (${PP_SIZE}) * TP_SIZE (${TP_SIZE}) = $((PP_SIZE * TP_SIZE))"
	echo "       NUM_NODES (${NUM_NODES}) * GPUS_PER_NODE (${GPUS_PER_NODE}) = ${EXPECTED_GPUS}"
	echo "       These must be equal."
	exit 1
fi

# Use a short temp dir to avoid AF_UNIX 107-byte socket path limit.
# PBS on Polaris sets $TMPDIR to very long paths like:
#   /var/tmp/pbs.7101514.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/
# which causes Ray socket paths to exceed the OS limit.
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${PBS_JOBID%%.*}}"
export RAY_TMPDIR
mkdir -p "${RAY_TMPDIR}"

# GPU visibility: expose GPUS_PER_NODE devices per node (0,1,...,N-1)
_default_cvd="$(seq -s, 0 $((GPUS_PER_NODE - 1)))"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${_default_cvd}}"
unset _default_cvd

# Benchmark output directory (under PBS job working directory)
RESULTS_DIR="${RESULTS_DIR:-${PBS_O_WORKDIR:-.}/results_${PBS_JOBID:-manual}}"

# Server port for online serving benchmark
SERVE_PORT="${SERVE_PORT:-8000}"

# Max model length
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Number of prompts / warmups for serving benchmark
NUM_PROMPTS="${NUM_PROMPTS:-500}"
NUM_WARMUPS="${NUM_WARMUPS:-10}"

# =============================================================================
# NETWORK CONFIGURATION (Polaris Slingshot)
# =============================================================================
# Polaris uses HPE Slingshot interconnect. The high-speed network interfaces
# are typically named hsn0, hsn1, etc. All inter-node traffic (NCCL, Gloo,
# Ray, and vLLM control plane) MUST use the same IP family or Ray's
# placement-group node-affinity constraint ("node:<ip>" resource) will not
# be satisfiable -- vLLM requests `node:<VLLM_HOST_IP>` for the driver
# bundle, and that IP has to match the IP Ray registered the node under.
#
# The interface name used for both NCCL/Gloo and Ray/VLLM_HOST_IP:
RAY_IFNAME="${RAY_IFNAME:-hsn0}"

export NCCL_SOCKET_IFNAME="${RAY_IFNAME}"
export GLOO_SOCKET_IFNAME="${RAY_IFNAME}"
export NCCL_DEBUG=WARN

# Disable Ray usage stats
export RAY_USAGE_STATS_ENABLED=0

# =============================================================================
# RAY EXECUTOR BACKEND
# =============================================================================
# vLLM's default RayDistributedExecutor uses Ray Compiled Graph (aDAG) for PP.
# On Polaris at PP=6 this hits a RayChannelTimeoutError ~300s into the first
# decode step: the first cross-node NCCL send/recv via RayPPCommunicator stalls
# during lazy ring setup, the last-PP-rank output never lands, and vLLM's
# `ray.get(ref, timeout=RAY_CGRAPH_get_timeout)` fires.
#
# Switch to RayExecutorV2, which subclasses MultiprocExecutor and uses
# MessageQueue (shm_broadcast) + plain Ray actor RPC. That removes the aDAG
# code path (and the RAY_CGRAPH_get_timeout read) entirely.
# See vllm/v1/executor/abstract.py:61 and vllm/v1/executor/ray_executor_v2.py.
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND="${VLLM_USE_RAY_V2_EXECUTOR_BACKEND:-1}"

# Fallbacks. Uncomment if V2 *also* hangs at the first decode step, which
# would mean the real culprit is NCCL-over-Slingshot (not Ray aDAG), and we
# need ALCF-level networking diagnostics rather than vLLM-level knobs.
#   1. Longer aDAG read timeout (only relevant if you disable V2 and go back
#      to the default RayDistributedExecutor):
# export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-1800}"
#   2. Verbose NCCL output to locate the hanging peer pair on hsn0:
# export NCCL_DEBUG=INFO

# =============================================================================
# THREAD LIMITS
# =============================================================================
# Polaris compute nodes have 32 physical CPU cores per node. With TP=4 GPUs
# per node, Ray spawns 4 worker processes plus several helper/driver/actor
# processes per node. If each process lets OpenBLAS/OMP/MKL default to the
# full core count (32), total thread creation quickly exceeds RLIMIT_NPROC
# and you see errors like:
#
#   OpenBLAS blas_thread_init: pthread_create failed for thread 26 of 32:
#     Resource temporarily unavailable
#   OpenBLAS blas_thread_init: RLIMIT_NPROC 2060880 current, 2060880 max
#
# which fail the Ray actor import and cascade into a placement-group hang.
#
# Cap BLAS/OMP threads to a sane per-worker budget. 32 cores / 4 workers = 8
# threads per worker, but we pick a conservative default of 4 to leave
# headroom for driver/actor processes, tokenizers' rayon pool, and torch's
# internal threadpools.
NUM_THREADS_PER_WORKER="${NUM_THREADS_PER_WORKER:-4}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
# HuggingFace tokenizers uses Rayon; cap its pool too.
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
# Match PyTorch's intra-op / inter-op pools.
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"

# =============================================================================
# NODE DISCOVERY FROM PBS
# =============================================================================

if [[ -z "${PBS_NODEFILE:-}" ]]; then
	echo "ERROR: PBS_NODEFILE is not set. This script must be submitted via qsub."
	echo "Usage: qsub qsub_polaris.sh"
	exit 1
fi

# Get unique hostnames preserving PBS allocation order.
# PBS runs this script on the first node, so ALL_NODES[0] is the head.
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
# We MUST use the Slingshot (${RAY_IFNAME}) IP for Ray / VLLM_HOST_IP, not the
# default hostname IP returned by `getent hosts`. On Polaris, `getent hosts
# <node>` typically returns a management-network IP, while NCCL/Gloo traffic
# goes over hsn0 (10.201.x.x). When Ray registers a node under one IP but
# vLLM's EngineCore (which probes its own IP via a UDP socket to 8.8.8.8)
# discovers a different IP, vLLM's placement-group request
# `node:<current_ip>: 0.001` can never be satisfied and you will see:
#
#   Waiting for creating a placement group of specs for 30 seconds.
#   specs=[{'node:10.201.3.102': 0.001, 'GPU': 1.0}, ...]
#
# forever. The fix is to force Ray to register each node under its hsn0 IP
# AND to export VLLM_HOST_IP=<hsn0-ip> everywhere so the two match.

# Return the IPv4 address of the local interface ${RAY_IFNAME}.
local_ifname_ip() {
	local ifname="${1:-${RAY_IFNAME}}"
	# Prefer `ip -4`; fall back to `hostname -I` scanning if `ip` is unavailable.
	if command -v ip &>/dev/null; then
		ip -4 -o addr show dev "${ifname}" 2>/dev/null |
			awk '{print $4}' | cut -d'/' -f1 | head -n1
	else
		# Fallback: look up via /sys/class/net... last resort.
		awk '{print $1}' "/sys/class/net/${ifname}/address" 2>/dev/null || true
	fi
}

# Return the IPv4 address of ${RAY_IFNAME} on a remote host via mpiexec.
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

echo "============================================="
echo "  PBS ${NUM_NODES}-Node DeepSeek-R1 Benchmark"
echo "============================================="
echo "  Job ID:         ${PBS_JOBID:-N/A}"
echo "  Head node:      ${HEAD_HOST} (${HEAD_IP})"
for i in "${!WORKER_HOSTS[@]}"; do
	printf "  Worker %-9s %s (%s)\n" "$((i + 1)):" "${WORKER_HOSTS[$i]}" "${WORKER_IPS[$i]}"
done
echo "  Model:          ${MODEL}"
echo "  Quantization:   ${QUANTIZATION:-none}"
echo "  TP size:        ${TP_SIZE}"
echo "  PP size:        ${PP_SIZE}"
echo "  EP size:        ${EP_SIZE} (derived: TP * DP, DP=1)"
echo "  Expected GPUs:  ${EXPECTED_GPUS}"
echo "  Max model len:  ${MAX_MODEL_LEN}"
echo "  Threads/worker: ${NUM_THREADS_PER_WORKER} (OMP/BLAS/MKL/Rayon)"
echo "  Cache root:     ${CACHE_ROOT}"
echo "  Results dir:    ${RESULTS_DIR}"
echo "============================================="

# =============================================================================
# CLEANUP TRAP
# =============================================================================
# Ensure Ray is stopped on all nodes and background jobs are killed on exit.

WORKER_PIDS=()
SERVER_PID=""

cleanup() {
	echo ""
	echo "Cleaning up..."

	# Kill the vLLM server if running
	if [[ -n "${SERVER_PID}" ]]; then
		echo "  Stopping vLLM server (PID ${SERVER_PID})..."
		kill "${SERVER_PID}" 2>/dev/null || true
		wait "${SERVER_PID}" 2>/dev/null || true
	fi

	# Stop Ray on all nodes
	echo "  Stopping Ray on head node (${HEAD_HOST})..."
	ray stop --force 2>/dev/null || true

	for WHOST in "${WORKER_HOSTS[@]}"; do
		echo "  Stopping Ray on ${WHOST}..."
		mpiexec -n 1 --ppn 1 --hosts "${WHOST}" -- bash -c \
			"source '${VENV_DIR}/bin/activate' && ray stop --force" 2>/dev/null || true
	done

	# Kill background mpiexec worker sessions
	for WPID in "${WORKER_PIDS[@]}"; do
		if [[ -n "${WPID}" ]]; then
			kill "${WPID}" 2>/dev/null || true
			wait "${WPID}" 2>/dev/null || true
		fi
	done

	# Clean up the Ray temp directory on all nodes
	if [[ -n "${RAY_TMPDIR:-}" ]]; then
		echo "  Removing Ray temp dir ${RAY_TMPDIR}..."
		rm -rf "${RAY_TMPDIR}" 2>/dev/null || true
		for WHOST in "${WORKER_HOSTS[@]}"; do
			mpiexec -n 1 --ppn 1 --hosts "${WHOST}" -- bash -c \
				"rm -rf '${RAY_TMPDIR}'" 2>/dev/null || true
		done
	fi

	echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

# =============================================================================
# STEP 1: Start Ray Head Node (on this node)
# =============================================================================

echo ""
echo "[Step 1/5] Starting Ray head node on ${HEAD_HOST} (${HEAD_IP}:${RAY_PORT})..."

# VLLM_HOST_IP MUST match the IP Ray registers the node under (--node-ip-address
# below); otherwise vLLM's placement-group request `node:<VLLM_HOST_IP>: 0.001`
# for the driver bundle will be unsatisfiable.
export VLLM_HOST_IP="${HEAD_IP}"

ray stop --force 2>/dev/null || true
sleep 2

ray start --head \
	--node-ip-address="${HEAD_IP}" \
	--port="${RAY_PORT}" \
	--num-gpus="${GPUS_PER_NODE}" \
	--temp-dir="${RAY_TMPDIR}" \
	--dashboard-host=0.0.0.0

echo "Ray head started. Dashboard: http://${HEAD_IP}:8265"

# Tell vLLM drivers and every subprocess they spawn to ATTACH to this Ray
# cluster rather than fall through to ray.init(address=None) and start a
# brand-new local Ray instance. The CUDA branch of
# vllm/v1/executor/ray_utils.py:initialize_ray_cluster calls
# `ray.init(address=ray_address)` with ray_address=None, which (unlike the
# ROCm/XPU branch that uses address='auto') does NOT auto-discover an
# existing local cluster. Without RAY_ADDRESS, EngineCore logs
# "Started a local Ray instance." and only sees the 4 head-node GPUs.
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
echo "Exported RAY_ADDRESS=${RAY_ADDRESS} for vLLM subprocesses."

# =============================================================================
# STEP 2: Start Ray Workers on Nodes 1..N-1 (via mpiexec)
# =============================================================================

echo ""
echo "[Step 2/5] Starting Ray workers on ${#WORKER_HOSTS[@]} node(s): ${WORKER_HOSTS[*]}"

# Helper: launch a Ray worker on a remote node via mpiexec.
# The worker runs with --block so the mpiexec session stays alive until Ray stops.
launch_worker() {
	local worker_host="$1"
	local worker_ip="$2"
	local worker_label="$3"

	mpiexec -n 1 --ppn 1 --hosts "${worker_host}" -- bash -c "
		set -euo pipefail

		export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'
		export VLLM_HOST_IP='${worker_ip}'
		export RAY_USAGE_STATS_ENABLED=0
		export RAY_TMPDIR='${RAY_TMPDIR}'
		mkdir -p '${RAY_TMPDIR}'

		# Persistent compile caches (shared filesystem, must match head
		# node so FileCacheManager's per-key flock actually serializes
		# across all ranks in the cluster).
		export TRITON_CACHE_DIR='${TRITON_CACHE_DIR}'
		export TORCHINDUCTOR_CACHE_DIR='${TORCHINDUCTOR_CACHE_DIR}'
		export VLLM_CACHE_ROOT='${VLLM_CACHE_ROOT}'
		export TRITON_CACHE_MANAGER='${TRITON_CACHE_MANAGER}'

		# Ray executor backend: keep workers in sync with the head-node env so
		# any worker-side code that re-reads VLLM_USE_RAY_V2_EXECUTOR_BACKEND
		# sees the same value. Selection itself happens on the driver.
		export VLLM_USE_RAY_V2_EXECUTOR_BACKEND='${VLLM_USE_RAY_V2_EXECUTOR_BACKEND}'

		# Thread limits (must match head node; see RLIMIT_NPROC note above).
		export OMP_NUM_THREADS='${OMP_NUM_THREADS}'
		export OPENBLAS_NUM_THREADS='${OPENBLAS_NUM_THREADS}'
		export MKL_NUM_THREADS='${MKL_NUM_THREADS}'
		export NUMEXPR_NUM_THREADS='${NUMEXPR_NUM_THREADS}'
		export VECLIB_MAXIMUM_THREADS='${VECLIB_MAXIMUM_THREADS}'
		export RAYON_NUM_THREADS='${RAYON_NUM_THREADS}'
		export TORCH_NUM_THREADS='${TORCH_NUM_THREADS}'

		# NCCL/Gloo must use the same interface as on the head node.
		export NCCL_SOCKET_IFNAME='${RAY_IFNAME}'
		export GLOO_SOCKET_IFNAME='${RAY_IFNAME}'
		export NCCL_DEBUG='${NCCL_DEBUG}'

		# Load CUDA runtime libraries
		module load cuda/12.9

		# Activate the venv so ray is available on the worker node
		source '${VENV_DIR}/bin/activate'

		echo '[${worker_label}] Stopping any existing Ray processes...'
		ray stop --force 2>/dev/null || true
		sleep 2

		echo '[${worker_label}] Joining Ray cluster at ${HEAD_IP}:${RAY_PORT}...'
		ELAPSED=0
		RETRY_INTERVAL=5
		JOIN_TIMEOUT=300

		while true; do
			if ray start \\
				--address='${HEAD_IP}:${RAY_PORT}' \\
				--node-ip-address='${worker_ip}' \\
				--num-gpus=${GPUS_PER_NODE} \\
				--temp-dir='${RAY_TMPDIR}' \\
				--block; then
				echo '[${worker_label}] Disconnected from cluster.'
				exit 0
			fi

			ELAPSED=\$((ELAPSED + RETRY_INTERVAL))
			if [[ \"\${ELAPSED}\" -ge \"\${JOIN_TIMEOUT}\" ]]; then
				echo '[${worker_label}] ERROR: Timed out joining cluster after \${JOIN_TIMEOUT}s.'
				exit 1
			fi

			echo '[${worker_label}] Retrying in \${RETRY_INTERVAL}s... (\${ELAPSED}s elapsed)'
			sleep \"\${RETRY_INTERVAL}\"
		done
	"
}

# Launch all workers in background
for i in "${!WORKER_HOSTS[@]}"; do
	launch_worker "${WORKER_HOSTS[$i]}" "${WORKER_IPS[$i]}" "Worker$((i + 1))" &
	WORKER_PIDS+=($!)
done

echo "Worker mpiexec sessions launched (PIDs: ${WORKER_PIDS[*]})"

# =============================================================================
# STEP 3: Wait for All Nodes to Join the Cluster
# =============================================================================

echo ""
echo "[Step 3/5] Waiting for ${EXPECTED_NODES} nodes (${EXPECTED_GPUS} GPUs) to join..."

POLL_INTERVAL=5
ELAPSED=0

while true; do
	ACTIVE_NODES=$(python -c "
import ray
ray.init(address='auto', ignore_reinit_error=True)
nodes = ray.nodes()
alive = [n for n in nodes if n['Alive']]
total_gpus = sum(n['Resources'].get('GPU', 0) for n in alive)
print(f'{len(alive)},{int(total_gpus)}')
ray.shutdown()
" 2>/dev/null || echo "0,0")

	LIVE_NODES=$(echo "${ACTIVE_NODES}" | cut -d',' -f1)
	LIVE_GPUS=$(echo "${ACTIVE_NODES}" | cut -d',' -f2)

	echo "  Nodes: ${LIVE_NODES}/${EXPECTED_NODES}, GPUs: ${LIVE_GPUS}/${EXPECTED_GPUS} (${ELAPSED}s elapsed)"

	if [[ "${LIVE_GPUS}" -ge "${EXPECTED_GPUS}" ]]; then
		echo "All nodes joined! Cluster ready with ${LIVE_NODES} nodes and ${LIVE_GPUS} GPUs."
		break
	fi

	if [[ "${ELAPSED}" -ge "${RAY_CLUSTER_TIMEOUT}" ]]; then
		echo "ERROR: Timed out waiting for nodes after ${RAY_CLUSTER_TIMEOUT}s."
		echo "       Got ${LIVE_NODES} nodes / ${LIVE_GPUS} GPUs, expected ${EXPECTED_NODES} / ${EXPECTED_GPUS}."
		exit 1
	fi

	sleep "${POLL_INTERVAL}"
	ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

# =============================================================================
# STEP 4: Run Offline Latency Benchmarks
# =============================================================================

mkdir -p "${RESULTS_DIR}"

echo ""
echo "[Step 4/5] Running offline latency benchmarks..."
echo "  Engine config: PP=${PP_SIZE}, TP=${TP_SIZE}, EP=${EP_SIZE}"
echo ""

ENGINE_ARGS=(
	--model "${MODEL}"
	${QUANT_ARGS[@]+"${QUANT_ARGS[@]}"}
	--dtype "${DTYPE}"
	--tensor-parallel-size "${TP_SIZE}"
	--pipeline-parallel-size "${PP_SIZE}"
	--enable-expert-parallel
	--distributed-executor-backend ray
	--max-model-len "${MAX_MODEL_LEN}"
	--trust-remote-code
)

BATCH_SIZES=(1 2 4 8 16 32)
INPUT_LENS=(128 512 1024)
OUTPUT_LEN=128

for INPUT_LEN in "${INPUT_LENS[@]}"; do
	for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
		RESULT_FILE="${RESULTS_DIR}/latency_bs${BATCH_SIZE}_in${INPUT_LEN}_out${OUTPUT_LEN}.json"
		LOG_FILE="${RESULTS_DIR}/latency_bs${BATCH_SIZE}_in${INPUT_LEN}_out${OUTPUT_LEN}.log"

		echo "  >> Latency: batch_size=${BATCH_SIZE}, input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}"

		vllm bench latency \
			"${ENGINE_ARGS[@]}" \
			--batch-size "${BATCH_SIZE}" \
			--input-len "${INPUT_LEN}" \
			--output-len "${OUTPUT_LEN}" \
			--num-iters-warmup 5 \
			--num-iters 20 \
			--output-json "${RESULT_FILE}" \
			2>&1 | tee "${LOG_FILE}"

		echo "  >> Saved to ${RESULT_FILE}"
		echo ""
	done
done

echo "Offline latency benchmarks complete."
echo ""

# =============================================================================
# STEP 5: Online Serving Benchmark
# =============================================================================

echo "[Step 5/5] Starting vLLM server and running serving benchmarks..."

# Start the vLLM server in the background.
# Use a process group so we can cleanly kill the server and its children.
vllm serve "${MODEL}" \
	${QUANT_ARGS[@]+"${QUANT_ARGS[@]}"} \
	--dtype "${DTYPE}" \
	--tensor-parallel-size "${TP_SIZE}" \
	--pipeline-parallel-size "${PP_SIZE}" \
	--enable-expert-parallel \
	--distributed-executor-backend ray \
	--max-model-len "${MAX_MODEL_LEN}" \
	--trust-remote-code \
	--host 0.0.0.0 \
	--port "${SERVE_PORT}" \
	>"${RESULTS_DIR}/server.log" 2>&1 &

SERVER_PID=$!
echo "vLLM server starting in background (PID ${SERVER_PID})..."

# Wait for server readiness
BASE_URL="http://localhost:${SERVE_PORT}"
echo "Waiting for server to be ready at ${BASE_URL}/v1/models ..."

MAX_WAIT=600
ELAPSED=0
SERVER_POLL_INTERVAL=10

while true; do
	# Check if the server process is still alive
	if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
		echo "ERROR: vLLM server process died unexpectedly."
		wait "${SERVER_PID}" || true
		exit 1
	fi

	if curl -s --max-time 5 "${BASE_URL}/v1/models" >/dev/null 2>&1; then
		echo "Server is ready!"
		break
	fi

	ELAPSED=$((ELAPSED + SERVER_POLL_INTERVAL))
	if [[ "${ELAPSED}" -ge "${MAX_WAIT}" ]]; then
		echo "ERROR: Server not ready after ${MAX_WAIT}s."
		exit 1
	fi

	echo "  Not ready yet... retrying in ${SERVER_POLL_INTERVAL}s (${ELAPSED}s elapsed)"
	sleep "${SERVER_POLL_INTERVAL}"
done

# --- Run the serving benchmark sweep ---

REQUEST_RATES=(1 2 4 8 16 32 inf)
IO_CONFIGS=(
	"128:128"
	"512:128"
	"1024:128"
	"128:512"
	"512:512"
)

echo ""
echo "Starting serving benchmark sweep..."
echo ""

for IO_CONFIG in "${IO_CONFIGS[@]}"; do
	INPUT_LEN="${IO_CONFIG%%:*}"
	OUTPUT_LEN="${IO_CONFIG##*:}"

	for REQUEST_RATE in "${REQUEST_RATES[@]}"; do
		RR_LABEL="${REQUEST_RATE}"

		RESULT_LABEL="serving_in${INPUT_LEN}_out${OUTPUT_LEN}_rr${RR_LABEL}"
		RESULT_FILE="${RESULTS_DIR}/${RESULT_LABEL}.json"
		LOG_FILE="${RESULTS_DIR}/${RESULT_LABEL}.log"

		echo ">> Serving: input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}, request_rate=${REQUEST_RATE}"

		vllm bench serve \
			--backend openai \
			--base-url "${BASE_URL}" \
			--endpoint /v1/completions \
			--model "${MODEL}" \
			--dataset-name random \
			--input-len "${INPUT_LEN}" \
			--output-len "${OUTPUT_LEN}" \
			--num-prompts "${NUM_PROMPTS}" \
			--num-warmups "${NUM_WARMUPS}" \
			--request-rate "${REQUEST_RATE}" \
			--ignore-eos \
			--percentile-metrics ttft,tpot,itl,e2el \
			--metric-percentiles 50,90,95,99 \
			--save-result \
			--result-dir "${RESULTS_DIR}" \
			--result-filename "${RESULT_LABEL}.json" \
			--metadata \
			tp="${TP_SIZE}" \
			pp="${PP_SIZE}" \
			ep="${EP_SIZE}" \
			quantization="${QUANTIZATION:-none}" \
			input_len="${INPUT_LEN}" \
			output_len="${OUTPUT_LEN}" \
			request_rate="${REQUEST_RATE}" \
			2>&1 | tee "${LOG_FILE}"

		echo ">> Saved to ${RESULT_FILE}"
		echo ""
	done
done

# =============================================================================
# SUMMARY
# =============================================================================

echo "============================================="
echo "  All benchmarks complete!"
echo "============================================="
echo ""
echo "Job ID:      ${PBS_JOBID:-N/A}"
echo "Results in:  ${RESULTS_DIR}/"
echo ""
echo "Latency results:"
ls -1 "${RESULTS_DIR}"/latency_*.json 2>/dev/null || echo "  (no latency JSON files)"
echo ""
echo "Serving results:"
ls -1 "${RESULTS_DIR}"/serving_*.json 2>/dev/null || echo "  (no serving JSON files)"
echo ""
echo "Metrics recorded:"
echo "  - Offline: end-to-end latency per batch size / input length"
echo "  - Online:  TTFT, TPOT, ITL, E2EL at p50/p90/p95/p99"
echo "  - Online:  request throughput, token throughput"
echo ""
echo "Cleanup will run automatically via EXIT trap."

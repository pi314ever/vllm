#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# =============================================================================
# Shared body (MULTIPROCESSING backend) for PBS Pro benchmark wrappers on
# ALCF Polaris
# =============================================================================
#
# This file is a SOURCED LIBRARY, not a submittable script. It is the
# multiprocessing-backend counterpart to qsub_polaris_body.sh (Ray).
# Wrappers pick which body to source via `BACKEND=ray|mp`:
#   qsub_polaris_deepseek_v3_latency.sh
#   qsub_polaris_deepseek_v3_serving.sh
#   qsub_polaris_deepseek_v3_throughput.sh
#
# Why a separate file instead of branching the Ray body:
#   The Ray body does ~600 lines of cluster bring-up (head start, worker
#   mpiexec join loops, ray.nodes() poll) and Ray-idle-worker tuning that
#   are irrelevant under mp. Forking keeps each file focused. The
#   diagnostic / sampler / thread-limit infrastructure is duplicated
#   VERBATIM between the two files; keep them in sync when editing.
#
# Responsibilities of this body:
#   1. Discover allocated nodes from PBS_NODEFILE.
#   2. Per bench invocation: launch `vllm serve --headless` on every
#      worker node via mpiexec, then run the head-side bench/serve
#      command with matching `--nnodes/--node-rank/--master-addr/
#      --master-port`. torch.distributed rendezvous brings the group
#      online; when the head command returns, workers' MultiprocExecutor
#      monitor loop detects the broken process group and exits.
#   3. Optionally run offline latency benchmarks (Step 4; RUN_LATENCY=1).
#   4. Optionally start vLLM API server and run serving benchmarks
#      (Step 5; RUN_SERVING=1). Workers stay headless for the full sweep.
#   5. Optionally run offline throughput benchmarks (Step 6; RUN_THROUGHPUT=1).
#   6. Clean up any lingering worker vllm processes on exit.
#   7. Optionally wrap every bench invocation with the torch profiler
#      (PROFILE=1). Same knobs as the Ray body.
#
# Multi-node mp pattern (from
# https://docs.vllm.ai/en/stable/serving/parallelism_scaling/#running-vllm-with-multiprocessing):
#
#   Head (node_rank 0):
#     vllm bench latency ... \
#       --distributed-executor-backend mp \
#       --nnodes N --node-rank 0 \
#       --master-addr <HEAD_IP> --master-port <PORT>
#
#   Workers (node_rank 1..N-1):
#     vllm serve <model> ... --headless \
#       --distributed-executor-backend mp \
#       --nnodes N --node-rank I \
#       --master-addr <HEAD_IP> --master-port <PORT>
#
#   All engine args (model, quantization, dtype, TP, PP, MAX_MODEL_LEN,
#   --trust-remote-code, --enforce-eager, --enable-expert-parallel) MUST
#   match between head and workers or the MultiprocExecutor workers will
#   fail the VllmConfig consistency check at join time.
#
# Trade-off vs. Ray body:
#   + No persistent Ray cluster; simpler teardown (kill one mpiexec per
#     worker).
#   + No ray::IDLE pre-spawn pool; less cgroup pids pressure at baseline.
#   + No Ray actor RPC layer; Compiled Graph timeouts impossible.
#   - EACH offline bench invocation (Steps 4 and 6) relaunches workers
#     from scratch (model load every time). Ray reused the same actors
#     but the `vllm bench latency` CLI already recreates the LLM per
#     invocation, so the actual wall-clock delta is small (mpiexec
#     spawn vs. Ray actor create is negligible next to 671B model load).
#   - Step 5 (serving) keeps workers alive for the full sweep, so it
#     has the SAME amortization as Ray.
#
# Parallelism model (defaults): same as Ray body.
#   NUM_NODES=6, GPUS_PER_NODE=4  -> 24 GPUs total
#   PP_SIZE=6, TP_SIZE=4          -> PP across nodes, TP within each node
#   --enable-expert-parallel      -> EP size = TP_SIZE * DP_SIZE = 4 (DP=1)
#
# Hardware:      Polaris nodes x 4x A100-80GB
# Default model: deepseek-ai/DeepSeek-R1 (wrappers override).
#
# Usage (from a wrapper that sets BACKEND=mp before sourcing this file):
#   export BACKEND=mp
#   qsub qsub_polaris_deepseek_v3_latency.sh
#
# =============================================================================

set -euo pipefail

# =============================================================================
# DIRECT-SUBMIT GUARD
# =============================================================================
# Mirror of the guard in qsub_polaris_body.sh. This file has no #PBS
# directives and must not be submitted directly.
if [[ "${BASH_SOURCE[0]}" == "${0}" && -z "${PBS_NODEFILE:-}" ]]; then
	echo "ERROR: qsub_polaris_body_mp.sh is a sourced library, not a"
	echo "       submittable script. Submit one of the wrapper files with"
	echo "       BACKEND=mp set, e.g.:"
	echo "         qsub -v BACKEND=mp qsub_polaris_deepseek_v3_latency.sh"
	echo "         qsub -v BACKEND=mp qsub_polaris_deepseek_v3_serving.sh"
	echo "         qsub -v BACKEND=mp qsub_polaris_deepseek_v3_throughput.sh"
	exit 1
fi

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"

# Load CUDA runtime libraries (required for libcudart.so on compute nodes)
module load cuda/12.9

# =============================================================================
# PYTHON ENVIRONMENT SETUP
# =============================================================================
# Identical to the Ray body: ensure uv is available, create a venv if
# needed, activate it. vllm (+ any deps) is expected to already be
# installed in the venv.

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

echo "Environment ready: $(python --version), vllm $(vllm --version 2>/dev/null || echo 'unknown')"

# =============================================================================
# PERSISTENT COMPILE CACHES (Triton / Inductor / vLLM)
# =============================================================================
# Verbatim from the Ray body. First-run JIT compilation is the main
# source of process pressure during the initial decode step; parking
# caches on /grand means second and later jobs skip gcc entirely.
# Hardcoded to /grand/Intel/dhuang because this script is expected to be
# used by dhuang. Change if another user adopts it.
CACHE_ROOT="/grand/Intel/dhuang/vllm_polaris_cache"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"

mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}"

# Serialize concurrent compilation through Triton's file-lock cache
# manager. Same rationale as the Ray body.
export TRITON_CACHE_MANAGER="${TRITON_CACHE_MANAGER:-triton.runtime.cache:FileCacheManager}"

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Model configuration
MODEL="${MODEL:-/grand/Intel/dhuang/DeepSeek-V3/}"
QUANTIZATION="${QUANTIZATION-}" # e.g. "fp8"; empty (default) disables --quantization
DTYPE="${DTYPE:-auto}"

# Build conditional quantization args (used by ENGINE_ARGS, MP_COMMON_ENGINE_ARGS, vllm serve)
QUANT_ARGS=()
if [[ -n "${QUANTIZATION}" ]]; then
	QUANT_ARGS=(--quantization "${QUANTIZATION}")
fi

# Parallelism / cluster sizing. Same semantics as the Ray body.
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
NUM_NODES="${NUM_NODES:-6}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-6}"

EXPECTED_GPUS=$((NUM_NODES * GPUS_PER_NODE))
EP_SIZE="${TP_SIZE}" # derived; only valid while DP=1

# Sanity check: total GPU count must match PP * TP.
if (( PP_SIZE * TP_SIZE != EXPECTED_GPUS )); then
	echo "ERROR: Invalid parallelism configuration."
	echo "       PP_SIZE (${PP_SIZE}) * TP_SIZE (${TP_SIZE}) = $((PP_SIZE * TP_SIZE))"
	echo "       NUM_NODES (${NUM_NODES}) * GPUS_PER_NODE (${GPUS_PER_NODE}) = ${EXPECTED_GPUS}"
	echo "       These must be equal."
	exit 1
fi

# =============================================================================
# MULTIPROCESSING EXECUTOR CONFIG (replaces the RAY EXECUTOR BACKEND block)
# =============================================================================
# vLLM's MultiprocExecutor uses torch.distributed for cross-node
# coordination (TCPStore + gloo/NCCL process groups). Unlike Ray, there
# is no persistent cluster daemon; each bench invocation's engine
# rendezvouses via <MP_MASTER_ADDR>:<MP_MASTER_PORT> at startup and tears
# down at engine shutdown.
#
# MP_MASTER_PORT: torch.distributed rendezvous port. 29501 matches
# vLLM's ParallelConfig.master_port default (vllm/config/parallel.py:256)
# so we don't have to override it on the command line. Bound on the head
# node's hsn0 IP; workers connect as clients. Change if the port is
# already in use on your node or if you need to run multiple concurrent
# vllm jobs on the same allocation.
MP_MASTER_PORT="${MP_MASTER_PORT:-29501}"

# How long (seconds) to wait for each headless worker to print its
# torch.distributed rendezvous-ready log line before we declare the
# worker dead and abort the run. 600s is generous enough for a cold
# Inductor/Triton cache on DeepSeek-671B; a warm cache run typically
# hits "Launching vLLM ... headless multiproc executor" within ~60s.
MP_WORKER_READY_TIMEOUT="${MP_WORKER_READY_TIMEOUT:-600}"

# How long (seconds) to give each headless worker to exit on its own
# after the head-side bench command returns (torch.distributed tears
# down, MultiprocExecutor's worker monitor detects the broken group and
# calls sys.exit). If the worker is still alive after this, we send
# SIGTERM, then SIGKILL, then fall back to a per-node `pkill -f "vllm
# serve.*--headless"`.
MP_WORKER_SHUTDOWN_TIMEOUT="${MP_WORKER_SHUTDOWN_TIMEOUT:-60}"

# Per-run log/output namespace.
#   logs/<job_name>/run.log         -> all stdout/stderr from this script
#   logs/<job_name>/results/        -> RESULTS_DIR (benchmark outputs + diag)
#   logs/<job_name>/results/workers -> per-node vllm serve --headless logs
#
# Short, filesystem-friendly PBS job id (no host suffix). Same plumbing
# as the Ray body; exported so remote mpiexec subshells can read it for
# cgroup resolution without re-parsing $PBS_JOBID.
PBS_SHORT_JOBID="${PBS_JOBID%%.*}"
PBS_SHORT_JOBID="${PBS_SHORT_JOBID:-manual}"
export PBS_SHORT_JOBID

JOB_NAME="${JOB_NAME:-${PBS_JOBNAME:-vllm-bench-mp}.o${PBS_SHORT_JOBID}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${PBS_O_WORKDIR:-.}/logs/${JOB_NAME}}"
mkdir -p "${RUN_LOG_DIR}"

# Tee every byte written from this point forward to run.log (same
# rationale as the Ray body).
RUN_LOG_FILE="${RUN_LOG_DIR}/run.log"
exec > >(tee -a "${RUN_LOG_FILE}") 2>&1
echo "Run log teeing to: ${RUN_LOG_FILE}"

# Benchmark output directory.
RESULTS_DIR="${RESULTS_DIR:-${RUN_LOG_DIR}/results}"

# Diagnostics subdirectory for per-phase, per-node thread/pid dumps.
DIAG_DIR="${RESULTS_DIR}/diagnostics"
mkdir -p "${DIAG_DIR}"

# Per-node vllm serve --headless worker logs.
MP_WORKER_LOG_DIR="${RESULTS_DIR}/workers"
mkdir -p "${MP_WORKER_LOG_DIR}"

# Background-sampler kill-switch flag.
SAMPLER_FLAG="${DIAG_DIR}/.sampler_run"

# Server port for online serving benchmark.
SERVE_PORT="${SERVE_PORT:-8000}"

# Max model length.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Number of prompts / warmups for serving benchmark. Same sizing as the
# Ray body.
NUM_PROMPTS="${NUM_PROMPTS:-200}"
NUM_WARMUPS="${NUM_WARMUPS:-2}"

# GPU visibility: expose GPUS_PER_NODE devices per node (0,1,...,N-1)
_default_cvd="$(seq -s, 0 $((GPUS_PER_NODE - 1)))"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${_default_cvd}}"
unset _default_cvd

# =============================================================================
# BENCHMARK STEP GATES
# =============================================================================
# Identical semantics to the Ray body. Each step is independently
# gateable via RUN_LATENCY / RUN_SERVING / RUN_THROUGHPUT.
RUN_LATENCY="${RUN_LATENCY:-1}"
RUN_SERVING="${RUN_SERVING:-1}"
RUN_THROUGHPUT="${RUN_THROUGHPUT:-0}"

# =============================================================================
# PROFILING CONFIG
# =============================================================================
# Same knobs as the Ray body. See qsub_polaris_body.sh for the full
# rationale, trade-offs, and cost estimates. Only mp-specific difference:
# mp workers ARE torch processes too, so per-rank .pt.trace.json.gz files
# are emitted from the same MultiprocExecutor worker subprocesses the
# bench head talks to. No additional plumbing.
PROFILE="${PROFILE:-0}"
PROFILE_DIR="${PROFILE_DIR:-/grand/Intel/dhuang/vllm_polaris_profiles/${JOB_NAME}}"
PROFILE_NUM_PROMPTS="${PROFILE_NUM_PROMPTS:-5}"
PROFILE_LATENCY_WARMUP_ITERS="${PROFILE_LATENCY_WARMUP_ITERS:-2}"

if [[ "${PROFILE}" == "1" ]]; then
	# Bump internal RPC timeout so `vllm bench serve --profile` does
	# not abort while `/stop_profile` flushes per-rank traces.
	export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
fi

# =============================================================================
# NETWORK CONFIGURATION (Polaris Slingshot)
# =============================================================================
# Same as the Ray body. The interface is used both for NCCL/Gloo (intra-
# engine all-reduce + PP send/recv) AND for torch.distributed rendezvous
# over MP_MASTER_PORT.
RAY_IFNAME="${RAY_IFNAME:-hsn0}"

export NCCL_SOCKET_IFNAME="${RAY_IFNAME}"
export GLOO_SOCKET_IFNAME="${RAY_IFNAME}"
export NCCL_DEBUG=WARN

# =============================================================================
# THREAD LIMITS
# =============================================================================
# Verbatim from the Ray body. The cgroup pids.max cap is still 4096 on
# Polaris regardless of backend; uncapped OMP/BLAS defaults still blow
# the budget.
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
# Verbatim from the Ray body. The Inductor 128-way gcc fan-out and XALT
# ld wrapper both fire regardless of distributed backend.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export XALT_EXECUTABLE_TRACKING="${XALT_EXECUTABLE_TRACKING:-no}"

# =============================================================================
# ADDITIONAL THREAD-COUNT REDUCTION (targets cgroup pids.max pressure)
# =============================================================================
# Same knobs as the Ray body, minus the Ray-specific
# RAY_memory_monitor_refresh_ms and RAY_num_workers_soft_limit (the
# ray::IDLE pool doesn't exist in mp mode).

# NCCL socket-thread fan-out cap. Revert for production throughput runs.
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-1}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-1}"

# HuggingFace tokenizers Rayon pool.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# vLLM / HuggingFace telemetry: skip background usage-stats upload thread.
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"

# =============================================================================
# PYTORCH NCCL PER-PG THREAD MINIMIZATION
# =============================================================================
# Same as the Ray body. isend_tensor_dict / irecv_tensor_dict still calls
# into NCCL's size-N PP ProcessGroup and each unique (src, dst) pair
# still spawns a 2-rank subcomm under lazy init. The Ray-vs-mp choice
# doesn't affect this code path.
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
# IP resolution (same plumbing as the Ray body)
# -----------------------------------------------------------------------------
# We MUST use the Slingshot (${RAY_IFNAME}) IP for MP_MASTER_ADDR and
# VLLM_HOST_IP, not the default hostname IP returned by `getent hosts`.
# torch.distributed's TCPStore on the head listens on MP_MASTER_ADDR;
# workers dial in over the same IP. Mismatching the interface here
# strands workers in their torch.distributed rendezvous.

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

# VLLM_HOST_IP identifies this process's local interface IP for
# intra-engine IPC. Head-side exports; per-worker mpiexec commands
# export their own local IP.
export VLLM_HOST_IP="${HEAD_IP}"

echo "============================================="
echo "  PBS ${NUM_NODES}-Node Multiprocessing Benchmark"
echo "============================================="
echo "  Job ID:         ${PBS_JOBID:-N/A}"
echo "  Backend:        mp (MultiprocExecutor + torch.distributed)"
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
echo "  MP master:      ${HEAD_IP}:${MP_MASTER_PORT}"
echo "  Cache root:     ${CACHE_ROOT}"
echo "  Run log:        ${RUN_LOG_FILE}"
echo "  Results dir:    ${RESULTS_DIR}"
echo "============================================="

# =============================================================================
# CLEANUP TRAP
# =============================================================================
# Ensure headless workers and background jobs are killed on exit.
#
# Unlike the Ray body there's no persistent cluster to `ray stop`. The
# teardown is strictly:
#   1. Stop the background sampler (flush CSV rows).
#   2. Kill the foreground vLLM server (Step 5) if running.
#   3. Kill any tracked mp worker mpiexec sessions (both foreground and
#      the server-side background workers).
#   4. Belt-and-suspenders: `pkill -f "vllm serve.*--headless"` on every
#      worker host in case an mpiexec session exited cleanly but left
#      its vllm child behind.

# Tracked PIDs of mpiexec-launched headless workers for the CURRENT
# engine invocation. stop_mp_workers() resets this array.
MP_WORKER_PIDS=()

# Tracked PIDs of mpiexec-launched headless workers for the VLLM SERVE
# session (Step 5). Separate from MP_WORKER_PIDS so the cleanup path
# can tear both down if e.g. the bench fails mid-sweep.
MP_SERVE_WORKER_PIDS=()

SERVER_PID=""

cleanup() {
	echo ""
	echo "Cleaning up..."

	# Stop the background sampler first so its final CSV rows land
	# before the rest of the teardown creates noise.
	stop_background_sampler 2>/dev/null || true

	# Kill the vLLM server if running.
	if [[ -n "${SERVER_PID}" ]]; then
		echo "  Stopping vLLM server (PID ${SERVER_PID})..."
		kill "${SERVER_PID}" 2>/dev/null || true
		wait "${SERVER_PID}" 2>/dev/null || true
	fi

	# Kill any tracked mp worker mpiexec sessions.
	local pid
	for pid in "${MP_WORKER_PIDS[@]:-}" "${MP_SERVE_WORKER_PIDS[@]:-}"; do
		if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
			kill "${pid}" 2>/dev/null || true
		fi
	done
	for pid in "${MP_WORKER_PIDS[@]:-}" "${MP_SERVE_WORKER_PIDS[@]:-}"; do
		if [[ -n "${pid}" ]]; then
			wait "${pid}" 2>/dev/null || true
		fi
	done

	# Belt-and-suspenders: pkill any lingering `vllm serve --headless`
	# on worker hosts. pkill returns 1 when nothing matches; swallow.
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
# PROCESS-LIMIT DIAGNOSTICS (read-only)
# =============================================================================
# Ported VERBATIM from qsub_polaris_body.sh so cross-run diffs (Ray vs.
# mp) line up exactly. Keep the two functions in sync: any edit here
# should be mirrored in the Ray body and vice versa.
#
# The "--- Ray component PID tally ---" section is retained so the
# output columns match across Ray and mp runs; under mp the ray::IDLE /
# default_worker.py / raylet / gcs_server rows will all be 0, which
# is itself a useful diagnostic (confirms no stray Ray processes from
# a prior job on the same node).
#
# Phase labels (called from the all-nodes wrapper):
#   - 00_job_start            (very first snapshot, before anything runs)
#   - 01_pre_setup            (after env/module setup, before any mp work)
#   - 04_pre_engine           (start of Step 4, immediately before vllm bench)
#   - 05_pre_vllm_serve       (start of Step 5)
#   - FAILURE_rc<N>           (from the ERR trap on any command failure)
print_process_diagnostics() {
	local label="${1:-diagnostics}"
	local hostname
	hostname="$(hostname 2>/dev/null || echo unknown)"

	local safe_label
	safe_label="$(printf '%s' "${label}" | tr -c '[:alnum:]._-' '_' | sed -E 's/_+/_/g; s/^_//; s/_$//')"
	local out_file=""
	if [[ -n "${DIAG_DIR:-}" && -d "${DIAG_DIR}" ]]; then
		out_file="${DIAG_DIR}/${safe_label}_${hostname}.txt"
	fi

	{
		echo ""
		echo "=== Process limit diagnostics: ${label} (host ${hostname}) ==="

		echo "  ulimit -u  (soft nproc): $(ulimit -u 2>/dev/null || echo unknown)"
		echo "  ulimit -Hu (hard nproc): $(ulimit -Hu 2>/dev/null || echo unknown)"
		echo "  ulimit -s  (stack KB):   $(ulimit -s 2>/dev/null || echo unknown)"

		if [[ -r /proc/sys/kernel/pid_max ]]; then
			echo "  kernel pid_max:          $(cat /proc/sys/kernel/pid_max)"
		fi
		if [[ -r /proc/sys/kernel/threads-max ]]; then
			echo "  kernel threads-max:      $(cat /proc/sys/kernel/threads-max)"
		fi

		# Polaris-specific fast path: /sys/fs/cgroup/jobs/<short_jobid>/.
		# See the Ray body's print_process_diagnostics for full cgroup
		# resolution rationale (why we prefer this path over the
		# /proc/self/cgroup walk).
		local pids_max_file=""
		local pids_cur_file=""
		if [[ -n "${PBS_SHORT_JOBID:-}" && -r "/sys/fs/cgroup/jobs/${PBS_SHORT_JOBID}/pids.max" ]]; then
			pids_max_file="/sys/fs/cgroup/jobs/${PBS_SHORT_JOBID}/pids.max"
			pids_cur_file="/sys/fs/cgroup/jobs/${PBS_SHORT_JOBID}/pids.current"
		fi

		local cgpath=""
		if [[ -r /proc/self/cgroup ]]; then
			cgpath="$(awk -F: '$1 == "0" {print $3; exit}' /proc/self/cgroup)"
			if [[ -z "${cgpath}" ]]; then
				cgpath="$(awk -F: '$2 == "pids" {print $3; exit}' /proc/self/cgroup)"
			fi
		fi

		if [[ -z "${pids_max_file}" && -n "${cgpath}" ]]; then
			local probe="${cgpath}"
			while true; do
				if [[ -r "/sys/fs/cgroup${probe}/pids.max" ]]; then
					pids_max_file="/sys/fs/cgroup${probe}/pids.max"
					pids_cur_file="/sys/fs/cgroup${probe}/pids.current"
					break
				fi
				if [[ -r "/sys/fs/cgroup/pids${probe}/pids.max" ]]; then
					pids_max_file="/sys/fs/cgroup/pids${probe}/pids.max"
					pids_cur_file="/sys/fs/cgroup/pids${probe}/pids.current"
					break
				fi
				if [[ -z "${probe}" || "${probe}" == "/" ]]; then
					break
				fi
				probe="${probe%/*}"
			done
		fi
		if [[ -z "${pids_max_file}" ]]; then
			if [[ -r /sys/fs/cgroup/pids.max ]]; then
				pids_max_file="/sys/fs/cgroup/pids.max"
				pids_cur_file="/sys/fs/cgroup/pids.current"
			elif [[ -r /sys/fs/cgroup/pids/pids.max ]]; then
				pids_max_file="/sys/fs/cgroup/pids/pids.max"
				pids_cur_file="/sys/fs/cgroup/pids/pids.current"
			fi
		fi

		local pids_max_val="unknown"
		local pids_cur_val="unknown"
		if [[ -n "${pids_max_file}" ]]; then
			pids_max_val="$(cat "${pids_max_file}" 2>/dev/null || echo unknown)"
			pids_cur_val="$(cat "${pids_cur_file}" 2>/dev/null || echo unknown)"
			echo "  cgroup pids.max:         ${pids_max_val} (${pids_max_file})"
			echo "  cgroup pids.current:     ${pids_cur_val}"
		else
			echo "  cgroup pids.max:         not found (cgpath='${cgpath:-none}')"
		fi
		if [[ -n "${cgpath}" ]]; then
			echo "  /proc/self/cgroup path:  ${cgpath}"
		fi

		local pid_count thread_count
		pid_count="$(ps -u "${USER}" --no-headers 2>/dev/null | wc -l)"
		thread_count="$(ps -u "${USER}" -L --no-headers 2>/dev/null | wc -l || echo unknown)"
		echo "  pids (uid=${USER}):      ${pid_count}"
		echo "  threads (uid=${USER}):   ${thread_count}"
		echo "  nproc:                   $(nproc 2>/dev/null || echo unknown)"

		# Ray component tally (kept for cross-backend diff-ability;
		# all Ray rows read 0 under mp). Also tallies "vllm serve"
		# headless workers as a sanity check.
		echo "  --- Ray/mp component PID tally ---"
		local ps_args_snapshot
		ps_args_snapshot="$(ps -u "${USER}" -o pid,args --no-headers 2>/dev/null || true)"
		local -a tally_patterns=(
			"ray::IDLE"
			"default_worker.py"
			"raylet"
			"gcs_server"
			"plasma_store"
			"log_monitor"
			"dashboard_agent"
			"runtime_env_agent"
			"ray::"
			"vllm serve"
			"vllm bench"
			"VllmWorker"
			"vllm"
			"python"
		)
		local pat count
		for pat in "${tally_patterns[@]}"; do
			count="$(printf '%s\n' "${ps_args_snapshot}" | grep -Fc -- "${pat}" 2>/dev/null || true)"
			count="${count:-0}"
			printf "    %-24s %s\n" "${pat}" "${count}"
		done

		echo "  --- Top 40 processes by thread count (pid nlwp rss comm args) ---"
		ps -eLo pid,nlwp,rss,comm,args --no-headers -u "${USER}" 2>/dev/null |
			awk '!seen[$1]++' |
			sort -k2 -nr |
			head -40 |
			sed 's/^/    /'

		local avg_threads="0"
		if [[ "${pid_count}" -gt 0 ]] 2>/dev/null; then
			avg_threads="$(awk -v t="${thread_count}" -v p="${pid_count}" 'BEGIN{if(p>0) printf "%.1f", t/p; else print "0"}')"
		fi
		echo "  SUMMARY: label=${label} host=${hostname} pids=${pid_count} threads=${thread_count} avg=${avg_threads} cgroup_pids.current=${pids_cur_val}/${pids_max_val}"
		echo "=============================================================="
		echo ""
	} 2>&1 | if [[ -n "${out_file}" ]]; then
		tee -a "${out_file}"
	else
		cat
	fi
}

# -----------------------------------------------------------------------------
# print_process_diagnostics_all_nodes <label>
# -----------------------------------------------------------------------------
# Ported verbatim from qsub_polaris_body.sh (same function, same inline
# remote-bash snippet). Fans print_process_diagnostics out to every node
# via mpiexec.
print_process_diagnostics_all_nodes() {
	local label="${1:-all-nodes}"

	if [[ -z "${HEAD_HOST:-}" ]]; then
		print_process_diagnostics "${label}"
		return 0
	fi

	local hosts="${HEAD_HOST}"
	local h
	for h in "${WORKER_HOSTS[@]:-}"; do
		if [[ -n "${h}" ]]; then
			hosts+=",${h}"
		fi
	done

	local n_hosts
	n_hosts="$(awk -F, '{print NF}' <<<"${hosts}")"

	local q_label q_diag_dir q_user q_short_jobid
	q_label="$(printf '%q' "${label}")"
	q_diag_dir="$(printf '%q' "${DIAG_DIR}")"
	q_user="$(printf '%q' "${USER}")"
	q_short_jobid="$(printf '%q' "${PBS_SHORT_JOBID:-}")"

	# shellcheck disable=SC2016  # single quotes intentional; vars are
	# expanded in the outer shell via the concatenation below.
	local remote_script='
export DIAG_DIR='"${q_diag_dir}"'
export USER='"${q_user}"'
export REMOTE_LABEL='"${q_label}"'
export PBS_SHORT_JOBID='"${q_short_jobid}"'
# Minimal inline copy of print_process_diagnostics (kept in sync with
# the driver-side function manually; same copy as in qsub_polaris_body.sh).
print_process_diagnostics() {
	local label="${1:-diagnostics}"
	local hostname
	hostname="$(hostname 2>/dev/null || echo unknown)"
	local safe_label
	safe_label="$(printf "%s" "${label}" | tr -c "[:alnum:]._-" "_" | sed -E "s/_+/_/g; s/^_//; s/_$//")"
	local out_file=""
	if [[ -n "${DIAG_DIR:-}" && -d "${DIAG_DIR}" ]]; then
		out_file="${DIAG_DIR}/${safe_label}_${hostname}.txt"
	fi
	{
		echo ""
		echo "=== Process limit diagnostics: ${label} (host ${hostname}) ==="
		echo "  ulimit -u  (soft nproc): $(ulimit -u 2>/dev/null || echo unknown)"
		echo "  ulimit -Hu (hard nproc): $(ulimit -Hu 2>/dev/null || echo unknown)"
		echo "  ulimit -s  (stack KB):   $(ulimit -s 2>/dev/null || echo unknown)"
		if [[ -r /proc/sys/kernel/pid_max ]]; then
			echo "  kernel pid_max:          $(cat /proc/sys/kernel/pid_max)"
		fi
		if [[ -r /proc/sys/kernel/threads-max ]]; then
			echo "  kernel threads-max:      $(cat /proc/sys/kernel/threads-max)"
		fi
		local pids_max_file="" pids_cur_file=""
		if [[ -n "${PBS_SHORT_JOBID:-}" && -r "/sys/fs/cgroup/jobs/${PBS_SHORT_JOBID}/pids.max" ]]; then
			pids_max_file="/sys/fs/cgroup/jobs/${PBS_SHORT_JOBID}/pids.max"
			pids_cur_file="/sys/fs/cgroup/jobs/${PBS_SHORT_JOBID}/pids.current"
		fi
		local cgpath=""
		if [[ -r /proc/self/cgroup ]]; then
			cgpath="$(awk -F: "\$1 == \"0\" {print \$3; exit}" /proc/self/cgroup)"
			if [[ -z "${cgpath}" ]]; then
				cgpath="$(awk -F: "\$2 == \"pids\" {print \$3; exit}" /proc/self/cgroup)"
			fi
		fi
		if [[ -z "${pids_max_file}" && -n "${cgpath}" ]]; then
			local probe="${cgpath}"
			while true; do
				if [[ -r "/sys/fs/cgroup${probe}/pids.max" ]]; then
					pids_max_file="/sys/fs/cgroup${probe}/pids.max"
					pids_cur_file="/sys/fs/cgroup${probe}/pids.current"
					break
				fi
				if [[ -r "/sys/fs/cgroup/pids${probe}/pids.max" ]]; then
					pids_max_file="/sys/fs/cgroup/pids${probe}/pids.max"
					pids_cur_file="/sys/fs/cgroup/pids${probe}/pids.current"
					break
				fi
				if [[ -z "${probe}" || "${probe}" == "/" ]]; then break; fi
				probe="${probe%/*}"
			done
		fi
		if [[ -z "${pids_max_file}" ]]; then
			if [[ -r /sys/fs/cgroup/pids.max ]]; then
				pids_max_file="/sys/fs/cgroup/pids.max"
				pids_cur_file="/sys/fs/cgroup/pids.current"
			elif [[ -r /sys/fs/cgroup/pids/pids.max ]]; then
				pids_max_file="/sys/fs/cgroup/pids/pids.max"
				pids_cur_file="/sys/fs/cgroup/pids/pids.current"
			fi
		fi
		local pids_max_val="unknown" pids_cur_val="unknown"
		if [[ -n "${pids_max_file}" ]]; then
			pids_max_val="$(cat "${pids_max_file}" 2>/dev/null || echo unknown)"
			pids_cur_val="$(cat "${pids_cur_file}" 2>/dev/null || echo unknown)"
			echo "  cgroup pids.max:         ${pids_max_val} (${pids_max_file})"
			echo "  cgroup pids.current:     ${pids_cur_val}"
		else
			echo "  cgroup pids.max:         not found (cgpath=${cgpath:-none})"
		fi
		if [[ -n "${cgpath}" ]]; then
			echo "  /proc/self/cgroup path:  ${cgpath}"
		fi
		local pid_count thread_count
		pid_count="$(ps -u "${USER}" --no-headers 2>/dev/null | wc -l)"
		thread_count="$(ps -u "${USER}" -L --no-headers 2>/dev/null | wc -l || echo unknown)"
		echo "  pids (uid=${USER}):      ${pid_count}"
		echo "  threads (uid=${USER}):   ${thread_count}"
		echo "  nproc:                   $(nproc 2>/dev/null || echo unknown)"
		echo "  --- Ray/mp component PID tally ---"
		local ps_args_snapshot
		ps_args_snapshot="$(ps -u "${USER}" -o pid,args --no-headers 2>/dev/null || true)"
		local tally_patterns=(ray::IDLE default_worker.py raylet gcs_server plasma_store log_monitor dashboard_agent runtime_env_agent "ray::" "vllm serve" "vllm bench" VllmWorker vllm python)
		local pat count
		for pat in "${tally_patterns[@]}"; do
			count="$(printf "%s\n" "${ps_args_snapshot}" | grep -Fc -- "${pat}" 2>/dev/null || true)"
			count="${count:-0}"
			printf "    %-24s %s\n" "${pat}" "${count}"
		done
		echo "  --- Top 40 processes by thread count (pid nlwp rss comm args) ---"
		ps -eLo pid,nlwp,rss,comm,args --no-headers -u "${USER}" 2>/dev/null | awk "!seen[\$1]++" | sort -k2 -nr | head -40 | sed "s/^/    /"
		local avg_threads="0"
		if [[ "${pid_count}" -gt 0 ]] 2>/dev/null; then
			avg_threads="$(awk -v t="${thread_count}" -v p="${pid_count}" "BEGIN{if(p>0) printf \"%.1f\", t/p; else print \"0\"}")"
		fi
		echo "  SUMMARY: label=${label} host=${hostname} pids=${pid_count} threads=${thread_count} avg=${avg_threads} cgroup_pids.current=${pids_cur_val}/${pids_max_val}"
		echo "=============================================================="
		echo ""
	} 2>&1 | if [[ -n "${out_file}" ]]; then tee -a "${out_file}"; else cat; fi
}
print_process_diagnostics "${REMOTE_LABEL}"
'

	mpiexec -n "${n_hosts}" --ppn 1 --hosts "${hosts}" -- bash -c "${remote_script}" 2>&1 || true
}

# =============================================================================
# BACKGROUND THREAD/PID SAMPLER
# =============================================================================
# Ported verbatim from qsub_polaris_body.sh. Samples pids.current and
# thread counts every 2 seconds, per node, during the engine-init /
# benchmark phase.
#
# The sampler is backend-agnostic: its CSV columns include ray_idle and
# ray_total which will always read 0 under mp, but keeping them makes
# cross-backend trajectory diffs trivial to plot.

SAMPLER_MPIEXEC_PID=""

start_background_sampler() {
	if [[ -n "${SAMPLER_MPIEXEC_PID}" ]] && kill -0 "${SAMPLER_MPIEXEC_PID}" 2>/dev/null; then
		return 0
	fi

	: >"${SAMPLER_FLAG}"

	if [[ -z "${HEAD_HOST:-}" ]]; then
		echo "WARN: start_background_sampler called before node discovery; head-only."
	fi

	local hosts="${HEAD_HOST}"
	local h
	for h in "${WORKER_HOSTS[@]:-}"; do
		if [[ -n "${h}" ]]; then
			hosts+=",${h}"
		fi
	done
	local n_hosts
	n_hosts="$(awk -F, '{print NF}' <<<"${hosts}")"

	local q_diag_dir q_user q_flag q_short_jobid
	q_diag_dir="$(printf '%q' "${DIAG_DIR}")"
	q_user="$(printf '%q' "${USER}")"
	q_flag="$(printf '%q' "${SAMPLER_FLAG}")"
	q_short_jobid="$(printf '%q' "${PBS_SHORT_JOBID:-}")"

	local remote_script='
export DIAG_DIR='"${q_diag_dir}"'
export USER='"${q_user}"'
export FLAG='"${q_flag}"'
export PBS_SHORT_JOBID='"${q_short_jobid}"'
hostname="$(hostname 2>/dev/null || echo unknown)"
out="${DIAG_DIR}/sampler_${hostname}.csv"

pids_max_file=""
pids_cur_file=""
if [[ -n "${PBS_SHORT_JOBID:-}" && -r "/sys/fs/cgroup/jobs/${PBS_SHORT_JOBID}/pids.max" ]]; then
	pids_max_file="/sys/fs/cgroup/jobs/${PBS_SHORT_JOBID}/pids.max"
	pids_cur_file="/sys/fs/cgroup/jobs/${PBS_SHORT_JOBID}/pids.current"
fi
cgpath=""
if [[ -r /proc/self/cgroup ]]; then
	cgpath="$(awk -F: "\$1 == \"0\" {print \$3; exit}" /proc/self/cgroup)"
	if [[ -z "${cgpath}" ]]; then
		cgpath="$(awk -F: "\$2 == \"pids\" {print \$3; exit}" /proc/self/cgroup)"
	fi
fi
if [[ -z "${pids_max_file}" && -n "${cgpath}" ]]; then
	probe="${cgpath}"
	while true; do
		if [[ -r "/sys/fs/cgroup${probe}/pids.max" ]]; then
			pids_max_file="/sys/fs/cgroup${probe}/pids.max"
			pids_cur_file="/sys/fs/cgroup${probe}/pids.current"
			break
		fi
		if [[ -r "/sys/fs/cgroup/pids${probe}/pids.max" ]]; then
			pids_max_file="/sys/fs/cgroup/pids${probe}/pids.max"
			pids_cur_file="/sys/fs/cgroup/pids${probe}/pids.current"
			break
		fi
		if [[ -z "${probe}" || "${probe}" == "/" ]]; then break; fi
		probe="${probe%/*}"
	done
fi
if [[ -z "${pids_max_file}" ]]; then
	if [[ -r /sys/fs/cgroup/pids.max ]]; then
		pids_max_file="/sys/fs/cgroup/pids.max"
		pids_cur_file="/sys/fs/cgroup/pids.current"
	elif [[ -r /sys/fs/cgroup/pids/pids.max ]]; then
		pids_max_file="/sys/fs/cgroup/pids/pids.max"
		pids_cur_file="/sys/fs/cgroup/pids/pids.current"
	fi
fi

if [[ ! -s "${out}" ]]; then
	echo "epoch,hostname,pid_count,thread_count,pids_current,pids_max,ray_idle,ray_total" >"${out}"
fi

deadline=$(( $(date +%s) + 10800 ))
while [[ -e "${FLAG}" ]]; do
	now=$(date +%s)
	if (( now > deadline )); then break; fi
	pid_count="$(ps -u "${USER}" --no-headers 2>/dev/null | wc -l)"
	thread_count="$(ps -u "${USER}" -L --no-headers 2>/dev/null | wc -l)"
	cur="unknown"; max="unknown"
	if [[ -n "${pids_cur_file}" ]]; then
		cur="$(cat "${pids_cur_file}" 2>/dev/null || echo unknown)"
		max="$(cat "${pids_max_file}" 2>/dev/null || echo unknown)"
	fi
	ps_snap="$(ps -u "${USER}" -o args --no-headers 2>/dev/null || true)"
	ray_idle="$(printf "%s\n" "${ps_snap}" | grep -Fc -- "ray::IDLE" 2>/dev/null || true)"
	ray_idle="${ray_idle:-0}"
	ray_total="$(printf "%s\n" "${ps_snap}" | grep -Fc -- "ray::" 2>/dev/null || true)"
	ray_total="${ray_total:-0}"
	echo "${now},${hostname},${pid_count},${thread_count},${cur},${max},${ray_idle},${ray_total}" >>"${out}"
	sleep 2
done
'

	mpiexec -n "${n_hosts}" --ppn 1 --hosts "${hosts}" -- bash -c "${remote_script}" \
		>/dev/null 2>&1 &
	SAMPLER_MPIEXEC_PID=$!
	echo "Background thread/pid sampler started (mpiexec PID ${SAMPLER_MPIEXEC_PID}, hosts: ${hosts})"
}

stop_background_sampler() {
	if [[ ! -e "${SAMPLER_FLAG}" ]] && [[ -z "${SAMPLER_MPIEXEC_PID}" ]]; then
		return 0
	fi
	rm -f "${SAMPLER_FLAG}" 2>/dev/null || true

	if [[ -n "${SAMPLER_MPIEXEC_PID}" ]]; then
		local waited=0
		while [[ "${waited}" -lt 5 ]] && kill -0 "${SAMPLER_MPIEXEC_PID}" 2>/dev/null; do
			sleep 1
			waited=$((waited + 1))
		done
		if kill -0 "${SAMPLER_MPIEXEC_PID}" 2>/dev/null; then
			kill "${SAMPLER_MPIEXEC_PID}" 2>/dev/null || true
			wait "${SAMPLER_MPIEXEC_PID}" 2>/dev/null || true
		fi
		SAMPLER_MPIEXEC_PID=""
	fi
}

# =============================================================================
# ERR TRAP (failure-time diagnostic dump)
# =============================================================================
# Same as the Ray body: on any command failure, dump per-node thread
# state and stop the sampler before the set-e unwind continues. Also
# makes a best-effort attempt to tear down mp workers so a failure
# during e.g. Step 4's bench doesn't leave stale vllm serve --headless
# processes on the worker nodes for the next iteration.

_diag_on_err() {
	local rc=$?
	echo ""
	echo "!!! ERR trap: command failed with rc=${rc} at line ${BASH_LINENO[0]}"

	print_process_diagnostics_all_nodes "FAILURE_rc${rc}" || true
	stop_background_sampler || true

	# Best-effort cleanup of mp workers so the rc propagates through
	# cleanup() with fewer stragglers to chase.
	stop_mp_workers || true
	stop_mp_serve_workers || true

	return "${rc}"
}

trap _diag_on_err ERR

# =============================================================================
# MULTIPROCESSING WORKER HELPERS
# =============================================================================
# Launch `vllm serve --headless` on each non-head node. The head-side
# bench / serve command dials in over <HEAD_IP>:<MP_MASTER_PORT> to
# form the torch.distributed process group.
#
# Flow:
#   1. `launch_mp_workers <engine-args...>` spawns one mpiexec-scoped
#      bash per worker host, each running `vllm serve --headless ...`
#      in the foreground of that bash. Background PIDs of the mpiexec
#      sessions are tracked in MP_WORKER_PIDS.
#   2. Head-side bench command runs and blocks until torch.distributed
#      init completes (all nodes rendezvous). Forward passes proceed.
#   3. On head-side command exit, workers' MultiprocExecutor monitor
#      sees the dead process group (detected via either heartbeat
#      timeout or a dropped TCP connection to the leader), workers
#      call sys.exit(), the vllm serve --headless process tree ends,
#      and the mpiexec session returns.
#   4. `stop_mp_workers` waits briefly for that natural shutdown, then
#      SIGTERMs any stragglers, then falls back to a remote pkill.
#
# Two tracked PID arrays so Step 5 (long-lived) can coexist with the
# per-iter teardown used by Steps 4 and 6:
#   MP_WORKER_PIDS        - per-iter bench workers (stop after each run)
#   MP_SERVE_WORKER_PIDS  - long-lived `vllm serve` workers (stop after
#                           the whole sweep)

# Args (positional):
#   $1 : which PID array to populate. One of "bench" or "serve".
#        "bench" -> MP_WORKER_PIDS
#        "serve" -> MP_SERVE_WORKER_PIDS
#   $2 : log-file prefix under ${MP_WORKER_LOG_DIR} (e.g.
#        "latency_bs1_in512_out128" or "serve").
#   $3+ : engine args to pass to `vllm serve --headless` on each worker.
#         Must be identical to the head-side engine args (except for
#         --node-rank, which launch_mp_workers overrides per host).
launch_mp_workers() {
	local which_array="$1"
	local log_prefix="$2"
	shift 2

	# Pre-quote all remaining args so they survive the outer->inner
	# bash-c expansion intact. printf %q produces output that is safe
	# to pass through a single round of word-splitting.
	local worker_args_str
	worker_args_str="$(printf '%q ' "$@")"

	# Reset the target array.
	if [[ "${which_array}" == "serve" ]]; then
		MP_SERVE_WORKER_PIDS=()
	else
		MP_WORKER_PIDS=()
	fi

	local i whost wip wrank worker_log
	for i in "${!WORKER_HOSTS[@]}"; do
		whost="${WORKER_HOSTS[$i]}"
		wip="${WORKER_IPS[$i]}"
		wrank="$((i + 1))"
		worker_log="${MP_WORKER_LOG_DIR}/${log_prefix}_rank${wrank}_${whost}.log"

		echo "  Launching mp worker rank ${wrank} on ${whost} (${wip}); log: ${worker_log}"

		# The remote command sets up the exact same env the head has,
		# activates the venv, then execs `vllm serve --headless`. Using
		# `exec` so signals sent to the mpiexec parent forward cleanly
		# to the actual vllm process tree (not interposed by bash).
		mpiexec -n 1 --ppn 1 --hosts "${whost}" -- bash -c "
			set -euo pipefail

			export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'
			export VLLM_HOST_IP='${wip}'

			# Persistent compile caches (shared filesystem, must match
			# head node so FileCacheManager's per-key flock actually
			# serializes across all ranks in the cluster).
			export TRITON_CACHE_DIR='${TRITON_CACHE_DIR}'
			export TORCHINDUCTOR_CACHE_DIR='${TORCHINDUCTOR_CACHE_DIR}'
			export VLLM_CACHE_ROOT='${VLLM_CACHE_ROOT}'
			export TRITON_CACHE_MANAGER='${TRITON_CACHE_MANAGER}'

			# Thread limits (see THREAD LIMITS block in the driver).
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

			# Plumb the PBS short job id through for cgroup resolution
			# in worker-side diagnostics / sampler.
			export PBS_SHORT_JOBID='${PBS_SHORT_JOBID}'

			# NCCL/Gloo must use the same interface as on the head node.
			export NCCL_SOCKET_IFNAME='${RAY_IFNAME}'
			export GLOO_SOCKET_IFNAME='${RAY_IFNAME}'
			export NCCL_DEBUG='${NCCL_DEBUG}'

			# Optional: bumped RPC timeout when profiling.
			if [[ -n '${VLLM_RPC_TIMEOUT:-}' ]]; then
				export VLLM_RPC_TIMEOUT='${VLLM_RPC_TIMEOUT:-}'
			fi

			# Load CUDA runtime libraries.
			module load cuda/12.9

			# Activate the venv so vllm is on PATH.
			source '${VENV_DIR}/bin/activate'

			echo '[mp-worker rank=${wrank} host=${whost}] Starting vllm serve --headless'
			echo '[mp-worker rank=${wrank} host=${whost}] master=${HEAD_IP}:${MP_MASTER_PORT}'

			# exec so the vllm process replaces bash; signals from
			# mpiexec's process-manager flow to vllm directly.
			exec vllm serve ${worker_args_str} \\
				--nnodes ${NUM_NODES} \\
				--node-rank ${wrank} \\
				--master-addr '${HEAD_IP}' \\
				--master-port '${MP_MASTER_PORT}' \\
				--headless
		" >"${worker_log}" 2>&1 &

		if [[ "${which_array}" == "serve" ]]; then
			MP_SERVE_WORKER_PIDS+=($!)
		else
			MP_WORKER_PIDS+=($!)
		fi
	done

	if [[ "${which_array}" == "serve" ]]; then
		echo "Launched ${#MP_SERVE_WORKER_PIDS[@]} mp serve worker(s): ${MP_SERVE_WORKER_PIDS[*]}"
	else
		echo "Launched ${#MP_WORKER_PIDS[@]} mp bench worker(s): ${MP_WORKER_PIDS[*]}"
	fi
}

# Wait (up to MP_WORKER_READY_TIMEOUT) for each worker log to contain
# the "Launching vLLM ... headless multiproc executor" line, which is
# emitted by vllm/entrypoints/cli/serve.py:186 right before workers
# enter the torch.distributed init handshake. Returning from this
# function means the workers are alive and ready to rendezvous; the
# head-side bench/serve command can then be launched.
#
# Args:
#   $1 : which array to wait on, "bench" or "serve".
#   $2 : log-file prefix (must match what was passed to launch_mp_workers).
wait_mp_workers_ready() {
	local which_array="$1"
	local log_prefix="$2"
	local deadline=$(( $(date +%s) + MP_WORKER_READY_TIMEOUT ))

	# Pattern matches the INFO line printed by cli/serve.py:186:
	#   "Launching vLLM (v...) headless multiproc executor, with head node address ..."
	local ready_pattern="headless multiproc executor"

	# Indirect-expansion variable name so we can target either the
	# per-iter bench PID array or the long-lived serve PID array
	# without eval.
	local pids_var="MP_WORKER_PIDS[@]"
	if [[ "${which_array}" == "serve" ]]; then
		pids_var="MP_SERVE_WORKER_PIDS[@]"
	fi

	local i whost wrank worker_log ready_count
	while true; do
		ready_count=0
		for i in "${!WORKER_HOSTS[@]}"; do
			whost="${WORKER_HOSTS[$i]}"
			wrank="$((i + 1))"
			worker_log="${MP_WORKER_LOG_DIR}/${log_prefix}_rank${wrank}_${whost}.log"
			if [[ -f "${worker_log}" ]] && grep -Fq "${ready_pattern}" "${worker_log}" 2>/dev/null; then
				ready_count=$((ready_count + 1))
			fi
		done

		if (( ready_count >= ${#WORKER_HOSTS[@]} )); then
			echo "All ${#WORKER_HOSTS[@]} mp workers are rendezvous-ready."
			return 0
		fi

		# Detect premature worker death: if any tracked PID is gone
		# before we saw the ready log line, bail out rather than wait
		# for the timeout.
		local pid
		for pid in "${!pids_var}"; do
			if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
				echo "ERROR: mp worker PID ${pid} died during startup."
				echo "       Check ${MP_WORKER_LOG_DIR}/${log_prefix}_rank*_*.log for tracebacks."
				return 1
			fi
		done

		if (( $(date +%s) >= deadline )); then
			echo "ERROR: Only ${ready_count}/${#WORKER_HOSTS[@]} mp workers ready after ${MP_WORKER_READY_TIMEOUT}s."
			echo "       See ${MP_WORKER_LOG_DIR}/${log_prefix}_rank*_*.log for slow-starters."
			return 1
		fi

		echo "  Waiting for mp workers... ${ready_count}/${#WORKER_HOSTS[@]} ready"
		sleep 5
	done
}

# Stop per-iter bench workers tracked in MP_WORKER_PIDS. Idempotent.
# Normal flow: head-side bench command has just returned, workers
# should be in the process of exiting on their own. Give them
# MP_WORKER_SHUTDOWN_TIMEOUT to do so cleanly before SIGTERM-ing the
# mpiexec parent.
stop_mp_workers() {
	if [[ "${#MP_WORKER_PIDS[@]:-0}" -eq 0 ]]; then
		return 0
	fi

	echo "  Stopping ${#MP_WORKER_PIDS[@]} mp bench worker(s)..."

	local deadline=$(( $(date +%s) + MP_WORKER_SHUTDOWN_TIMEOUT ))
	local pid still_alive=0
	while (( $(date +%s) < deadline )); do
		still_alive=0
		for pid in "${MP_WORKER_PIDS[@]}"; do
			if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
				still_alive=$((still_alive + 1))
			fi
		done
		if (( still_alive == 0 )); then
			break
		fi
		sleep 1
	done

	if (( still_alive > 0 )); then
		echo "  ${still_alive} worker(s) still alive after ${MP_WORKER_SHUTDOWN_TIMEOUT}s; sending SIGTERM."
		for pid in "${MP_WORKER_PIDS[@]}"; do
			if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
				kill "${pid}" 2>/dev/null || true
			fi
		done
		sleep 2
		for pid in "${MP_WORKER_PIDS[@]}"; do
			if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
				kill -9 "${pid}" 2>/dev/null || true
			fi
		done
	fi

	for pid in "${MP_WORKER_PIDS[@]}"; do
		wait "${pid}" 2>/dev/null || true
	done
	MP_WORKER_PIDS=()

	# Belt-and-suspenders: remote pkill in case an mpiexec session
	# exited cleanly but left a vllm serve --headless child behind.
	local whost
	for whost in "${WORKER_HOSTS[@]}"; do
		mpiexec -n 1 --ppn 1 --hosts "${whost}" -- bash -c \
			"pkill -f 'vllm serve.*--headless' 2>/dev/null || true" \
			2>/dev/null || true
	done
}

# Stop long-lived serving workers tracked in MP_SERVE_WORKER_PIDS.
# Mirrors stop_mp_workers but targets the other array. Called at the
# end of Step 5 after `vllm serve` is torn down.
stop_mp_serve_workers() {
	if [[ "${#MP_SERVE_WORKER_PIDS[@]:-0}" -eq 0 ]]; then
		return 0
	fi

	echo "  Stopping ${#MP_SERVE_WORKER_PIDS[@]} mp serve worker(s)..."

	local deadline=$(( $(date +%s) + MP_WORKER_SHUTDOWN_TIMEOUT ))
	local pid still_alive=0
	while (( $(date +%s) < deadline )); do
		still_alive=0
		for pid in "${MP_SERVE_WORKER_PIDS[@]}"; do
			if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
				still_alive=$((still_alive + 1))
			fi
		done
		if (( still_alive == 0 )); then
			break
		fi
		sleep 1
	done

	if (( still_alive > 0 )); then
		echo "  ${still_alive} serve worker(s) still alive after ${MP_WORKER_SHUTDOWN_TIMEOUT}s; sending SIGTERM."
		for pid in "${MP_SERVE_WORKER_PIDS[@]}"; do
			if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
				kill "${pid}" 2>/dev/null || true
			fi
		done
		sleep 2
		for pid in "${MP_SERVE_WORKER_PIDS[@]}"; do
			if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
				kill -9 "${pid}" 2>/dev/null || true
			fi
		done
	fi

	for pid in "${MP_SERVE_WORKER_PIDS[@]}"; do
		wait "${pid}" 2>/dev/null || true
	done
	MP_SERVE_WORKER_PIDS=()

	local whost
	for whost in "${WORKER_HOSTS[@]}"; do
		mpiexec -n 1 --ppn 1 --hosts "${whost}" -- bash -c \
			"pkill -f 'vllm serve.*--headless' 2>/dev/null || true" \
			2>/dev/null || true
	done
}

# =============================================================================
# STEP 1: (no-op; kept for phase-numbering parity with the Ray body)
# =============================================================================
# The Ray body's Steps 1-3 (start Ray head, launch Ray workers, wait for
# cluster to join) have no mp equivalent because there is no persistent
# cross-node process. Workers are launched per-bench-invocation below
# (Steps 4/5/6). We still emit the 00/01 diagnostic snapshots so the
# cross-backend log-diff stays aligned.

echo ""
echo "[Step 1/5] No-op under mp backend (no persistent cluster bring-up)."

# 00_job_start: snapshot BEFORE any mp-specific work. On Polaris each
# job gets fresh nodes, so this should show a near-empty process list;
# anything unexpected here is a leaked process from a prior tenant.
print_process_diagnostics_all_nodes "00_job_start"

# 01_pre_setup: snapshot after env/module setup, just before the first
# bench launches workers. Baseline: nothing mp-related exists yet.
print_process_diagnostics_all_nodes "01_pre_setup"

# =============================================================================
# STEP 4-6 SHARED SETUP (always runs)
# =============================================================================

mkdir -p "${RESULTS_DIR}"

# Engine args shared between head-side invocations (vllm bench latency /
# throughput / serve on node_rank 0) AND worker-side invocations
# (vllm serve --headless on node_rank > 0).
#
# IMPORTANT: every value in this array must appear IDENTICALLY on the
# head and every worker or MultiprocExecutor's VllmConfig consistency
# check rejects the rendezvous. Knobs that MUST match include: model
# path, quantization, dtype, TP/PP sizes, --enable-expert-parallel,
# max-model-len, --trust-remote-code, --enforce-eager,
# --distributed-executor-backend.
#
# --node-rank and --master-addr/--master-port differ per node and are
# added by the caller (head uses node-rank=0; launch_mp_workers overrides
# per-worker).
MP_COMMON_ENGINE_ARGS=(
	--model "${MODEL}"
	${QUANT_ARGS[@]+"${QUANT_ARGS[@]}"}
	--dtype "${DTYPE}"
	--tensor-parallel-size "${TP_SIZE}"
	--pipeline-parallel-size "${PP_SIZE}"
	--enable-expert-parallel
	--distributed-executor-backend mp
	--max-model-len "${MAX_MODEL_LEN}"
	--trust-remote-code
	# --enforce-eager: same rationale as Ray body. Under Polaris's
	# cgroup pids.max=4096 cap, CUDA graph capture's transient thread
	# churn has pushed peak pids past the cap. Trade 1.5-2x speed for
	# surviving the graph-capture phase.
	--enforce-eager
)

# Head-side args = common args + this node's node-rank/master coords.
# Used directly by `vllm bench latency/throughput` (Step 4/6) AND by
# the head-side `vllm serve` (Step 5).
MP_HEAD_DIST_ARGS=(
	--nnodes "${NUM_NODES}"
	--node-rank 0
	--master-addr "${HEAD_IP}"
	--master-port "${MP_MASTER_PORT}"
)

# ENGINE_ARGS: used by the Step 4/6 offline bench invocations. Named
# to match the Ray body so the bench-invocation code below stays
# textually similar across backends.
ENGINE_ARGS=(
	"${MP_COMMON_ENGINE_ARGS[@]}"
	"${MP_HEAD_DIST_ARGS[@]}"
)

# Latency sweep matrix. Same semantics as the Ray body.
BATCH_SIZES_CSV="${BATCH_SIZES_CSV:-1,8,32}"
LATENCY_IO_CONFIGS_CSV="${LATENCY_IO_CONFIGS_CSV:-512:128}"
IFS=',' read -r -a BATCH_SIZES       <<<"${BATCH_SIZES_CSV}"
IFS=',' read -r -a LATENCY_IO_CONFIGS <<<"${LATENCY_IO_CONFIGS_CSV}"

LATENCY_WARMUP_ITERS="${LATENCY_WARMUP_ITERS:-2}"
LATENCY_ITERS="${LATENCY_ITERS:-10}"

# Profiling: same trim-down as the Ray body when PROFILE=1.
if [[ "${PROFILE}" == "1" ]]; then
	NUM_PROMPTS="${PROFILE_NUM_PROMPTS}"
	LATENCY_WARMUP_ITERS="${PROFILE_LATENCY_WARMUP_ITERS}"

	mkdir -p "${PROFILE_DIR}"

	echo ""
	echo "============================================="
	echo "  PROFILING ENABLED (PROFILE=1)"
	echo "============================================="
	echo "  Profile dir:       ${PROFILE_DIR}"
	echo "  Num prompts:       ${NUM_PROMPTS} (was PROFILE_NUM_PROMPTS)"
	echo "  Latency warmups:   ${LATENCY_WARMUP_ITERS} (was PROFILE_LATENCY_WARMUP_ITERS)"
	echo "  VLLM_RPC_TIMEOUT:  ${VLLM_RPC_TIMEOUT:-default}"
	echo "  Trace viewer:      https://ui.perfetto.dev/"
	echo "============================================="
	echo ""
fi

# 04_pre_engine: snapshot immediately before the first bench-driven mp
# worker launch.
print_process_diagnostics_all_nodes "04_pre_engine"

# Kick off the continuous 2s sampler on every node. Same lifecycle as
# the Ray body: sampler runs for the duration of Steps 4-6, torn down
# by the ERR trap, the cleanup trap, or the explicit call at the end
# of the script.
start_background_sampler

# =============================================================================
# STEP 4: Run Offline Latency Benchmarks
# =============================================================================
# Per (batch_size, input_len, output_len) iteration:
#   1. launch_mp_workers "bench" <label> <common-engine-args>
#   2. wait_mp_workers_ready "bench" <label>
#   3. vllm bench latency <common-engine-args> <head-dist-args> <per-iter-args>
#   4. stop_mp_workers
#
# Workers are torn down after each iteration so their VllmConfig doesn't
# drift against the next iteration's bench command (different --batch-size
# alone doesn't matter, but keeping teardown per-iter means a stuck worker
# on iteration K doesn't corrupt iteration K+1).

if [[ "${RUN_LATENCY}" == "1" ]]; then
	echo ""
	echo "[Step 4/6] Running offline latency benchmarks (mp backend)..."
	echo "  Engine config: PP=${PP_SIZE}, TP=${TP_SIZE}, EP=${EP_SIZE}"
	echo "  Sweep: batch_sizes=[${BATCH_SIZES[*]}] io_configs=[${LATENCY_IO_CONFIGS[*]}]"
	echo ""

	for IO_CONFIG in "${LATENCY_IO_CONFIGS[@]}"; do
		INPUT_LEN="${IO_CONFIG%%:*}"
		OUTPUT_LEN="${IO_CONFIG##*:}"

		for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
			RUN_LABEL="latency_bs${BATCH_SIZE}_in${INPUT_LEN}_out${OUTPUT_LEN}"
			RESULT_FILE="${RESULTS_DIR}/${RUN_LABEL}.json"
			LOG_FILE="${RESULTS_DIR}/${RUN_LABEL}.log"

			echo "  >> Latency: batch_size=${BATCH_SIZE}, input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}"

			# Per-invocation profiler args. Each (bs, in, out) gets
			# its own torch_profiler_dir (same convention as Ray body).
			LATENCY_PROFILE_ARGS=()
			if [[ "${PROFILE}" == "1" ]]; then
				LATENCY_PROFILE_DIR="${PROFILE_DIR}/${RUN_LABEL}"
				mkdir -p "${LATENCY_PROFILE_DIR}"
				LATENCY_PROFILE_ARGS=(
					--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${LATENCY_PROFILE_DIR}\",\"torch_profiler_record_shapes\":true,\"torch_profiler_with_stack\":false}"
					--profile
				)
				echo "  >> Profiler traces -> ${LATENCY_PROFILE_DIR}"
			fi

			# Launch headless workers on all N-1 non-head nodes.
			launch_mp_workers "bench" "${RUN_LABEL}" "${MP_COMMON_ENGINE_ARGS[@]}"

			# Wait for workers to reach torch.distributed rendezvous.
			if ! wait_mp_workers_ready "bench" "${RUN_LABEL}"; then
				echo "ERROR: mp workers failed to start for ${RUN_LABEL}; skipping."
				stop_mp_workers || true
				continue
			fi

			# Run the head-side bench. Workers are already blocked in
			# init_process_group; this call completes the rendezvous.
			vllm bench latency \
				"${ENGINE_ARGS[@]}" \
				--batch-size "${BATCH_SIZE}" \
				--input-len "${INPUT_LEN}" \
				--output-len "${OUTPUT_LEN}" \
				--num-iters-warmup "${LATENCY_WARMUP_ITERS}" \
				--num-iters "${LATENCY_ITERS}" \
				--output-json "${RESULT_FILE}" \
				${LATENCY_PROFILE_ARGS[@]+"${LATENCY_PROFILE_ARGS[@]}"} \
				2>&1 | tee "${LOG_FILE}"

			# Tear down workers (iteration-scoped).
			stop_mp_workers

			echo "  >> Saved to ${RESULT_FILE}"
			echo ""
		done
	done

	echo "Offline latency benchmarks complete."
	echo ""
else
	echo "[Step 4/6] Skipping offline latency benchmarks (RUN_LATENCY=${RUN_LATENCY})."
	echo ""
fi

# Snapshot just before vllm serve.
print_process_diagnostics_all_nodes "05_pre_vllm_serve"

# =============================================================================
# STEP 5: Online Serving Benchmark
# =============================================================================
# Unlike Steps 4/6, the `vllm serve` process stays alive for the whole
# serving sweep, so mp workers are launched ONCE, the sweep runs, then
# workers are stopped.

if [[ "${RUN_SERVING}" == "1" ]]; then
	echo "[Step 5/6] Starting vLLM server and running serving benchmarks (mp backend)..."

	# Per-server profiler args.
	SERVE_PROFILE_ARGS=()
	if [[ "${PROFILE}" == "1" ]]; then
		SERVE_PROFILE_DIR="${PROFILE_DIR}/serving"
		mkdir -p "${SERVE_PROFILE_DIR}"
		SERVE_PROFILE_ARGS=(
			--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${SERVE_PROFILE_DIR}\",\"torch_profiler_record_shapes\":true,\"torch_profiler_with_stack\":false}"
		)
		echo "  Profiler traces -> ${SERVE_PROFILE_DIR}"
	fi

	# Launch headless workers. They'll block on torch.distributed
	# rendezvous until the head-side vllm serve below completes init.
	launch_mp_workers "serve" "serve" "${MP_COMMON_ENGINE_ARGS[@]}"

	if ! wait_mp_workers_ready "serve" "serve"; then
		echo "ERROR: mp workers failed to start for serve; aborting Step 5."
		stop_mp_serve_workers || true
		exit 1
	fi

	# Start the vLLM server in the background on the head.
	# Note: --model is already in MP_COMMON_ENGINE_ARGS, so do NOT pass
	# it as a positional arg again (duplicate-arg error).
	vllm serve \
		"${MP_COMMON_ENGINE_ARGS[@]}" \
		"${MP_HEAD_DIST_ARGS[@]}" \
		--host 0.0.0.0 \
		--port "${SERVE_PORT}" \
		${SERVE_PROFILE_ARGS[@]+"${SERVE_PROFILE_ARGS[@]}"} \
		>"${RESULTS_DIR}/server.log" 2>&1 &

	SERVER_PID=$!
	echo "vLLM server starting in background (PID ${SERVER_PID})..."

	# Wait for server readiness.
	BASE_URL="http://localhost:${SERVE_PORT}"
	echo "Waiting for server to be ready at ${BASE_URL}/v1/models ..."

	MAX_WAIT=600
	ELAPSED=0
	SERVER_POLL_INTERVAL=10

	while true; do
		if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
			echo "ERROR: vLLM server process died unexpectedly."
			tail -n 50 "${RESULTS_DIR}/server.log" 2>/dev/null || true
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
	REQUEST_RATES_CSV="${REQUEST_RATES_CSV:-1,8,inf}"
	IO_CONFIGS_CSV="${IO_CONFIGS_CSV:-512:128,128:512}"
	IFS=',' read -r -a REQUEST_RATES <<<"${REQUEST_RATES_CSV}"
	IFS=',' read -r -a IO_CONFIGS    <<<"${IO_CONFIGS_CSV}"

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

			BENCH_SERVE_PROFILE_ARGS=()
			if [[ "${PROFILE}" == "1" ]]; then
				BENCH_SERVE_PROFILE_ARGS=(--profile)
			fi

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
				${BENCH_SERVE_PROFILE_ARGS[@]+"${BENCH_SERVE_PROFILE_ARGS[@]}"} \
				--metadata \
				tp="${TP_SIZE}" \
				pp="${PP_SIZE}" \
				ep="${EP_SIZE}" \
				backend=mp \
				quantization="${QUANTIZATION:-none}" \
				input_len="${INPUT_LEN}" \
				output_len="${OUTPUT_LEN}" \
				request_rate="${REQUEST_RATE}" \
				2>&1 | tee "${LOG_FILE}"

			echo ">> Saved to ${RESULT_FILE}"
			echo ""
		done
	done

	echo "Online serving benchmarks complete."
	echo ""

	# Tear down the server, then the headless workers.
	if [[ -n "${SERVER_PID}" ]]; then
		echo "  Stopping vLLM server (PID ${SERVER_PID})..."
		kill "${SERVER_PID}" 2>/dev/null || true
		wait "${SERVER_PID}" 2>/dev/null || true
		SERVER_PID=""
	fi
	stop_mp_serve_workers

else
	echo "[Step 5/6] Skipping online serving benchmarks (RUN_SERVING=${RUN_SERVING})."
	echo ""
fi

# =============================================================================
# STEP 6: Offline Throughput Benchmark
# =============================================================================
# Same per-iter worker lifecycle as Step 4 (launch / wait / bench /
# stop). Sweep is over IO shapes only (vllm bench throughput has no
# --batch-size arg).

if [[ "${RUN_THROUGHPUT}" == "1" ]]; then
	echo "[Step 6/6] Running offline throughput benchmarks (mp backend)..."
	echo "  Engine config: PP=${PP_SIZE}, TP=${TP_SIZE}, EP=${EP_SIZE}"
	# Fall back to the same default IO_CONFIGS as Step 5 if we get
	# here with RUN_SERVING=0 (in which case Step 5 never populated
	# the IO_CONFIGS array).
	if [[ "${#IO_CONFIGS[@]:-0}" -eq 0 ]]; then
		IO_CONFIGS_CSV="${IO_CONFIGS_CSV:-512:128,128:512}"
		IFS=',' read -r -a IO_CONFIGS <<<"${IO_CONFIGS_CSV}"
	fi
	echo "  Sweep: io_configs=[${IO_CONFIGS[*]}]"
	echo ""

	for IO_CONFIG in "${IO_CONFIGS[@]}"; do
		INPUT_LEN="${IO_CONFIG%%:*}"
		OUTPUT_LEN="${IO_CONFIG##*:}"

		RUN_LABEL="throughput_in${INPUT_LEN}_out${OUTPUT_LEN}"
		RESULT_FILE="${RESULTS_DIR}/${RUN_LABEL}.json"
		LOG_FILE="${RESULTS_DIR}/${RUN_LABEL}.log"

		echo "  >> Throughput: input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}"

		THROUGHPUT_PROFILE_ARGS=()
		if [[ "${PROFILE}" == "1" ]]; then
			THROUGHPUT_PROFILE_DIR="${PROFILE_DIR}/${RUN_LABEL}"
			mkdir -p "${THROUGHPUT_PROFILE_DIR}"
			THROUGHPUT_PROFILE_ARGS=(
				--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${THROUGHPUT_PROFILE_DIR}\",\"torch_profiler_record_shapes\":true,\"torch_profiler_with_stack\":false}"
				--profile
			)
			echo "  >> Profiler traces -> ${THROUGHPUT_PROFILE_DIR}"
		fi

		launch_mp_workers "bench" "${RUN_LABEL}" "${MP_COMMON_ENGINE_ARGS[@]}"

		if ! wait_mp_workers_ready "bench" "${RUN_LABEL}"; then
			echo "ERROR: mp workers failed to start for ${RUN_LABEL}; skipping."
			stop_mp_workers || true
			continue
		fi

		vllm bench throughput \
			"${ENGINE_ARGS[@]}" \
			--dataset-name random \
			--input-len "${INPUT_LEN}" \
			--output-len "${OUTPUT_LEN}" \
			--num-prompts "${NUM_PROMPTS}" \
			--output-json "${RESULT_FILE}" \
			${THROUGHPUT_PROFILE_ARGS[@]+"${THROUGHPUT_PROFILE_ARGS[@]}"} \
			2>&1 | tee "${LOG_FILE}"

		stop_mp_workers

		echo "  >> Saved to ${RESULT_FILE}"
		echo ""
	done

	echo "Offline throughput benchmarks complete."
	echo ""
else
	echo "[Step 6/6] Skipping offline throughput benchmarks (RUN_THROUGHPUT=${RUN_THROUGHPUT})."
	echo ""
fi

# =============================================================================
# SUMMARY
# =============================================================================

# Stop the sampler before printing summary so its final CSV rows flush.
stop_background_sampler || true

echo "============================================="
echo "  All benchmarks complete!"
echo "============================================="
echo ""
echo "Job ID:      ${PBS_JOBID:-N/A}"
echo "Backend:     mp (MultiprocExecutor + torch.distributed)"
echo "Results in:  ${RESULTS_DIR}/"
echo ""
echo "Latency results:"
ls -1 "${RESULTS_DIR}"/latency_*.json 2>/dev/null || echo "  (no latency JSON files)"
echo ""
echo "Serving results:"
ls -1 "${RESULTS_DIR}"/serving_*.json 2>/dev/null || echo "  (no serving JSON files)"
echo ""
echo "Throughput results:"
ls -1 "${RESULTS_DIR}"/throughput_*.json 2>/dev/null || echo "  (no throughput JSON files)"
echo ""
echo "Profiler traces:"
if [[ "${PROFILE}" == "1" ]]; then
	if [[ -d "${PROFILE_DIR}" ]]; then
		echo "  Profile dir: ${PROFILE_DIR}"
		find "${PROFILE_DIR}" -name '*.pt.trace.json*' 2>/dev/null | head -20 |
			sed 's/^/    /' || echo "    (no trace files found)"
		echo "  View traces directly at https://ui.perfetto.dev/"
		echo "  (drag-and-drop any .pt.trace.json.gz file; no untar needed)"
	else
		echo "  (PROFILE=1 but ${PROFILE_DIR} was not created)"
	fi
else
	echo "  (profiling disabled; submit with 'qsub -v PROFILE=1 ...' to enable)"
fi
echo ""
echo "Per-node mp worker logs: ${MP_WORKER_LOG_DIR}/"
echo ""
echo "Metrics recorded:"
echo "  - Offline latency:    end-to-end latency per (batch size, input length)"
echo "  - Online serving:     TTFT, TPOT, ITL, E2EL at p50/p90/p95/p99"
echo "  - Offline throughput: total tokens/s per IO shape"
echo "  - Online:  request throughput, token throughput"
echo ""
echo "Cleanup will run automatically via EXIT trap."

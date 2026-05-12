#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# =============================================================================
# PBS Pro Job Script - Single-Node Qwen3-0.6B vLLM Process Sanity Check
# =============================================================================
#
# Purpose: verify that the local vLLM build actually runs end-to-end on a
# Polaris A100 node BEFORE attempting multi-node Ray runs with DeepSeek-R1
# (see qsub_polaris.sh). This script runs a tiny, fast dense model
# (/grand/Intel/dhuang/Qwen3-0.6B) with single-node parallelism so any
# failures isolate to the vLLM process itself, not Ray, Slingshot,
# MoE/EP, or FP8 quantization.
#
# Differences from qsub_polaris.sh:
#   - Single node only (no PBS_NODEFILE parsing, no mpiexec, no Ray).
#   - --distributed-executor-backend mp (vLLM's MultiprocExecutor) instead
#     of ray. mp spawns worker subprocesses locally via torch.multiprocessing;
#     no Ray actors, no placement groups, no cross-node networking.
#   - Model: Qwen3-0.6B (dense, bf16/fp16). No quantization, no EP flag.
#   - TP=2, PP=2 -> 4 GPUs on one node. Exercises intra-node TP (NVLink)
#     and in-process PP without touching hsn0.
#   - Adds an offline throughput benchmark (vllm bench throughput) in
#     addition to latency and serving, since it runs cheaply on a tiny
#     model and it's the third main bench entry point users call.
#
# Hardware:  1 x Polaris node, 4x A100-80GB
# Model:     /grand/Intel/dhuang/Qwen3-0.6B (local HF checkpoint)
#
# Usage:
#   # Edit the -A project allocation below, then:
#   qsub qsub_polaris_qwen3_06b.sh
#
# =============================================================================

# =============================================================================
# PBS DIRECTIVES
# =============================================================================
# `debug` queue: 1 node, up to 1h walltime on Polaris. Use `debug-scaling`
# or `prod` for more nodes; this script is single-node by design.

#PBS -l select=1:system=polaris:ncpus=32:ngpus=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -q debug
#PBS -A Intel
#PBS -N vllm-qwen3-06b-bench
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
# Reuses the same venv as qsub_polaris.sh by default, so both scripts hit
# the same installed vllm and the same warm compile caches.

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
# See qsub_polaris.sh for the full rationale. Sharing the cache root with
# the multi-node driver means a warm Inductor / Triton cache populated by
# either script benefits the other.

CACHE_ROOT="/grand/Intel/dhuang/vllm_polaris_cache"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"

mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}"

# Serialize concurrent gcc stubs across the 4 TP*PP worker ranks (see the
# FileCacheManager rationale in qsub_polaris.sh).
export TRITON_CACHE_MANAGER="${TRITON_CACHE_MANAGER:-triton.runtime.cache:FileCacheManager}"

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Local HF checkpoint on Polaris /grand. Qwen3-0.6B is a tiny ~0.6B-param
# dense causal LM; no quantization flag required. --trust-remote-code is
# the safe default for Qwen checkpoints.
MODEL="${MODEL:-/grand/Intel/dhuang/Qwen3-0.6B}"
DTYPE="${DTYPE:-auto}"

# Parallelism on a single Polaris node (4 x A100-80GB).
#   TP=2 exercises intra-node tensor parallelism over NVLink.
#   PP=2 exercises in-process pipeline parallelism through MultiprocExecutor.
# Invariant: TP * PP == GPUS_PER_NODE.
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-2}"
EXPECTED_GPUS="${GPUS_PER_NODE}"

# Distributed executor backend. vLLM accepts:
#   "ray" | "mp" | "uni" | "external_launcher"  (vllm/config/parallel.py:35)
# "mp" = MultiprocExecutor: spawns worker subprocesses directly on this
# node via torch.multiprocessing. No Ray. Single cleanest backend to
# validate the vllm process itself against.
DIST_BACKEND="${DIST_BACKEND:-mp}"

# Optional: skip torch.compile + CUDA graph capture. Leave OFF (default)
# for a real sanity check so we exercise the compile code path. If the
# job dies with pthread_create EAGAIN or fork retry errors during engine
# init, set ENFORCE_EAGER=1 (same rationale as qsub_polaris.sh's
# --enforce-eager usage under the cgroup pids.max=4096 cap).
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
EAGER_ARGS=()
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
	EAGER_ARGS=(--enforce-eager)
fi

if (( PP_SIZE * TP_SIZE != EXPECTED_GPUS )); then
	echo "ERROR: Invalid parallelism configuration."
	echo "       PP_SIZE (${PP_SIZE}) * TP_SIZE (${TP_SIZE}) = $((PP_SIZE * TP_SIZE))"
	echo "       GPUS_PER_NODE = ${EXPECTED_GPUS}"
	echo "       These must be equal on a single-node run."
	exit 1
fi

# GPU visibility: expose all GPUS_PER_NODE devices; workers inherit.
_default_cvd="$(seq -s, 0 $((GPUS_PER_NODE - 1)))"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${_default_cvd}}"
unset _default_cvd

# Benchmark output directory (under PBS job working directory)
RESULTS_DIR="${RESULTS_DIR:-${PBS_O_WORKDIR:-.}/results_qwen3_06b_${PBS_JOBID:-manual}}"
mkdir -p "${RESULTS_DIR}"

# Server port for online serving benchmark
SERVE_PORT="${SERVE_PORT:-8000}"

# Max model length. Qwen3-0.6B has a large native context; 4096 is plenty
# for the (input_len=512, output_len=128) sweep below.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Number of prompts / warmups for serving benchmark. Sized for the 1h
# debug queue.
NUM_PROMPTS="${NUM_PROMPTS:-200}"
NUM_WARMUPS="${NUM_WARMUPS:-2}"

# =============================================================================
# NETWORK & THREAD LIMITS
# =============================================================================
# Single-node run, so inter-node IF selection is academic, but NCCL still
# comes up for intra-node TP all-reduces. Keeping the same hsn0 exports
# as the multi-node driver is harmless: NCCL uses CUDA IPC / P2P over
# NVLink for co-located ranks regardless of NCCL_SOCKET_IFNAME.
RAY_IFNAME="${RAY_IFNAME:-hsn0}"
export NCCL_SOCKET_IFNAME="${RAY_IFNAME}"
export GLOO_SOCKET_IFNAME="${RAY_IFNAME}"
export NCCL_DEBUG=WARN

# Thread limits. Same rationale as qsub_polaris.sh: 32 cores / 4 workers
# with each worker letting OpenBLAS/OMP/MKL default to the full core
# count trips RLIMIT_NPROC and the per-job cgroup pids.max. Cap at a
# conservative 4 threads per worker and leave headroom for driver /
# tokenizer pools.
NUM_THREADS_PER_WORKER="${NUM_THREADS_PER_WORKER:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-${NUM_THREADS_PER_WORKER}}"

# Inductor + ALCF XALT fan-out caps. These are the fixes for the
# gcc / ld "fork: retry: Resource temporarily unavailable" failure
# mode documented in qsub_polaris.sh. They cost a bit of cold-compile
# time and buy a lot of thread-budget headroom.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export XALT_EXECUTABLE_TRACKING="${XALT_EXECUTABLE_TRACKING:-no}"

# Tokenizers / telemetry / NCCL thread-budget caps (same as multi-node
# driver; see its ADDITIONAL THREAD-COUNT REDUCTION block for details).
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-1}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-1}"

echo "============================================="
echo "  Single-Node Qwen3-0.6B vLLM Sanity Benchmark"
echo "============================================="
echo "  Job ID:         ${PBS_JOBID:-N/A}"
echo "  Host:           $(hostname)"
echo "  Model:          ${MODEL}"
echo "  DType:          ${DTYPE}"
echo "  TP size:        ${TP_SIZE}"
echo "  PP size:        ${PP_SIZE}"
echo "  GPUs:           ${EXPECTED_GPUS}"
echo "  Backend:        ${DIST_BACKEND} (MultiprocExecutor, no Ray)"
echo "  Enforce eager:  ${ENFORCE_EAGER} (1 skips torch.compile + CUDA graphs)"
echo "  Max model len:  ${MAX_MODEL_LEN}"
echo "  Threads/worker: ${NUM_THREADS_PER_WORKER}"
echo "  Cache root:     ${CACHE_ROOT}"
echo "  Results dir:    ${RESULTS_DIR}"
echo "============================================="

# =============================================================================
# CLEANUP TRAP
# =============================================================================
# Single-node version: just kill the background vllm server if we started
# one. No Ray / no workers to tear down.

SERVER_PID=""

cleanup() {
	echo ""
	echo "Cleaning up..."

	if [[ -n "${SERVER_PID}" ]]; then
		echo "  Stopping vLLM server (PID ${SERVER_PID})..."
		kill "${SERVER_PID}" 2>/dev/null || true
		wait "${SERVER_PID}" 2>/dev/null || true
	fi

	echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

# =============================================================================
# SHARED ENGINE ARGS
# =============================================================================
# Kept in one place so latency / throughput / serve configure the engine
# identically. No quantization, no --enable-expert-parallel (Qwen3-0.6B
# is a dense model, not MoE).
ENGINE_ARGS=(
	--model "${MODEL}"
	--dtype "${DTYPE}"
	--tensor-parallel-size "${TP_SIZE}"
	--pipeline-parallel-size "${PP_SIZE}"
	--distributed-executor-backend "${DIST_BACKEND}"
	--max-model-len "${MAX_MODEL_LEN}"
	--trust-remote-code
	${EAGER_ARGS[@]+"${EAGER_ARGS[@]}"}
)

# =============================================================================
# STEP 1: Offline Latency Benchmark
# =============================================================================

echo ""
echo "[Step 1/3] Offline latency sweep..."
echo "  Engine config: TP=${TP_SIZE}, PP=${PP_SIZE}, backend=${DIST_BACKEND}"
echo ""

# Latency sweep matrix. Three batch sizes at one representative (input,
# output) shape: small / medium / large concurrency at fixed context.
# Each (batch_size, input_len, output_len) triple triggers a fresh
# engine init (model load, profile, KV cache, warmup).
#
# LATENCY_IO_CONFIGS_CSV uses the same "<input>:<output>,..." format as
# Step 3's IO_CONFIGS (e.g. "512:128,128:512").
BATCH_SIZES=(1 8 32)
LATENCY_IO_CONFIGS_CSV="${LATENCY_IO_CONFIGS_CSV:-512:128}"
IFS=',' read -r -a LATENCY_IO_CONFIGS <<<"${LATENCY_IO_CONFIGS_CSV}"

LATENCY_WARMUP_ITERS="${LATENCY_WARMUP_ITERS:-2}"
LATENCY_ITERS="${LATENCY_ITERS:-10}"

for IO_CONFIG in "${LATENCY_IO_CONFIGS[@]}"; do
	INPUT_LEN="${IO_CONFIG%%:*}"
	OUTPUT_LEN="${IO_CONFIG##*:}"

	for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
		RESULT_FILE="${RESULTS_DIR}/latency_bs${BATCH_SIZE}_in${INPUT_LEN}_out${OUTPUT_LEN}.json"
		LOG_FILE="${RESULTS_DIR}/latency_bs${BATCH_SIZE}_in${INPUT_LEN}_out${OUTPUT_LEN}.log"

		echo "  >> Latency: batch_size=${BATCH_SIZE}, input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}"

		vllm bench latency \
			"${ENGINE_ARGS[@]}" \
			--batch-size "${BATCH_SIZE}" \
			--input-len "${INPUT_LEN}" \
			--output-len "${OUTPUT_LEN}" \
			--num-iters-warmup "${LATENCY_WARMUP_ITERS}" \
			--num-iters "${LATENCY_ITERS}" \
			--output-json "${RESULT_FILE}" \
			2>&1 | tee "${LOG_FILE}"

		echo "  >> Saved to ${RESULT_FILE}"
		echo ""
	done
done

echo "Offline latency benchmarks complete."

# =============================================================================
# STEP 2: Offline Throughput Benchmark
# =============================================================================
# Sanity-checks max steady-state throughput on a saturated engine. Uses
# the `random` synthetic dataset so this step has no external dataset
# dependency.

echo ""
echo "[Step 2/3] Offline throughput run..."

THROUGHPUT_NUM_PROMPTS="${THROUGHPUT_NUM_PROMPTS:-200}"
THROUGHPUT_INPUT_LEN="${THROUGHPUT_INPUT_LEN:-512}"
THROUGHPUT_OUTPUT_LEN="${THROUGHPUT_OUTPUT_LEN:-128}"

TP_RESULT_FILE="${RESULTS_DIR}/throughput_in${THROUGHPUT_INPUT_LEN}_out${THROUGHPUT_OUTPUT_LEN}.json"
TP_LOG_FILE="${RESULTS_DIR}/throughput_in${THROUGHPUT_INPUT_LEN}_out${THROUGHPUT_OUTPUT_LEN}.log"

echo "  >> Throughput: num_prompts=${THROUGHPUT_NUM_PROMPTS}, input_len=${THROUGHPUT_INPUT_LEN}, output_len=${THROUGHPUT_OUTPUT_LEN}"

vllm bench throughput \
	"${ENGINE_ARGS[@]}" \
	--dataset-name random \
	--input-len "${THROUGHPUT_INPUT_LEN}" \
	--output-len "${THROUGHPUT_OUTPUT_LEN}" \
	--num-prompts "${THROUGHPUT_NUM_PROMPTS}" \
	--output-json "${TP_RESULT_FILE}" \
	2>&1 | tee "${TP_LOG_FILE}"

echo "  >> Saved to ${TP_RESULT_FILE}"
echo "Offline throughput complete."

# =============================================================================
# STEP 3: Online Serving Benchmark
# =============================================================================

echo ""
echo "[Step 3/3] Starting vLLM server and running serving benchmarks..."

# Start the vLLM server in the background.
vllm serve "${MODEL}" \
	--dtype "${DTYPE}" \
	--tensor-parallel-size "${TP_SIZE}" \
	--pipeline-parallel-size "${PP_SIZE}" \
	--distributed-executor-backend "${DIST_BACKEND}" \
	--max-model-len "${MAX_MODEL_LEN}" \
	--trust-remote-code \
	${EAGER_ARGS[@]+"${EAGER_ARGS[@]}"} \
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
	if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
		echo "ERROR: vLLM server process died unexpectedly. Tail of server.log:"
		tail -n 50 "${RESULTS_DIR}/server.log" || true
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

# Serving sweep: three request rates x two IO configs = 6 runs. Sized to
# run in under ~15 minutes on a tiny model.
REQUEST_RATES=(1 8 inf)
IO_CONFIGS=(
	"512:128"
	"128:512"
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
			backend="${DIST_BACKEND}" \
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
ls -1 "${RESULTS_DIR}"/latency_*.json 2>/dev/null || echo "  (none)"
echo ""
echo "Throughput results:"
ls -1 "${RESULTS_DIR}"/throughput_*.json 2>/dev/null || echo "  (none)"
echo ""
echo "Serving results:"
ls -1 "${RESULTS_DIR}"/serving_*.json 2>/dev/null || echo "  (none)"
echo ""
echo "Metrics recorded:"
echo "  - Offline latency:    end-to-end per batch size / input length"
echo "  - Offline throughput: requests/s and tokens/s on random dataset"
echo "  - Online serving:     TTFT, TPOT, ITL, E2EL at p50/p90/p95/p99"
echo "  - Online serving:     request throughput, token throughput"
echo ""
echo "Cleanup will run automatically via EXIT trap."

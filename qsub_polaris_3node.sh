#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# =============================================================================
# PBS Pro Job Script - 3-Node DeepSeek-R1 Benchmark on ALCF Polaris
# =============================================================================
#
# Single qsub-submittable script that orchestrates the full 3-node benchmark:
#   1. Discovers allocated nodes from PBS_NODEFILE
#   2. Launches Ray head on node 0, Ray workers on nodes 1 & 2 (via SSH)
#   3. Runs offline latency benchmarks (vllm bench latency)
#   4. Starts vLLM API server and runs online serving benchmarks
#   5. Cleans up Ray on all nodes on exit
#
# Configuration: PP=3, TP=4, EP enabled (ep_size=4 for MoE layers)
# Hardware:      3 Polaris nodes x 4x A100-80GB = 12 GPUs total
# Model:         deepseek-ai/DeepSeek-R1 (671B MoE, FP8)
#
# Usage:
#   # Edit the -A project allocation below, then:
#   qsub qsub_polaris_3node.sh
#
#   # Or override settings at submission time:
#   qsub -v MODEL=/path/to/model,MAX_MODEL_LEN=8192 qsub_polaris_3node.sh
#
# =============================================================================

# =============================================================================
# PBS DIRECTIVES
# =============================================================================

#PBS -l select=3:system=polaris:ncpus=32:ngpus=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -q debug-scaling
#PBS -A Intel
#PBS -N vllm-deepseek-r1-bench
#PBS -j oe
#PBS -V

set -euo pipefail

# Change to the directory from which qsub was invoked
cd "${PBS_O_WORKDIR:-$(pwd)}"

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Model configuration
MODEL="${MODEL:-/grand/Intel/dhuang/Deepseek-V3-0324}"
QUANTIZATION="${QUANTIZATION:-}"  # e.g. "fp8"; leave empty for no quantization
DTYPE="${DTYPE:-auto}"

# Build conditional quantization args (used by ENGINE_ARGS and vllm serve)
QUANT_ARGS=()
if [[ -n "${QUANTIZATION}" ]]; then
	QUANT_ARGS=(--quantization "${QUANTIZATION}")
fi

# Parallelism configuration
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-3}"

# Ray configuration
RAY_PORT="${RAY_PORT:-6379}"
EXPECTED_NODES=3
EXPECTED_GPUS=12                                  # 3 nodes * 4 GPUs
RAY_CLUSTER_TIMEOUT="${RAY_CLUSTER_TIMEOUT:-600}" # seconds

# GPU visibility (4 GPUs per node on Polaris)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

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
# are typically named hsn0, hsn1, etc. Uncomment and adjust if needed.
#
# export NCCL_SOCKET_IFNAME=hsn0
# export GLOO_SOCKET_IFNAME=hsn0
# export NCCL_DEBUG=WARN

# Disable Ray usage stats
export RAY_USAGE_STATS_ENABLED=0

# =============================================================================
# NODE DISCOVERY FROM PBS
# =============================================================================

if [[ -z "${PBS_NODEFILE:-}" ]]; then
	echo "ERROR: PBS_NODEFILE is not set. This script must be submitted via qsub."
	echo "Usage: qsub qsub_polaris_3node.sh"
	exit 1
fi

# Get unique hostnames preserving PBS allocation order.
# PBS runs this script on the first node, so ALL_NODES[0] is the head.
mapfile -t ALL_NODES < <(awk '!seen[$0]++' "${PBS_NODEFILE}")

if [[ "${#ALL_NODES[@]}" -lt 3 ]]; then
	echo "ERROR: Expected 3 nodes, but PBS allocated ${#ALL_NODES[@]}:"
	printf "  %s\n" "${ALL_NODES[@]}"
	exit 1
fi

HEAD_HOST="${ALL_NODES[0]}"
WORKER1_HOST="${ALL_NODES[1]}"
WORKER2_HOST="${ALL_NODES[2]}"

# Resolve hostnames to IPs for VLLM_HOST_IP / Ray communication.
# On Polaris, the PBS hostnames are directly resolvable.
resolve_ip() {
	local host="$1"
	# getent hosts returns "<ip> <hostname> [aliases...]"; grab the first field.
	# Fall back to the hostname itself if resolution fails.
	getent hosts "${host}" | awk '{print $1; exit}' || echo "${host}"
}

HEAD_IP="$(resolve_ip "${HEAD_HOST}")"
WORKER1_IP="$(resolve_ip "${WORKER1_HOST}")"
WORKER2_IP="$(resolve_ip "${WORKER2_HOST}")"

echo "============================================="
echo "  PBS 3-Node DeepSeek-R1 Benchmark"
echo "============================================="
echo "  Job ID:         ${PBS_JOBID:-N/A}"
echo "  Head node:      ${HEAD_HOST} (${HEAD_IP})"
echo "  Worker 1:       ${WORKER1_HOST} (${WORKER1_IP})"
echo "  Worker 2:       ${WORKER2_HOST} (${WORKER2_IP})"
echo "  Model:          ${MODEL}"
echo "  Quantization:   ${QUANTIZATION:-none}"
echo "  TP size:        ${TP_SIZE}"
echo "  PP size:        ${PP_SIZE}"
echo "  EP:             enabled"
echo "  Expected GPUs:  ${EXPECTED_GPUS}"
echo "  Max model len:  ${MAX_MODEL_LEN}"
echo "  Results dir:    ${RESULTS_DIR}"
echo "============================================="

# =============================================================================
# CLEANUP TRAP
# =============================================================================
# Ensure Ray is stopped on all nodes and background jobs are killed on exit.

WORKER1_PID=""
WORKER2_PID=""
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

	for WHOST in "${WORKER1_HOST}" "${WORKER2_HOST}"; do
		echo "  Stopping Ray on ${WHOST}..."
		ssh "${WHOST}" "ray stop --force" 2>/dev/null || true
	done

	# Kill background SSH sessions
	for WPID in "${WORKER1_PID}" "${WORKER2_PID}"; do
		if [[ -n "${WPID}" ]]; then
			kill "${WPID}" 2>/dev/null || true
			wait "${WPID}" 2>/dev/null || true
		fi
	done

	echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

# =============================================================================
# STEP 1: Start Ray Head Node (on this node)
# =============================================================================

echo ""
echo "[Step 1/5] Starting Ray head node on ${HEAD_HOST} (${HEAD_IP}:${RAY_PORT})..."

export VLLM_HOST_IP="${HEAD_IP}"

ray stop --force 2>/dev/null || true
sleep 2

ray start --head \
	--port="${RAY_PORT}" \
	--num-gpus="${TP_SIZE}" \
	--dashboard-host=0.0.0.0

echo "Ray head started. Dashboard: http://${HEAD_IP}:8265"

# =============================================================================
# STEP 2: Start Ray Workers on Nodes 1 & 2 (via SSH)
# =============================================================================

echo ""
echo "[Step 2/5] Starting Ray workers on ${WORKER1_HOST} and ${WORKER2_HOST}..."

# Helper: launch a Ray worker on a remote node via SSH.
# The worker runs with --block so the SSH session stays alive until Ray stops.
launch_worker() {
	local worker_host="$1"
	local worker_ip="$2"
	local worker_label="$3"

	ssh "${worker_host}" bash -l <<-WORKER_EOF
		set -euo pipefail

		export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
		export VLLM_HOST_IP="${worker_ip}"
		export RAY_USAGE_STATS_ENABLED=0

		echo "[${worker_label}] Stopping any existing Ray processes..."
		ray stop --force 2>/dev/null || true
		sleep 2

		echo "[${worker_label}] Joining Ray cluster at ${HEAD_IP}:${RAY_PORT}..."
		ELAPSED=0
		RETRY_INTERVAL=5
		JOIN_TIMEOUT=300

		while true; do
			if ray start \
				--address="${HEAD_IP}:${RAY_PORT}" \
				--num-gpus=${TP_SIZE} \
				--block; then
				echo "[${worker_label}] Disconnected from cluster."
				exit 0
			fi

			ELAPSED=\$((ELAPSED + RETRY_INTERVAL))
			if [[ "\${ELAPSED}" -ge "\${JOIN_TIMEOUT}" ]]; then
				echo "[${worker_label}] ERROR: Timed out joining cluster after \${JOIN_TIMEOUT}s."
				exit 1
			fi

			echo "[${worker_label}] Retrying in \${RETRY_INTERVAL}s... (\${ELAPSED}s elapsed)"
			sleep "\${RETRY_INTERVAL}"
		done
	WORKER_EOF
}

# Launch workers in background
launch_worker "${WORKER1_HOST}" "${WORKER1_IP}" "Worker1" &
WORKER1_PID=$!

launch_worker "${WORKER2_HOST}" "${WORKER2_IP}" "Worker2" &
WORKER2_PID=$!

echo "Worker SSH sessions launched (PIDs: ${WORKER1_PID}, ${WORKER2_PID})"

# =============================================================================
# STEP 3: Wait for All Nodes to Join the Cluster
# =============================================================================

echo ""
echo "[Step 3/5] Waiting for ${EXPECTED_NODES} nodes (${EXPECTED_GPUS} GPUs) to join..."

POLL_INTERVAL=5
ELAPSED=0

while true; do
	ACTIVE_NODES=$(python3 -c "
import ray
ray.init(address='auto', ignore_reinit_error=True)
nodes = ray.nodes()
alive = [n for n in nodes if n['Alive']]
total_gpus = sum(n['Resources'].get('GPU', 0) for n in alive)
print(f'{len(alive)},{int(total_gpus)}')
ray.shutdown()
" 2>/dev/null || echo "0,0")

	NUM_NODES=$(echo "${ACTIVE_NODES}" | cut -d',' -f1)
	NUM_GPUS=$(echo "${ACTIVE_NODES}" | cut -d',' -f2)

	echo "  Nodes: ${NUM_NODES}/${EXPECTED_NODES}, GPUs: ${NUM_GPUS}/${EXPECTED_GPUS} (${ELAPSED}s elapsed)"

	if [[ "${NUM_GPUS}" -ge "${EXPECTED_GPUS}" ]]; then
		echo "All nodes joined! Cluster ready with ${NUM_NODES} nodes and ${NUM_GPUS} GPUs."
		break
	fi

	if [[ "${ELAPSED}" -ge "${RAY_CLUSTER_TIMEOUT}" ]]; then
		echo "ERROR: Timed out waiting for nodes after ${RAY_CLUSTER_TIMEOUT}s."
		echo "       Got ${NUM_NODES} nodes / ${NUM_GPUS} GPUs, expected ${EXPECTED_NODES} / ${EXPECTED_GPUS}."
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
echo "  Engine config: PP=${PP_SIZE}, TP=${TP_SIZE}, EP=on"
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
	--disable-log-requests
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
	--disable-log-requests \
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
			ep=true \
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

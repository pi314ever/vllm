#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# =============================================================================
# Shared body for PBS Pro benchmark wrappers on ALCF Polaris
# =============================================================================
#
# This file is a SOURCED LIBRARY, not a submittable script. It is `source`d
# from thin wrapper files that supply their own #PBS directives and per-run
# environment overrides, e.g.:
#   qsub_polaris_deepseek_v3_latency_bs1.sh
#   qsub_polaris_deepseek_v3_latency_bs8.sh
#   qsub_polaris_deepseek_v3_latency_bs32.sh
#   qsub_polaris_deepseek_v3_serving.sh
#   qsub_polaris_deepseek_v3_throughput.sh
#
# Each wrapper:
#   1. Declares its own #PBS directives (select, walltime, queue, -N, etc.).
#   2. Exports config-specific overrides (BATCH_SIZES_CSV, RUN_LATENCY,
#      RUN_SERVING, RUN_THROUGHPUT, MODEL, ...).
#   3. Ends with `source "${PBS_O_WORKDIR}/qsub_polaris_body.sh"`.
#
# Responsibilities of this body:
#   1. Discover allocated nodes from PBS_NODEFILE.
#   2. Launch Ray head on node 0, Ray workers on nodes 1..N-1 (via mpiexec).
#   3. Optionally run offline latency benchmarks (Step 4; RUN_LATENCY=1).
#   4. Optionally start vLLM API server and run serving benchmarks
#      (Step 5; RUN_SERVING=1).
#   5. Optionally run offline throughput benchmarks (Step 6; RUN_THROUGHPUT=1).
#   6. Optionally run lm-evaluation-harness accuracy benchmarks (Step 7;
#      RUN_LM_EVAL=1). lm_eval drives a vllm.LLM(...) inside its own
#      vllm wrapper; the existing Ray cluster is reused via RAY_ADDRESS.
#   7. Clean up Ray on all nodes on exit.
#   8. Optionally wrap every bench invocation with the torch profiler and
#      emit Perfetto-viewable traces (PROFILE=1). See the PROFILING CONFIG
#      block below for knobs and trade-offs.
#
# Parallelism model (defaults):
#   NUM_NODES=6, GPUS_PER_NODE=4  -> 24 GPUs total
#   PP_SIZE=6, TP_SIZE=4          -> PP across nodes, TP within each node
#   --enable-expert-parallel      -> EP size = TP_SIZE * DP_SIZE = 4 (DP=1)
#
# NOTE on EP: vLLM exposes expert parallelism as a boolean flag
# (--enable-expert-parallel). The effective EP group size is TP * DP. To get
# EP=M with DP=1, set TP_SIZE=M. To get EP=M with DP>1, size TP and DP so
# that TP*DP=M and PP*TP*DP = NUM_NODES*GPUS_PER_NODE (DP is not configured
# by this script; extend QUANT_ARGS / ENGINE_ARGS if you need it).
#
# Hardware:      Polaris nodes x 4x A100-80GB
# Default model: deepseek-ai/DeepSeek-R1 (671B MoE, FP8); wrappers override.
#
# Usage:
#   # Submit a single wrapper:
#   qsub qsub_polaris_deepseek_v3_latency_bs8.sh
#
#   # Submit the whole DeepSeek-V3 suite:
#   for f in qsub_polaris_deepseek_v3_*.sh; do qsub "$f"; done
#
#   # Override a knob at submission time (qsub -v strips whitespace, so CSVs
#   # use comma separators, e.g. BATCH_SIZES_CSV=1,8,32):
#   qsub -v BATCH_SIZES_CSV=4,16,64 qsub_polaris_deepseek_v3_latency_bs8.sh
#
#   # Enable torch-profiler tracing for every bench invocation (emits
#   # gzipped Perfetto traces to /grand/Intel/dhuang/vllm_polaris_profiles/
#   # by default; see PROFILING CONFIG below):
#   qsub -v PROFILE=1 qsub_polaris_deepseek_v3_latency_bs8.sh
#
# =============================================================================

set -euo pipefail

# =============================================================================
# DIRECT-SUBMIT GUARD
# =============================================================================
# This file is not meant to be submitted directly (it has no #PBS directives
# of its own). If someone runs `qsub qsub_polaris_body.sh` or
# `bash qsub_polaris_body.sh`, PBS_NODEFILE is unset (no job context) AND
# BASH_SOURCE[0] == $0 (this file is the top-level script, not sourced).
# In that case, refuse with a clear message pointing at the wrapper files.
#
# When sourced from a wrapper, BASH_SOURCE[0] is this file's path but $0 is
# the wrapper's path, so the guard is skipped.
if [[ "${BASH_SOURCE[0]}" == "${0}" && -z "${PBS_NODEFILE:-}" ]]; then
	echo "ERROR: qsub_polaris_body.sh is a sourced library, not a"
	echo "       submittable script. Submit one of the wrapper files:"
	echo "         qsub qsub_polaris_deepseek_v3_latency_bs1.sh"
	echo "         qsub qsub_polaris_deepseek_v3_latency_bs8.sh"
	echo "         qsub qsub_polaris_deepseek_v3_latency_bs32.sh"
	echo "         qsub qsub_polaris_deepseek_v3_serving.sh"
	echo "         qsub qsub_polaris_deepseek_v3_throughput.sh"
	echo "         qsub qsub_polaris_deepseek_v3_lm_eval.sh"
	echo "       Or submit the whole suite:"
	echo "         for f in qsub_polaris_deepseek_v3_*.sh; do qsub \"\$f\"; done"
	exit 1
fi

# =============================================================================
# MULTI-INSTANCE PARTITIONING
# =============================================================================
# Two knobs that let multiple instances of this body run concurrently
# inside a single PBS allocation, each owning a disjoint slice of nodes:
#
#   NODE_OFFSET     : index into the unique-hosts list parsed from
#                     PBS_NODEFILE. The first node of THIS instance is
#                     ALL_NODES[NODE_OFFSET]; the next NUM_NODES-1 nodes
#                     after it are workers. Default 0 (single-instance
#                     behaviour, head = first allocated node).
#   INSTANCE_SUFFIX : string appended to every per-instance namespace
#                     (RAY_TMPDIR, JOB_NAME, and everything that derives
#                     from JOB_NAME: RUN_LOG_DIR, RESULTS_DIR, DIAG_DIR,
#                     SAMPLER_FLAG, PROFILE_DIR). Default empty (single-
#                     instance behaviour, paths use the bare PBS jobid).
#
# When sourcing/dispatching this body twice from a multi-instance wrapper,
# set NODE_OFFSET and INSTANCE_SUFFIX to disjoint values per instance
# (e.g. 0/"_A" and NUM_NODES/"_B"). All other shared state (caches on
# /grand, ports on each head's local stack) is naturally disjoint because
# each instance occupies different physical nodes.
NODE_OFFSET="${NODE_OFFSET:-0}"
INSTANCE_SUFFIX="${INSTANCE_SUFFIX:-}"

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
MODEL="${MODEL:-/grand/Intel/dhuang/DeepSeek-V3/}"
QUANTIZATION="${QUANTIZATION-}" # e.g. "fp8"; empty (default) disables --quantization
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
NUM_NODES="${NUM_NODES:-6}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-6}"

# Ray configuration
RAY_PORT="${RAY_PORT:-6379}"
EXPECTED_NODES="${NUM_NODES}"
EXPECTED_GPUS=$((NUM_NODES * GPUS_PER_NODE))
EP_SIZE="${TP_SIZE}" # derived; only valid while DP=1
RAY_CLUSTER_TIMEOUT="${RAY_CLUSTER_TIMEOUT:-600}" # seconds

# Ray CPU slots per node. This caps Ray's idle-worker pre-spawn pool,
# which is the single largest consumer of PBS cgroup pids budget on
# Polaris. See the "RAY IDLE WORKER POOL LIMIT" block below for the
# full rationale. TL;DR: raylet defaults --maximum_startup_concurrency
# from `ncpus`, which on Polaris is 32 (physical) or 64 (SMT), and
# then pre-spawns up to 64 `ray::IDLE` Python workers per node. At
# ~48 threads each that is ~3000 threads of dead weight inside the
# 4096 pids.max cap, leaving the vLLM engine <600 threads of headroom
# before CUDA graph capture / Triton JIT / lazy NCCL P2P blows past
# the limit with EAGAIN on pthread_create. Setting --num-cpus to a
# small value (matching GPUS_PER_NODE) reduces the idle pool to a
# handful of workers. vLLM's Ray placement-group bundles request
# `{"GPU": 1.0}` only (see vllm/v1/executor/ray_utils.py:635), and
# worker remote kwargs use num_cpus=0 (ray_executor_v2.py:344), so
# reducing --num-cpus does NOT interfere with vLLM scheduling.
NUM_RAY_CPUS_PER_NODE="${NUM_RAY_CPUS_PER_NODE:-${GPUS_PER_NODE}}"

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
#
# INSTANCE_SUFFIX disambiguates concurrent instances inside one PBS job
# (see MULTI-INSTANCE PARTITIONING block above). For single-instance
# runs the suffix is empty and this resolves to the prior path.
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${PBS_JOBID%%.*}${INSTANCE_SUFFIX}}"
export RAY_TMPDIR
mkdir -p "${RAY_TMPDIR}"

# Same fix shape for vLLM's ZMQ IPC sockets. vllm/utils/network_utils.py's
# get_open_zmq_ipc_path() builds:
#     ipc://${VLLM_RPC_BASE_PATH}/<uuid4>
# and VLLM_RPC_BASE_PATH defaults to tempfile.gettempdir(), which on
# PBS-Polaris reads $TMPDIR (the same long /var/tmp/pbs.<jobid>.<long-host>/
# <uuid>/tmp/ path that broke Ray). Appending /<uuid4> (37 chars) blows
# past the kernel's 107-byte sockaddr_un.sun_path limit and zmq.bind()
# rejects it with:
#     ZMQError: ipc path "..." is longer than 107 characters
# (zmq.IPC_PATH_MAX_LEN). This bites `vllm bench latency` because it runs
# in offline mode (client_local_only=True) and so the engine-client
# address comes from get_open_zmq_ipc_path() rather than a TCP URI.
#
# Pinning a short, job-scoped base path here keeps the socket name well
# under the limit (e.g. /tmp/vllm_rpc_7158834_A/<uuid> ~ 60 chars).
VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH:-/tmp/vllm_rpc_${PBS_JOBID%%.*}${INSTANCE_SUFFIX}}"
export VLLM_RPC_BASE_PATH
mkdir -p "${VLLM_RPC_BASE_PATH}"

# GPU visibility: expose GPUS_PER_NODE devices per node (0,1,...,N-1)
_default_cvd="$(seq -s, 0 $((GPUS_PER_NODE - 1)))"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${_default_cvd}}"
unset _default_cvd

# Short, filesystem-friendly PBS job id (no host suffix). Used as the
# unique per-run namespace component below. For manual / no-PBS runs
# this falls back to "manual". Exported so remote mpiexec subshells
# (diagnostics, sampler) can read it for cgroup resolution without
# re-parsing $PBS_JOBID.
PBS_SHORT_JOBID="${PBS_JOBID%%.*}"
PBS_SHORT_JOBID="${PBS_SHORT_JOBID:-manual}"
export PBS_SHORT_JOBID

# Per-run log/output namespace.
#   logs/<job_name>/run.log         -> all stdout/stderr from this script
#   logs/<job_name>/results/        -> RESULTS_DIR (benchmark outputs + diag)
#
# <job_name> = "${PBS_JOBNAME}.o${PBS_SHORT_JOBID}${INSTANCE_SUFFIX}",
# matching PBS's own default output file convention (<jobname>.o<jobid>)
# so a user scanning `ls logs/` can correlate with the PBS job list
# easily while still getting per-run history (no cross-run overwrites).
# INSTANCE_SUFFIX (default empty) disambiguates concurrent instances
# inside one PBS job (see MULTI-INSTANCE PARTITIONING block above).
JOB_NAME="${JOB_NAME:-${PBS_JOBNAME:-vllm-bench}.o${PBS_SHORT_JOBID}${INSTANCE_SUFFIX}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${PBS_O_WORKDIR:-.}/logs/${JOB_NAME}}"
mkdir -p "${RUN_LOG_DIR}"

# Tee every byte written from this point forward to run.log. PBS still
# maintains its own <jobname>.o<jobid> spool file in PBS_O_WORKDIR
# (redundant but harmless backup; users can ignore it).
#
# Using `exec > >(tee ...)` with `set -e` works fine: the tee runs in
# a background process-substitution subshell, and bash's signal /
# pipefail handling is unaffected. The cleanup trap kills the
# explicitly-tracked worker / sampler PIDs; the tee process exits
# naturally when its stdin closes at end-of-script.
RUN_LOG_FILE="${RUN_LOG_DIR}/run.log"
exec > >(tee -a "${RUN_LOG_FILE}") 2>&1
echo "Run log teeing to: ${RUN_LOG_FILE}"

# Benchmark output directory. Nested under ${RUN_LOG_DIR} so all
# per-run artifacts (run.log, results/, diagnostics/) live in one
# namespace. Override RESULTS_DIR at submission time to decouple.
RESULTS_DIR="${RESULTS_DIR:-${RUN_LOG_DIR}/results}"

# Diagnostics subdirectory for per-phase, per-node thread/pid dumps.
# Created early so the very first diagnostic call at job start
# (before ray stop) has a place to write.
DIAG_DIR="${RESULTS_DIR}/diagnostics"
mkdir -p "${DIAG_DIR}"

# Background-sampler kill-switch flag. The sampler loop (launched in
# Step 4) runs while this file exists and exits cleanly when it is
# removed by the cleanup / ERR trap or the normal end-of-step code.
SAMPLER_FLAG="${DIAG_DIR}/.sampler_run"

# Server port for online serving benchmark.
#
# The default is derived per-job from PBS_SHORT_JOBID rather than
# hard-coded to 8000, to sidestep SO_REUSEPORT collisions with stale
# listeners left over from prior jobs that happened to land on the
# same Polaris head node. (See vllm/entrypoints/openai/api_server.py
# create_server_socket(): the API-server socket has both SO_REUSEADDR
# and SO_REUSEPORT, so a leftover process from a previous run can
# silently shadow our listener and absorb /health traffic, while the
# pre-flight ss check below only catches sockets that have already
# called listen().) The derivation is deterministic so an operator
# can predict the port from the job id when SSH'ing onto the head
# node mid-run; the RANDOM fallback covers the "manual" case where
# the body is sourced outside PBS.
#
# Range 30000-39999 sits below the top of the typical Linux ephemeral
# range (32768-60999); the kernel only auto-assigns a specific
# ephemeral port if free at bind time, and vLLM's SO_REUSEADDR makes
# any rare race benign in practice. Override at submission with
# `qsub -v SERVE_PORT=<N>` if you need a fixed port.
if [[ "${PBS_SHORT_JOBID}" =~ ^[0-9]+$ ]]; then
	SERVE_PORT="${SERVE_PORT:-$((30000 + (PBS_SHORT_JOBID % 10000)))}"
else
	SERVE_PORT="${SERVE_PORT:-$((30000 + RANDOM % 10000))}"
fi
echo "Server port: ${SERVE_PORT} (derived from PBS_SHORT_JOBID=${PBS_SHORT_JOBID})"

# Max model length. Optional; empty (default) omits --max-model-len so vLLM
# falls back to the model config's max_position_embeddings. Set to an int
# (e.g. `qsub -v MAX_MODEL_LEN=4096 ...`) to cap context length, which
# reduces KV-cache reservation and speeds up engine init on large models.
MAX_MODEL_LEN="${MAX_MODEL_LEN-}"

# Build conditional max-model-len args (used by ENGINE_ARGS and vllm serve)
MAX_MODEL_LEN_ARGS=()
if [[ -n "${MAX_MODEL_LEN}" ]]; then
	MAX_MODEL_LEN_ARGS=(--max-model-len "${MAX_MODEL_LEN}")
fi

# Number of prompts / warmups for serving benchmark.
# Sized to fit within Polaris's 1h debug-scaling queue walltime. Raise
# when submitting to a longer-walltime queue (e.g. prod).
NUM_PROMPTS="${NUM_PROMPTS:-200}"
NUM_WARMUPS="${NUM_WARMUPS:-2}"

# =============================================================================
# BENCHMARK STEP GATES
# =============================================================================
# Each of Steps 4, 5, 6, 7 can be independently turned on or off by the
# wrapper. A wrapper that only wants the latency sweep sets
# RUN_LATENCY=1, RUN_SERVING=0, RUN_THROUGHPUT=0, RUN_LM_EVAL=0 before
# `source`ing this body. Defaults leave latency + serving on (matches
# the pre-refactor monolith), throughput off (it's opt-in), and lm_eval
# off (added later for accuracy benchmarks; opt-in via the dedicated
# qsub_polaris_deepseek_v3_lm_eval.sh wrapper).
#
# Steps 1-3 (Ray cluster bring-up and 00-04 diagnostics) always run;
# a benchmark job that disables all four RUN_* knobs still brings up
# the cluster and takes process-limit snapshots, which is useful for
# pure infrastructure debugging.
RUN_LATENCY="${RUN_LATENCY:-1}"
RUN_SERVING="${RUN_SERVING:-1}"
RUN_THROUGHPUT="${RUN_THROUGHPUT:-0}"
RUN_LM_EVAL="${RUN_LM_EVAL:-0}"

# -----------------------------------------------------------------------------
# Step 7: lm-evaluation-harness (accuracy) configuration
# -----------------------------------------------------------------------------
# Only consulted when RUN_LM_EVAL=1 (default 0). Defaults reproduce the
# DeepSeek-V3 reference accuracy benchmark:
#   lm_eval --model vllm \
#     --model_args pretrained=...,trust_remote_code=True,enforce_eager=True,
#                  tensor_parallel_size=...,max_num_batched_tokens=4096,
#                  max_model_len=4096,moe_backend=triton,
#                  enable_expert_parallel=True \
#     --tasks gsm8k --limit 128 --num_fewshot 8
# MODEL, TP_SIZE and MAX_MODEL_LEN are sourced from the body's top-level
# config (so a single qsub -v override updates the Ray cluster sizing
# AND the lm_eval engine). Everything below is lm_eval-specific.
#
# LM_EVAL_TASKS              : comma-separated lm-eval-harness task list.
# LM_EVAL_LIMIT              : per-task example cap. Empty/unset (default)
#                              passes no --limit flag to lm_eval, which
#                              evaluates the full task split. Set to an
#                              int (e.g. `qsub -v LM_EVAL_LIMIT=128`) for
#                              a quick smoke run; lm_eval also accepts a
#                              float in (0,1) for fractional caps.
# LM_EVAL_NUM_FEWSHOT        : few-shot context count.
# LM_EVAL_MAX_NUM_BATCHED_TOKENS : engine batch budget. Bounded above
#                              by MAX_MODEL_LEN to keep prefill within
#                              KV cache.
# LM_EVAL_MOE_BACKEND        : MoE kernel backend. vLLM auto-selects
#                              the CUDA backend on A100 (faster), but
#                              the reference command pins triton; keep
#                              the reference default to make Polaris
#                              numbers comparable to other sites.
LM_EVAL_TASKS="${LM_EVAL_TASKS:-gsm8k}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT-}"
LM_EVAL_NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT:-8}"
LM_EVAL_MAX_NUM_BATCHED_TOKENS="${LM_EVAL_MAX_NUM_BATCHED_TOKENS:-4096}"
LM_EVAL_MOE_BACKEND="${LM_EVAL_MOE_BACKEND:-triton}"

# =============================================================================
# PROFILING CONFIG
# =============================================================================
# Torch-profiler integration. When PROFILE=1, every enabled bench step below
# (Step 4 latency / Step 5 serving / Step 6 throughput) is invoked with
# vLLM's torch-profiler wiring:
#
#   --profiler-config '{"profiler":"torch","torch_profiler_dir":"<dir>",
#                       "torch_profiler_record_shapes":true,
#                       "torch_profiler_with_stack":false}'
#   --profile
#
# Field choices (vs. vllm/config/profiler.py defaults):
#   * torch_profiler_record_shapes=true  (default false): records tensor
#     shapes so the trace distinguishes e.g. a prefill vs decode matmul
#     with different (M,K) shapes. Small overhead, useful for reading
#     traces on a heterogeneous PP/TP deployment.
#   * torch_profiler_with_stack=false   (default true):  skips Python
#     stack capture. Stacks are large and mostly redundant once you
#     have shapes; omitting them trims trace size meaningfully on 24
#     ranks.
#   * torch_profiler_use_gzip=true, torch_profiler_dump_cuda_time_total=true,
#     torch_profiler_with_memory=false, torch_profiler_with_flops=false
#     are all left at their defaults.
#
# Mechanism:
#   * `vllm/benchmarks/{latency,throughput,serve}.py` all accept --profile
#     and thread --profiler-config through to the engine. See
#     vllm/config/profiler.py:33 (ProfilerConfig) for the full field list.
#   * The driver's ProfilerConfig is transported to every worker over Ray
#     via VllmConfig, so worker-side tracing "just works" with no extra
#     env plumbing in launch_worker(). Verified at
#     vllm/v1/worker/gpu_worker.py:143-151.
#   * Each worker emits one `rank_<N>.pt.trace.json.gz` file per run into
#     torch_profiler_dir. Driver-side AsyncLLM front-end traces are also
#     written to the same directory (vllm/v1/engine/async_llm.py:179-196).
#   * Traces are viewable directly at https://ui.perfetto.dev/ (no untar
#     required; the UI accepts .json.gz).
#
# Cost / trade-offs on DeepSeek-V3 @ PP=6, TP=4 (24 ranks):
#   * Traces are large. Expect O(100 MB) per rank per short run, so a
#     single (bs, input_len) pair can produce ~2-24 GB of traces. The
#     whole Step 4 sweep with 3 batch sizes x 1 input length = ~3x that.
#     Plan for tens of GB on disk when PROFILE=1.
#   * `/stop_profile` synchronously flushes every rank's trace before
#     returning. For a 671B model this can take several minutes; we
#     bump VLLM_RPC_TIMEOUT below to keep the bench client from
#     timing out mid-flush.
#   * Profiling adds runtime overhead (doc warns "will significantly
#     slow down the inference"; see docs/contributing/profiling.md:4).
#     We reduce the per-step workload to the minimum that still covers
#     engine init + a couple of forward passes:
#       - `vllm bench latency --profile` already internally runs exactly
#         ONE profiled iteration after warmup and then returns
#         (vllm/benchmarks/latency.py:139-149). `--num-iters` is ignored
#         in this path; only --num-iters-warmup matters.
#       - `vllm bench throughput --profile` wraps the full --num-prompts
#         batch. Overridden to PROFILE_NUM_PROMPTS.
#       - `vllm bench serve --profile` hits /start_profile before the
#         sweep and /stop_profile after (serve.py:751,1107). Overridden
#         to PROFILE_NUM_PROMPTS per request rate.
#   * First-run cold compile still runs inside the profiled region,
#     so the first trace will include gcc/ld fork noise from Triton
#     JIT. Re-run on a warm TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR
#     (already set to /grand/.../vllm_polaris_cache above) to get a
#     clean steady-state trace.
#
# Filesystem layout:
#   PROFILE_DIR/
#     latency_bs<BS>_in<IN>_out<OUT>/      (one dir per Step 4 invocation)
#     throughput_in<IN>_out<OUT>/          (one dir per Step 6 invocation)
#     serving/                             (single dir; start_profile has
#                                           no prefix arg so all Step 5
#                                           runs share this dir)
#   Traces land on /grand by default so they're visible from every node
#   AND don't eat the repo checkout's /home quota. Override at submit
#   time via `qsub -v PROFILE_DIR=/path/to/out` if needed.
#
# Knobs:
PROFILE="${PROFILE:-0}"
PROFILE_DIR="${PROFILE_DIR:-/grand/Intel/dhuang/vllm_polaris_profiles/${JOB_NAME}}"
# Reduced workload sizes used only when PROFILE=1. Kept small because
# every rank flushes its own multi-hundred-MB trace on stop and trace
# volume scales linearly with workload. Override at submit time if you
# want a longer-steady-state trace (and have the disk / walltime for
# it).
PROFILE_NUM_PROMPTS="${PROFILE_NUM_PROMPTS:-5}"
PROFILE_LATENCY_WARMUP_ITERS="${PROFILE_LATENCY_WARMUP_ITERS:-2}"

if [[ "${PROFILE}" == "1" ]]; then
	# Bump the internal RPC timeout so that the client side of
	# `vllm bench serve --profile` does not abort while `/stop_profile`
	# is still flushing per-rank trace files. 30 minutes matches the
	# upper bound cited in docs/contributing/profiling.md. Honor any
	# pre-existing user override (don't clobber).
	export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
fi

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
# SUBPROCESS FAN-OUT LIMITS (Inductor + ALCF XALT)
# =============================================================================
# On Polaris we have repeatedly hit:
#
#   /opt/cray/pe/lmod/lmod/init/bash: fork: retry: Resource temporarily unavailable
#   /soft/xalt/.../bin/ld: fork: retry: Resource temporarily unavailable
#   collect2: error: ld returned 254 exit status
#   ... subprocess.CalledProcessError: ... gcc ... cuda_utils.c ...
#
# during torch.compile / Triton JIT on worker startup. Root causes:
#
# 1. torch._inductor.async_compile defaults TORCHINDUCTOR_COMPILE_THREADS
#    to min(32, nproc). With TP=4 workers per node on 32-core Polaris
#    nodes, that is 4 * 32 = 128 gcc-invoking helper processes per node
#    on top of the main worker / Ray actor / driver processes.
#
# 2. On ALCF, /usr/bin/ld is shadowed by /soft/xalt/.../bin/ld, a bash
#    wrapper that re-sources Lmod init on every ld invocation. Each
#    gcc->ld call therefore fans out into several extra fork()s
#    (bash + module-source subshells) before the real linker runs.
#
# Together these saturate the per-user / cgroup pids budget and the
# first synchronous compile_module_from_src("cuda_utils") call in the
# main worker dies with EAGAIN from fork().
#
# Fix 1: force Inductor to compile serially in each worker. First-run
# cold-compile is slower, but results land in TORCHINDUCTOR_CACHE_DIR
# (shared on /grand) so later runs skip it.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

# Fix 2: disable XALT executable tracking so the ld wrapper short-
# circuits to a plain exec of the real linker and stops re-sourcing
# Lmod on every link. This is a documented ALCF escape hatch; it only
# affects XALT's telemetry, not job correctness or allowed software.
export XALT_EXECUTABLE_TRACKING="${XALT_EXECUTABLE_TRACKING:-no}"

# =============================================================================
# ADDITIONAL THREAD-COUNT REDUCTION (targets cgroup pids.max pressure)
# =============================================================================
# After the Inductor + XALT fixes, the engine made it all the way through
# model init and CUDA graph capture, but then died at libzmq's
# `pthread_create` in the EngineCore output-dispatcher thread with
# "Resource temporarily unavailable". Diagnostics showed ulimit -u at
# 2.06M so RLIMIT_NPROC is not the cap; the binding limit must be the
# PBS cgroup's pids.max (which we couldn't read at cgroup root and
# which the expanded diagnostics will now walk /proc/self/cgroup to
# find).
#
# ZMQ's own footprint inside EngineCore is already at its floor: every
# context uses io_threads=1, which is libzmq's minimum for TCP
# sockets. The real headroom is in other subsystems' thread pools.
# Each knob below targets a specific source; together they should
# free ~30-80 threads per node at steady state.

# NCCL: default socket-thread fan-out (NCCL_SOCKET_NTHREADS *
# NCCL_NSOCKS_PERTHREAD) is tuned for hundred-GB throughput. On
# 16-GPU DeepSeek at benchmark loads it's overkill and contributes
# ~30-50 threads per node. May reduce inter-node all-reduce bandwidth;
# revert both for production-throughput serving runs.
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-1}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-1}"

# HuggingFace tokenizers: disable Rayon parallelism entirely. At
# NUM_PROMPTS=200 the tokenizer throughput is not a bottleneck and
# the Rayon pool is pure thread-budget waste. Emits a one-time HF
# warning post-fork; harmless.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Ray: disable the memory-monitor thread. At DeepSeek-671B /
# A100-80GB / 16-GPU scale the workload is statically sized, so
# OOM-watching buys us nothing and the thread counts against the
# cgroup pids budget.
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"

# vLLM / HuggingFace telemetry: skip background usage-stats upload
# thread. Two spellings because different upstream code paths honor
# different env var names.
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"

# =============================================================================
# RAY IDLE WORKER POOL LIMIT
# =============================================================================
# Ray's raylet pre-spawns a pool of idle Python workers per node. The size
# of that pool is driven by `ray start --num-cpus` (or, if unset, the
# raylet's auto-discovered CPU count via Ray's `--maximum_startup_concurrency`
# which defaults to 2x physical cores on Polaris -> 64). Once vLLM's
# placement-group request fires during engine init, Ray balloons the
# idle pool to match `num_cpus`, and each `ray::IDLE` Python worker
# carries ~48 threads (torch + CUDA context + Ray internal pools). On
# a 6-node run with pids.max=4096 per job cgroup, this chews up ~3500
# of 4096 pids per node BEFORE the engine starts any compute, leaving
# <600 threads of headroom for CUDA graph capture, Triton JIT
# (gcc+ld forks), and lazy 2-rank NCCL subcomm creation from
# unbatched PP P2P ops. The result is EAGAIN on pthread_create at
# first forward pass, worker death, and a 30-min downstream gloo
# recv timeout.
#
# Setting RAY_num_workers_soft_limit caps the soft limit of idle
# Python workers Ray maintains, independent of --num-cpus. With
# both knobs set, the idle pool stabilizes at ~NUM_RAY_CPUS_PER_NODE
# workers per node -- saving ~2800 threads per node at the DeepSeek
# scale above.
export RAY_num_workers_soft_limit="${RAY_num_workers_soft_limit:-${NUM_RAY_CPUS_PER_NODE}}"

# =============================================================================
# PYTORCH NCCL PER-PG THREAD MINIMIZATION
# =============================================================================
# vLLM's isend_tensor_dict / irecv_tensor_dict (parallel_state.py:899,1002)
# calls torch.distributed.isend/irecv directly on the PP device_group
# (NCCL, size=PP_SIZE) -- it does NOT route through pynccl for CUDA
# PP tensors. In PyTorch's default lazy-init mode, each unique
# (src, dst) pair inside a size-N NCCL ProcessGroup causes a NEW
# 2-rank NCCL subcommunicator to be created on first use
# ("An unbatched P2P op (send/recv) was called on this ProcessGroup
# with size N. In lazy initialization mode, this will result in a
# new 2-rank NCCL communicator to be created."). Each new subcomm
# spawns:
#   1. a ProcessGroupNCCL watchdog thread,
#   2. a NCCL proxy / service thread (libnccl),
#   3. a heartbeat monitor thread (TORCH_NCCL_ENABLE_MONITORING, on
#      by default),
#   4. optional trace-dump threads.
#
# At PP=6, TP=4 there are up to 5 send pairs x 4 TP ranks = 20 new
# 2-rank subcomms cluster-wide, each adding ~3 threads -> ~60 extra
# threads cluster-wide, ~10/node, appearing AT first forward pass
# (CUDA graph capture + Triton JIT) when the thread budget is
# already most strained.
#
# We cannot disable the watchdog (#1) -- it's required for error
# propagation. We can turn off #3 and #4, which are telemetry-only:
#   - TORCH_NCCL_ENABLE_MONITORING=0 : disables heartbeat monitor
#     thread. No correctness impact; only disables PyTorch's passive
#     NCCL hang-detection telemetry.
#   - TORCH_NCCL_TRACE_BUFFER_SIZE=0 : already default on recent
#     torch, set explicitly to suppress any trace-capture thread.
#   - TORCH_NCCL_DUMP_ON_TIMEOUT=0   : belt-and-suspenders against
#     any trace-dump-on-timeout worker.
#
# NOT setting TORCH_NCCL_ASYNC_ERROR_HANDLING -- leaving it at the
# PyTorch default so NCCL errors still surface as Python exceptions
# (vs. silent deadlocks).
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-0}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-0}"

# =============================================================================
# NODE DISCOVERY FROM PBS
# =============================================================================

if [[ -z "${PBS_NODEFILE:-}" ]]; then
	echo "ERROR: PBS_NODEFILE is not set. This script must be submitted via qsub."
	echo "Usage: qsub qsub_polaris.sh"
	exit 1
fi

# Get unique hostnames preserving PBS allocation order.
# PBS runs this script on the first node by default, so ALL_NODES[0] is
# the natural head for single-instance runs (NODE_OFFSET=0). Multi-
# instance wrappers dispatch the body onto the per-instance head via
# mpiexec so non-zero NODE_OFFSETs still resolve correctly.
mapfile -t ALL_NODES < <(awk '!seen[$0]++' "${PBS_NODEFILE}")

# Total nodes this body needs from the allocation = NUM_NODES at the
# given NODE_OFFSET. Multi-instance wrappers may allocate more than
# NUM_NODES total (e.g. select=12, two instances of NUM_NODES=6); the
# guard below only checks that THIS instance's slice exists.
REQUIRED_NODES=$((NODE_OFFSET + NUM_NODES))
if [[ "${#ALL_NODES[@]}" -lt "${REQUIRED_NODES}" ]]; then
	echo "ERROR: Need ${REQUIRED_NODES} unique nodes (NODE_OFFSET=${NODE_OFFSET} + NUM_NODES=${NUM_NODES}),"
	echo "       but PBS allocated ${#ALL_NODES[@]}:"
	printf "  %s\n" "${ALL_NODES[@]}"
	exit 1
fi

HEAD_HOST="${ALL_NODES[NODE_OFFSET]}"
# Take exactly NUM_NODES-1 workers starting one slot past HEAD_HOST.
# Slice form: "${ARR[@]:start:count}" -> elements [start, start+count).
WORKER_HOSTS=("${ALL_NODES[@]:$((NODE_OFFSET + 1)):$((NUM_NODES - 1))}")

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
echo "  Instance:       offset=${NODE_OFFSET} suffix=${INSTANCE_SUFFIX:-<none>}"
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
echo "  Max model len:  ${MAX_MODEL_LEN:-model default}"
echo "  Threads/worker: ${NUM_THREADS_PER_WORKER} (OMP/BLAS/MKL/Rayon)"
echo "  Ray CPUs/node:  ${NUM_RAY_CPUS_PER_NODE} (caps ray::IDLE pool)"
echo "  Cache root:     ${CACHE_ROOT}"
echo "  Run log:        ${RUN_LOG_FILE}"
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

	# Stop the background sampler first so its final CSV rows land
	# before the Ray teardown spins down the nodes. Safe to call even
	# if the sampler was never started.
	stop_background_sampler 2>/dev/null || true

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

	# Clean up the vLLM RPC IPC socket directory on all nodes. vLLM
	# normally unlinks its own sockets, but a crashed engine can leave
	# stale entries; remove the whole job-scoped dir defensively. Only
	# the head ever opens sockets here today, but the worker sweep is
	# cheap and future-proofs against engine processes that bind on
	# the workers (e.g. multi-engine setups).
	if [[ -n "${VLLM_RPC_BASE_PATH:-}" ]]; then
		echo "  Removing vLLM RPC base dir ${VLLM_RPC_BASE_PATH}..."
		rm -rf "${VLLM_RPC_BASE_PATH}" 2>/dev/null || true
		for WHOST in "${WORKER_HOSTS[@]}"; do
			mpiexec -n 1 --ppn 1 --hosts "${WHOST}" -- bash -c \
				"rm -rf '${VLLM_RPC_BASE_PATH}'" 2>/dev/null || true
		done
	fi

	echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

# =============================================================================
# PROCESS-LIMIT DIAGNOSTICS (read-only)
# =============================================================================
# Captures the binding pids/threads limit shape at various points in the
# job's lifecycle. Useful when workers are dying with EAGAIN-shaped errors
# (fork / pthread_create / libzmq "Resource temporarily unavailable").
# Everything here is read-only (ulimit reads, cat on /sys/fs/cgroup and
# /proc, ps). No limits are ever raised.
#
# print_process_diagnostics runs on ONE host (the current one). Output is
# emitted to stdout AND tee'd to ${DIAG_DIR}/<safe_label>_<hostname>.txt
# so we can diff across phases and across nodes after the job finishes.
#
# print_process_diagnostics_all_nodes fans the single-node function out to
# every node via mpiexec. Use this at phase boundaries; it's the only way
# to see per-node thread pressure (the cgroup pids.max is per-node).
#
# Called at (via the all-nodes wrapper):
#   - 00_job_start            (before ray stop / module load)
#   - 01_pre_ray              (after env/module setup, before ray start)
#   - 02_head_ray_up          (after head ray start, before workers join)
#   - 03_cluster_ready        (end of Step 3, all nodes joined)
#   - 04_pre_engine           (start of Step 4, immediately before vllm bench)
#   - 05_pre_vllm_serve       (start of Step 5; only reached on success path)
#   - FAILURE_rc<N>           (from the ERR trap on any command failure)
print_process_diagnostics() {
	local label="${1:-diagnostics}"
	local hostname
	hostname="$(hostname 2>/dev/null || echo unknown)"

	# Sanitize label for use as a filename component. Everything outside
	# [A-Za-z0-9._-] collapses to '_' so labels like "FAILURE (rc=1)" and
	# "cluster ready (post-Step 3)" become well-behaved file paths.
	local safe_label
	safe_label="$(printf '%s' "${label}" | tr -c '[:alnum:]._-' '_' | sed -E 's/_+/_/g; s/^_//; s/_$//')"
	local out_file=""
	if [[ -n "${DIAG_DIR:-}" && -d "${DIAG_DIR}" ]]; then
		out_file="${DIAG_DIR}/${safe_label}_${hostname}.txt"
	fi

	# Emit to stdout and (if DIAG_DIR exists) also to the per-phase file.
	# Using a group redirect with tee keeps the PBS .oe stream intact
	# while producing per-node artifacts for offline analysis.
	{
		echo ""
		echo "=== Process limit diagnostics: ${label} (host ${hostname}) ==="

		# ulimits
		echo "  ulimit -u  (soft nproc): $(ulimit -u 2>/dev/null || echo unknown)"
		echo "  ulimit -Hu (hard nproc): $(ulimit -Hu 2>/dev/null || echo unknown)"
		echo "  ulimit -s  (stack KB):   $(ulimit -s 2>/dev/null || echo unknown)"

		# Kernel-wide ceilings
		if [[ -r /proc/sys/kernel/pid_max ]]; then
			echo "  kernel pid_max:          $(cat /proc/sys/kernel/pid_max)"
		fi
		if [[ -r /proc/sys/kernel/threads-max ]]; then
			echo "  kernel threads-max:      $(cat /proc/sys/kernel/threads-max)"
		fi

		# Find the actual cgroup pids.max for THIS process. PBS on Polaris
		# puts each job in a per-job cgroup subtree, so the root
		# /sys/fs/cgroup/pids.max (if it exists at all) is not the binding
		# limit.
		#
		# Polaris-specific fast path: the binding cgroup is
		# /sys/fs/cgroup/jobs/<short_jobid>/. Try this path FIRST because
		# the /proc/self/cgroup walk below can land in a sibling cgroup
		# (e.g. palsd.service under nested mpiexec) whose pids.max reads
		# "max" -- yielding a misleading diagnostic even when the binding
		# cap is 4096. If the Polaris path is unreadable, fall through
		# to the generic walk for portability.
		local pids_max_file=""
		local pids_cur_file=""
		if [[ -n "${PBS_SHORT_JOBID:-}" && -r "/sys/fs/cgroup/jobs/${PBS_SHORT_JOBID}/pids.max" ]]; then
			pids_max_file="/sys/fs/cgroup/jobs/${PBS_SHORT_JOBID}/pids.max"
			pids_cur_file="/sys/fs/cgroup/jobs/${PBS_SHORT_JOBID}/pids.current"
		fi

		local cgpath=""
		if [[ -r /proc/self/cgroup ]]; then
			# cgroup v2 line:  "0::/pbs_jobid/..."
			# cgroup v1 pids:  "N:pids:/pbs_jobid/..."
			cgpath="$(awk -F: '$1 == "0" {print $3; exit}' /proc/self/cgroup)"
			if [[ -z "${cgpath}" ]]; then
				cgpath="$(awk -F: '$2 == "pids" {print $3; exit}' /proc/self/cgroup)"
			fi
		fi

		if [[ -z "${pids_max_file}" && -n "${cgpath}" ]]; then
			# Walk up the cgroup hierarchy looking for the nearest pids.max.
			local probe="${cgpath}"
			while true; do
				if [[ -r "/sys/fs/cgroup${probe}/pids.max" ]]; then
					pids_max_file="/sys/fs/cgroup${probe}/pids.max"
					pids_cur_file="/sys/fs/cgroup${probe}/pids.current"
					break
				fi
				# cgroup v1 pids controller path
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
		# Fallbacks to cgroup root if the walk failed.
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

		# Current usage by this user on this host.
		local pid_count thread_count
		pid_count="$(ps -u "${USER}" --no-headers 2>/dev/null | wc -l)"
		# Per-thread listing (`ps -L`) counts each OS thread once.
		thread_count="$(ps -u "${USER}" -L --no-headers 2>/dev/null | wc -l || echo unknown)"
		echo "  pids (uid=${USER}):      ${pid_count}"
		echo "  threads (uid=${USER}):   ${thread_count}"
		echo "  nproc:                   $(nproc 2>/dev/null || echo unknown)"

		# -----------------------------------------------------------------
		# Ray component tally. Each bucket is a substring match against the
		# full command line of every process owned by ${USER}. Matches are
		# mutually non-exclusive at the regex level, but the Ray process
		# names are distinctive enough that each bucket effectively counts
		# one component. This block identifies whether the ~60-PID excess
		# (observed: 86 PIDs at post-cluster-ready) is Ray idle-worker
		# pre-spawn or something else entirely.
		# -----------------------------------------------------------------
		echo "  --- Ray component PID tally ---"
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
			"vllm"
			"python"
		)
		local pat count
		for pat in "${tally_patterns[@]}"; do
			count="$(printf '%s\n' "${ps_args_snapshot}" | grep -Fc -- "${pat}" 2>/dev/null || true)"
			# grep -c returns 0 matches as "0"; guard against empty on error.
			count="${count:-0}"
			printf "    %-24s %s\n" "${pat}" "${count}"
		done

		# -----------------------------------------------------------------
		# Top 40 processes by thread count (NLWP). This is the core
		# per-process diagnostic: it identifies WHICH process owns the
		# bulk of the threads at each phase, which no amount of env-var
		# tuning can tell us. Read from top to bottom; the first few
		# rows usually account for the majority of the budget.
		#
		# Columns: pid, nlwp (thread count), rss (KB), comm, args
		# Sort: descending by nlwp (column 2 numeric).
		# -----------------------------------------------------------------
		echo "  --- Top 40 processes by thread count (pid nlwp rss comm args) ---"
		ps -eLo pid,nlwp,rss,comm,args --no-headers -u "${USER}" 2>/dev/null |
			awk '!seen[$1]++' |
			sort -k2 -nr |
			head -40 |
			sed 's/^/    /'

		# -----------------------------------------------------------------
		# One-line summary for quick scanning across phases/nodes.
		# Format designed for grep-ability.
		# -----------------------------------------------------------------
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
# Fans print_process_diagnostics out to every node in the allocation via
# mpiexec. Each node writes its own ${DIAG_DIR}/<label>_<hostname>.txt and
# tees to mpiexec's merged stdout. Falls back to single-node (head) mode if
# node discovery hasn't run yet (e.g. very early in the script).
#
# The remote shell re-enters the current script so print_process_diagnostics
# is in scope. We pass DIAG_DIR, USER, and the label through the environment
# because they're needed inside the function.
print_process_diagnostics_all_nodes() {
	local label="${1:-all-nodes}"

	# If node discovery hasn't completed yet (HEAD_HOST not set), fall
	# back to a single-node call on the current host. This is safe for
	# the very first "job start" snapshot that we may want to take
	# before we've parsed PBS_NODEFILE.
	if [[ -z "${HEAD_HOST:-}" ]]; then
		print_process_diagnostics "${label}"
		return 0
	fi

	# Build the comma-separated host list. PBS runs the script on the
	# head, so HEAD_HOST is always in the list.
	local hosts="${HEAD_HOST}"
	local h
	for h in "${WORKER_HOSTS[@]:-}"; do
		if [[ -n "${h}" ]]; then
			hosts+=",${h}"
		fi
	done

	local n_hosts
	n_hosts="$(awk -F, '{print NF}' <<<"${hosts}")"

	# Remote command: source env, activate venv, re-source this very
	# script in "functions-only" mode by marking DIAG_SOURCE_ONLY=1 and
	# having the top of the script early-exit when it's set. Simpler
	# alternative: inline the minimal subset needed by
	# print_process_diagnostics. We pick the inline approach to avoid
	# re-sourcing the whole qsub script (which would re-run `ray stop`
	# etc. with disastrous consequences).
	#
	# The remote bash -c receives:
	#   - DIAG_DIR (shared /grand path, visible from all nodes)
	#   - label, USER
	# and re-defines a slim version of the diagnostic function inline.
	# This duplicates code, but keeps the remote call hermetic and avoids
	# any chance of accidentally re-executing the driver script.
	#
	# NOTE on quoting: we build the remote snippet with printf %q for the
	# label so spaces/parens in labels ("cluster ready (post-Step 3)")
	# survive the mpiexec bash -c round-trip.
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
# Re-define print_process_diagnostics locally. This is a minimal copy
# of the driver-script function, kept in sync manually. If you update
# the driver-side function signature/output format, update this copy
# too or the per-node files will diverge.
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
		# Polaris-specific fast path: the binding cgroup is
		# /sys/fs/cgroup/jobs/<short_jobid>/. Try this path FIRST because
		# the /proc/self/cgroup walk below, under nested mpiexec/palsd,
		# can land in a sibling cgroup whose pids.max reads "max".
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
		echo "  --- Ray component PID tally ---"
		local ps_args_snapshot
		ps_args_snapshot="$(ps -u "${USER}" -o pid,args --no-headers 2>/dev/null || true)"
		local tally_patterns=(ray::IDLE default_worker.py raylet gcs_server plasma_store log_monitor dashboard_agent runtime_env_agent "ray::" vllm python)
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

	# Run on all nodes in parallel. --ppn 1 ensures one invocation per
	# host. Any node that fails its diagnostic (e.g. missing /proc paths
	# or hostname resolution errors) is logged but does not fail the
	# main job: we wrap with `|| true` so missing data on one node does
	# not mask the data we do get from the others.
	mpiexec -n "${n_hosts}" --ppn 1 --hosts "${hosts}" -- bash -c "${remote_script}" 2>&1 || true
}

# =============================================================================
# BACKGROUND THREAD/PID SAMPLER
# =============================================================================
# Samples pids.current (from the cgroup file resolved on each node) and
# thread counts every 2 seconds, per node, for the duration of the
# engine-init / benchmark phase. Writes CSV rows to
# ${DIAG_DIR}/sampler_<hostname>.csv so we can reconstruct the *trajectory*
# of thread growth during the engine startup that currently dies with
# "Resource temporarily unavailable".
#
# Lifecycle:
#   - start_background_sampler    : called at the start of Step 4
#   - stop_background_sampler     : called on success, on ERR trap, and
#                                   from cleanup() as a final belt-and-
#                                   suspenders guarantee.
#
# Control mechanism: a shared flag file (${SAMPLER_FLAG}) sitting on the
# shared /grand filesystem, visible from every node. The remote loops
# exit when the flag disappears. This avoids the need to track remote
# PIDs across mpiexec sessions.

SAMPLER_MPIEXEC_PID=""

start_background_sampler() {
	# No-op if the sampler is already running.
	if [[ -n "${SAMPLER_MPIEXEC_PID}" ]] && kill -0 "${SAMPLER_MPIEXEC_PID}" 2>/dev/null; then
		return 0
	fi

	# Drop the control flag so remote loops see it at startup.
	: >"${SAMPLER_FLAG}"

	# Require HEAD_HOST / WORKER_HOSTS to have been populated by node
	# discovery; otherwise we'd sample only the head and silently miss
	# the workers.
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

	# Remote loop: every 2s, append a CSV row with
	#   epoch,hostname,pid_count,thread_count,cgroup_pids_current,cgroup_pids_max,ray_idle,ray_total
	# while the flag file exists. Exits cleanly when the flag is removed.
	# Inline cgroup resolution (duplicated from print_process_diagnostics)
	# so the loop body is self-contained.
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

# Resolve the binding cgroup pids.max / pids.current files once.
# Polaris-specific fast path: prefer /sys/fs/cgroup/jobs/<short_jobid>/
# over the /proc/self/cgroup walk. Under nested mpiexec/palsd, the
# walk can land in a sibling cgroup whose pids.max reads "max", which
# gives a CSV with pids_max=max (misleading) even when the binding
# cap is 4096. If the Polaris path is unreadable (non-Polaris site,
# or cgroup moved), fall through to the portable walk.
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

# Header (written once; append-safe across re-entries).
if [[ ! -s "${out}" ]]; then
	echo "epoch,hostname,pid_count,thread_count,pids_current,pids_max,ray_idle,ray_total" >"${out}"
fi

# Sample loop. 2s cadence; exits when flag disappears or after a hard
# ceiling of 3h (safety cap in case the flag file gets stranded).
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

	# Launch the mpiexec in background; track its PID so we can wait on
	# it in stop_background_sampler. Output goes to /dev/null because the
	# useful data lands in the per-node CSVs.
	mpiexec -n "${n_hosts}" --ppn 1 --hosts "${hosts}" -- bash -c "${remote_script}" \
		>/dev/null 2>&1 &
	SAMPLER_MPIEXEC_PID=$!
	echo "Background thread/pid sampler started (mpiexec PID ${SAMPLER_MPIEXEC_PID}, hosts: ${hosts})"
}

stop_background_sampler() {
	# Idempotent: safe to call multiple times (normal exit + cleanup trap
	# + ERR trap all call it).
	if [[ ! -e "${SAMPLER_FLAG}" ]] && [[ -z "${SAMPLER_MPIEXEC_PID}" ]]; then
		return 0
	fi
	# Remove the flag so remote loops exit on their next iteration
	# (within 2s). This is the clean shutdown path.
	rm -f "${SAMPLER_FLAG}" 2>/dev/null || true

	if [[ -n "${SAMPLER_MPIEXEC_PID}" ]]; then
		# Wait up to 5s for graceful exit, then force-kill the mpiexec.
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
# When any command in the driver script exits non-zero under `set -e`, this
# trap fires, dumps the per-process thread state on every node, stops the
# background sampler so its final CSV rows flush to disk, then returns the
# original exit code so the `set -e` unwind continues normally.
#
# This is the single most valuable data point for the current failure mode:
# the thread budget is hit during engine init, and we want to see *which*
# process was fattest at the moment of death, on every node.

_diag_on_err() {
	local rc=$?
	echo ""
	echo "!!! ERR trap: command failed with rc=${rc} at line ${BASH_LINENO[0]}"

	# Dump per-node diagnostics. Guard against recursion: if the
	# diagnostic itself fails (e.g. mpiexec is down), we swallow the
	# error to preserve the original rc.
	print_process_diagnostics_all_nodes "FAILURE_rc${rc}" || true

	# Flush sampler CSVs before we unwind.
	stop_background_sampler || true

	# Preserve the original exit code. With `set -e`, the script will
	# now unwind via the EXIT trap (cleanup).
	return "${rc}"
}

trap _diag_on_err ERR

# =============================================================================
# STEP 1: Start Ray Head Node (on this node)
# =============================================================================

echo ""
echo "[Step 1/7] Starting Ray head node on ${HEAD_HOST} (${HEAD_IP}:${RAY_PORT})..."

# VLLM_HOST_IP MUST match the IP Ray registers the node under (--node-ip-address
# below); otherwise vLLM's placement-group request `node:<VLLM_HOST_IP>: 0.001`
# for the driver bundle will be unsatisfiable.
export VLLM_HOST_IP="${HEAD_IP}"

# -----------------------------------------------------------------------------
# Process-limit diagnostics (job start)
# -----------------------------------------------------------------------------
# The previous "not readable" output for cgroup pids.max was because the
# old block only checked the cgroup root, while PBS on Polaris places
# each job in a per-job subtree under /sys/fs/cgroup/. The function
# below walks /proc/self/cgroup to find the binding pids.max.
#
# 00_job_start: snapshot taken BEFORE `ray stop --force` runs, so we
# capture any leaked processes from prior jobs on each compute node.
# User asserts Polaris reallocates nodes per job (so this should be
# empty of Ray/vLLM processes), but verifying is free.
print_process_diagnostics_all_nodes "00_job_start"

ray stop --force 2>/dev/null || true
sleep 2

# 01_pre_ray: snapshot AFTER ray stop / module load / venv activation,
# immediately before we bring the new Ray head up. Baseline: everything
# before our new Ray cluster exists.
print_process_diagnostics_all_nodes "01_pre_ray"

# Dashboard is disabled to reduce thread count under Polaris's cgroup
# pids.max=4096 cap. The dashboard aiohttp server + its per-node agent
# together contribute ~30-80 threads to the budget and we don't use
# the UI in non-interactive benchmark jobs. Re-enable with
# --include-dashboard=true --dashboard-host=0.0.0.0 if you need the
# cluster UI for live debugging.
ray start --head \
	--node-ip-address="${HEAD_IP}" \
	--port="${RAY_PORT}" \
	--num-cpus="${NUM_RAY_CPUS_PER_NODE}" \
	--num-gpus="${GPUS_PER_NODE}" \
	--temp-dir="${RAY_TMPDIR}" \
	--include-dashboard=false

echo "Ray head started. (dashboard disabled, --num-cpus=${NUM_RAY_CPUS_PER_NODE} to cap idle worker pool)"

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
echo "[Step 2/7] Starting Ray workers on ${#WORKER_HOSTS[@]} node(s): ${WORKER_HOSTS[*]}"

# 02_head_ray_up: snapshot AFTER the head-node ray start completes but
# BEFORE any worker joins. Isolates the head-only Ray infrastructure
# overhead (raylet + gcs_server + object store + log monitor + any
# pre-spawned idle Python workers). Workers show baseline state.
print_process_diagnostics_all_nodes "02_head_ray_up"

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

		# vLLM ZMQ IPC base dir: same job-scoped short path as the head
		# (see SHORT vLLM RPC IPC BASE PATH block in the driver). Must
		# exist locally on every node because shm_broadcast.MessageQueue
		# binds an ipc:// socket via get_open_zmq_ipc_path() inside each
		# RayWorkerProc actor on its own host. Without this mkdir, the
		# bind fails with 'No such file or directory' on workers.
		export VLLM_RPC_BASE_PATH='${VLLM_RPC_BASE_PATH}'
		mkdir -p '${VLLM_RPC_BASE_PATH}'

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

		# Subprocess fan-out caps (must match head node; see the
		# SUBPROCESS FAN-OUT LIMITS block in the driver script).
		# - Inductor: compile serially per worker (no 32-way subprocess pool).
		# - XALT: skip Lmod re-source on every ld invocation.
		export TORCHINDUCTOR_COMPILE_THREADS='${TORCHINDUCTOR_COMPILE_THREADS}'
		export XALT_EXECUTABLE_TRACKING='${XALT_EXECUTABLE_TRACKING}'

		# Additional thread-count caps (must match head node; see the
		# ADDITIONAL THREAD-COUNT REDUCTION block in the driver script).
		# Targets cgroup pids.max pressure surfaced as libzmq
		# pthread_create EAGAIN. Tokenizers/Ray/telemetry knobs are
		# safe; NCCL caps reduce inter-node throughput, revert for
		# production runs.
		export NCCL_SOCKET_NTHREADS='${NCCL_SOCKET_NTHREADS}'
		export NCCL_NSOCKS_PERTHREAD='${NCCL_NSOCKS_PERTHREAD}'
		export TOKENIZERS_PARALLELISM='${TOKENIZERS_PARALLELISM}'
		export RAY_memory_monitor_refresh_ms='${RAY_memory_monitor_refresh_ms}'
		export VLLM_NO_USAGE_STATS='${VLLM_NO_USAGE_STATS}'
		export DO_NOT_TRACK='${DO_NOT_TRACK}'

		# Ray idle worker pool cap (see RAY IDLE WORKER POOL LIMIT
		# block in the driver script). The worker also passes
		# --num-cpus=${NUM_RAY_CPUS_PER_NODE} to ray start below;
		# RAY_num_workers_soft_limit is a belt-and-suspenders env
		# knob that caps the idle pool independent of --num-cpus.
		export RAY_num_workers_soft_limit='${RAY_num_workers_soft_limit}'

		# PyTorch NCCL per-PG thread minimization (see PYTORCH NCCL
		# PER-PG THREAD MINIMIZATION block in the driver script).
		# Disables the optional heartbeat-monitor / trace-buffer /
		# dump-on-timeout threads that each new 2-rank NCCL subcomm
		# would otherwise spawn under lazy P2P init.
		export TORCH_NCCL_ENABLE_MONITORING='${TORCH_NCCL_ENABLE_MONITORING}'
		export TORCH_NCCL_TRACE_BUFFER_SIZE='${TORCH_NCCL_TRACE_BUFFER_SIZE}'
		export TORCH_NCCL_DUMP_ON_TIMEOUT='${TORCH_NCCL_DUMP_ON_TIMEOUT}'

		# Plumb the PBS short job id through so worker-side diagnostics
		# / sampler can resolve the per-job cgroup at
		# /sys/fs/cgroup/jobs/<short_jobid>/ directly.
		export PBS_SHORT_JOBID='${PBS_SHORT_JOBID}'

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
				--num-cpus=${NUM_RAY_CPUS_PER_NODE} \\
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
echo "[Step 3/7] Waiting for ${EXPECTED_NODES} nodes (${EXPECTED_GPUS} GPUs) to join..."

POLL_INTERVAL=10
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

# Snapshot process/thread counts after the full Ray cluster is up. If
# this snapshot is already near cgroup pids.max, we know the vLLM engine
# startup cannot possibly succeed and we should bail before wasting GPU
# time on a run that's going to die in pthread_create.
print_process_diagnostics_all_nodes "03_cluster_ready"

# =============================================================================
# STEP 4-6 SHARED SETUP (always runs)
# =============================================================================
# Everything in this block is required regardless of which benchmark
# steps are enabled: output directories, engine args, sweep matrices,
# the final pre-engine diagnostic snapshot, and the background sampler.
# Individual benchmark steps below (4/5/6) are each gated by their
# RUN_* flag.

mkdir -p "${RESULTS_DIR}"

ENGINE_ARGS=(
	--model "${MODEL}"
	${QUANT_ARGS[@]+"${QUANT_ARGS[@]}"}
	--dtype "${DTYPE}"
	--tensor-parallel-size "${TP_SIZE}"
	--pipeline-parallel-size "${PP_SIZE}"
	--enable-expert-parallel
	--distributed-executor-backend ray
	${MAX_MODEL_LEN_ARGS[@]+"${MAX_MODEL_LEN_ARGS[@]}"}
	--trust-remote-code
	# --enforce-eager skips torch.compile AND CUDA graph capture.
	# Required under Polaris's cgroup pids.max=4096 cap: CUDA graph
	# capture spawns hundreds of transient threads (one batch-shape at
	# a time, but the total thread churn pushes peak pids past the
	# cap). Cluster-ready diagnostics landed at 3498/4096 already, so
	# we have no room for the graph-capture transient. Trade-off:
	# 1.5-2x slower at small batch; benchmark numbers should be
	# annotated as "eager-mode" until ALCF raises the pids.max limit.
	--enforce-eager
)

# Latency sweep matrix.
#
# Each (BATCH_SIZE, INPUT_LEN, OUTPUT_LEN) triple triggers a full
# `vllm bench latency` invocation, which in turn spins up a fresh vLLM
# engine (model load, profile, KV cache, warmup, CUDA graph capture).
# On 16x A100 DeepSeek that's ~60-90s per invocation with a warm
# Inductor/Triton cache.
#
# The engine-init cost dominates the actual benchmark cost, so we keep
# the sweep small: three batch sizes that span the light/medium/heavy
# concurrency regimes, at a single representative (input, output) shape.
#
# Latency batch sizes and I/O shapes are env-overridable via comma-
# separated strings. `qsub -v` discards whitespace, so CSVs (no spaces)
# are the friendly wire format.
#
# LATENCY_IO_CONFIGS_CSV replaces the older INPUT_LENS_CSV + OUTPUT_LEN
# pair so that each latency invocation can independently vary BOTH the
# prompt length and the generation length. Format matches Steps 5/6's
# IO_CONFIGS_CSV: "<input>:<output>" pairs, comma-separated
# (e.g. "512:128,1024:4096"). The bash arrays used by the loop are
# derived from the CSVs once here.
BATCH_SIZES_CSV="${BATCH_SIZES_CSV:-1,8,32}"
LATENCY_IO_CONFIGS_CSV="${LATENCY_IO_CONFIGS_CSV:-512:128}"
IFS=',' read -r -a BATCH_SIZES       <<<"${BATCH_SIZES_CSV}"
IFS=',' read -r -a LATENCY_IO_CONFIGS <<<"${LATENCY_IO_CONFIGS_CSV}"

# Iteration counts. 2 warmup + 10 measured iters is the smallest window
# where per-iter variance stays tight enough to report p50/p99 latency
# meaningfully. Don't reduce further without also widening the iter-to-
# iter variance bounds in downstream analysis.
LATENCY_WARMUP_ITERS="${LATENCY_WARMUP_ITERS:-2}"
LATENCY_ITERS="${LATENCY_ITERS:-10}"

# -----------------------------------------------------------------------------
# PROFILING: per-run setup
# -----------------------------------------------------------------------------
# When PROFILE=1, trim the bench workloads down to a profiler-friendly size
# (a single latency iteration after warmup is all `vllm bench latency
# --profile` actually runs; keeping NUM_PROMPTS small bounds serving +
# throughput trace volume) and create the shared output directory.
#
# These reassignments intentionally override any caller-supplied
# NUM_PROMPTS / LATENCY_WARMUP_ITERS only when PROFILE=1 AND the caller
# has not explicitly set PROFILE_NUM_PROMPTS / PROFILE_LATENCY_WARMUP_ITERS
# to a different value. That lets a submitter write
#   qsub -v PROFILE=1,PROFILE_NUM_PROMPTS=32 ...
# to override the profile defaults.
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
	echo "  Latency iters:     IGNORED (vllm bench latency --profile runs"
	echo "                     exactly 1 profiled iteration; see"
	echo "                     vllm/benchmarks/latency.py:139-149)"
	echo "  VLLM_RPC_TIMEOUT:  ${VLLM_RPC_TIMEOUT:-default}"
	echo "  Trace viewer:      https://ui.perfetto.dev/"
	echo "============================================="
	echo ""
fi

# 04_pre_engine: snapshot immediately before the first vllm bench
# launches the engine. If the engine init dies (as it has been doing),
# this is the last clean "before" state we can compare the FAILURE
# snapshot against to see exactly which process ballooned.
print_process_diagnostics_all_nodes "04_pre_engine"

# Kick off the continuous 2s sampler on every node. It writes
# ${DIAG_DIR}/sampler_<hostname>.csv and is torn down by the ERR trap,
# by the EXIT/cleanup trap, or by the explicit stop_background_sampler
# call at the end of the script.
start_background_sampler

# =============================================================================
# STEP 4: Run Offline Latency Benchmarks
# =============================================================================

if [[ "${RUN_LATENCY}" == "1" ]]; then
	echo ""
	echo "[Step 4/7] Running offline latency benchmarks..."
	echo "  Engine config: PP=${PP_SIZE}, TP=${TP_SIZE}, EP=${EP_SIZE}"
	echo "  Sweep: batch_sizes=[${BATCH_SIZES[*]}] io_configs=[${LATENCY_IO_CONFIGS[*]}]"
	echo ""

	for IO_CONFIG in "${LATENCY_IO_CONFIGS[@]}"; do
		INPUT_LEN="${IO_CONFIG%%:*}"
		OUTPUT_LEN="${IO_CONFIG##*:}"

		for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
			RESULT_FILE="${RESULTS_DIR}/latency_bs${BATCH_SIZE}_in${INPUT_LEN}_out${OUTPUT_LEN}.json"
			LOG_FILE="${RESULTS_DIR}/latency_bs${BATCH_SIZE}_in${INPUT_LEN}_out${OUTPUT_LEN}.log"

			echo "  >> Latency: batch_size=${BATCH_SIZE}, input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}"

			# Per-invocation profiler args. Each (bs, in, out) gets its
			# own torch_profiler_dir so per-rank trace files (named
			# rank_<N>.pt.trace.json.gz by TorchProfilerWrapper) don't
			# collide across invocations. Empty array when PROFILE=0.
			LATENCY_PROFILE_ARGS=()
			if [[ "${PROFILE}" == "1" ]]; then
				LATENCY_PROFILE_DIR="${PROFILE_DIR}/latency_bs${BATCH_SIZE}_in${INPUT_LEN}_out${OUTPUT_LEN}"
				mkdir -p "${LATENCY_PROFILE_DIR}"
				LATENCY_PROFILE_ARGS=(
					--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${LATENCY_PROFILE_DIR}\",\"torch_profiler_record_shapes\":true,\"torch_profiler_with_stack\":true}"
					--profile
				)
				echo "  >> Profiler traces -> ${LATENCY_PROFILE_DIR}"
			fi

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

			echo "  >> Saved to ${RESULT_FILE}"
			echo ""
		done
	done

	echo "Offline latency benchmarks complete."
	echo ""
else
	echo "[Step 4/7] Skipping offline latency benchmarks (RUN_LATENCY=${RUN_LATENCY})."
	echo ""
fi

# Snapshot just before vllm serve. Between Step 4 (offline latency) and
# here, the offline engine has torn itself down but some processes may
# linger briefly. If pids.current is close to pids.max at this point,
# the `vllm serve` subprocess tree below is likely to fail in the same
# zmq pthread_create path we've hit before.
print_process_diagnostics_all_nodes "05_pre_vllm_serve"

# =============================================================================
# STEP 5: Online Serving Benchmark
# =============================================================================

if [[ "${RUN_SERVING}" == "1" ]]; then
	echo "[Step 5/7] Starting vLLM server and running serving benchmarks..."

# Per-server profiler args. The server reads --profiler-config once at
# launch and uses it for every /start_profile the bench client sends.
# Because `vllm bench serve --profile` does not pass a profile_prefix
# through /start_profile (vllm/benchmarks/serve.py:751-760), all
# per-rate trace files land in the same directory, distinguished only
# by trace timestamp. Acceptable: the bench client waits for
# /stop_profile to return before issuing the next /start_profile, so
# trace files from different rates do not interleave.
SERVE_PROFILE_ARGS=()
if [[ "${PROFILE}" == "1" ]]; then
	SERVE_PROFILE_DIR="${PROFILE_DIR}/serving"
	mkdir -p "${SERVE_PROFILE_DIR}"
	SERVE_PROFILE_ARGS=(
		--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${SERVE_PROFILE_DIR}\",\"torch_profiler_record_shapes\":true,\"torch_profiler_with_stack\":false}"
	)
	echo "  Profiler traces -> ${SERVE_PROFILE_DIR}"
fi

# Pre-flight: refuse to start if ${SERVE_PORT} is already in LISTEN state
# on the head node. vLLM sets SO_REUSEPORT on the API-server socket
# (vllm/entrypoints/openai/api_server.py:503), so a stale listener (e.g.
# from a prior failed run on the same head node) would silently share
# the port and answer benchmark traffic on our behalf, while our new
# server stays bound-but-not-listening for the entire ~15 min model
# load. There is no readiness probe that can recover from this once it
# has happened, so we fail fast here instead.
if ss -ltn "sport = :${SERVE_PORT}" 2>/dev/null | grep -q LISTEN; then
	echo "ERROR: Port ${SERVE_PORT} already in LISTEN state on $(hostname)."
	echo "       Refusing to start vLLM serve to avoid SO_REUSEPORT collision."
	ss -ltnp "sport = :${SERVE_PORT}" 2>/dev/null || true
	exit 1
fi

# Start the vLLM server in the background.
# Use a process group so we can cleanly kill the server and its children.
vllm serve "${MODEL}" \
	${QUANT_ARGS[@]+"${QUANT_ARGS[@]}"} \
	--dtype "${DTYPE}" \
	--tensor-parallel-size "${TP_SIZE}" \
	--pipeline-parallel-size "${PP_SIZE}" \
	--enable-expert-parallel \
	--distributed-executor-backend ray \
	${MAX_MODEL_LEN_ARGS[@]+"${MAX_MODEL_LEN_ARGS[@]}"} \
	--trust-remote-code \
	--enforce-eager \
	--host 0.0.0.0 \
	--port "${SERVE_PORT}" \
	${SERVE_PROFILE_ARGS[@]+"${SERVE_PROFILE_ARGS[@]}"} \
	>"${RESULTS_DIR}/server.log" 2>&1 &

SERVER_PID=$!
echo "vLLM server starting in background (PID ${SERVER_PID})..."

# Wait for server readiness via a two-stage probe:
#   Stage 1: poll GET /health until 200. /health is gated by
#     AsyncLLM.check_health (vllm/v1/engine/async_llm.py:866-869) and
#     only becomes reachable once uvicorn starts serving in
#     build_and_serve() (vllm/entrypoints/openai/api_server.py:604,
#     vllm/entrypoints/launcher.py:82), which runs only after engine
#     init + KV cache alloc + warmup are complete on every Ray worker
#     (vllm/v1/engine/core.py:116, 126, 1411-1416). curl -f rejects
#     5xx, so a 503 from EngineDeadError is treated as not-ready.
#   Stage 2: confirm we are actually talking to OUR engine (not a
#     stale SO_REUSEPORT-shadowed listener that pre-flight didn't
#     catch) by generating one real token. Same shape of probe as
#     vllm/benchmarks/lib/ready_checker.py:18-79 uses internally when
#     --ready-check-timeout-sec > 0.
#
# MAX_WAIT=1800: DeepSeek-V3 cold-start on Polaris (lustre first-touch)
# observed at ~893s (model load + init engine + warmup); warm-cache
# runs land closer to ~5 min. 30 min ceiling leaves ~25 min of the 1h
# walltime for the sweep + cleanup.
BASE_URL="http://localhost:${SERVE_PORT}"
echo "Waiting for server to be ready at ${BASE_URL}/health ..."

MAX_WAIT=1800
ELAPSED=0
SERVER_POLL_INTERVAL=10

# Per-iteration port-state diagnostics. The pre-flight ss check above
# only sees a single instant; if /health stays unreachable while the
# server appears alive, we want a self-contained record of who, if
# anyone, is bound to ${SERVE_PORT} on this host and what curl is
# actually getting back. Snapshots fire at: post-fork (once),
# on-success, on-server-death, on-timeout, and every 30s while the
# poll is stuck waiting for /health. Both `ss` and `lsof` are invoked
# with `|| true` so missing tools degrade to a noisy log line rather
# than aborting the job.
PORT_DIAG_LOG="${DIAG_DIR}/serve_port_${SERVE_PORT}_poll.log"
LAST_PORT_SNAPSHOT=0
CURL_EXIT="NA"
snapshot_port_state() {
	local label="$1"
	{
		echo "=== $(date -Iseconds) ${label} elapsed=${ELAPSED}s curl_exit=${CURL_EXIT} ==="
		echo "-- ss -tnap sport=:${SERVE_PORT} --"
		ss -tnap "sport = :${SERVE_PORT}" 2>&1 || true
		echo "-- lsof -nP -iTCP:${SERVE_PORT} --"
		lsof -nP -iTCP:"${SERVE_PORT}" 2>&1 || true
		echo
	} >>"${PORT_DIAG_LOG}"
}

snapshot_port_state "00_post_fork"

while true; do
	# Check if the server process is still alive
	if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
		echo "ERROR: vLLM server process died unexpectedly."
		snapshot_port_state "fatal_server_died"
		tail -n 100 "${RESULTS_DIR}/server.log" 2>/dev/null || true
		wait "${SERVER_PID}" || true
		exit 1
	fi

	if curl -fsS --max-time 5 "${BASE_URL}/health" >/dev/null 2>&1; then
		CURL_EXIT=0
		snapshot_port_state "ready"
		echo "Server /health passed."
		break
	else
		CURL_EXIT=$?
	fi

	ELAPSED=$((ELAPSED + SERVER_POLL_INTERVAL))
	if [[ "${ELAPSED}" -ge "${MAX_WAIT}" ]]; then
		echo "ERROR: Server /health not 200 after ${MAX_WAIT}s (last curl exit=${CURL_EXIT})."
		snapshot_port_state "fatal_timeout"
		tail -n 100 "${RESULTS_DIR}/server.log" 2>/dev/null || true
		exit 1
	fi

	if (( ELAPSED - LAST_PORT_SNAPSHOT >= 30 )); then
		snapshot_port_state "polling"
		LAST_PORT_SNAPSHOT=${ELAPSED}
	fi

	echo "  Not ready yet... retrying in ${SERVER_POLL_INTERVAL}s (${ELAPSED}s elapsed, last curl exit=${CURL_EXIT})"
	sleep "${SERVER_POLL_INTERVAL}"
done

# Stage 2: prove we're talking to OUR engine by generating one token.
echo "Validating engine via POST /v1/completions (max_tokens=1)..."
if ! curl -fsS --max-time 30 \
	-H 'Content-Type: application/json' \
	-d "{\"model\": \"${MODEL}\", \"prompt\": \".\", \"max_tokens\": 1, \"temperature\": 0}" \
	"${BASE_URL}/v1/completions" >/dev/null; then
	echo "ERROR: Probe POST /v1/completions failed."
	tail -n 100 "${RESULTS_DIR}/server.log" 2>/dev/null || true
	exit 1
fi
echo "Server is ready!"

# --- Run the serving benchmark sweep ---
#
# Sweep size is tuned for Polaris's 1h debug-scaling queue. At
# NUM_PROMPTS=200, each config takes roughly:
#   - request_rate=1:    ~200 s (prompt arrivals throttled to 1/s)
#   - request_rate=8:    ~25 s
#   - request_rate=inf:  ~15-30 s (saturated; bound by server throughput)
# Three rates x two IO configs = 6 runs ~= 8-12 min total, including
# brief per-config ramp-up.
#
# Request rates and IO shapes are env-overridable via comma-separated
# strings (qsub -v friendly). IO shapes are "<input>:<output>" pairs.
REQUEST_RATES_CSV="${REQUEST_RATES_CSV:-1,8,inf}"
IO_CONFIGS_CSV="${IO_CONFIGS_CSV:-512:128,128:512}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
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

		# Per-invocation --profile flag. Triggers the server's
		# /start_profile and /stop_profile endpoints around this
		# sweep iteration. Empty when PROFILE=0.
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
			--max-concurrency "${MAX_CONCURRENCY}" \
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

else
	echo "[Step 5/7] Skipping online serving benchmarks (RUN_SERVING=${RUN_SERVING})."
	echo ""
fi

# =============================================================================
# STEP 6: Offline Throughput Benchmark
# =============================================================================
# `vllm bench throughput` spins up a fresh `LLM` engine per invocation and
# submits NUM_PROMPTS requests as a single saturated batch, measuring total
# prompt+generation tokens/s. Unlike latency, throughput does not iterate
# over --batch-size (max batching is bounded by the engine's runtime limits),
# so the sweep is over IO shapes only.
#
# CLI flags (see vllm/benchmarks/throughput.py:add_cli_args):
#   --dataset-name random  (synthesizes random prompts)
#   --input-len N --output-len N
#   --num-prompts N        (total prompts submitted in the batch)
#   --output-json PATH
# Engine config flags (model, TP/PP, quantization, enforce-eager) are shared
# with Steps 4/5 via ENGINE_ARGS.

if [[ "${RUN_THROUGHPUT}" == "1" ]]; then
	echo "[Step 6/7] Running offline throughput benchmarks..."
	echo "  Engine config: PP=${PP_SIZE}, TP=${TP_SIZE}, EP=${EP_SIZE}"
	echo "  Sweep: io_configs=[${IO_CONFIGS[*]}]"
	echo ""

	for IO_CONFIG in "${IO_CONFIGS[@]}"; do
		INPUT_LEN="${IO_CONFIG%%:*}"
		OUTPUT_LEN="${IO_CONFIG##*:}"

		RESULT_LABEL="throughput_in${INPUT_LEN}_out${OUTPUT_LEN}"
		RESULT_FILE="${RESULTS_DIR}/${RESULT_LABEL}.json"
		LOG_FILE="${RESULTS_DIR}/${RESULT_LABEL}.log"

		echo "  >> Throughput: input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}"

		# Per-invocation profiler args. Same per-config subdir pattern
		# as Step 4 so trace files from different (in, out) shapes
		# don't collide. Empty when PROFILE=0.
		THROUGHPUT_PROFILE_ARGS=()
		if [[ "${PROFILE}" == "1" ]]; then
			THROUGHPUT_PROFILE_DIR="${PROFILE_DIR}/throughput_in${INPUT_LEN}_out${OUTPUT_LEN}"
			mkdir -p "${THROUGHPUT_PROFILE_DIR}"
			THROUGHPUT_PROFILE_ARGS=(
				--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${THROUGHPUT_PROFILE_DIR}\",\"torch_profiler_record_shapes\":true,\"torch_profiler_with_stack\":false}"
				--profile
			)
			echo "  >> Profiler traces -> ${THROUGHPUT_PROFILE_DIR}"
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

		echo "  >> Saved to ${RESULT_FILE}"
		echo ""
	done

	echo "Offline throughput benchmarks complete."
	echo ""
else
	echo "[Step 6/7] Skipping offline throughput benchmarks (RUN_THROUGHPUT=${RUN_THROUGHPUT})."
	echo ""
fi

# =============================================================================
# STEP 7: lm-evaluation-harness Accuracy Benchmark
# =============================================================================
# `lm_eval --model vllm` instantiates a vllm.LLM(...) inside lm-eval-
# harness's vllm wrapper (`lm_eval/models/vllm_causallms.py`). The LLM
# auto-detects the existing Ray cluster via RAY_ADDRESS (exported in
# Step 1) and runs a TP=${TP_SIZE} engine across all ${EXPECTED_GPUS}
# GPUs. lm_eval then drives one logits / generation request per few-shot
# example through the LLM's generate() API and scores the responses
# against the task harness.
#
# Differences vs. Steps 4/5/6:
#   - lm_eval owns the engine lifecycle. We do NOT pass ENGINE_ARGS;
#     instead we build a comma-separated --model_args string that
#     lm_eval forwards verbatim to vllm.LLM(...). MODEL, TP_SIZE,
#     MAX_MODEL_LEN, etc. are sourced from the body's top-level config
#     so a single qsub -v override updates both the Ray cluster sizing
#     AND the lm_eval engine.
#   - PROFILE=1 is NOT wired through here: lm_eval has no
#     --profiler-config / --profile flag, so trace capture would
#     require extending lm-eval-harness's vllm wrapper. Out of scope
#     for the accuracy job; profile a generate() invocation via
#     Step 4 / 6 if you need traces.
#
# Knobs (see BENCHMARK STEP GATES section): LM_EVAL_TASKS,
# LM_EVAL_LIMIT, LM_EVAL_NUM_FEWSHOT, LM_EVAL_MAX_NUM_BATCHED_TOKENS,
# LM_EVAL_MOE_BACKEND.

if [[ "${RUN_LM_EVAL}" == "1" ]]; then
	echo "[Step 7/7] Running lm-evaluation-harness accuracy sweep..."
	echo ""

	# Ensure lm-eval is installed in the venv. Idempotent: skips the
	# install if the package is already present so warm jobs don't pay
	# the pip resolution cost. The body does NOT auto-install vllm or
	# ray (matches the existing convention); we handle lm_eval here
	# because its dependency footprint is small and bundling it with
	# the wrapper avoids requiring submitters to remember a manual
	# install step.
	if ! python -c "import lm_eval" 2>/dev/null; then
		echo "  Installing lm-eval into ${VENV_DIR}..."
		uv pip install -q lm-eval
	fi

	# Snapshot per-node thread/pid state immediately before lm_eval
	# spawns the LLM. Mirrors the 04_pre_engine snapshot taken before
	# Steps 4/5/6 -- gives us a clean "before" diagnostic for the
	# RUN_LM_EVAL=1 path.
	print_process_diagnostics_all_nodes "07_pre_lm_eval"

	# Build lm_eval --model_args. Comma-separated key=value pairs that
	# lm_eval's vllm wrapper passes straight to vllm.LLM(...). Mirrors
	# the reference DeepSeek-V3 accuracy command with our Polaris-tuned
	# defaults substituted in.
	#
	# Both tensor_parallel_size and pipeline_parallel_size are forwarded
	# unconditionally so the lm_eval engine matches the topology Steps
	# 1-3 brought up on the Ray cluster. PP_SIZE=1 is a vLLM no-op, so
	# pure-TP runs are unaffected. The PP_SIZE * TP_SIZE == EXPECTED_GPUS
	# invariant is enforced upstream (qsub_polaris_body.sh:245), so no
	# additional validation is needed here.
	#
	# NOTE: keep order stable for grep-friendly logging; the comma
	# separator must NOT be followed by spaces (lm_eval's parser splits
	# on bare commas only).
	LM_EVAL_MODEL_ARGS="pretrained=${MODEL}"
	LM_EVAL_MODEL_ARGS+=",trust_remote_code=True"
	LM_EVAL_MODEL_ARGS+=",enforce_eager=True"
	LM_EVAL_MODEL_ARGS+=",tensor_parallel_size=${TP_SIZE}"
	LM_EVAL_MODEL_ARGS+=",pipeline_parallel_size=${PP_SIZE}"
	# Pin the distributed executor to Ray. lm-eval-harness's vllm wrapper
	# spawns vllm.LLM(...) inside the lm_eval driver process, which on a
	# multi-node Polaris allocation is a Ray client (Steps 1-3 brought
	# the cluster up and exported RAY_ADDRESS). vLLM's auto-detection of
	# the executor backend can pick MultiprocExecutor when it sees a
	# single host's GPUs, so we force `distributed_executor_backend=ray`
	# explicitly to guarantee the engine attaches to the existing Ray
	# cluster and uses all NUM_NODES * GPUS_PER_NODE GPUs (matches the
	# `--distributed-executor-backend ray` flag used by Steps 4/5/6's
	# vllm bench CLIs at qsub_polaris_body.sh:1810,2020).
	LM_EVAL_MODEL_ARGS+=",distributed_executor_backend=ray"
	LM_EVAL_MODEL_ARGS+=",max_num_batched_tokens=${LM_EVAL_MAX_NUM_BATCHED_TOKENS}"
	if [[ -n "${MAX_MODEL_LEN}" ]]; then
		LM_EVAL_MODEL_ARGS+=",max_model_len=${MAX_MODEL_LEN}"
	fi
	LM_EVAL_MODEL_ARGS+=",moe_backend=${LM_EVAL_MOE_BACKEND}"
	LM_EVAL_MODEL_ARGS+=",enable_expert_parallel=True"

	# Sanitize the comma-separated task list for use as a filename
	# fragment ("gsm8k,arc_easy" -> "gsm8k_arc_easy"). Result-label
	# also encodes limit / few-shot so re-runs with different sweep
	# parameters don't overwrite each other's logs. When LM_EVAL_LIMIT
	# is empty the run targets the full task split; we encode that as
	# "limfull" so the resulting filename remains grep-friendly and
	# distinct from any limited run.
	LM_EVAL_TASKS_LABEL="${LM_EVAL_TASKS//,/_}"
	if [[ -n "${LM_EVAL_LIMIT}" ]]; then
		LM_EVAL_LIMIT_LABEL="lim${LM_EVAL_LIMIT}"
	else
		LM_EVAL_LIMIT_LABEL="limfull"
	fi
	LM_EVAL_RESULT_LABEL="lm_eval_${LM_EVAL_TASKS_LABEL}_${LM_EVAL_LIMIT_LABEL}_fs${LM_EVAL_NUM_FEWSHOT}"
	LM_EVAL_LOG_FILE="${RESULTS_DIR}/${LM_EVAL_RESULT_LABEL}.log"
	# lm_eval writes per-task JSON / JSONL outputs under --output_path.
	# Use a per-run subdirectory so multiple sweeps in one job don't
	# clobber each other's artifacts.
	LM_EVAL_OUTPUT_PATH="${RESULTS_DIR}/${LM_EVAL_RESULT_LABEL}"
	mkdir -p "${LM_EVAL_OUTPUT_PATH}"

	# Build --limit args conditionally. When LM_EVAL_LIMIT is empty,
	# omit the flag entirely so lm_eval evaluates the full task split
	# (its own default). Passing --limit "" would error.
	LM_EVAL_LIMIT_ARGS=()
	if [[ -n "${LM_EVAL_LIMIT}" ]]; then
		LM_EVAL_LIMIT_ARGS=(--limit "${LM_EVAL_LIMIT}")
	fi

	echo "  Model args:        ${LM_EVAL_MODEL_ARGS}"
	echo "  Tasks:             ${LM_EVAL_TASKS}"
	echo "  Limit:             ${LM_EVAL_LIMIT:-(full split)}"
	echo "  Few-shot:          ${LM_EVAL_NUM_FEWSHOT}"
	echo "  Log file:          ${LM_EVAL_LOG_FILE}"
	echo "  Output dir:        ${LM_EVAL_OUTPUT_PATH}"
	echo ""

	lm_eval \
		--model vllm \
		--model_args "${LM_EVAL_MODEL_ARGS}" \
		--tasks "${LM_EVAL_TASKS}" \
		${LM_EVAL_LIMIT_ARGS[@]+"${LM_EVAL_LIMIT_ARGS[@]}"} \
		--num_fewshot "${LM_EVAL_NUM_FEWSHOT}" \
		--output_path "${LM_EVAL_OUTPUT_PATH}" \
		2>&1 | tee "${LM_EVAL_LOG_FILE}"

	echo ""
	echo "lm-evaluation-harness accuracy sweep complete."
	echo "  Log:               ${LM_EVAL_LOG_FILE}"
	echo "  JSON results:      ${LM_EVAL_OUTPUT_PATH}/"
	echo ""
else
	echo "[Step 7/7] Skipping lm-evaluation-harness sweep (RUN_LM_EVAL=${RUN_LM_EVAL})."
	echo ""
fi

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
echo "Throughput results:"
ls -1 "${RESULTS_DIR}"/throughput_*.json 2>/dev/null || echo "  (no throughput JSON files)"
echo ""
echo "lm-eval results:"
# lm_eval writes its harness JSON output(s) under per-run subdirs
# (created in Step 7) and its tee'd console log in
# lm_eval_<task>_lim<N>_fs<K>.log. List both so a submitter can find
# the structured per-task JSON and the human-readable log easily.
if compgen -G "${RESULTS_DIR}/lm_eval_*.log" >/dev/null 2>&1; then
	ls -1 "${RESULTS_DIR}"/lm_eval_*.log 2>/dev/null
fi
if compgen -G "${RESULTS_DIR}/lm_eval_*/*" >/dev/null 2>&1; then
	# One level deep: per-run output dir contents (results JSON + jsonl).
	ls -1d "${RESULTS_DIR}"/lm_eval_*/ 2>/dev/null
fi
if ! compgen -G "${RESULTS_DIR}/lm_eval_*" >/dev/null 2>&1; then
	echo "  (no lm-eval results)"
fi
echo ""
echo "Profiler traces:"
if [[ "${PROFILE}" == "1" ]]; then
	# List up to 20 per-rank trace files so the run log has an easy
	# pointer. The full tree lives under ${PROFILE_DIR} if a submitter
	# needs everything.
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
echo "Metrics recorded:"
echo "  - Offline latency:    end-to-end latency per (batch size, input length)"
echo "  - Online serving:     TTFT, TPOT, ITL, E2EL at p50/p90/p95/p99"
echo "  - Offline throughput: total tokens/s per IO shape"
echo "  - Online:  request throughput, token throughput"
echo "  - lm-eval:            per-task accuracy / correctness (acc, acc_norm,"
echo "                        exact_match, ...; see lm_eval_*/results_*.json)"
echo ""
echo "Cleanup will run automatically via EXIT trap."

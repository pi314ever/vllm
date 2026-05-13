#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# =============================================================================
# PBS Pro wrapper: DeepSeek-V3 offline throughput benchmark
# =============================================================================
#
# Runs ONLY Step 6 (offline throughput) of qsub_polaris_body.sh. Each
# IO shape triggers a fresh `vllm bench throughput` invocation, which
# spins up a new LLM engine, submits NUM_PROMPTS random prompts, and
# reports aggregate tokens/s.
#
# Unlike latency, `vllm bench throughput` does not take --batch-size
# (max batching is engine-governed), so the sweep dimension is IO
# shape only. BATCH_SIZES_CSV is therefore unused by this wrapper.
#
# Submit:
#   qsub qsub_polaris_deepseek_v3_throughput.sh
#
# Submit-time overrides (qsub -v; CSVs use commas, no spaces):
#   qsub -v IO_CONFIGS_CSV=512:128,2048:256 qsub_polaris_deepseek_v3_throughput.sh
#   qsub -v NUM_PROMPTS=500 qsub_polaris_deepseek_v3_throughput.sh
#   qsub -v BACKEND=mp qsub_polaris_deepseek_v3_throughput.sh
#
# Backend selection:
#   BACKEND=ray (default) -> sources qsub_polaris_body.sh (Ray cluster).
#   BACKEND=mp            -> sources qsub_polaris_body_mp.sh
#                            (torch.distributed + MultiprocExecutor, no Ray).
#
# =============================================================================

#PBS -l select=6:system=polaris:ncpus=32:ngpus=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -q debug-scaling
#PBS -A Intel
#PBS -N ds-v3-tput
#PBS -j oe
#PBS -V

# Config-specific overrides. All of these honor any qsub -v values the
# user supplied, so the wrapper's defaults are "only if not overridden".
export MODEL="${MODEL:-/grand/Intel/dhuang/DeepSeek-V3-0324/}"
export RUN_LATENCY="${RUN_LATENCY:-0}"
export RUN_SERVING="${RUN_SERVING:-0}"
export RUN_THROUGHPUT="${RUN_THROUGHPUT:-1}"

# Backend selection. Default is ray (matches pre-existing behaviour so
# submitters with no BACKEND set see no change). Set BACKEND=mp to use
# the multiprocessing body instead of the Ray body.
BACKEND="${BACKEND:-ray}"
case "${BACKEND}" in
	ray) BODY_SCRIPT="qsub_polaris_body.sh" ;;
	mp)  BODY_SCRIPT="qsub_polaris_body_mp.sh" ;;
	*)
		echo "ERROR: Unknown BACKEND=${BACKEND}. Expected 'ray' or 'mp'."
		exit 1
		;;
esac

# Shared body. Assumes this wrapper was submitted from the repo root so
# PBS_O_WORKDIR resolves to the directory containing the body scripts.
# shellcheck source=/dev/null
source "${PBS_O_WORKDIR}/${BODY_SCRIPT}"

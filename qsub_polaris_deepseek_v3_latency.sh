#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# =============================================================================
# PBS Pro wrapper: DeepSeek-V3 offline latency benchmark, batch size 1
# =============================================================================
#
# Runs ONLY Step 4 (offline latency) of qsub_polaris_body.sh with a
# single batch size. Steps 5 (serving) and 6 (throughput) are skipped
# so the whole engine load + single-batch latency sweep fits in the
# 1h debug-scaling walltime.
#
# Submit:
#   qsub qsub_polaris_deepseek_v3_latency.sh
#
# Submit-time overrides (qsub -v):
#   qsub -v LATENCY_IO_CONFIGS_CSV=512:128,1024:4096 qsub_polaris_deepseek_v3_latency.sh
#   qsub -v BACKEND=mp qsub_polaris_deepseek_v3_latency.sh
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
#PBS -N ds-v3-lat-bs1
#PBS -j oe
#PBS -V

# Config-specific overrides. All of these honor any qsub -v values the
# user supplied, so the wrapper's defaults are "only if not overridden".
export MODEL="${MODEL:-/grand/Intel/dhuang/DeepSeek-V3-0324/}"
export BATCH_SIZES_CSV="${BATCH_SIZES_CSV:-1}"
# (input:output) pairs, comma-separated. Matches Step 5/6 IO_CONFIGS_CSV
# syntax. bs1 targets a long-context, long-generation interactive shape
# (1024 prompt / 4096 decode) to stress single-stream decode latency.
export LATENCY_IO_CONFIGS_CSV="${LATENCY_IO_CONFIGS_CSV:-1024:4096,4096:1024}"
export RUN_LATENCY="${RUN_LATENCY:-1}"
export RUN_SERVING="${RUN_SERVING:-0}"
export RUN_THROUGHPUT="${RUN_THROUGHPUT:-0}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-6000}"

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

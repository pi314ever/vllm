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
#   qsub qsub_polaris_deepseek_v3_latency_bs1.sh
#
# Submit-time overrides (qsub -v):
#   qsub -v INPUT_LENS_CSV=512,2048 qsub_polaris_deepseek_v3_latency_bs1.sh
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
export INPUT_LENS_CSV="${INPUT_LENS_CSV:-1024}"
export OUTPUT_LEN="${OUTPUT_LEN:-4096}"
export RUN_LATENCY="${RUN_LATENCY:-1}"
export RUN_SERVING="${RUN_SERVING:-0}"
export RUN_THROUGHPUT="${RUN_THROUGHPUT:-0}"

# Shared body. Assumes this wrapper was submitted from the repo root so
# PBS_O_WORKDIR resolves to the directory containing qsub_polaris_body.sh.
source "${PBS_O_WORKDIR}/qsub_polaris_body.sh"

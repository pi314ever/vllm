#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# =============================================================================
# PBS Pro wrapper: DeepSeek-V3 online serving benchmark (full sweep)
# =============================================================================
#
# Runs ONLY Step 5 (online serving) of qsub_polaris_body.sh. Starts
# `vllm serve` once, then sweeps all (request_rate, io_shape) pairs
# against the same server. Steps 4 (offline latency) and 6 (offline
# throughput) are skipped.
#
# Defaults sweep 3 request rates (1, 8, inf) x 2 IO shapes
# (512:128, 128:512) = 6 configs, fits comfortably in 1h walltime.
#
# Submit:
#   qsub qsub_polaris_deepseek_v3_serving.sh
#
# Submit-time overrides (qsub -v; CSVs use commas, no spaces):
#   qsub -v REQUEST_RATES_CSV=1,inf qsub_polaris_deepseek_v3_serving.sh
#   qsub -v IO_CONFIGS_CSV=512:128 qsub_polaris_deepseek_v3_serving.sh
#
# =============================================================================

#PBS -l select=6:system=polaris:ncpus=32:ngpus=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -q debug-scaling
#PBS -A Intel
#PBS -N ds-v3-serve
#PBS -j oe
#PBS -V

# Config-specific overrides. All of these honor any qsub -v values the
# user supplied, so the wrapper's defaults are "only if not overridden".
export MODEL="${MODEL:-/grand/Intel/dhuang/DeepSeek-V3-0324/}"
export RUN_LATENCY="${RUN_LATENCY:-0}"
export RUN_SERVING="${RUN_SERVING:-1}"
export RUN_THROUGHPUT="${RUN_THROUGHPUT:-0}"

# Shared body. Assumes this wrapper was submitted from the repo root so
# PBS_O_WORKDIR resolves to the directory containing qsub_polaris_body.sh.
source "${PBS_O_WORKDIR}/qsub_polaris_body.sh"

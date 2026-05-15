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
#   qsub -v BACKEND=mp qsub_polaris_deepseek_v3_serving.sh
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
#PBS -N ds-v3-serve
#PBS -j oe
#PBS -V

# Config-specific overrides. All of these honor any qsub -v values the
# user supplied, so the wrapper's defaults are "only if not overridden".
export MODEL="${MODEL:-/grand/Intel/dhuang/DeepSeek-V3/}"
export IO_CONFIGS_CSV="${IO_CONFIGS_CSV:-1024:1024}"
export REQUEST_RATES_CSV="${REQUEST_RATES_CSV:-inf}"
export RUN_LATENCY="${RUN_LATENCY:-0}"
export RUN_SERVING="${RUN_SERVING:-1}"
export RUN_THROUGHPUT="${RUN_THROUGHPUT:-0}"

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

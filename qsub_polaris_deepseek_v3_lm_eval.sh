#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# =============================================================================
# PBS Pro wrapper: DeepSeek-V3 lm-evaluation-harness accuracy benchmark
# =============================================================================
#
# Runs ONLY Step 7 (lm-evaluation-harness) of qsub_polaris_body.sh on a
# 6-node allocation (24 A100s) with the Polaris-tuned default topology
# TP=4 PP=6 EP=4. Steps 4-6 (offline latency, online serving, offline
# throughput) are skipped so the engine cold-load + the accuracy sweep
# fits in the 1h debug-scaling walltime.
#
# Polaris-tuned variant of the reference DeepSeek-V3 accuracy command
# (see "Deviations vs. the reference command" below for the full diff):
#
#   pip install ray lm_eval
#   lm_eval \
#     --model vllm \
#     --model_args pretrained=/grand/Intel/dhuang/DeepSeek-V3/, \
#                  trust_remote_code=True,enforce_eager=True, \
#                  tensor_parallel_size=4,pipeline_parallel_size=6, \
#                  max_num_batched_tokens=4096,max_model_len=4096, \
#                  moe_backend=marlin,enable_expert_parallel=True \
#     --tasks gsm8k --num_fewshot 8
#
# Upstream reference command (for attribution; NOT what this wrapper
# runs by default -- see override path below):
#
#   lm_eval --model vllm \
#     --model_args pretrained=/mnt/data/deepseek-ai/DeepSeek-V3/, \
#                  trust_remote_code=True,enforce_eager=True, \
#                  tensor_parallel_size=16,max_num_batched_tokens=4096, \
#                  max_model_len=4096,moe_backend=triton, \
#                  enable_expert_parallel=True \
#     --tasks gsm8k --limit 128 --num_fewshot 8
#
# Deviations vs. the reference command:
#   * MODEL path rewritten to /grand/Intel/dhuang/DeepSeek-V3/ (Polaris
#     path; the reference command's /mnt/data/... path does not exist on
#     Polaris). Override at submit time with `qsub -v MODEL=...` if the
#     checkpoint moves.
#   * Engine is launched on a 6-node Ray cluster (NUM_NODES=6,
#     GPUS_PER_NODE=4 = 24 GPUs total) brought up by qsub_polaris_body.sh
#     Steps 1-3. lm_eval picks up the cluster via RAY_ADDRESS exported by
#     the body so the underlying vllm.LLM(...) sees all 24 GPUs.
#   * Topology: TP=4 (intra-node, NVLink) x PP=6 (inter-node, Slingshot
#     hsn0). NVLink is intra-node only on Polaris, so cross-node TP all-
#     reduces over Slingshot are meaningfully slower than the same TP=4
#     on one node. The reference command's TP=16 (which would span all 4
#     nodes of a 4-node allocation via Slingshot) is therefore replaced
#     with TP=4/PP=6: TP stays inside one node, PP eats the cross-node
#     hop instead of TP all-reduce. The PP_SIZE * TP_SIZE == 24 invariant
#     is enforced by qsub_polaris_body.sh:245.
#   * moe_backend defaulted to `marlin` (Polaris-tuned default in
#     LM_EVAL_MOE_BACKEND below). Override with `qsub -v LM_EVAL_MOE_BACKEND=triton`
#     to match the reference command's MoE kernel exactly.
#
# To reproduce the upstream reference layout (TP=16, PP=1, 4 nodes / 16
# GPUs), submit with:
#
#   qsub -v TP_SIZE=16,PP_SIZE=1 \
#        -l select=4:system=polaris:ncpus=32:ngpus=4 \
#        qsub_polaris_deepseek_v3_lm_eval.sh
#
# Submit:
#   qsub qsub_polaris_deepseek_v3_lm_eval.sh
#
# Submit-time overrides (qsub -v; CSVs use commas, no spaces):
#   qsub -v LM_EVAL_TASKS=gsm8k,arc_easy qsub_polaris_deepseek_v3_lm_eval.sh
#   qsub -v LM_EVAL_LIMIT=64 qsub_polaris_deepseek_v3_lm_eval.sh
#   qsub -v LM_EVAL_NUM_FEWSHOT=4 qsub_polaris_deepseek_v3_lm_eval.sh
#   qsub -v LM_EVAL_MOE_BACKEND=cuda qsub_polaris_deepseek_v3_lm_eval.sh
#
# Topology overrides (must be paired with a matching `-l select=...`
# so PP_SIZE * TP_SIZE == NUM_NODES * 4):
#   qsub -v TP_SIZE=4,PP_SIZE=6 \
#        qsub_polaris_deepseek_v3_lm_eval.sh                            # default (Polaris-tuned)
#   qsub -v TP_SIZE=16,PP_SIZE=1 \
#        -l select=4:system=polaris:ncpus=32:ngpus=4 \
#        qsub_polaris_deepseek_v3_lm_eval.sh                            # upstream reference layout
#
# Backend: Ray only. The reference command's `pip install ray` makes Ray
# the intended distributed-executor backend. The MultiprocExecutor body
# (qsub_polaris_body_mp.sh) is NOT supported by this wrapper because
# lm-eval-harness's vllm wrapper does not expose --headless rendezvous
# args; Ray is the only multi-node path that works without modifying
# lm-eval-harness itself.
#
# =============================================================================

# =============================================================================
# PBS DIRECTIVES
# =============================================================================
# 6 nodes on Polaris debug-scaling queue. Sizing rationale:
#   - Polaris-tuned default topology is TP=4 x PP=6 -> 24 GPUs total.
#   - Polaris has 4 GPUs / node -> 24 / 4 = 6 nodes.
#   - The body's invariant PP_SIZE * TP_SIZE == NUM_NODES * GPUS_PER_NODE
#     (qsub_polaris_body.sh:245) ties this directive to the TP_SIZE /
#     PP_SIZE defaults below; if you override either at submit time, you
#     must also override `-l select=...` so the GPU count matches.
#   - 1h walltime is enough for one cold model load (~3-5 min on a warm
#     /grand cache) plus a 128-example gsm8k sweep at 8-shot.

#PBS -l select=6:system=polaris:ncpus=32:ngpus=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -q debug-scaling
#PBS -A Intel
#PBS -N ds-v3-lmeval
#PBS -j oe
#PBS -V

# =============================================================================
# CONFIG-SPECIFIC OVERRIDES
# =============================================================================
# All of these honor any qsub -v values the user supplied, so the
# wrapper's defaults are "only if not overridden".

# Model checkpoint (Polaris /grand path; reference command's /mnt/data
# path is not valid on Polaris).
export MODEL="${MODEL:-/grand/Intel/dhuang/DeepSeek-V3/}"

# Engine context length. Caps prefill + generation at 4096 tokens, which
# is comfortably above gsm8k's 8-shot prompt length (~1k tokens) plus
# generation budget (~256 tokens).
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Topology: TP intra-node (NVLink), PP across nodes (Slingshot hsn0).
# Polaris-tuned default is TP=4 x PP=6 = 24 GPUs, matching the 6-node
# allocation in the PBS directive above. The body enforces the
# invariant PP_SIZE * TP_SIZE == NUM_NODES * GPUS_PER_NODE
# (qsub_polaris_body.sh:245), so any override here also requires a
# matching `-l select=...` override.
#
# Both sizes are forwarded to lm_eval's vllm wrapper as
# tensor_parallel_size / pipeline_parallel_size by Step 7 of the body
# (qsub_polaris_body.sh:2306-2316), so the lm_eval engine matches the
# Ray-cluster topology Steps 1-3 brought up.
#
# To reproduce the upstream reference layout (TP=16, PP=1, 4 nodes /
# 16 GPUs) submit with:
#   qsub -v TP_SIZE=16,PP_SIZE=1 \
#        -l select=4:system=polaris:ncpus=32:ngpus=4 \
#        qsub_polaris_deepseek_v3_lm_eval.sh
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-6}"

# Step gates: only Step 7 (lm-eval) runs.
export RUN_LATENCY="${RUN_LATENCY:-0}"
export RUN_SERVING="${RUN_SERVING:-0}"
export RUN_THROUGHPUT="${RUN_THROUGHPUT:-0}"
export RUN_LM_EVAL="${RUN_LM_EVAL:-1}"

# lm-evaluation-harness configuration. Defaults match the reference
# DeepSeek-V3 accuracy command's task / few-shot selection plus the
# model_args knobs (max_num_batched_tokens, moe_backend) it supplied.
#
# LM_EVAL_LIMIT is intentionally NOT defaulted: a missing LM_EVAL_LIMIT
# (the value if no `qsub -v LM_EVAL_LIMIT=...` is passed) drops the
# --limit flag from the lm_eval invocation, which evaluates the full
# task split. Set explicitly for a quick smoke run (the reference
# command used `--limit 128`):
#   qsub -v LM_EVAL_LIMIT=128 qsub_polaris_deepseek_v3_lm_eval.sh
export LM_EVAL_TASKS="${LM_EVAL_TASKS:-gsm8k}"
export LM_EVAL_NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT:-8}"
export LM_EVAL_MAX_NUM_BATCHED_TOKENS="${LM_EVAL_MAX_NUM_BATCHED_TOKENS:-4096}"
export LM_EVAL_MOE_BACKEND="${LM_EVAL_MOE_BACKEND:-marlin}"

# Shared body. Assumes this wrapper was submitted from the repo root so
# PBS_O_WORKDIR resolves to the directory containing the body scripts.
# shellcheck source=/dev/null
source "${PBS_O_WORKDIR}/qsub_polaris_body.sh"

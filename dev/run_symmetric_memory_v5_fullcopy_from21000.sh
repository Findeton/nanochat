#!/usr/bin/env bash
set -euo pipefail

cd "${NANOCHAT_DIR:-/nanochat}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# V5: arbitrary-copy memory training.
#
# This is one 100k-step run with three phases:
#   A: memory/interface exact-copy stabilization
#   B: short low-LR full-trunk copy adaptation
#   C: memory/interface trajectory and guardrail polish
#
# Phase B uses a smaller batch because full-trunk gradients are much heavier
# than the memory/interface-only phases on the 24GB GPU.

BASE_TAG="${BASE_TAG:-d12-memory-symmetric-v4-long100k-b16-data2-from4000}"
BASE_STEP="${BASE_STEP:-21000}"
OUTPUT_TAG="${OUTPUT_TAG:-d12-memory-symmetric-v5-fullcopy-gradclean-from21000}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
PHASE_A_STEPS="${PHASE_A_STEPS:-30000}"
PHASE_B_STEPS="${PHASE_B_STEPS:-10000}"
PHASE_C_STEPS="${PHASE_C_STEPS:-60000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PHASE_B_BATCH_SIZE="${PHASE_B_BATCH_SIZE:-4}"
PHASE_C_BATCH_SIZE="${PHASE_C_BATCH_SIZE:-16}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
SAVE_OPTIMIZER_CHECKPOINTS="${SAVE_OPTIMIZER_CHECKPOINTS:-0}"
EPISODE_CACHE_SIZE="${EPISODE_CACHE_SIZE:-524288}"
GC_EVERY="${GC_EVERY:-0}"

PHASE3_RECOVERY_LOSS_WEIGHT="${PHASE3_RECOVERY_LOSS_WEIGHT:-0.35}"
PHASE3_RECOVERY_BATCH_FRAC="${PHASE3_RECOVERY_BATCH_FRAC:-0.35}"
PHASE3_RECOVERY_STEPS="${PHASE3_RECOVERY_STEPS:-20}"
PHASE3_RECOVERY_TEMPERATURE="${PHASE3_RECOVERY_TEMPERATURE:-0.34}"
PHASE3_RECOVERY_TOP_K="${PHASE3_RECOVERY_TOP_K:-16}"
PHASE3_RECOVERY_MAX_TARGET_TOKENS="${PHASE3_RECOVERY_MAX_TARGET_TOKENS:-12}"
PHASE3_RECOVERY_ANCHOR_LOSS_WEIGHT="${PHASE3_RECOVERY_ANCHOR_LOSS_WEIGHT:-0.55}"
PHASE3_RECOVERY_KEY_TAIL_TOKENS="${PHASE3_RECOVERY_KEY_TAIL_TOKENS:-2}"
PHASE3_RECOVERY_POST_KEY_TOKENS="${PHASE3_RECOVERY_POST_KEY_TOKENS:-4}"

CURRICULUM_FORCE_REGEN="${CURRICULUM_FORCE_REGEN:-0}"
CURRICULUM_LIVE_SESSION_SIZE="${CURRICULUM_LIVE_SESSION_SIZE:-160000}"
CURRICULUM_IDENTITY_SIZE="${CURRICULUM_IDENTITY_SIZE:-60000}"
CURRICULUM_ANSWER_SIZE="${CURRICULUM_ANSWER_SIZE:-100000}"
CURRICULUM_BINDING_SIZE="${CURRICULUM_BINDING_SIZE:-140000}"
CURRICULUM_STRUCTURED_SIZE="${CURRICULUM_STRUCTURED_SIZE:-50000}"
CURRICULUM_STRUCTURED_SINGLE_SIZE="${CURRICULUM_STRUCTURED_SINGLE_SIZE:-60000}"
CURRICULUM_STRUCTURED_PAIR_SIZE="${CURRICULUM_STRUCTURED_PAIR_SIZE:-60000}"
CURRICULUM_REASONING_SIZE="${CURRICULUM_REASONING_SIZE:-90000}"
CURRICULUM_DRAG_SIZE="${CURRICULUM_DRAG_SIZE:-90000}"
CURRICULUM_HEURISTIC_SIZE="${CURRICULUM_HEURISTIC_SIZE:-100000}"
CURRICULUM_GUARDRAIL_SIZE="${CURRICULUM_GUARDRAIL_SIZE:-160000}"
CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE="${CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE:-160000}"
CURRICULUM_V4_EXACT_COPY_SIZE="${CURRICULUM_V4_EXACT_COPY_SIZE:-450000}"
CURRICULUM_V4_BRANCHPOINT_COPY_SIZE="${CURRICULUM_V4_BRANCHPOINT_COPY_SIZE:-350000}"
CURRICULUM_V4_NO_MEMORY_SIZE="${CURRICULUM_V4_NO_MEMORY_SIZE:-320000}"
CURRICULUM_V4_NORMAL_CHAT_SIZE="${CURRICULUM_V4_NORMAL_CHAT_SIZE:-350000}"
CURRICULUM_V4_IRRELEVANT_MEMORY_CHAT_SIZE="${CURRICULUM_V4_IRRELEVANT_MEMORY_CHAT_SIZE:-320000}"
CURRICULUM_V4_FAILURE_REPLAY_SIZE="${CURRICULUM_V4_FAILURE_REPLAY_SIZE:-250000}"
CURRICULUM_V5_ARBITRARY_CHAR_SIZE="${CURRICULUM_V5_ARBITRARY_CHAR_SIZE:-750000}"
CURRICULUM_V5_ARBITRARY_TOKEN_SIZE="${CURRICULUM_V5_ARBITRARY_TOKEN_SIZE:-750000}"
CURRICULUM_V5_MEMORY_VARIETY_SIZE="${CURRICULUM_V5_MEMORY_VARIETY_SIZE:-600000}"
CURRICULUM_V5_NO_MEMORY_SIZE="${CURRICULUM_V5_NO_MEMORY_SIZE:-450000}"
CURRICULUM_V5_IRRELEVANT_MEMORY_SIZE="${CURRICULUM_V5_IRRELEVANT_MEMORY_SIZE:-450000}"
CURRICULUM_V5_RANDOM_KEY_BINDING_SIZE="${CURRICULUM_V5_RANDOM_KEY_BINDING_SIZE:-900000}"
CURRICULUM_EVAL_SIZE="${CURRICULUM_EVAL_SIZE:-1024}"
CURRICULUM_V4_EVAL_SIZE="${CURRICULUM_V4_EVAL_SIZE:-4096}"
CURRICULUM_V5_EVAL_SIZE="${CURRICULUM_V5_EVAL_SIZE:-4096}"

if [[ "$((PHASE_A_STEPS + PHASE_B_STEPS + PHASE_C_STEPS))" -ne "${TRAIN_STEPS}" ]]; then
  echo "Phase steps must sum to TRAIN_STEPS" >&2
  exit 1
fi

LOG="${LOG:-$PWD/dev/logs/${OUTPUT_TAG}.train.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo
echo "Starting V5 arbitrary-copy memory training at $(date -Is)"
echo "Base checkpoint: ${BASE_TAG} @ ${BASE_STEP}"
echo "Output tag: ${OUTPUT_TAG}"
echo "Train steps: ${TRAIN_STEPS} = A:${PHASE_A_STEPS} B:${PHASE_B_STEPS} C:${PHASE_C_STEPS}"
echo "Batch sizes: A:${BATCH_SIZE} B:${PHASE_B_BATCH_SIZE} C:${PHASE_C_BATCH_SIZE}"
echo "Phase B generated recovery: weight=${PHASE3_RECOVERY_LOSS_WEIGHT}, frac=${PHASE3_RECOVERY_BATCH_FRAC}, steps=${PHASE3_RECOVERY_STEPS}, temp=${PHASE3_RECOVERY_TEMPERATURE}, top_k=${PHASE3_RECOVERY_TOP_K}"
echo "Episode cache size: ${EPISODE_CACHE_SIZE} with rolling refresh"
echo "Checkpointing: every ${SAVE_EVERY}, keep last ${KEEP_LAST_CHECKPOINTS}, optimizer=${SAVE_OPTIMIZER_CHECKPOINTS}"
echo "Log: ${LOG}"
echo

if [[ "${CURRICULUM_FORCE_REGEN}" == "1" \
  || ! -f data/kv_campaign/curriculum_v5_arbitrary_char_copy.jsonl \
  || ! -f data/kv_campaign/curriculum_v5_arbitrary_token_copy.jsonl \
  || ! -f data/kv_campaign/curriculum_v5_memory_variety.jsonl \
  || ! -f data/kv_campaign/curriculum_v5_no_memory_guardrail.jsonl \
  || ! -f data/kv_campaign/curriculum_v5_irrelevant_memory_guardrail.jsonl ]]; then
  python3 scripts/prepare_memory_curriculum_data.py \
    --output-dir data/kv_campaign \
    --live-session-size "${CURRICULUM_LIVE_SESSION_SIZE}" \
    --identity-size "${CURRICULUM_IDENTITY_SIZE}" \
    --answer-size "${CURRICULUM_ANSWER_SIZE}" \
    --binding-size "${CURRICULUM_BINDING_SIZE}" \
    --structured-size "${CURRICULUM_STRUCTURED_SIZE}" \
    --structured-single-size "${CURRICULUM_STRUCTURED_SINGLE_SIZE}" \
    --structured-pair-size "${CURRICULUM_STRUCTURED_PAIR_SIZE}" \
    --reasoning-size "${CURRICULUM_REASONING_SIZE}" \
    --drag-size "${CURRICULUM_DRAG_SIZE}" \
    --heuristic-size "${CURRICULUM_HEURISTIC_SIZE}" \
    --guardrail-size "${CURRICULUM_GUARDRAIL_SIZE}" \
    --distractor-guardrail-size "${CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE}" \
    --v4-exact-copy-size "${CURRICULUM_V4_EXACT_COPY_SIZE}" \
    --v4-branchpoint-copy-size "${CURRICULUM_V4_BRANCHPOINT_COPY_SIZE}" \
    --v4-no-memory-size "${CURRICULUM_V4_NO_MEMORY_SIZE}" \
    --v4-normal-chat-size "${CURRICULUM_V4_NORMAL_CHAT_SIZE}" \
    --v4-irrelevant-memory-chat-size "${CURRICULUM_V4_IRRELEVANT_MEMORY_CHAT_SIZE}" \
    --v4-failure-replay-size "${CURRICULUM_V4_FAILURE_REPLAY_SIZE}" \
    --v5-arbitrary-char-size "${CURRICULUM_V5_ARBITRARY_CHAR_SIZE}" \
    --v5-arbitrary-token-size "${CURRICULUM_V5_ARBITRARY_TOKEN_SIZE}" \
    --v5-memory-variety-size "${CURRICULUM_V5_MEMORY_VARIETY_SIZE}" \
    --v5-no-memory-size "${CURRICULUM_V5_NO_MEMORY_SIZE}" \
    --v5-irrelevant-memory-size "${CURRICULUM_V5_IRRELEVANT_MEMORY_SIZE}" \
    --v5-random-key-binding-size "${CURRICULUM_V5_RANDOM_KEY_BINDING_SIZE}" \
    --eval-size "${CURRICULUM_EVAL_SIZE}" \
    --v4-eval-size "${CURRICULUM_V4_EVAL_SIZE}" \
    --v5-eval-size "${CURRICULUM_V5_EVAL_SIZE}"
fi

if [[ ! -f data/kv_campaign/curriculum_v5_random_key_binding.jsonl ]]; then
  python3 - <<PY
import random
from pathlib import Path

from scripts.prepare_memory_curriculum_data import (
    build_rows,
    has_trainable_final_pair,
    make_v5_random_key_binding,
    write_jsonl,
)

size = int("${CURRICULUM_V5_RANDOM_KEY_BINDING_SIZE}")
path = Path("data/kv_campaign/curriculum_v5_random_key_binding.jsonl")
rng = random.Random(20260426 + 55)
rows = build_rows(rng, size, [make_v5_random_key_binding])
write_jsonl(path, rows)
valid_rows = sum(1 for row in rows if has_trainable_final_pair(row))
print(f"Wrote {path}: {len(rows):,} rows, {valid_rows:,} trainable by chat_memory")
PY
fi

OPTIMIZER_SAVE_ARGS=()
if [[ "${SAVE_OPTIMIZER_CHECKPOINTS}" == "1" ]]; then
  OPTIMIZER_SAVE_ARGS=(--save-optimizer-checkpoints)
fi

python3 -u scripts/chat_memory.py \
  --run dummy \
  --device-type "${DEVICE_TYPE}" \
  --model-tag "${BASE_TAG}" \
  --model-step "${BASE_STEP}" \
  --output-tag "${OUTPUT_TAG}" \
  --num-iterations "${TRAIN_STEPS}" \
  --phase1-steps 0 \
  --phase2-steps "${PHASE_A_STEPS}" \
  --phase3-steps "${PHASE_B_STEPS}" \
  --phase35-steps 0 \
  --phase36-steps 0 \
  --phase37-steps 0 \
  --phase38-steps "${PHASE_C_STEPS}" \
  --device-batch-size "${BATCH_SIZE}" \
  --phase2-device-batch-size "${BATCH_SIZE}" \
  --phase3-device-batch-size "${PHASE_B_BATCH_SIZE}" \
  --phase38-device-batch-size "${PHASE_C_BATCH_SIZE}" \
  --phase3-trunk-trainable \
  --phase3-only-trunk-trainable \
  --phase3-recovery-phase-only \
  --phase3-recovery-loss-weight "${PHASE3_RECOVERY_LOSS_WEIGHT}" \
  --phase3-recovery-batch-frac "${PHASE3_RECOVERY_BATCH_FRAC}" \
  --phase3-recovery-steps "${PHASE3_RECOVERY_STEPS}" \
  --phase3-recovery-temperature "${PHASE3_RECOVERY_TEMPERATURE}" \
  --phase3-recovery-top-k "${PHASE3_RECOVERY_TOP_K}" \
  --phase3-recovery-max-target-tokens "${PHASE3_RECOVERY_MAX_TARGET_TOKENS}" \
  --phase3-recovery-anchor-loss-weight "${PHASE3_RECOVERY_ANCHOR_LOSS_WEIGHT}" \
  --phase3-recovery-key-tail-tokens "${PHASE3_RECOVERY_KEY_TAIL_TOKENS}" \
  --phase3-recovery-post-key-tokens "${PHASE3_RECOVERY_POST_KEY_TOKENS}" \
  --max-seq-len 128 \
  --episode-cache-size "${EPISODE_CACHE_SIZE}" \
  --episode-cache-refresh \
  --episode-cache-reshuffle \
  --gc-every "${GC_EVERY}" \
  --save-every "${SAVE_EVERY}" \
  --keep-last-checkpoints "${KEEP_LAST_CHECKPOINTS}" \
  "${OPTIMIZER_SAVE_ARGS[@]}" \
  --target-selection final \
  --context-mode full \
  --memory-build-mode user \
  --joint-top-layers 2 \
  --gate-lr 0.0000045 \
  --memory-lr 0.0000018 \
  --interface-lr 0.00000005 \
  --train-lr 0.000000010 \
  --weighted-answer-ce-weight 0.92 \
  --weighted-template-ce 0.58 \
  --weighted-fact-ce 1.35 \
  --weighted-rest-ce 0.96 \
  --fact-span-margin-loss-weight 0.050 \
  --fact-span-margin 1.25 \
  --guardrail-kl-weight 0.95 \
  --guardrail-temperature 1.40 \
  --write-diversity-loss-weight 0.020 \
  --memory-utility-loss-weight 0.075 \
  --memory-utility-margin 0.28 \
  --anchor-margin-loss-weight 0.050 \
  --anchor-margin 0.65 \
  --anchor-loss-tokens 5 \
  --memory-probe-loss-weight 0.015 \
  --key-token-ce-loss-weight 1.00 \
  --key-token-rank-loss-weight 0.22 \
  --key-token-rank-margin 1.05 \
  --key-token-utility-loss-weight 0.18 \
  --key-token-utility-margin 0.85 \
  --deference-start-phase phase2 \
  --phase2-answer-start-ce-loss-weight 0.85 \
  --phase2-answer-start-tokens 16 \
  --first-fact-token-multiplier 2.60 \
  --habit-confuser-loss-weight 0.12 \
  --habit-confuser-margin 1.20 \
  --habit-confuser-top-k 10 \
  --span-contrast-loss-weight 0.18 \
  --span-contrast-margin 0.78 \
  --span-utility-loss-weight 0.045 \
  --span-utility-margin 0.34 \
  --span-contrast-top-k 8 \
  --span-contrast-max-tokens 18 \
  --span-hard-token-loss-weight 0.22 \
  --span-hard-token-margin 1.05 \
  --span-hard-token-top-k 5 \
  --post-key-loss-weight 0.32 \
  --post-key-repeat-margin 1.50 \
  --post-key-max-groups 5 \
  --phase38-trajectory-loss-weight 0.16 \
  --phase38-trajectory-batch-frac 0.26 \
  --phase38-trajectory-margin 0.55 \
  --phase38-trajectory-gold-ce-weight 0.78 \
  --phase38-sample-steps 20 \
  --phase38-temperature 0.28 \
  --phase38-high-temperature 0.62 \
  --phase38-high-temperature-frac 0.22 \
  --phase38-top-k 16 \
  --phase38-min-prefix-tokens 2 \
  --phase38-answer-tokens 24 \
  --phase38-max-span-tokens 14 \
  --phase38-post-key-tokens 8 \
  --phase38-no-memory-confuser-top-k 18 \
  --phase38-hard-token-loss-weight 0.34 \
  --phase38-hard-token-margin 0.92 \
  --phase38-hard-token-top-k 5 \
  --phase38-hard-token-post-key 2 \
  --phase38-default-branch-loss-weight 0.26 \
  --phase38-default-branch-margin 0.92 \
  --phase38-default-recovery-ce-weight 0.74 \
  --phase38-default-recovery-tokens 10 \
  --phase38-probe-binding-loss-weight 0.07 \
  --phase38-probe-binding-margin 0.70 \
  --phase38-probe-binding-max-spans 5 \
  --phase38-branchpoint-loss-weight 0.34 \
  --phase38-branchpoint-batch-frac 0.30 \
  --phase38-branchpoint-margin 1.05 \
  --phase38-branchpoint-recovery-ce-weight 0.72 \
  --phase38-branchpoint-recovery-tokens 12 \
  --phase38-branchpoint-answer-start-tokens 8 \
  --phase38-branchpoint-steps 20 \
  --phase38-branchpoint-temperature 0.34 \
  --phase38-branchpoint-top-k 16 \
  --custom-json data/kv_campaign/curriculum_v5_arbitrary_char_copy.jsonl \
  --custom-json data/kv_campaign/curriculum_v5_arbitrary_token_copy.jsonl \
  --custom-json data/kv_campaign/curriculum_v5_memory_variety.jsonl \
  --custom-json data/kv_campaign/curriculum_v5_random_key_binding.jsonl \
  --custom-json data/kv_campaign/curriculum_v5_no_memory_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_v5_irrelevant_memory_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_v4_exact_copy.jsonl \
  --custom-json data/kv_campaign/curriculum_v4_branchpoint_copy.jsonl \
  --custom-json data/kv_campaign/curriculum_v4_no_memory_deference.jsonl \
  --custom-json data/kv_campaign/curriculum_v4_normal_chat_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_v4_irrelevant_memory_chat_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_v4_failure_replay.jsonl \
  --custom-json data/kv_campaign/curriculum_live_session_names.jsonl \
  --custom-json data/kv_campaign/curriculum_answer_realization.jsonl \
  --custom-json data/kv_campaign/curriculum_binding_current.jsonl \
  --custom-json data/kv_campaign/curriculum_memory_reasoning.jsonl \
  --custom-json data/kv_campaign/curriculum_contextual_drag.jsonl \
  --custom-json data/kv_campaign/curriculum_heuristic_override.jsonl \
  --custom-json data/kv_campaign/curriculum_language_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_irrelevant_memory_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_structured_single_field.jsonl \
  --custom-json data/kv_campaign/curriculum_structured_pair_fields.jsonl \
  --custom-json data/kv_campaign/curriculum_structured_fields.jsonl

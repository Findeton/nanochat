#!/usr/bin/env bash
set -euo pipefail

cd "${NANOCHAT_DIR:-/nanochat}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"

# V4 long-run continuation.
#
# This is intentionally a single stage: we resume from the best 16k V4
# checkpoint and keep training in phase3_8_trajectory_preference. The data mix
# is weighted by shard size rather than by repeatedly loading the same JSONL,
# so the run can use millions of distinct rows without multiplying RAM.

BASE_TAG="${BASE_TAG:-d12-memory-symmetric-v4-32k-normal100x-from3600}"
BASE_STEP="${BASE_STEP:-16000}"
OUTPUT_TAG="${OUTPUT_TAG:-d12-memory-symmetric-v4-long100k-b16-from16000}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
SAVE_OPTIMIZER_CHECKPOINTS="${SAVE_OPTIMIZER_CHECKPOINTS:-0}"
EPISODE_CACHE_SIZE="${EPISODE_CACHE_SIZE:-524288}"
GC_EVERY="${GC_EVERY:-0}"

CURRICULUM_FORCE_REGEN="${CURRICULUM_FORCE_REGEN:-0}"
CURRICULUM_LIVE_SESSION_SIZE="${CURRICULUM_LIVE_SESSION_SIZE:-220000}"
CURRICULUM_IDENTITY_SIZE="${CURRICULUM_IDENTITY_SIZE:-80000}"
CURRICULUM_ANSWER_SIZE="${CURRICULUM_ANSWER_SIZE:-140000}"
CURRICULUM_BINDING_SIZE="${CURRICULUM_BINDING_SIZE:-180000}"
CURRICULUM_STRUCTURED_SIZE="${CURRICULUM_STRUCTURED_SIZE:-60000}"
CURRICULUM_STRUCTURED_SINGLE_SIZE="${CURRICULUM_STRUCTURED_SINGLE_SIZE:-70000}"
CURRICULUM_STRUCTURED_PAIR_SIZE="${CURRICULUM_STRUCTURED_PAIR_SIZE:-70000}"
CURRICULUM_REASONING_SIZE="${CURRICULUM_REASONING_SIZE:-110000}"
CURRICULUM_DRAG_SIZE="${CURRICULUM_DRAG_SIZE:-120000}"
CURRICULUM_HEURISTIC_SIZE="${CURRICULUM_HEURISTIC_SIZE:-120000}"
CURRICULUM_GUARDRAIL_SIZE="${CURRICULUM_GUARDRAIL_SIZE:-180000}"
CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE="${CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE:-180000}"
CURRICULUM_V4_EXACT_COPY_SIZE="${CURRICULUM_V4_EXACT_COPY_SIZE:-700000}"
CURRICULUM_V4_BRANCHPOINT_COPY_SIZE="${CURRICULUM_V4_BRANCHPOINT_COPY_SIZE:-500000}"
CURRICULUM_V4_NO_MEMORY_SIZE="${CURRICULUM_V4_NO_MEMORY_SIZE:-450000}"
CURRICULUM_V4_NORMAL_CHAT_SIZE="${CURRICULUM_V4_NORMAL_CHAT_SIZE:-450000}"
CURRICULUM_V4_IRRELEVANT_MEMORY_CHAT_SIZE="${CURRICULUM_V4_IRRELEVANT_MEMORY_CHAT_SIZE:-380000}"
CURRICULUM_V4_FAILURE_REPLAY_SIZE="${CURRICULUM_V4_FAILURE_REPLAY_SIZE:-300000}"
CURRICULUM_EVAL_SIZE="${CURRICULUM_EVAL_SIZE:-1024}"
CURRICULUM_V4_EVAL_SIZE="${CURRICULUM_V4_EVAL_SIZE:-4096}"

LOG="${LOG:-$PWD/dev/logs/${OUTPUT_TAG}.train.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo
echo "Starting V4 long single-stage memory training at $(date -Is)"
echo "Base checkpoint: ${BASE_TAG} @ ${BASE_STEP}"
echo "Output tag: ${OUTPUT_TAG}"
echo "Train steps: ${TRAIN_STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Episode cache size: ${EPISODE_CACHE_SIZE} with rolling refresh"
echo "Checkpointing: every ${SAVE_EVERY}, keep last ${KEEP_LAST_CHECKPOINTS}, optimizer=${SAVE_OPTIMIZER_CHECKPOINTS}"
echo "Log: ${LOG}"
echo

if [[ "${CURRICULUM_FORCE_REGEN}" == "1" \
  || ! -f data/kv_campaign/curriculum_v4_exact_copy.jsonl \
  || ! -f data/kv_campaign/curriculum_v4_branchpoint_copy.jsonl \
  || ! -f data/kv_campaign/curriculum_v4_no_memory_deference.jsonl \
  || ! -f data/kv_campaign/curriculum_v4_normal_chat_guardrail.jsonl \
  || ! -f data/kv_campaign/curriculum_v4_irrelevant_memory_chat_guardrail.jsonl \
  || ! -f data/kv_campaign/curriculum_v4_failure_replay.jsonl ]]; then
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
    --eval-size "${CURRICULUM_EVAL_SIZE}" \
    --v4-eval-size "${CURRICULUM_V4_EVAL_SIZE}"
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
  --phase2-steps 0 \
  --phase3-steps 0 \
  --phase35-steps 0 \
  --phase36-steps 0 \
  --phase37-steps 0 \
  --phase38-steps "${TRAIN_STEPS}" \
  --device-batch-size "${BATCH_SIZE}" \
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
  --gate-lr 0.00006 \
  --memory-lr 0.000022 \
  --interface-lr 0.00000022 \
  --train-lr 0.00000007 \
  --weighted-answer-ce-weight 1.05 \
  --weighted-template-ce 0.55 \
  --weighted-fact-ce 1.65 \
  --weighted-rest-ce 0.90 \
  --fact-span-margin-loss-weight 0.07 \
  --fact-span-margin 1.20 \
  --guardrail-kl-weight 1.05 \
  --guardrail-temperature 1.35 \
  --write-diversity-loss-weight 0.020 \
  --memory-utility-loss-weight 0.10 \
  --memory-utility-margin 0.25 \
  --anchor-margin-loss-weight 0.07 \
  --anchor-margin 0.60 \
  --anchor-loss-tokens 4 \
  --memory-probe-loss-weight 0.020 \
  --key-token-ce-loss-weight 1.25 \
  --key-token-rank-loss-weight 0.34 \
  --key-token-rank-margin 0.95 \
  --key-token-utility-loss-weight 0.25 \
  --key-token-utility-margin 0.75 \
  --deference-start-phase phase2 \
  --phase2-answer-start-ce-loss-weight 1.15 \
  --phase2-answer-start-tokens 16 \
  --first-fact-token-multiplier 3.00 \
  --habit-confuser-loss-weight 0.16 \
  --habit-confuser-margin 1.15 \
  --habit-confuser-top-k 8 \
  --span-contrast-loss-weight 0.24 \
  --span-contrast-margin 0.70 \
  --span-utility-loss-weight 0.06 \
  --span-utility-margin 0.30 \
  --span-contrast-top-k 6 \
  --span-contrast-max-tokens 16 \
  --span-hard-token-loss-weight 0.32 \
  --span-hard-token-margin 0.95 \
  --span-hard-token-top-k 4 \
  --post-key-loss-weight 0.46 \
  --post-key-repeat-margin 1.35 \
  --post-key-max-groups 4 \
  --phase38-trajectory-loss-weight 0.16 \
  --phase38-trajectory-batch-frac 0.30 \
  --phase38-trajectory-margin 0.55 \
  --phase38-trajectory-gold-ce-weight 0.78 \
  --phase38-sample-steps 20 \
  --phase38-temperature 0.28 \
  --phase38-high-temperature 0.62 \
  --phase38-high-temperature-frac 0.24 \
  --phase38-top-k 16 \
  --phase38-min-prefix-tokens 2 \
  --phase38-answer-tokens 24 \
  --phase38-max-span-tokens 12 \
  --phase38-post-key-tokens 8 \
  --phase38-no-memory-confuser-top-k 16 \
  --phase38-hard-token-loss-weight 0.30 \
  --phase38-hard-token-margin 0.85 \
  --phase38-hard-token-top-k 4 \
  --phase38-hard-token-post-key 2 \
  --phase38-default-branch-loss-weight 0.24 \
  --phase38-default-branch-margin 0.85 \
  --phase38-default-recovery-ce-weight 0.72 \
  --phase38-default-recovery-tokens 10 \
  --phase38-probe-binding-loss-weight 0.06 \
  --phase38-probe-binding-margin 0.65 \
  --phase38-probe-binding-max-spans 4 \
  --phase38-branchpoint-loss-weight 0.34 \
  --phase38-branchpoint-batch-frac 0.34 \
  --phase38-branchpoint-margin 1.00 \
  --phase38-branchpoint-recovery-ce-weight 0.72 \
  --phase38-branchpoint-recovery-tokens 12 \
  --phase38-branchpoint-answer-start-tokens 8 \
  --phase38-branchpoint-steps 20 \
  --phase38-branchpoint-temperature 0.34 \
  --phase38-branchpoint-top-k 16 \
  --skip-smoltalk \
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
  --custom-json data/kv_campaign/curriculum_structured_fields.jsonl \
  --custom-json data/kv_campaign/curriculum_identity_varied.jsonl

echo
echo "V4 long training complete at $(date -Is)"
echo "Final checkpoint tag: ${OUTPUT_TAG}"

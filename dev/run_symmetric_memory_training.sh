#!/usr/bin/env bash
set -euo pipefail

cd "${NANOCHAT_DIR:-/nanochat}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"

# Override these when launching on a GPU box, for example:
#   BASE_TAG=d12 BASE_STEP=967 OUTPUT_TAG=d12-memory-symmetric-stage1 bash dev/run_symmetric_memory_training.sh
BASE_TAG="${BASE_TAG:-d12}"
BASE_STEP="${BASE_STEP:-967}"
OUTPUT_TAG="${OUTPUT_TAG:-d12-memory-symmetric-stage1}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
PHASE1_STEPS="${PHASE1_STEPS:-1000}"
PHASE2_STEPS="${PHASE2_STEPS:-1000}"
PHASE38_STEPS="${PHASE38_STEPS:-1000}"
STEPS="${STEPS:-$((PHASE1_STEPS + PHASE2_STEPS + PHASE38_STEPS))}"
SAVE_EVERY="${SAVE_EVERY:-250}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPISODE_CACHE_SIZE="${EPISODE_CACHE_SIZE:-65536}"
GC_EVERY="${GC_EVERY:-0}"
USE_CURRICULUM_DATA="${USE_CURRICULUM_DATA:-1}"
CURRICULUM_FORCE_REGEN="${CURRICULUM_FORCE_REGEN:-0}"
CURRICULUM_IDENTITY_SIZE="${CURRICULUM_IDENTITY_SIZE:-20000}"
CURRICULUM_ANSWER_SIZE="${CURRICULUM_ANSWER_SIZE:-18000}"
CURRICULUM_BINDING_SIZE="${CURRICULUM_BINDING_SIZE:-20000}"
CURRICULUM_STRUCTURED_SIZE="${CURRICULUM_STRUCTURED_SIZE:-20000}"
CURRICULUM_STRUCTURED_SINGLE_SIZE="${CURRICULUM_STRUCTURED_SINGLE_SIZE:-12000}"
CURRICULUM_STRUCTURED_PAIR_SIZE="${CURRICULUM_STRUCTURED_PAIR_SIZE:-12000}"
CURRICULUM_REASONING_SIZE="${CURRICULUM_REASONING_SIZE:-12000}"
CURRICULUM_DRAG_SIZE="${CURRICULUM_DRAG_SIZE:-12000}"
CURRICULUM_HEURISTIC_SIZE="${CURRICULUM_HEURISTIC_SIZE:-12000}"
CURRICULUM_GUARDRAIL_SIZE="${CURRICULUM_GUARDRAIL_SIZE:-14000}"
CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE="${CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE:-14000}"
CURRICULUM_EVAL_SIZE="${CURRICULUM_EVAL_SIZE:-512}"
INCLUDE_LEGACY_KV_STAGES="${INCLUDE_LEGACY_KV_STAGES:-0}"
INCLUDE_SMOLTALK="${INCLUDE_SMOLTALK:-0}"
PHASE38_BRANCHPOINT_LOSS_WEIGHT="${PHASE38_BRANCHPOINT_LOSS_WEIGHT:-0.25}"
PHASE38_BRANCHPOINT_BATCH_FRAC="${PHASE38_BRANCHPOINT_BATCH_FRAC:-0.25}"
PHASE38_BRANCHPOINT_MARGIN="${PHASE38_BRANCHPOINT_MARGIN:-1.5}"
PHASE38_BRANCHPOINT_ANSWER_START_TOKENS="${PHASE38_BRANCHPOINT_ANSWER_START_TOKENS:-0}"
LOG="${LOG:-$PWD/dev/logs/${OUTPUT_TAG}.train.log}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

if [[ "${USE_CURRICULUM_DATA}" == "1" ]]; then
  if [[ "${CURRICULUM_FORCE_REGEN}" == "1" || ! -f data/kv_campaign/curriculum_answer_realization.jsonl || ! -f data/kv_campaign/curriculum_irrelevant_memory_guardrail.jsonl || ! -f data/kv_campaign/curriculum_contextual_drag.jsonl || ! -f data/kv_campaign/curriculum_heuristic_override.jsonl ]]; then
    python3 scripts/prepare_memory_curriculum_data.py \
      --output-dir data/kv_campaign \
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
      --eval-size "${CURRICULUM_EVAL_SIZE}"
  fi
fi

SMOLTALK_ARGS=(--skip-smoltalk)
if [[ "${INCLUDE_SMOLTALK}" == "1" ]]; then
  SMOLTALK_ARGS=()
fi

CUSTOM_JSON_ARGS=()
if [[ "${USE_CURRICULUM_DATA}" == "1" ]]; then
  CUSTOM_JSON_ARGS+=(
    --custom-json data/kv_campaign/curriculum_identity_varied.jsonl
    --custom-json data/kv_campaign/curriculum_answer_realization.jsonl
    --custom-json data/kv_campaign/curriculum_answer_realization.jsonl
    --custom-json data/kv_campaign/curriculum_binding_current.jsonl
    --custom-json data/kv_campaign/curriculum_binding_current.jsonl
    --custom-json data/kv_campaign/curriculum_structured_single_field.jsonl
    --custom-json data/kv_campaign/curriculum_structured_single_field.jsonl
    --custom-json data/kv_campaign/curriculum_structured_pair_fields.jsonl
    --custom-json data/kv_campaign/curriculum_structured_fields.jsonl
    --custom-json data/kv_campaign/curriculum_memory_reasoning.jsonl
    --custom-json data/kv_campaign/curriculum_memory_reasoning.jsonl
    --custom-json data/kv_campaign/curriculum_contextual_drag.jsonl
    --custom-json data/kv_campaign/curriculum_heuristic_override.jsonl
    --custom-json data/kv_campaign/curriculum_language_guardrail.jsonl
    --custom-json data/kv_campaign/curriculum_language_guardrail.jsonl
    --custom-json data/kv_campaign/curriculum_irrelevant_memory_guardrail.jsonl
    --custom-json data/kv_campaign/curriculum_irrelevant_memory_guardrail.jsonl
  )
fi
if [[ "${INCLUDE_LEGACY_KV_STAGES}" == "1" ]]; then
  CUSTOM_JSON_ARGS+=(
    --custom-json data/kv_campaign/stage1_identity.jsonl
    --custom-json data/kv_campaign/stage2_structured.jsonl
    --custom-json data/kv_campaign/stage3_mixed.jsonl
  )
fi

echo
echo "Starting symmetric memory training at $(date -Is)"
echo "Base checkpoint: ${BASE_TAG} @ ${BASE_STEP}"
echo "Output tag: ${OUTPUT_TAG}"
echo "Steps: ${STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Episode cache size: ${EPISODE_CACHE_SIZE}"
echo "Curriculum data: ${USE_CURRICULUM_DATA} (legacy=${INCLUDE_LEGACY_KV_STAGES}, smoltalk=${INCLUDE_SMOLTALK})"
echo "Phase 3.8 branchpoint: weight=${PHASE38_BRANCHPOINT_LOSS_WEIGHT}, frac=${PHASE38_BRANCHPOINT_BATCH_FRAC}, margin=${PHASE38_BRANCHPOINT_MARGIN}, answer_start_tokens=${PHASE38_BRANCHPOINT_ANSWER_START_TOKENS}"
echo "Log: ${LOG}"
echo

python3 -u scripts/chat_memory.py \
  --run dummy \
  --device-type "${DEVICE_TYPE}" \
  --model-tag "${BASE_TAG}" \
  --model-step "${BASE_STEP}" \
  --output-tag "${OUTPUT_TAG}" \
  --num-iterations "${STEPS}" \
  --phase1-steps "${PHASE1_STEPS}" \
  --phase2-steps "${PHASE2_STEPS}" \
  --phase38-steps "${PHASE38_STEPS}" \
  --device-batch-size "${BATCH_SIZE}" \
  --max-seq-len 128 \
  --episode-cache-size "${EPISODE_CACHE_SIZE}" \
  --gc-every "${GC_EVERY}" \
  --gate-lr 0.0005 \
  --memory-lr 0.00025 \
  --interface-lr 0.000002 \
  --train-lr 0.0000005 \
  --save-every "${SAVE_EVERY}" \
  --target-selection final \
  --context-mode full \
  --memory-build-mode turn \
  --joint-top-layers 2 \
  --weighted-answer-ce-weight 0.45 \
  --weighted-template-ce 0.35 \
  --weighted-fact-ce 1.0 \
  --weighted-rest-ce 0.70 \
  --fact-span-margin-loss-weight 0.25 \
  --fact-span-margin 2.5 \
  --guardrail-kl-weight 0.35 \
  --write-diversity-loss-weight 0.05 \
  --memory-utility-loss-weight 0.45 \
  --memory-utility-margin 0.5 \
  --anchor-margin-loss-weight 0.25 \
  --anchor-margin 2.0 \
  --anchor-loss-tokens 3 \
  --memory-probe-loss-weight 0.10 \
  --key-token-ce-loss-weight 0.75 \
  --key-token-rank-loss-weight 0.75 \
  --key-token-rank-margin 2.0 \
  --key-token-utility-loss-weight 1.0 \
  --key-token-utility-margin 2.0 \
  --deference-start-phase phase2 \
  --phase2-answer-start-ce-loss-weight 0.30 \
  --phase2-answer-start-tokens 10 \
  --first-fact-token-multiplier 2.5 \
  --habit-confuser-loss-weight 0.40 \
  --habit-confuser-margin 3.0 \
  --habit-confuser-top-k 8 \
  --span-contrast-loss-weight 0.30 \
  --span-contrast-margin 1.5 \
  --span-utility-loss-weight 0.20 \
  --span-utility-margin 1.0 \
  --span-contrast-top-k 4 \
  --span-contrast-max-tokens 12 \
  --span-hard-token-loss-weight 0.50 \
  --span-hard-token-margin 4.0 \
  --span-hard-token-top-k 3 \
  --post-key-loss-weight 0.25 \
  --post-key-repeat-margin 3.0 \
  --post-key-max-groups 4 \
  --phase38-trajectory-loss-weight 0.35 \
  --phase38-trajectory-batch-frac 0.25 \
  --phase38-trajectory-margin 1.0 \
  --phase38-trajectory-gold-ce-weight 0.35 \
  --phase38-sample-steps 24 \
  --phase38-temperature 0.30 \
  --phase38-high-temperature 0.60 \
  --phase38-high-temperature-frac 0.25 \
  --phase38-top-k 16 \
  --phase38-min-prefix-tokens 2 \
  --phase38-answer-tokens 32 \
  --phase38-max-span-tokens 12 \
  --phase38-post-key-tokens 8 \
  --phase38-no-memory-confuser-top-k 16 \
  --phase38-hard-token-loss-weight 0.35 \
  --phase38-hard-token-margin 1.5 \
  --phase38-hard-token-top-k 3 \
  --phase38-hard-token-post-key 2 \
  --phase38-default-branch-loss-weight 0.25 \
  --phase38-default-branch-margin 2.0 \
  --phase38-default-recovery-ce-weight 0.35 \
  --phase38-default-recovery-tokens 8 \
  --phase38-probe-binding-loss-weight 0.40 \
  --phase38-probe-binding-margin 2.0 \
  --phase38-probe-binding-max-spans 4 \
  --phase38-branchpoint-loss-weight "${PHASE38_BRANCHPOINT_LOSS_WEIGHT}" \
  --phase38-branchpoint-batch-frac "${PHASE38_BRANCHPOINT_BATCH_FRAC}" \
  --phase38-branchpoint-margin "${PHASE38_BRANCHPOINT_MARGIN}" \
  --phase38-branchpoint-recovery-ce-weight 0.20 \
  --phase38-branchpoint-recovery-tokens 8 \
  --phase38-branchpoint-answer-start-tokens "${PHASE38_BRANCHPOINT_ANSWER_START_TOKENS}" \
  --phase38-branchpoint-steps 24 \
  --phase38-branchpoint-temperature 0.35 \
  --phase38-branchpoint-top-k 16 \
  "${SMOLTALK_ARGS[@]}" \
  "${CUSTOM_JSON_ARGS[@]}"

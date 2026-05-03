#!/usr/bin/env bash
set -euo pipefail

cd "${NANOCHAT_DIR:-/nanochat}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"

# Continuation recipe for the symmetric memory model after retrieval already
# works. It intentionally spends less pressure on "make memory huge" margins and
# more pressure on answer shape, normal-language guardrails, and generated
# branch recovery.
BASE_TAG="${BASE_TAG:-d12-memory-symmetric-curriculum-frombranch1000}"
BASE_STEP="${BASE_STEP:-3000}"
OUTPUT_TAG="${OUTPUT_TAG:-d12-memory-symmetric-realization-curriculum-from3000}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
PHASE2_STEPS="${PHASE2_STEPS:-750}"
PHASE38_STEPS="${PHASE38_STEPS:-750}"
STEPS="${STEPS:-$((PHASE2_STEPS + PHASE38_STEPS))}"
SAVE_EVERY="${SAVE_EVERY:-250}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPISODE_CACHE_SIZE="${EPISODE_CACHE_SIZE:-65536}"
GC_EVERY="${GC_EVERY:-0}"
CURRICULUM_FORCE_REGEN="${CURRICULUM_FORCE_REGEN:-0}"
INCLUDE_SMOLTALK="${INCLUDE_SMOLTALK:-0}"
LOG="${LOG:-$PWD/dev/logs/${OUTPUT_TAG}.train.log}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

if [[ "${CURRICULUM_FORCE_REGEN}" == "1" || ! -f data/kv_campaign/curriculum_answer_realization.jsonl || ! -f data/kv_campaign/curriculum_irrelevant_memory_guardrail.jsonl || ! -f data/kv_campaign/curriculum_contextual_drag.jsonl || ! -f data/kv_campaign/curriculum_heuristic_override.jsonl ]]; then
  python3 scripts/prepare_memory_curriculum_data.py \
    --output-dir data/kv_campaign \
    --identity-size "${CURRICULUM_IDENTITY_SIZE:-20000}" \
    --answer-size "${CURRICULUM_ANSWER_SIZE:-24000}" \
    --binding-size "${CURRICULUM_BINDING_SIZE:-22000}" \
    --structured-size "${CURRICULUM_STRUCTURED_SIZE:-16000}" \
    --structured-single-size "${CURRICULUM_STRUCTURED_SINGLE_SIZE:-18000}" \
    --structured-pair-size "${CURRICULUM_STRUCTURED_PAIR_SIZE:-16000}" \
    --reasoning-size "${CURRICULUM_REASONING_SIZE:-18000}" \
    --drag-size "${CURRICULUM_DRAG_SIZE:-18000}" \
    --heuristic-size "${CURRICULUM_HEURISTIC_SIZE:-18000}" \
    --guardrail-size "${CURRICULUM_GUARDRAIL_SIZE:-22000}" \
    --distractor-guardrail-size "${CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE:-22000}" \
    --eval-size "${CURRICULUM_EVAL_SIZE:-512}"
fi

SMOLTALK_ARGS=(--skip-smoltalk)
if [[ "${INCLUDE_SMOLTALK}" == "1" ]]; then
  SMOLTALK_ARGS=()
fi

echo
echo "Starting symmetric memory realization curriculum at $(date -Is)"
echo "Base checkpoint: ${BASE_TAG} @ ${BASE_STEP}"
echo "Output tag: ${OUTPUT_TAG}"
echo "Steps: ${STEPS} (phase2=${PHASE2_STEPS}, phase38=${PHASE38_STEPS})"
echo "Batch size: ${BATCH_SIZE}; episode cache: ${EPISODE_CACHE_SIZE}; smoltalk=${INCLUDE_SMOLTALK}"
echo "Log: ${LOG}"
echo

python3 -u scripts/chat_memory.py \
  --run dummy \
  --device-type "${DEVICE_TYPE}" \
  --model-tag "${BASE_TAG}" \
  --model-step "${BASE_STEP}" \
  --output-tag "${OUTPUT_TAG}" \
  --num-iterations "${STEPS}" \
  --phase1-steps 0 \
  --phase2-steps "${PHASE2_STEPS}" \
  --phase38-steps "${PHASE38_STEPS}" \
  --device-batch-size "${BATCH_SIZE}" \
  --max-seq-len 128 \
  --episode-cache-size "${EPISODE_CACHE_SIZE}" \
  --gc-every "${GC_EVERY}" \
  --gate-lr 0.00025 \
  --memory-lr 0.00012 \
  --interface-lr 0.000001 \
  --train-lr 0.00000025 \
  --save-every "${SAVE_EVERY}" \
  --target-selection final \
  --context-mode full \
  --memory-build-mode turn \
  --joint-top-layers 2 \
  --weighted-answer-ce-weight 0.60 \
  --weighted-template-ce 0.45 \
  --weighted-fact-ce 1.0 \
  --weighted-rest-ce 0.95 \
  --fact-span-margin-loss-weight 0.10 \
  --fact-span-margin 1.5 \
  --guardrail-kl-weight 0.55 \
  --guardrail-temperature 1.25 \
  --write-diversity-loss-weight 0.04 \
  --memory-utility-loss-weight 0.20 \
  --memory-utility-margin 0.3 \
  --anchor-margin-loss-weight 0.12 \
  --anchor-margin 1.0 \
  --anchor-loss-tokens 2 \
  --memory-probe-loss-weight 0.05 \
  --key-token-ce-loss-weight 0.55 \
  --key-token-rank-loss-weight 0.35 \
  --key-token-rank-margin 1.0 \
  --key-token-utility-loss-weight 0.35 \
  --key-token-utility-margin 1.0 \
  --deference-start-phase phase2 \
  --phase2-answer-start-ce-loss-weight 0.55 \
  --phase2-answer-start-tokens 12 \
  --first-fact-token-multiplier 2.0 \
  --habit-confuser-loss-weight 0.25 \
  --habit-confuser-margin 2.0 \
  --habit-confuser-top-k 8 \
  --span-contrast-loss-weight 0.25 \
  --span-contrast-margin 1.0 \
  --span-utility-loss-weight 0.10 \
  --span-utility-margin 0.5 \
  --span-contrast-top-k 4 \
  --span-contrast-max-tokens 12 \
  --span-hard-token-loss-weight 0.25 \
  --span-hard-token-margin 2.5 \
  --span-hard-token-top-k 3 \
  --post-key-loss-weight 0.45 \
  --post-key-repeat-margin 2.5 \
  --post-key-max-groups 4 \
  --phase38-trajectory-loss-weight 0.20 \
  --phase38-trajectory-batch-frac 0.25 \
  --phase38-trajectory-margin 0.75 \
  --phase38-trajectory-gold-ce-weight 0.45 \
  --phase38-sample-steps 24 \
  --phase38-temperature 0.25 \
  --phase38-high-temperature 0.55 \
  --phase38-high-temperature-frac 0.20 \
  --phase38-top-k 16 \
  --phase38-min-prefix-tokens 2 \
  --phase38-answer-tokens 32 \
  --phase38-max-span-tokens 12 \
  --phase38-post-key-tokens 8 \
  --phase38-no-memory-confuser-top-k 16 \
  --phase38-hard-token-loss-weight 0.25 \
  --phase38-hard-token-margin 1.0 \
  --phase38-hard-token-top-k 3 \
  --phase38-hard-token-post-key 2 \
  --phase38-default-branch-loss-weight 0.20 \
  --phase38-default-branch-margin 1.0 \
  --phase38-default-recovery-ce-weight 0.45 \
  --phase38-default-recovery-tokens 8 \
  --phase38-probe-binding-loss-weight 0.20 \
  --phase38-probe-binding-margin 1.0 \
  --phase38-probe-binding-max-spans 4 \
  --phase38-branchpoint-loss-weight 0.20 \
  --phase38-branchpoint-batch-frac 0.25 \
  --phase38-branchpoint-margin 1.0 \
  --phase38-branchpoint-recovery-ce-weight 0.35 \
  --phase38-branchpoint-recovery-tokens 10 \
  --phase38-branchpoint-answer-start-tokens 4 \
  --phase38-branchpoint-steps 24 \
  --phase38-branchpoint-temperature 0.30 \
  --phase38-branchpoint-top-k 16 \
  "${SMOLTALK_ARGS[@]}" \
  --custom-json data/kv_campaign/curriculum_answer_realization.jsonl \
  --custom-json data/kv_campaign/curriculum_answer_realization.jsonl \
  --custom-json data/kv_campaign/curriculum_memory_reasoning.jsonl \
  --custom-json data/kv_campaign/curriculum_memory_reasoning.jsonl \
  --custom-json data/kv_campaign/curriculum_contextual_drag.jsonl \
  --custom-json data/kv_campaign/curriculum_contextual_drag.jsonl \
  --custom-json data/kv_campaign/curriculum_heuristic_override.jsonl \
  --custom-json data/kv_campaign/curriculum_heuristic_override.jsonl \
  --custom-json data/kv_campaign/curriculum_identity_varied.jsonl \
  --custom-json data/kv_campaign/curriculum_binding_current.jsonl \
  --custom-json data/kv_campaign/curriculum_binding_current.jsonl \
  --custom-json data/kv_campaign/curriculum_structured_single_field.jsonl \
  --custom-json data/kv_campaign/curriculum_structured_single_field.jsonl \
  --custom-json data/kv_campaign/curriculum_structured_pair_fields.jsonl \
  --custom-json data/kv_campaign/curriculum_structured_fields.jsonl \
  --custom-json data/kv_campaign/curriculum_language_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_language_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_irrelevant_memory_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_irrelevant_memory_guardrail.jsonl

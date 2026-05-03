#!/usr/bin/env bash
set -euo pipefail

cd "${NANOCHAT_DIR:-/nanochat}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"

# Four-hour continuation after the symmetric memory checkpoint has learned
# retrieval. This recipe intentionally trains policy/calibration: when to use
# memory, how to bind the right field/current value, and how to realize the
# remembered fact as a normal answer.
BASE_TAG="${BASE_TAG:-d12-memory-symmetric-curriculum-frombranch1000}"
BASE_STEP="${BASE_STEP:-3000}"
OUTPUT_TAG="${OUTPUT_TAG:-d12-memory-symmetric-policy4h-from3000}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"

# Observed on the A6000-class box:
# - phase2/interface steps: ~0.7-0.8s/step
# - phase3.8 trajectory steps: ~2.4-2.7s/step
# This fits in ~4h including checkpoint/eval overhead.
PHASE2_STEPS="${PHASE2_STEPS:-2500}"
PHASE38_STEPS="${PHASE38_STEPS:-4000}"
STEPS="${STEPS:-$((PHASE2_STEPS + PHASE38_STEPS))}"
SAVE_EVERY="${SAVE_EVERY:-250}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-4}"
SAVE_OPTIMIZER_CHECKPOINTS="${SAVE_OPTIMIZER_CHECKPOINTS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPISODE_CACHE_SIZE="${EPISODE_CACHE_SIZE:-131072}"
GC_EVERY="${GC_EVERY:-0}"

CURRICULUM_FORCE_REGEN="${CURRICULUM_FORCE_REGEN:-0}"
CURRICULUM_IDENTITY_SIZE="${CURRICULUM_IDENTITY_SIZE:-20000}"
CURRICULUM_ANSWER_SIZE="${CURRICULUM_ANSWER_SIZE:-60000}"
CURRICULUM_BINDING_SIZE="${CURRICULUM_BINDING_SIZE:-60000}"
CURRICULUM_STRUCTURED_SIZE="${CURRICULUM_STRUCTURED_SIZE:-30000}"
CURRICULUM_STRUCTURED_SINGLE_SIZE="${CURRICULUM_STRUCTURED_SINGLE_SIZE:-40000}"
CURRICULUM_STRUCTURED_PAIR_SIZE="${CURRICULUM_STRUCTURED_PAIR_SIZE:-40000}"
CURRICULUM_REASONING_SIZE="${CURRICULUM_REASONING_SIZE:-35000}"
CURRICULUM_DRAG_SIZE="${CURRICULUM_DRAG_SIZE:-50000}"
CURRICULUM_HEURISTIC_SIZE="${CURRICULUM_HEURISTIC_SIZE:-80000}"
CURRICULUM_GUARDRAIL_SIZE="${CURRICULUM_GUARDRAIL_SIZE:-80000}"
CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE="${CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE:-80000}"
CURRICULUM_EVAL_SIZE="${CURRICULUM_EVAL_SIZE:-512}"

# SmolTalk is useful ordinary-chat preservation data, but downloading it can
# dominate startup on a fresh GPU container. Set INCLUDE_SMOLTALK=1 when the
# dataset cache is already present or network startup time is acceptable.
INCLUDE_SMOLTALK="${INCLUDE_SMOLTALK:-0}"

LOG="${LOG:-$PWD/dev/logs/${OUTPUT_TAG}.train.log}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

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

SMOLTALK_ARGS=(--skip-smoltalk)
if [[ "${INCLUDE_SMOLTALK}" == "1" ]]; then
  SMOLTALK_ARGS=()
fi

OPTIMIZER_SAVE_ARGS=()
if [[ "${SAVE_OPTIMIZER_CHECKPOINTS}" == "1" ]]; then
  OPTIMIZER_SAVE_ARGS=(--save-optimizer-checkpoints)
fi

echo
echo "Starting symmetric memory policy 4h continuation at $(date -Is)"
echo "Base checkpoint: ${BASE_TAG} @ ${BASE_STEP}"
echo "Output tag: ${OUTPUT_TAG}"
echo "Steps: ${STEPS} (phase2=${PHASE2_STEPS}, phase38=${PHASE38_STEPS})"
echo "Batch size: ${BATCH_SIZE}; episode cache: ${EPISODE_CACHE_SIZE}; smoltalk=${INCLUDE_SMOLTALK}"
echo "Checkpointing: save_every=${SAVE_EVERY}; keep_last=${KEEP_LAST_CHECKPOINTS}; save_optimizer=${SAVE_OPTIMIZER_CHECKPOINTS}"
echo "LRs: gate=1.5e-4 memory=6e-5 interface=6e-7 trunk=1.5e-7"
echo "Curriculum sizes: identity=${CURRICULUM_IDENTITY_SIZE}, answer=${CURRICULUM_ANSWER_SIZE}, binding=${CURRICULUM_BINDING_SIZE}, drag=${CURRICULUM_DRAG_SIZE}, heuristic=${CURRICULUM_HEURISTIC_SIZE}, guardrail=${CURRICULUM_GUARDRAIL_SIZE}+${CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE}"
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
  --gate-lr 0.00015 \
  --memory-lr 0.00006 \
  --interface-lr 0.0000006 \
  --train-lr 0.00000015 \
  --save-every "${SAVE_EVERY}" \
  --keep-last-checkpoints "${KEEP_LAST_CHECKPOINTS}" \
  "${OPTIMIZER_SAVE_ARGS[@]}" \
  --target-selection final \
  --context-mode full \
  --memory-build-mode turn \
  --joint-top-layers 2 \
  --weighted-answer-ce-weight 0.75 \
  --weighted-template-ce 0.55 \
  --weighted-fact-ce 1.0 \
  --weighted-rest-ce 1.10 \
  --fact-span-margin-loss-weight 0.08 \
  --fact-span-margin 1.25 \
  --guardrail-kl-weight 0.75 \
  --guardrail-temperature 1.35 \
  --write-diversity-loss-weight 0.03 \
  --memory-utility-loss-weight 0.12 \
  --memory-utility-margin 0.25 \
  --anchor-margin-loss-weight 0.08 \
  --anchor-margin 0.75 \
  --anchor-loss-tokens 2 \
  --memory-probe-loss-weight 0.03 \
  --key-token-ce-loss-weight 0.45 \
  --key-token-rank-loss-weight 0.20 \
  --key-token-rank-margin 0.75 \
  --key-token-utility-loss-weight 0.20 \
  --key-token-utility-margin 0.75 \
  --deference-start-phase phase2 \
  --phase2-answer-start-ce-loss-weight 0.75 \
  --phase2-answer-start-tokens 14 \
  --first-fact-token-multiplier 1.8 \
  --habit-confuser-loss-weight 0.20 \
  --habit-confuser-margin 1.5 \
  --habit-confuser-top-k 8 \
  --span-contrast-loss-weight 0.22 \
  --span-contrast-margin 0.75 \
  --span-utility-loss-weight 0.06 \
  --span-utility-margin 0.35 \
  --span-contrast-top-k 4 \
  --span-contrast-max-tokens 12 \
  --span-hard-token-loss-weight 0.20 \
  --span-hard-token-margin 2.0 \
  --span-hard-token-top-k 3 \
  --post-key-loss-weight 0.60 \
  --post-key-repeat-margin 2.0 \
  --post-key-max-groups 4 \
  --phase38-trajectory-loss-weight 0.16 \
  --phase38-trajectory-batch-frac 0.22 \
  --phase38-trajectory-margin 0.50 \
  --phase38-trajectory-gold-ce-weight 0.55 \
  --phase38-sample-steps 24 \
  --phase38-temperature 0.25 \
  --phase38-high-temperature 0.50 \
  --phase38-high-temperature-frac 0.15 \
  --phase38-top-k 16 \
  --phase38-min-prefix-tokens 2 \
  --phase38-answer-tokens 32 \
  --phase38-max-span-tokens 12 \
  --phase38-post-key-tokens 8 \
  --phase38-no-memory-confuser-top-k 16 \
  --phase38-hard-token-loss-weight 0.20 \
  --phase38-hard-token-margin 0.75 \
  --phase38-hard-token-top-k 3 \
  --phase38-hard-token-post-key 2 \
  --phase38-default-branch-loss-weight 0.16 \
  --phase38-default-branch-margin 0.75 \
  --phase38-default-recovery-ce-weight 0.55 \
  --phase38-default-recovery-tokens 10 \
  --phase38-probe-binding-loss-weight 0.08 \
  --phase38-probe-binding-margin 0.50 \
  --phase38-probe-binding-max-spans 4 \
  --phase38-branchpoint-loss-weight 0.16 \
  --phase38-branchpoint-batch-frac 0.22 \
  --phase38-branchpoint-margin 0.75 \
  --phase38-branchpoint-recovery-ce-weight 0.50 \
  --phase38-branchpoint-recovery-tokens 12 \
  --phase38-branchpoint-answer-start-tokens 6 \
  --phase38-branchpoint-steps 24 \
  --phase38-branchpoint-temperature 0.28 \
  --phase38-branchpoint-top-k 16 \
  "${SMOLTALK_ARGS[@]}" \
  --custom-json data/kv_campaign/curriculum_heuristic_override.jsonl \
  --custom-json data/kv_campaign/curriculum_heuristic_override.jsonl \
  --custom-json data/kv_campaign/curriculum_heuristic_override.jsonl \
  --custom-json data/kv_campaign/curriculum_irrelevant_memory_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_irrelevant_memory_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_irrelevant_memory_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_language_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_language_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_language_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_answer_realization.jsonl \
  --custom-json data/kv_campaign/curriculum_answer_realization.jsonl \
  --custom-json data/kv_campaign/curriculum_answer_realization.jsonl \
  --custom-json data/kv_campaign/curriculum_contextual_drag.jsonl \
  --custom-json data/kv_campaign/curriculum_contextual_drag.jsonl \
  --custom-json data/kv_campaign/curriculum_binding_current.jsonl \
  --custom-json data/kv_campaign/curriculum_binding_current.jsonl \
  --custom-json data/kv_campaign/curriculum_structured_single_field.jsonl \
  --custom-json data/kv_campaign/curriculum_structured_single_field.jsonl \
  --custom-json data/kv_campaign/curriculum_structured_pair_fields.jsonl \
  --custom-json data/kv_campaign/curriculum_structured_pair_fields.jsonl \
  --custom-json data/kv_campaign/curriculum_structured_fields.jsonl \
  --custom-json data/kv_campaign/curriculum_memory_reasoning.jsonl \
  --custom-json data/kv_campaign/curriculum_memory_reasoning.jsonl \
  --custom-json data/kv_campaign/curriculum_identity_varied.jsonl

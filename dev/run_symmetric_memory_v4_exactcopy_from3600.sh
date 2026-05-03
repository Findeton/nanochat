#!/usr/bin/env bash
set -euo pipefail

cd "${NANOCHAT_DIR:-/nanochat}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"

# V4: exact remembered-value copying.
#
# This run starts from the best symmetric live-variety checkpoint and doubles
# the proposed training budget. The data generator creates hundreds of
# thousands of dedicated exact-copy examples before repeats:
#   - short/weird names such as Buh/Qo/Vex
#   - long pseudo-words such as Taloobrook-style compounds
#   - hyphenated names such as Pebble-Cloud variants
#   - non-name remembered values such as labels/codes
#   - no-memory/deference rows to suppress fluent invented defaults
#
# The script chains three checkpoints. The final model is saved under
# ${OUTPUT_TAG}; intermediate stage tags are kept separate for analysis.

BASE_TAG="${BASE_TAG:-d12-memory-symmetric-live-variety-from3750}"
BASE_STEP="${BASE_STEP:-3600}"
OUTPUT_TAG="${OUTPUT_TAG:-d12-memory-symmetric-v4-exactcopy-from3600}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"

STAGE1_STEPS="${STAGE1_STEPS:-2400}"
STAGE2_STEPS="${STAGE2_STEPS:-2400}"
STAGE3_STEPS="${STAGE3_STEPS:-4800}"
START_STAGE="${START_STAGE:-1}"
STAGE2_PHASE2_STEPS="${STAGE2_PHASE2_STEPS:-800}"
STAGE3_PHASE2_STEPS="${STAGE3_PHASE2_STEPS:-800}"
SAVE_EVERY="${SAVE_EVERY:-500}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
SAVE_OPTIMIZER_CHECKPOINTS="${SAVE_OPTIMIZER_CHECKPOINTS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPISODE_CACHE_SIZE="${EPISODE_CACHE_SIZE:-196608}"
GC_EVERY="${GC_EVERY:-0}"

CURRICULUM_FORCE_REGEN="${CURRICULUM_FORCE_REGEN:-1}"
CURRICULUM_LIVE_SESSION_SIZE="${CURRICULUM_LIVE_SESSION_SIZE:-220000}"
CURRICULUM_IDENTITY_SIZE="${CURRICULUM_IDENTITY_SIZE:-70000}"
CURRICULUM_ANSWER_SIZE="${CURRICULUM_ANSWER_SIZE:-100000}"
CURRICULUM_BINDING_SIZE="${CURRICULUM_BINDING_SIZE:-70000}"
CURRICULUM_STRUCTURED_SIZE="${CURRICULUM_STRUCTURED_SIZE:-35000}"
CURRICULUM_STRUCTURED_SINGLE_SIZE="${CURRICULUM_STRUCTURED_SINGLE_SIZE:-45000}"
CURRICULUM_STRUCTURED_PAIR_SIZE="${CURRICULUM_STRUCTURED_PAIR_SIZE:-45000}"
CURRICULUM_REASONING_SIZE="${CURRICULUM_REASONING_SIZE:-55000}"
CURRICULUM_DRAG_SIZE="${CURRICULUM_DRAG_SIZE:-60000}"
CURRICULUM_HEURISTIC_SIZE="${CURRICULUM_HEURISTIC_SIZE:-90000}"
CURRICULUM_GUARDRAIL_SIZE="${CURRICULUM_GUARDRAIL_SIZE:-110000}"
CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE="${CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE:-110000}"
CURRICULUM_V4_EXACT_COPY_SIZE="${CURRICULUM_V4_EXACT_COPY_SIZE:-450000}"
CURRICULUM_V4_BRANCHPOINT_COPY_SIZE="${CURRICULUM_V4_BRANCHPOINT_COPY_SIZE:-250000}"
CURRICULUM_V4_NO_MEMORY_SIZE="${CURRICULUM_V4_NO_MEMORY_SIZE:-200000}"
CURRICULUM_V4_NORMAL_CHAT_SIZE="${CURRICULUM_V4_NORMAL_CHAT_SIZE:-220000}"
CURRICULUM_V4_IRRELEVANT_MEMORY_CHAT_SIZE="${CURRICULUM_V4_IRRELEVANT_MEMORY_CHAT_SIZE:-220000}"
CURRICULUM_EVAL_SIZE="${CURRICULUM_EVAL_SIZE:-768}"
CURRICULUM_V4_EVAL_SIZE="${CURRICULUM_V4_EVAL_SIZE:-2048}"

LOG="${LOG:-$PWD/dev/logs/${OUTPUT_TAG}.train.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

if [[ "${CURRICULUM_FORCE_REGEN}" == "1" || ! -f data/kv_campaign/curriculum_v4_exact_copy.jsonl || ! -f data/kv_campaign/curriculum_v4_branchpoint_copy.jsonl || ! -f data/kv_campaign/curriculum_v4_no_memory_deference.jsonl || ! -f data/kv_campaign/curriculum_v4_normal_chat_guardrail.jsonl || ! -f data/kv_campaign/curriculum_v4_irrelevant_memory_chat_guardrail.jsonl ]]; then
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
    --eval-size "${CURRICULUM_EVAL_SIZE}" \
    --v4-eval-size "${CURRICULUM_V4_EVAL_SIZE}"
fi

OPTIMIZER_SAVE_ARGS=()
if [[ "${SAVE_OPTIMIZER_CHECKPOINTS}" == "1" ]]; then
  OPTIMIZER_SAVE_ARGS=(--save-optimizer-checkpoints)
fi

COMMON_ARGS=(
  --run dummy
  --device-type "${DEVICE_TYPE}"
  --device-batch-size "${BATCH_SIZE}"
  --max-seq-len 128
  --episode-cache-size "${EPISODE_CACHE_SIZE}"
  --gc-every "${GC_EVERY}"
  --save-every "${SAVE_EVERY}"
  --keep-last-checkpoints "${KEEP_LAST_CHECKPOINTS}"
  "${OPTIMIZER_SAVE_ARGS[@]}"
  --target-selection final
  --context-mode full
  --memory-build-mode user
  --joint-top-layers 2
  --gate-lr 0.00009
  --memory-lr 0.000035
  --interface-lr 0.00000035
  --train-lr 0.00000010
  --weighted-answer-ce-weight 1.00
  --weighted-template-ce 0.50
  --weighted-fact-ce 1.60
  --weighted-rest-ce 0.85
  --fact-span-margin-loss-weight 0.08
  --fact-span-margin 1.20
  --guardrail-kl-weight 0.90
  --guardrail-temperature 1.35
  --write-diversity-loss-weight 0.025
  --memory-utility-loss-weight 0.12
  --memory-utility-margin 0.25
  --anchor-margin-loss-weight 0.08
  --anchor-margin 0.60
  --anchor-loss-tokens 4
  --memory-probe-loss-weight 0.025
  --key-token-ce-loss-weight 1.20
  --key-token-rank-loss-weight 0.35
  --key-token-rank-margin 1.00
  --key-token-utility-loss-weight 0.28
  --key-token-utility-margin 0.85
  --deference-start-phase phase2
  --phase2-answer-start-ce-loss-weight 1.05
  --phase2-answer-start-tokens 16
  --first-fact-token-multiplier 3.0
  --habit-confuser-loss-weight 0.18
  --habit-confuser-margin 1.25
  --habit-confuser-top-k 8
  --span-contrast-loss-weight 0.28
  --span-contrast-margin 0.80
  --span-utility-loss-weight 0.07
  --span-utility-margin 0.35
  --span-contrast-top-k 6
  --span-contrast-max-tokens 16
  --span-hard-token-loss-weight 0.35
  --span-hard-token-margin 1.80
  --span-hard-token-top-k 4
  --post-key-loss-weight 0.50
  --post-key-repeat-margin 1.75
  --post-key-max-groups 4
  --skip-smoltalk
)

DATA_EXACT=(
  --custom-json data/kv_campaign/curriculum_v4_exact_copy.jsonl
  --custom-json data/kv_campaign/curriculum_v4_exact_copy.jsonl
  --custom-json data/kv_campaign/curriculum_v4_exact_copy.jsonl
  --custom-json data/kv_campaign/curriculum_v4_exact_copy.jsonl
  --custom-json data/kv_campaign/curriculum_live_session_names.jsonl
  --custom-json data/kv_campaign/curriculum_live_session_names.jsonl
  --custom-json data/kv_campaign/curriculum_answer_realization.jsonl
  --custom-json data/kv_campaign/curriculum_v4_no_memory_deference.jsonl
  --custom-json data/kv_campaign/curriculum_language_guardrail.jsonl
)

DATA_BRANCH=(
  --custom-json data/kv_campaign/curriculum_v4_branchpoint_copy.jsonl
  --custom-json data/kv_campaign/curriculum_v4_branchpoint_copy.jsonl
  --custom-json data/kv_campaign/curriculum_v4_exact_copy.jsonl
  --custom-json data/kv_campaign/curriculum_v4_exact_copy.jsonl
  --custom-json data/kv_campaign/curriculum_live_session_names.jsonl
  --custom-json data/kv_campaign/curriculum_answer_realization.jsonl
  --custom-json data/kv_campaign/curriculum_contextual_drag.jsonl
  --custom-json data/kv_campaign/curriculum_heuristic_override.jsonl
  --custom-json data/kv_campaign/curriculum_v4_no_memory_deference.jsonl
  --custom-json data/kv_campaign/curriculum_v4_normal_chat_guardrail.jsonl
  --custom-json data/kv_campaign/curriculum_v4_irrelevant_memory_chat_guardrail.jsonl
  --custom-json data/kv_campaign/curriculum_v4_irrelevant_memory_chat_guardrail.jsonl
  --custom-json data/kv_campaign/curriculum_irrelevant_memory_guardrail.jsonl
  --custom-json data/kv_campaign/curriculum_language_guardrail.jsonl
)

DATA_LIVE=(
  --custom-json data/kv_campaign/curriculum_v4_exact_copy.jsonl
  --custom-json data/kv_campaign/curriculum_v4_branchpoint_copy.jsonl
  --custom-json data/kv_campaign/curriculum_v4_no_memory_deference.jsonl
  --custom-json data/kv_campaign/curriculum_v4_no_memory_deference.jsonl
  --custom-json data/kv_campaign/curriculum_live_session_names.jsonl
  --custom-json data/kv_campaign/curriculum_live_session_names.jsonl
  --custom-json data/kv_campaign/curriculum_answer_realization.jsonl
  --custom-json data/kv_campaign/curriculum_memory_reasoning.jsonl
  --custom-json data/kv_campaign/curriculum_binding_current.jsonl
  --custom-json data/kv_campaign/curriculum_contextual_drag.jsonl
  --custom-json data/kv_campaign/curriculum_heuristic_override.jsonl
  --custom-json data/kv_campaign/curriculum_v4_normal_chat_guardrail.jsonl
  --custom-json data/kv_campaign/curriculum_v4_normal_chat_guardrail.jsonl
  --custom-json data/kv_campaign/curriculum_v4_normal_chat_guardrail.jsonl
  --custom-json data/kv_campaign/curriculum_v4_irrelevant_memory_chat_guardrail.jsonl
  --custom-json data/kv_campaign/curriculum_v4_irrelevant_memory_chat_guardrail.jsonl
  --custom-json data/kv_campaign/curriculum_language_guardrail.jsonl
  --custom-json data/kv_campaign/curriculum_language_guardrail.jsonl
  --custom-json data/kv_campaign/curriculum_irrelevant_memory_guardrail.jsonl
  --custom-json data/kv_campaign/curriculum_structured_single_field.jsonl
  --custom-json data/kv_campaign/curriculum_structured_pair_fields.jsonl
)

run_stage() {
  local stage_name="$1"
  local base_tag="$2"
  local base_step="$3"
  local output_tag="$4"
  local steps="$5"
  local phase2_steps="$6"
  local phase38_steps="$7"
  shift 7

  echo
  echo "=== ${stage_name} ==="
  echo "Started: $(date -Is)"
  echo "Base: ${base_tag} @ ${base_step}"
  echo "Output: ${output_tag}"
  echo "Steps: ${steps} (phase2=${phase2_steps}, phase38=${phase38_steps})"
  echo

  python3 -u scripts/chat_memory.py \
    "${COMMON_ARGS[@]}" \
    --model-tag "${base_tag}" \
    --model-step "${base_step}" \
    --output-tag "${output_tag}" \
    --num-iterations "${steps}" \
    --phase1-steps 0 \
    --phase2-steps "${phase2_steps}" \
    --phase38-steps "${phase38_steps}" \
    --phase38-trajectory-loss-weight 0.12 \
    --phase38-trajectory-batch-frac 0.22 \
    --phase38-trajectory-margin 0.45 \
    --phase38-trajectory-gold-ce-weight 0.65 \
    --phase38-sample-steps 24 \
    --phase38-temperature 0.25 \
    --phase38-high-temperature 0.55 \
    --phase38-high-temperature-frac 0.20 \
    --phase38-top-k 16 \
    --phase38-min-prefix-tokens 2 \
    --phase38-answer-tokens 32 \
    --phase38-max-span-tokens 16 \
    --phase38-post-key-tokens 8 \
    --phase38-no-memory-confuser-top-k 16 \
    --phase38-hard-token-loss-weight 0.28 \
    --phase38-hard-token-margin 0.80 \
    --phase38-hard-token-top-k 4 \
    --phase38-hard-token-post-key 2 \
    --phase38-default-branch-loss-weight 0.18 \
    --phase38-default-branch-margin 0.70 \
    --phase38-default-recovery-ce-weight 0.60 \
    --phase38-default-recovery-tokens 10 \
    --phase38-probe-binding-loss-weight 0.05 \
    --phase38-probe-binding-margin 0.40 \
    --phase38-probe-binding-max-spans 4 \
    --phase38-branchpoint-loss-weight 0.26 \
    --phase38-branchpoint-batch-frac 0.30 \
    --phase38-branchpoint-margin 0.85 \
    --phase38-branchpoint-recovery-ce-weight 0.70 \
    --phase38-branchpoint-recovery-tokens 14 \
    --phase38-branchpoint-answer-start-tokens 8 \
    --phase38-branchpoint-steps 24 \
    --phase38-branchpoint-temperature 0.32 \
    --phase38-branchpoint-top-k 16 \
    "$@"
}

STAGE1_TAG="${STAGE1_TAG:-${OUTPUT_TAG}-stage1}"
STAGE2_TAG="${STAGE2_TAG:-${OUTPUT_TAG}-stage2}"
STAGE2_BASE_TAG="${STAGE2_BASE_TAG:-${STAGE1_TAG}}"
STAGE2_BASE_STEP="${STAGE2_BASE_STEP:-${STAGE1_STEPS}}"
STAGE3_BASE_TAG="${STAGE3_BASE_TAG:-${STAGE2_TAG}}"
STAGE3_BASE_STEP="${STAGE3_BASE_STEP:-${STAGE2_STEPS}}"

echo
echo "Starting V4 exact-copy memory training at $(date -Is)"
echo "Base checkpoint: ${BASE_TAG} @ ${BASE_STEP}"
echo "Final output tag: ${OUTPUT_TAG}"
echo "Total planned steps: $((STAGE1_STEPS + STAGE2_STEPS + STAGE3_STEPS))"
echo "Start stage: ${START_STAGE}"
echo "Dedicated V4 examples: exact=${CURRICULUM_V4_EXACT_COPY_SIZE}, branch=${CURRICULUM_V4_BRANCHPOINT_COPY_SIZE}, no-memory=${CURRICULUM_V4_NO_MEMORY_SIZE}, normal-chat=${CURRICULUM_V4_NORMAL_CHAT_SIZE}, irrelevant-memory-chat=${CURRICULUM_V4_IRRELEVANT_MEMORY_CHAT_SIZE}"
echo "Broader curriculum examples: live=${CURRICULUM_LIVE_SESSION_SIZE}, answer=${CURRICULUM_ANSWER_SIZE}, guardrails=$((CURRICULUM_GUARDRAIL_SIZE + CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE))"
echo "Log: ${LOG}"
echo

if (( START_STAGE <= 1 )); then
  run_stage "v4a_exact_copy_grounding" "${BASE_TAG}" "${BASE_STEP}" "${STAGE1_TAG}" "${STAGE1_STEPS}" "${STAGE1_STEPS}" 0 "${DATA_EXACT[@]}"
fi
if (( START_STAGE <= 2 )); then
  run_stage "v4b_branchpoint_copy" "${STAGE2_BASE_TAG}" "${STAGE2_BASE_STEP}" "${STAGE2_TAG}" "${STAGE2_STEPS}" "${STAGE2_PHASE2_STEPS}" "$((STAGE2_STEPS - STAGE2_PHASE2_STEPS))" "${DATA_BRANCH[@]}"
fi
if (( START_STAGE <= 3 )); then
  run_stage "v4c_live_recall_and_deference" "${STAGE3_BASE_TAG}" "${STAGE3_BASE_STEP}" "${OUTPUT_TAG}" "${STAGE3_STEPS}" "${STAGE3_PHASE2_STEPS}" "$((STAGE3_STEPS - STAGE3_PHASE2_STEPS))" "${DATA_LIVE[@]}"
fi

echo
echo "V4 training complete at $(date -Is)"
echo "Final checkpoint tag: ${OUTPUT_TAG}"

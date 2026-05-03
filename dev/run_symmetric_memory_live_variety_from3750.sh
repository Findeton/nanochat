#!/usr/bin/env bash
set -euo pipefail

cd "${NANOCHAT_DIR:-/nanochat}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"

# Continuation focused on the deployment failure:
#   process 1: memorize a user-only fact, persist memory
#   process 2: ask a fresh read-only recall question
#
# The key differences from the previous 4h policy run are:
# - memory is built with --memory-build-mode user, matching chat_cli --write-mode user
# - a high-cardinality live-session names shard supplies tens of thousands of
#   arbitrary pet names, including hyphenated and invented names
# - ordinary-language guardrails remain in the mix so the model does not turn
#   every answer into a pet-name lookup.

BASE_TAG="${BASE_TAG:-d12-memory-symmetric-policy4h-from2750-resume}"
BASE_STEP="${BASE_STEP:-3750}"
OUTPUT_TAG="${OUTPUT_TAG:-d12-memory-symmetric-live-variety-from3750}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"

PHASE2_STEPS="${PHASE2_STEPS:-1200}"
PHASE38_STEPS="${PHASE38_STEPS:-2400}"
STEPS="${STEPS:-$((PHASE2_STEPS + PHASE38_STEPS))}"
SAVE_EVERY="${SAVE_EVERY:-250}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-4}"
SAVE_OPTIMIZER_CHECKPOINTS="${SAVE_OPTIMIZER_CHECKPOINTS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPISODE_CACHE_SIZE="${EPISODE_CACHE_SIZE:-131072}"
GC_EVERY="${GC_EVERY:-0}"

CURRICULUM_FORCE_REGEN="${CURRICULUM_FORCE_REGEN:-1}"
CURRICULUM_LIVE_SESSION_SIZE="${CURRICULUM_LIVE_SESSION_SIZE:-160000}"
CURRICULUM_IDENTITY_SIZE="${CURRICULUM_IDENTITY_SIZE:-50000}"
CURRICULUM_ANSWER_SIZE="${CURRICULUM_ANSWER_SIZE:-70000}"
CURRICULUM_BINDING_SIZE="${CURRICULUM_BINDING_SIZE:-60000}"
CURRICULUM_STRUCTURED_SIZE="${CURRICULUM_STRUCTURED_SIZE:-30000}"
CURRICULUM_STRUCTURED_SINGLE_SIZE="${CURRICULUM_STRUCTURED_SINGLE_SIZE:-40000}"
CURRICULUM_STRUCTURED_PAIR_SIZE="${CURRICULUM_STRUCTURED_PAIR_SIZE:-40000}"
CURRICULUM_REASONING_SIZE="${CURRICULUM_REASONING_SIZE:-45000}"
CURRICULUM_DRAG_SIZE="${CURRICULUM_DRAG_SIZE:-50000}"
CURRICULUM_HEURISTIC_SIZE="${CURRICULUM_HEURISTIC_SIZE:-80000}"
CURRICULUM_GUARDRAIL_SIZE="${CURRICULUM_GUARDRAIL_SIZE:-80000}"
CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE="${CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE:-80000}"
CURRICULUM_EVAL_SIZE="${CURRICULUM_EVAL_SIZE:-768}"

INCLUDE_SMOLTALK="${INCLUDE_SMOLTALK:-0}"
LOG="${LOG:-$PWD/dev/logs/${OUTPUT_TAG}.train.log}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

if [[ "${CURRICULUM_FORCE_REGEN}" == "1" || ! -f data/kv_campaign/curriculum_live_session_names.jsonl || ! -f data/kv_campaign/curriculum_answer_realization.jsonl || ! -f data/kv_campaign/curriculum_irrelevant_memory_guardrail.jsonl ]]; then
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
echo "Starting symmetric live-variety memory continuation at $(date -Is)"
echo "Base checkpoint: ${BASE_TAG} @ ${BASE_STEP}"
echo "Output tag: ${OUTPUT_TAG}"
echo "Steps: ${STEPS} (phase2=${PHASE2_STEPS}, phase38=${PHASE38_STEPS})"
echo "Memory build mode: user (matches chat_cli --write-mode user)"
echo "Batch size: ${BATCH_SIZE}; episode cache: ${EPISODE_CACHE_SIZE}; smoltalk=${INCLUDE_SMOLTALK}"
echo "Curriculum sizes: live=${CURRICULUM_LIVE_SESSION_SIZE}, identity=${CURRICULUM_IDENTITY_SIZE}, answer=${CURRICULUM_ANSWER_SIZE}, guardrail=${CURRICULUM_GUARDRAIL_SIZE}+${CURRICULUM_DISTRACTOR_GUARDRAIL_SIZE}"
echo "Checkpointing: save_every=${SAVE_EVERY}; keep_last=${KEEP_LAST_CHECKPOINTS}; save_optimizer=${SAVE_OPTIMIZER_CHECKPOINTS}"
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
  --gate-lr 0.00012 \
  --memory-lr 0.00005 \
  --interface-lr 0.0000005 \
  --train-lr 0.00000012 \
  --save-every "${SAVE_EVERY}" \
  --keep-last-checkpoints "${KEEP_LAST_CHECKPOINTS}" \
  "${OPTIMIZER_SAVE_ARGS[@]}" \
  --target-selection final \
  --context-mode full \
  --memory-build-mode user \
  --joint-top-layers 2 \
  --weighted-answer-ce-weight 0.85 \
  --weighted-template-ce 0.60 \
  --weighted-fact-ce 1.0 \
  --weighted-rest-ce 1.10 \
  --fact-span-margin-loss-weight 0.06 \
  --fact-span-margin 1.0 \
  --guardrail-kl-weight 0.80 \
  --guardrail-temperature 1.35 \
  --write-diversity-loss-weight 0.03 \
  --memory-utility-loss-weight 0.10 \
  --memory-utility-margin 0.20 \
  --anchor-margin-loss-weight 0.06 \
  --anchor-margin 0.50 \
  --anchor-loss-tokens 2 \
  --memory-probe-loss-weight 0.02 \
  --key-token-ce-loss-weight 0.55 \
  --key-token-rank-loss-weight 0.18 \
  --key-token-rank-margin 0.60 \
  --key-token-utility-loss-weight 0.18 \
  --key-token-utility-margin 0.60 \
  --deference-start-phase phase2 \
  --phase2-answer-start-ce-loss-weight 0.90 \
  --phase2-answer-start-tokens 14 \
  --first-fact-token-multiplier 2.2 \
  --habit-confuser-loss-weight 0.16 \
  --habit-confuser-margin 1.25 \
  --habit-confuser-top-k 8 \
  --span-contrast-loss-weight 0.18 \
  --span-contrast-margin 0.60 \
  --span-utility-loss-weight 0.04 \
  --span-utility-margin 0.25 \
  --span-contrast-top-k 4 \
  --span-contrast-max-tokens 12 \
  --span-hard-token-loss-weight 0.18 \
  --span-hard-token-margin 1.5 \
  --span-hard-token-top-k 3 \
  --post-key-loss-weight 0.55 \
  --post-key-repeat-margin 1.75 \
  --post-key-max-groups 4 \
  --phase38-trajectory-loss-weight 0.16 \
  --phase38-trajectory-batch-frac 0.25 \
  --phase38-trajectory-margin 0.45 \
  --phase38-trajectory-gold-ce-weight 0.55 \
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
  --phase38-hard-token-loss-weight 0.18 \
  --phase38-hard-token-margin 0.60 \
  --phase38-hard-token-top-k 3 \
  --phase38-hard-token-post-key 2 \
  --phase38-default-branch-loss-weight 0.16 \
  --phase38-default-branch-margin 0.60 \
  --phase38-default-recovery-ce-weight 0.55 \
  --phase38-default-recovery-tokens 10 \
  --phase38-probe-binding-loss-weight 0.06 \
  --phase38-probe-binding-margin 0.40 \
  --phase38-probe-binding-max-spans 4 \
  --phase38-branchpoint-loss-weight 0.18 \
  --phase38-branchpoint-batch-frac 0.25 \
  --phase38-branchpoint-margin 0.60 \
  --phase38-branchpoint-recovery-ce-weight 0.55 \
  --phase38-branchpoint-recovery-tokens 12 \
  --phase38-branchpoint-answer-start-tokens 8 \
  --phase38-branchpoint-steps 24 \
  --phase38-branchpoint-temperature 0.30 \
  --phase38-branchpoint-top-k 16 \
  "${SMOLTALK_ARGS[@]}" \
  --custom-json data/kv_campaign/curriculum_live_session_names.jsonl \
  --custom-json data/kv_campaign/curriculum_live_session_names.jsonl \
  --custom-json data/kv_campaign/curriculum_live_session_names.jsonl \
  --custom-json data/kv_campaign/curriculum_live_session_names.jsonl \
  --custom-json data/kv_campaign/curriculum_answer_realization.jsonl \
  --custom-json data/kv_campaign/curriculum_answer_realization.jsonl \
  --custom-json data/kv_campaign/curriculum_memory_reasoning.jsonl \
  --custom-json data/kv_campaign/curriculum_contextual_drag.jsonl \
  --custom-json data/kv_campaign/curriculum_heuristic_override.jsonl \
  --custom-json data/kv_campaign/curriculum_heuristic_override.jsonl \
  --custom-json data/kv_campaign/curriculum_identity_varied.jsonl \
  --custom-json data/kv_campaign/curriculum_binding_current.jsonl \
  --custom-json data/kv_campaign/curriculum_structured_single_field.jsonl \
  --custom-json data/kv_campaign/curriculum_structured_pair_fields.jsonl \
  --custom-json data/kv_campaign/curriculum_structured_fields.jsonl \
  --custom-json data/kv_campaign/curriculum_language_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_language_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_irrelevant_memory_guardrail.jsonl \
  --custom-json data/kv_campaign/curriculum_irrelevant_memory_guardrail.jsonl

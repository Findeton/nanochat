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
LOG="${LOG:-$PWD/dev/logs/${OUTPUT_TAG}.train.log}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo
echo "Starting symmetric memory training at $(date -Is)"
echo "Base checkpoint: ${BASE_TAG} @ ${BASE_STEP}"
echo "Output tag: ${OUTPUT_TAG}"
echo "Steps: ${STEPS}"
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
  --device-batch-size 2 \
  --max-seq-len 128 \
  --gate-lr 0.0005 \
  --memory-lr 0.00025 \
  --interface-lr 0.000002 \
  --train-lr 0.0000005 \
  --save-every "${SAVE_EVERY}" \
  --skip-smoltalk \
  --custom-json data/kv_campaign/stage1_identity.jsonl \
  --custom-json data/kv_campaign/stage2_structured.jsonl \
  --custom-json data/kv_campaign/stage2_structured.jsonl \
  --custom-json data/kv_campaign/stage3_mixed.jsonl \
  --custom-json data/kv_campaign/stage3_mixed.jsonl \
  --target-selection final \
  --context-mode full \
  --memory-build-mode turn \
  --joint-top-layers 2 \
  --weighted-answer-ce-weight 0.45 \
  --weighted-template-ce 0.15 \
  --weighted-fact-ce 1.0 \
  --weighted-rest-ce 0.30 \
  --fact-span-margin-loss-weight 0.25 \
  --fact-span-margin 2.5 \
  --guardrail-kl-weight 0.20 \
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
  --phase38-trajectory-loss-weight 0.75 \
  --phase38-trajectory-batch-frac 0.50 \
  --phase38-trajectory-margin 1.0 \
  --phase38-trajectory-gold-ce-weight 0.20 \
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
  --phase38-hard-token-loss-weight 0.75 \
  --phase38-hard-token-margin 1.5 \
  --phase38-hard-token-top-k 3 \
  --phase38-hard-token-post-key 2 \
  --phase38-default-branch-loss-weight 0.50 \
  --phase38-default-branch-margin 2.0 \
  --phase38-default-recovery-ce-weight 0.20 \
  --phase38-default-recovery-tokens 8 \
  --phase38-probe-binding-loss-weight 0.40 \
  --phase38-probe-binding-margin 2.0 \
  --phase38-probe-binding-max-spans 4

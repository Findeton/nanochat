#!/usr/bin/env bash
set -euo pipefail

cd /nanochat
export PYTHONPATH=/nanochat

TAG="d12-memory-phase38-trajectory-from36-500"
LOG="/nanochat/dev/logs/${TAG}.train.log"
mkdir -p /nanochat/dev/logs

exec > >(tee -a "$LOG") 2>&1

echo "Starting ${TAG} at $(date -Is)"
echo "Base checkpoint: d12-memory-phase36-branch-recovery-from35-250 step 500"
echo "Log: ${LOG}"

python3 -u scripts/chat_memory.py \
  --run dummy \
  --device-type cuda \
  --model-tag d12-memory-phase36-branch-recovery-from35-250 \
  --model-step 500 \
  --output-tag "${TAG}" \
  --num-iterations 500 \
  --phase38-steps 500 \
  --device-batch-size 2 \
  --max-seq-len 128 \
  --gate-lr 0.001 \
  --memory-lr 0.0005 \
  --interface-lr 0.000005 \
  --train-lr 0.000001 \
  --save-every 250 \
  --skip-smoltalk \
  --custom-json data/kv_campaign/stage1_identity.jsonl \
  --custom-json data/kv_campaign/stage2_structured.jsonl \
  --custom-json data/kv_campaign/stage3_mixed.jsonl \
  --target-selection final \
  --context-mode full \
  --memory-build-mode turn \
  --joint-top-layers 2 \
  --weighted-answer-ce-weight 0.5 \
  --weighted-template-ce 0.2 \
  --weighted-fact-ce 1.0 \
  --weighted-rest-ce 0.35 \
  --fact-span-margin-loss-weight 0.25 \
  --fact-span-margin 2.5 \
  --guardrail-kl-weight 0.2 \
  --write-diversity-loss-weight 0.05 \
  --memory-utility-loss-weight 0.5 \
  --memory-utility-margin 0.5 \
  --anchor-margin-loss-weight 0.25 \
  --anchor-margin 2.0 \
  --anchor-loss-tokens 3 \
  --memory-probe-loss-weight 0.1 \
  --key-token-ce-loss-weight 0.75 \
  --key-token-rank-loss-weight 0.75 \
  --key-token-rank-margin 2.0 \
  --key-token-utility-loss-weight 1.0 \
  --key-token-utility-margin 2.0 \
  --deference-start-phase phase2 \
  --phase2-answer-start-ce-loss-weight 0.3 \
  --phase2-answer-start-tokens 10 \
  --first-fact-token-multiplier 2.0 \
  --habit-confuser-loss-weight 0.4 \
  --habit-confuser-margin 3.0 \
  --habit-confuser-top-k 8 \
  --span-contrast-loss-weight 0.25 \
  --span-contrast-margin 1.5 \
  --span-utility-loss-weight 0.2 \
  --span-utility-margin 1.0 \
  --span-contrast-top-k 4 \
  --span-contrast-max-tokens 12 \
  --post-key-loss-weight 0.10 \
  --post-key-repeat-margin 3.0 \
  --post-key-max-groups 4 \
  --phase3-recovery-loss-weight 0.0 \
  --phase35-branch-loss-weight 0.0 \
  --phase35-stop-loss-weight 0.0 \
  --phase36-branch-recovery-loss-weight 0.0 \
  --phase37-branch-loss-weight 0.0 \
  --phase37-branch-batch-frac 0.0 \
  --phase38-trajectory-loss-weight 1.0 \
  --phase38-trajectory-batch-frac 0.75 \
  --phase38-trajectory-margin 1.0 \
  --phase38-trajectory-gold-ce-weight 0.20 \
  --phase38-sample-steps 24 \
  --phase38-temperature 0.30 \
  --phase38-top-k 12 \
  --phase38-min-prefix-tokens 3 \
  --phase38-answer-tokens 32 \
  --phase38-max-span-tokens 12 \
  --phase38-post-key-tokens 8 \
  --phase38-no-memory-confuser-top-k 16 \
  --phase38-probe-binding-loss-weight 0.5 \
  --phase38-probe-binding-margin 2.0 \
  --phase38-probe-binding-max-spans 4

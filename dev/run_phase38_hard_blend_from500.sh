#!/usr/bin/env bash
set -euo pipefail

cd /nanochat
export PYTHONPATH=/nanochat

TAG="d12-memory-phase38-default-branch-lr3-from500"
LOG="/nanochat/dev/logs/${TAG}.train.log"
mkdir -p /nanochat/dev/logs

exec > >(tee -a "$LOG") 2>&1

echo "Starting ${TAG} at $(date -Is)"
echo "Base checkpoint: d12-memory-phase38-trajectory-from36-500 step 500"
echo "Schedule: 3x lower LR, softened phase36/phase37 blend pressure, explicit no-memory/default branch recovery"
echo "Log: ${LOG}"

python3 -u scripts/chat_memory.py \
  --run dummy \
  --device-type cuda \
  --model-tag d12-memory-phase38-trajectory-from36-500 \
  --model-step 500 \
  --output-tag "${TAG}" \
  --num-iterations 1000 \
  --phase38-steps 1000 \
  --device-batch-size 2 \
  --max-seq-len 128 \
  --gate-lr 0.0003333333 \
  --memory-lr 0.0001666667 \
  --interface-lr 0.0000016667 \
  --train-lr 0.0000003333 \
  --save-every 250 \
  --skip-smoltalk \
  --custom-json data/kv_campaign/stage1_identity.jsonl \
  --custom-json data/kv_campaign/stage2_structured.jsonl \
  --custom-json data/kv_campaign/stage2_structured.jsonl \
  --custom-json data/kv_campaign/stage2_structured.jsonl \
  --custom-json data/kv_campaign/stage3_mixed.jsonl \
  --custom-json data/kv_campaign/stage3_mixed.jsonl \
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
  --guardrail-kl-weight 0.2 \
  --write-diversity-loss-weight 0.05 \
  --memory-utility-loss-weight 0.45 \
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
  --first-fact-token-multiplier 2.5 \
  --habit-confuser-loss-weight 0.4 \
  --habit-confuser-margin 3.0 \
  --habit-confuser-top-k 8 \
  --span-contrast-loss-weight 0.30 \
  --span-contrast-margin 1.5 \
  --span-utility-loss-weight 0.20 \
  --span-utility-margin 1.0 \
  --span-contrast-top-k 4 \
  --span-contrast-max-tokens 12 \
  --span-hard-token-loss-weight 0.65 \
  --span-hard-token-margin 4.0 \
  --span-hard-token-top-k 3 \
  --post-key-loss-weight 0.25 \
  --post-key-repeat-margin 3.0 \
  --post-key-max-groups 4 \
  --phase3-recovery-loss-weight 0.0 \
  --phase35-branch-loss-weight 0.20 \
  --phase35-branch-margin 5.0 \
  --phase35-branch-tokens 2 \
  --phase35-branch-no-memory-top-k 12 \
  --phase35-stop-loss-weight 0.15 \
  --phase35-stop-margin 4.0 \
  --phase36-branch-recovery-loss-weight 0.35 \
  --phase36-branch-recovery-batch-frac 0.20 \
  --phase36-branch-recovery-temperature 0.35 \
  --phase36-branch-recovery-top-k 12 \
  --phase36-branch-min-prefix-tokens 2 \
  --phase36-branch-recovery-max-target-tokens 12 \
  --phase36-branch-recovery-post-key-tokens 8 \
  --phase36-branch-margin-loss-weight 0.60 \
  --phase36-branch-margin 5.0 \
  --phase36-branch-anchor-tokens 3 \
  --phase36-no-memory-confuser-top-k 16 \
  --phase37-branch-loss-weight 0.40 \
  --phase37-branch-batch-frac 0.20 \
  --phase37-branch-ce-weight 0.25 \
  --phase37-branch-margin 2.0 \
  --phase37-branch-max-confusers 4 \
  --phase37-stop-loss-weight 0.25 \
  --phase37-stop-margin 5.0 \
  --phase37-sample-steps 16 \
  --phase37-temperature 0.35 \
  --phase37-top-k 12 \
  --phase37-min-prefix-tokens 2 \
  --phase37-max-span-tokens 12 \
  --phase37-post-key-tokens 8 \
  --phase37-no-memory-confuser-top-k 16 \
  --phase38-trajectory-loss-weight 0.80 \
  --phase38-trajectory-batch-frac 0.55 \
  --phase38-trajectory-margin 1.0 \
  --phase38-trajectory-gold-ce-weight 0.20 \
  --phase38-sample-steps 24 \
  --phase38-temperature 0.30 \
  --phase38-high-temperature 0.60 \
  --phase38-high-temperature-frac 0.30 \
  --phase38-top-k 16 \
  --phase38-min-prefix-tokens 2 \
  --phase38-answer-tokens 32 \
  --phase38-max-span-tokens 12 \
  --phase38-post-key-tokens 8 \
  --phase38-no-memory-confuser-top-k 16 \
  --phase38-hard-token-loss-weight 1.0 \
  --phase38-hard-token-margin 1.5 \
  --phase38-hard-token-top-k 3 \
  --phase38-hard-token-post-key 2 \
  --phase38-default-branch-loss-weight 1.25 \
  --phase38-default-branch-margin 4.0 \
  --phase38-default-recovery-ce-weight 0.35 \
  --phase38-default-recovery-tokens 8 \
  --phase38-probe-binding-loss-weight 0.4 \
  --phase38-probe-binding-margin 2.0 \
  --phase38-probe-binding-max-spans 4

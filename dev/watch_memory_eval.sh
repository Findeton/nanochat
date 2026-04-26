#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/nanochat}"
TAG="${1:-d12-memory-keyloss-phase1-gpu}"
CKPT_DIR="${CKPT_DIR:-/root/.cache/nanochat/chatsft_checkpoints/${TAG}}"
LOG_DIR="${ROOT}/dev/logs"
ANALYSIS_DIR="${ROOT}/dev/analysis"
LOG_FILE="${LOG_DIR}/${TAG}-eval-watcher.log"
LOCK_DIR="${LOG_DIR}/${TAG}-eval-watcher.lock"
STEPS="${STEPS:-500 1000 1500 2000 2500 3000}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
LIMIT="${LIMIT:-5}"
DET_TOKENS="${DET_TOKENS:-24}"
SAMP_TOKENS="${SAMP_TOKENS:-24}"
SAMP_TEMP="${SAMP_TEMP:-0.2}"
SAMP_TOP_K="${SAMP_TOP_K:-8}"
SAMP_SEEDS="${SAMP_SEEDS:-1}"

mkdir -p "${LOG_DIR}" "${ANALYSIS_DIR}"
exec >> "${LOG_FILE}" 2>&1

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "[$(date -Is)] watcher already running for ${TAG}"
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

log() {
  echo "[$(date -Is)] $*"
}

dataset_args=(
  --dataset "${ROOT}/data/kv_campaign/stage1_identity.jsonl"
  --dataset "${ROOT}/data/kv_campaign/stage2_structured.jsonl"
  --dataset "${ROOT}/data/kv_campaign/stage3_mixed.jsonl"
)

cd "${ROOT}" || exit 1
export PYTHONPATH="${ROOT}"

log "watcher started tag=${TAG} device=${DEVICE_TYPE} limit=${LIMIT} steps=${STEPS}"

for step in ${STEPS}; do
  model_file=$(printf "%s/model_%06d.pt" "${CKPT_DIR}" "${step}")
  meta_file=$(printf "%s/meta_%06d.json" "${CKPT_DIR}" "${step}")
  suite_json=$(printf "%s/%s_step%06d_suite_limit%s.json" "${ANALYSIS_DIR}" "${TAG}" "${step}" "${LIMIT}")
  diagnose_json=$(printf "%s/%s_step%06d_diagnose_limit%s.json" "${ANALYSIS_DIR}" "${TAG}" "${step}" "${LIMIT}")

  if [[ -f "${suite_json}" && -f "${diagnose_json}" ]]; then
    log "step ${step}: eval outputs already exist"
    continue
  fi

  log "step ${step}: waiting for ${model_file}"
  while [[ ! -f "${model_file}" || ! -f "${meta_file}" ]]; do
    sleep 60
  done

  # The trainer writes model and metadata before optimizer state; give the filesystem
  # a small quiet period before loading the checkpoint from another process.
  sleep 20
  log "step ${step}: checkpoint detected, starting evaluation"

  if [[ ! -f "${suite_json}" ]]; then
    log "step ${step}: running eval_memory_suite -> ${suite_json}"
    python3 scripts/eval_memory_suite.py \
      --source sft \
      --model-tag "${TAG}" \
      --step "${step}" \
      --device-type "${DEVICE_TYPE}" \
      "${dataset_args[@]}" \
      --limit "${LIMIT}" \
      --deterministic-max-tokens "${DET_TOKENS}" \
      --sampled-max-tokens "${SAMP_TOKENS}" \
      --sampled-temperature "${SAMP_TEMP}" \
      --sampled-top-k "${SAMP_TOP_K}" \
      --sampled-seeds "${SAMP_SEEDS}" \
      --failures-to-print 2 \
      --json-output "${suite_json}"
    log "step ${step}: eval_memory_suite done"
  fi

  if [[ ! -f "${diagnose_json}" ]]; then
    log "step ${step}: running diagnose_memory_checkpoint -> ${diagnose_json}"
    python3 scripts/diagnose_memory_checkpoint.py \
      --source sft \
      --model-tag "${TAG}" \
      --step "${step}" \
      --device-type "${DEVICE_TYPE}" \
      "${dataset_args[@]}" \
      --limit "${LIMIT}" \
      --deterministic-max-tokens "${DET_TOKENS}" \
      --failures-to-keep 2 \
      --ablation-mode full \
      --ablation-mode no_memory_state \
      --ablation-mode no_memory_read \
      --save-json "${diagnose_json}"
    log "step ${step}: diagnose_memory_checkpoint done"
  fi
done

log "watcher finished all configured steps"

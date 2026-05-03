#!/usr/bin/env bash
set -euo pipefail

# Resume V5 from the latest stable checkpoint after discovering that the first
# Phase B pass did not enable generated-prefix recovery. This run starts directly
# in Phase B, then continues into Phase C.

export BASE_TAG="${BASE_TAG:-d12-memory-symmetric-v5-fullcopy-gradclean-from21000}"
export BASE_STEP="${BASE_STEP:-40000}"
export OUTPUT_TAG="${OUTPUT_TAG:-d12-memory-symmetric-v5-phaseB-recovery-from40000}"

export TRAIN_STEPS="${TRAIN_STEPS:-70000}"
export PHASE_A_STEPS="${PHASE_A_STEPS:-0}"
export PHASE_B_STEPS="${PHASE_B_STEPS:-10000}"
export PHASE_C_STEPS="${PHASE_C_STEPS:-60000}"

# Full-trunk recovery has extra generated-prefix forwards; keep it small enough
# to avoid the 24GB GPU cliff. Phase C previously OOMed at 16, so restart at 12.
export BATCH_SIZE="${BATCH_SIZE:-12}"
export PHASE_B_BATCH_SIZE="${PHASE_B_BATCH_SIZE:-2}"
export PHASE_C_BATCH_SIZE="${PHASE_C_BATCH_SIZE:-12}"

export PHASE3_RECOVERY_LOSS_WEIGHT="${PHASE3_RECOVERY_LOSS_WEIGHT:-0.35}"
export PHASE3_RECOVERY_BATCH_FRAC="${PHASE3_RECOVERY_BATCH_FRAC:-0.35}"
export PHASE3_RECOVERY_STEPS="${PHASE3_RECOVERY_STEPS:-20}"
export PHASE3_RECOVERY_TEMPERATURE="${PHASE3_RECOVERY_TEMPERATURE:-0.34}"
export PHASE3_RECOVERY_TOP_K="${PHASE3_RECOVERY_TOP_K:-16}"
export PHASE3_RECOVERY_MAX_TARGET_TOKENS="${PHASE3_RECOVERY_MAX_TARGET_TOKENS:-12}"
export PHASE3_RECOVERY_ANCHOR_LOSS_WEIGHT="${PHASE3_RECOVERY_ANCHOR_LOSS_WEIGHT:-0.55}"

exec "$(dirname "$0")/run_symmetric_memory_v5_fullcopy_from21000.sh"

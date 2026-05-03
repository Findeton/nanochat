#!/usr/bin/env bash
set -euo pipefail

# Continue V5 directly in Phase C after adding random-key binding guardrails.
# The base checkpoint is supplied at launch time from the latest completed step.

export BASE_TAG="${BASE_TAG:-d12-memory-symmetric-v5-phaseB-recovery-from40000}"
export BASE_STEP="${BASE_STEP:-12000}"
export OUTPUT_TAG="${OUTPUT_TAG:-d12-memory-symmetric-v5-keyguard-phaseC-from${BASE_STEP}}"

export TRAIN_STEPS="${TRAIN_STEPS:-60000}"
export PHASE_A_STEPS="${PHASE_A_STEPS:-0}"
export PHASE_B_STEPS="${PHASE_B_STEPS:-0}"
export PHASE_C_STEPS="${PHASE_C_STEPS:-60000}"

export BATCH_SIZE="${BATCH_SIZE:-12}"
export PHASE_B_BATCH_SIZE="${PHASE_B_BATCH_SIZE:-2}"
export PHASE_C_BATCH_SIZE="${PHASE_C_BATCH_SIZE:-12}"

export CURRICULUM_V5_RANDOM_KEY_BINDING_SIZE="${CURRICULUM_V5_RANDOM_KEY_BINDING_SIZE:-900000}"

exec "$(dirname "$0")/run_symmetric_memory_v5_fullcopy_from21000.sh"

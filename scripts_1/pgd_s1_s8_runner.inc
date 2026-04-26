# Shared runner for scripts/run_pgd_s1_s8_*.sh.
# Source this file from a dataset/config entrypoint; do not run it directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${DATASET_KEY:?DATASET_KEY must be set before sourcing pgd_s1_s8_runner.inc}"
: "${ROOT_PATH:?ROOT_PATH must be set before sourcing pgd_s1_s8_runner.inc}"

EXP_NAME="${EXP_NAME:-pgd_${DATASET_KEY}}"

# Keep outputs outside scripts/ and preserve the existing output layout.
PROJECT_DIR="${PROJECT_DIR:-$PROJECT_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
TEACHER_OUTPUT_ROOT="${TEACHER_OUTPUT_ROOT:-$OUTPUT_ROOT}"

# Keep step-3 pruning/recovery disabled by default for all S1-S8 batch runs.
# Override STEP3_PRUNING=1 and set STEP3_PRUNING_EPOCHS>0 only when needed.
STEP3_PRUNING="${STEP3_PRUNING:-0}"

export_if_set() {
  local name="$1"
  if [ "${!name+x}" = "x" ]; then
    export "$name"
  fi
}

for variable_name in \
  DATASET_KEY ROOT_PATH EXP_NAME PROJECT_DIR OUTPUT_ROOT TEACHER_OUTPUT_ROOT \
  STEP3_PRUNING STEP3_PRUNING_EPOCHS TEACHER_MODEL MAX_EPOCHS_TEACHER MAX_EPOCHS_STUDENT \
  BATCH_SIZE NUM_WORKERS PATCH_SIZE_H PATCH_SIZE_W \
  USE_KD_OUTPUT USE_SPARSITY USE_FEATURE_DISTILL USE_AUX_LOSS \
  LAMBDA_DISTILL LAMBDA_SPARSITY LAMBDA_FEAT LAMBDA_AUX \
  FEATURE_LAYERS SEG_LOSS_METHOD DISTILL_LOSS_METHOD DISTILL_TEMPERATURE LOSS_METHOD \
  SPLIT_SEED PREPARE_SPLITS PREPARE_SPLITS_OVERWRITE PREPARE_ANALYSIS CONDA_ENV CUDA_MODULE; do
  export_if_set "$variable_name"
done

# S1 and S5 static strategies run all these ratios. Override with:
# STATIC_PRUNE_RATES="0.2 0.4" bash scripts/run_....sh
STATIC_PRUNE_RATES_STRING="${STATIC_PRUNE_RATES:-0.05 0.1 0.3 0.5 0.75}"
read -r -a STATIC_RATE_LIST <<< "$STATIC_PRUNE_RATES_STRING"

run_pgd_strategy_once() {
  local strategy="$1"
  local ratio="${2:-}"

  export PRUNE_STRATEGY="$strategy"
  if [ -n "$ratio" ]; then
    export PRUNE_RATE="$ratio"
    echo ">>> Dataset=$DATASET_KEY strategy=$strategy prune_rate=$ratio"
  else
    unset PRUNE_RATE || true
    echo ">>> Dataset=$DATASET_KEY strategy=$strategy prune_rate=auto"
  fi

  bash "$PROJECT_ROOT/run_pgd_common.sh"
}

for strategy in S1 S2 S3 S4 S5 S6 S7 S8; do
  case "$strategy" in
    S1|S5)
      for ratio in "${STATIC_RATE_LIST[@]}"; do
        run_pgd_strategy_once "$strategy" "$ratio"
      done
      ;;
    *)
      run_pgd_strategy_once "$strategy"
      ;;
  esac
done

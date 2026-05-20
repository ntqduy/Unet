# Shared runner for PGD S1-S12 dataset entrypoints.
# Source this file from a dataset/config entrypoint; do not run it directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${DATASET_KEY:?DATASET_KEY must be set before sourcing pgd_s1_s12_runner.inc}"
: "${ROOT_PATH:?ROOT_PATH must be set before sourcing pgd_s1_s12_runner.inc}"

EXP_NAME="${EXP_NAME:-pgd_${DATASET_KEY}}"

# Keep outputs outside scripts/ and preserve the existing output layout.
PROJECT_DIR="${PROJECT_DIR:-$PROJECT_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
TEACHER_OUTPUT_ROOT="${TEACHER_OUTPUT_ROOT:-$OUTPUT_ROOT}"

# Keep step-3 pruning/recovery disabled by default for all S1-S12 batch runs.
# Override STEP3_PRUNING=1 and set STEP3_PRUNING_EPOCHS>0 only when needed.
STEP3_PRUNING="${STEP3_PRUNING:-0}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-20}"

# DEVICE can be one GPU id ("0") or multiple ids ("0 1 2 3").
# With multiple ids, each PGD strategy/rate run is launched on the next GPU
# and the runner keeps at most one active job per listed device.
# By default the first run is executed synchronously to create/reuse the shared
# teacher checkpoint before parallel S-runs start. Set PGD_TEACHER_PREPARE_FIRST=0
# if the teacher checkpoint is already available and you want full parallelism.
DEVICE_STRING="${DEVICE:-0}"
DEVICE_STRING="${DEVICE_STRING//,/ }"
read -r -a DEVICE_LIST <<< "$DEVICE_STRING"
if [ "${#DEVICE_LIST[@]}" -eq 0 ]; then
  DEVICE_LIST=("0")
fi
PGD_TEACHER_PREPARE_FIRST="${PGD_TEACHER_PREPARE_FIRST:-1}"
PGD_RUN_INDEX=0
PGD_ACTIVE_PIDS=()

wait_for_pgd_slot() {
  local max_jobs="${#DEVICE_LIST[@]}"
  while [ "${#PGD_ACTIVE_PIDS[@]}" -ge "$max_jobs" ]; do
    local pid="${PGD_ACTIVE_PIDS[0]}"
    wait "$pid"
    PGD_ACTIVE_PIDS=("${PGD_ACTIVE_PIDS[@]:1}")
  done
}

wait_for_all_pgd_jobs() {
  local pid
  for pid in "${PGD_ACTIVE_PIDS[@]}"; do
    wait "$pid"
  done
  PGD_ACTIVE_PIDS=()
}

# Main paper loss variants only:
#   seg_only = student fine-tunes with segmentation loss only
#   seg_kd   = student fine-tunes with segmentation + teacher output distillation
# Override with LOSS_VARIANTS="seg_only" or LOSS_VARIANTS="seg_kd" if needed.
LOSS_VARIANTS_STRING="${LOSS_VARIANTS:-seg_only seg_kd}"
read -r -a LOSS_VARIANT_LIST <<< "$LOSS_VARIANTS_STRING"
DEFAULT_LAMBDA_DISTILL="${LAMBDA_DISTILL:-0.3}"
DEFAULT_DISTILL_LOSS_METHOD="${DISTILL_LOSS_METHOD:-mse}"

export_if_set() {
  local name="$1"
  if [ "${!name+x}" = "x" ]; then
    export "$name"
  fi
}

for variable_name in \
  DATASET_KEY ROOT_PATH EXP_NAME PROJECT_DIR OUTPUT_ROOT TEACHER_OUTPUT_ROOT \
  SKIP_IF_LAST_EXISTS FORCE_RETRAIN_TEACHER FORCE_REPRUNE FORCE_RETRAIN_STUDENT \
  STEP3_PRUNING STEP3_PRUNING_EPOCHS TEACHER_MODEL MAX_EPOCHS_TEACHER MAX_EPOCHS_STUDENT \
  BATCH_SIZE NUM_WORKERS PATCH_SIZE_H PATCH_SIZE_W \
  STUDENT_VARIANT \
  USE_KD_OUTPUT USE_SPARSITY USE_FEATURE_DISTILL USE_AUX_LOSS \
  LAMBDA_DISTILL LAMBDA_SPARSITY LAMBDA_FEAT LAMBDA_AUX \
  FEATURE_LAYERS SEG_LOSS_METHOD DISTILL_LOSS_METHOD DISTILL_TEMPERATURE LOSS_METHOD \
  DEVICE PGD_TEACHER_PREPARE_FIRST EARLY_STOP_PATIENCE \
  SPLIT_SEED PREPARE_SPLITS PREPARE_SPLITS_OVERWRITE PREPARE_ANALYSIS CONDA_ENV CUDA_MODULE; do
  export_if_set "$variable_name"
done

# S1, S5, and S9 static strategies run all these ratios. Override with:
# STATIC_PRUNE_RATES="0.2 0.4" bash scripts/run_....sh
STATIC_PRUNE_RATES_STRING="${STATIC_PRUNE_RATES:-0.05 0.1 0.3 0.5 0.7 0.9}"
read -r -a STATIC_RATE_LIST <<< "$STATIC_PRUNE_RATES_STRING"

run_pgd_strategy_once() {
  local strategy="$1"
  local ratio="${2:-}"
  local device="${DEVICE_LIST[$((PGD_RUN_INDEX % ${#DEVICE_LIST[@]}))]}"
  PGD_RUN_INDEX=$((PGD_RUN_INDEX + 1))
  local run_number="$PGD_RUN_INDEX"

  export PRUNE_STRATEGY="$strategy"
  export DEVICE="$device"
  if [ -n "$ratio" ]; then
    export PRUNE_RATE="$ratio"
    echo ">>> Dataset=$DATASET_KEY strategy=$strategy prune_rate=$ratio device=$device"
  else
    unset PRUNE_RATE || true
    echo ">>> Dataset=$DATASET_KEY strategy=$strategy prune_rate=auto device=$device"
  fi

  if [ "${#DEVICE_LIST[@]}" -gt 1 ] && { [ "$PGD_TEACHER_PREPARE_FIRST" != "1" ] || [ "$run_number" -gt 1 ]; }; then
    wait_for_pgd_slot
    DEVICE="$device" bash "$PROJECT_ROOT/run_pgd_common.sh" &
    PGD_ACTIVE_PIDS+=("$!")
  else
    DEVICE="$device" bash "$PROJECT_ROOT/run_pgd_common.sh"
  fi
}

for loss_variant in "${LOSS_VARIANT_LIST[@]}"; do
  case "$loss_variant" in
    seg_only)
      export USE_KD_OUTPUT=0
      export LAMBDA_DISTILL=0
      export USE_SPARSITY=0
      export USE_FEATURE_DISTILL=0
      export USE_AUX_LOSS=0
      export LOSS_METHOD=""
      ;;
    seg_kd)
      export USE_KD_OUTPUT=1
      export LAMBDA_DISTILL="$DEFAULT_LAMBDA_DISTILL"
      export USE_SPARSITY=0
      export USE_FEATURE_DISTILL=0
      export USE_AUX_LOSS=0
      export DISTILL_LOSS_METHOD="$DEFAULT_DISTILL_LOSS_METHOD"
      export LOSS_METHOD=""
      ;;
    *)
      echo "Unsupported LOSS_VARIANTS entry: $loss_variant. Use seg_only and/or seg_kd."
      exit 1
      ;;
  esac

  echo "============================================================"
  echo "Dataset: $DATASET_KEY"
  echo "Loss variant: $loss_variant"
  echo "Strategies: S1-S12"
  echo "============================================================"

  for strategy in S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S12; do
    case "$strategy" in
      S1|S5|S9)
        for ratio in "${STATIC_RATE_LIST[@]}"; do
          run_pgd_strategy_once "$strategy" "$ratio"
        done
        ;;
      *)
        run_pgd_strategy_once "$strategy"
      ;;
    esac
  done
  wait_for_all_pgd_jobs
done

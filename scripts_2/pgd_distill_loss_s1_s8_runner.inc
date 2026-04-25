# Shared runner for scripts_2/run_pgd_distill_loss_s1_s8_*.sh.
# Source from a dataset entrypoint; output stays in the project-level outputs/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${DATASET_KEY:?DATASET_KEY must be set before sourcing pgd_distill_loss_s1_s8_runner.inc}"
: "${ROOT_PATH:?ROOT_PATH must be set before sourcing pgd_distill_loss_s1_s8_runner.inc}"

EXP_NAME="${EXP_NAME:-pgd_${DATASET_KEY}}"

# Preserve the existing output layout:
# outputs/pgd_unet/<dataset>/<teacher>_teacher/<loss_tag>/<output_dir>/
PROJECT_DIR="${PROJECT_DIR:-$PROJECT_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
TEACHER_OUTPUT_ROOT="${TEACHER_OUTPUT_ROOT:-$OUTPUT_ROOT}"

# Keep step-3 pruning/recovery disabled by default for this batch.
STEP3_PRUNING="${STEP3_PRUNING:-0}"

# Normal paper loss:
# Ltotal = Lseg + lambda_distill * Ldistill + lambda_sparsity * Lsparsity
USE_KD_OUTPUT="${USE_KD_OUTPUT:-1}"
USE_SPARSITY="${USE_SPARSITY:-1}"
USE_FEATURE_DISTILL="${USE_FEATURE_DISTILL:-0}"
USE_AUX_LOSS="${USE_AUX_LOSS:-0}"

# Four KD losses for Table III loss study.
# mse is intentionally excluded because the default scripts/ batch already
# covers the original/old MSE distillation.
# Override if needed:
# DISTILL_METHODS="ce dice kl hybrid" bash scripts_2/run_....sh
DISTILL_METHODS_STRING="${DISTILL_METHODS:-ce dice kl hybrid}"
read -r -a DISTILL_METHOD_LIST <<< "$DISTILL_METHODS_STRING"

# S1 and S5 static strategies run all these ratios.
STATIC_PRUNE_RATES_STRING="${STATIC_PRUNE_RATES:-0.05 0.1 0.3 0.5 0.75}"
read -r -a STATIC_RATE_LIST <<< "$STATIC_PRUNE_RATES_STRING"

distill_label() {
  case "$1" in
    ce) echo "CE" ;;
    dice) echo "Dice" ;;
    kl) echo "KL" ;;
    hybrid) echo "Hybrid" ;;
    *) echo "$1" ;;
  esac
}

run_pgd_strategy_once() {
  local strategy="$1"
  local ratio="${2:-}"

  export PRUNE_STRATEGY="$strategy"
  if [ -n "$ratio" ]; then
    export PRUNE_RATE="$ratio"
    echo ">>> Dataset=$DATASET_KEY distill=$DISTILL_LOSS_METHOD strategy=$strategy prune_rate=$ratio"
  else
    unset PRUNE_RATE || true
    echo ">>> Dataset=$DATASET_KEY distill=$DISTILL_LOSS_METHOD strategy=$strategy prune_rate=auto"
  fi

  source "$PROJECT_ROOT/run_pgd_common.sh"
}

for distill_method in "${DISTILL_METHOD_LIST[@]}"; do
  if [ "$distill_method" = "mse" ]; then
    echo "Skipping distill_method=mse because scripts_2 is reserved for non-MSE KD loss study."
    continue
  fi
  export DISTILL_LOSS_METHOD="$distill_method"
  export LOSS_METHOD="Proposed + $(distill_label "$distill_method") KD"

  echo "============================================================"
  echo "Dataset: $DATASET_KEY"
  echo "Distillation loss: $DISTILL_LOSS_METHOD"
  echo "Loss label: $LOSS_METHOD"
  echo "Formula: Lseg + lambda_distill * Ldistill + lambda_sparsity * Lsparsity"
  echo "============================================================"

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
done

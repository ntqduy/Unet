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

# These batch scripts are meant to test recovery/distillation losses, so step 3
# is enabled by default. Override STEP3_PRUNING=0 if you need pruning-only runs.
STEP3_PRUNING="${STEP3_PRUNING:-1}"

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

  source "$PROJECT_ROOT/run_pgd_common.sh"
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

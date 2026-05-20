# Shared runner for basic model dataset entrypoints.
# DEVICE can be one id ("0") or multiple ids ("0 1 2 3"). With multiple ids,
# each model is launched on the next GPU and at most one job runs per GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${DATASET:?DATASET must be set before sourcing basic_models_runner.inc}"

IFS=' ' read -r -a MODELS <<< "${MODELS:-unet resunet vnet unetr unet_resnet152 att_unet r2unet unet_plus_plus}"
DEVICE_STRING="${DEVICE:-0}"
DEVICE_STRING="${DEVICE_STRING//,/ }"
read -r -a DEVICE_LIST <<< "$DEVICE_STRING"
if [ "${#DEVICE_LIST[@]}" -eq 0 ]; then
  DEVICE_LIST=("0")
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BASE_LR="${BASE_LR:-0.01}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ENCODER_PRETRAINED="${ENCODER_PRETRAINED:-1}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

BASIC_PIDS=()
MODEL_INDEX=0

wait_for_basic_slot() {
  local max_jobs="${#DEVICE_LIST[@]}"
  while [ "${#BASIC_PIDS[@]}" -ge "$max_jobs" ]; do
    local pid="${BASIC_PIDS[0]}"
    wait "$pid"
    BASIC_PIDS=("${BASIC_PIDS[@]:1}")
  done
}

wait_for_all_basic_jobs() {
  local pid
  for pid in "${BASIC_PIDS[@]}"; do
    wait "$pid"
  done
  BASIC_PIDS=()
}

run_basic_model_once() {
  local model="$1"
  local device="${DEVICE_LIST[$((MODEL_INDEX % ${#DEVICE_LIST[@]}))]}"
  MODEL_INDEX=$((MODEL_INDEX + 1))
  local args=(
    --datasets "$DATASET"
    --models "$model"
    --output-root "$OUTPUT_ROOT"
    --device "$device"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --base-lr "$BASE_LR"
    --num-workers "$NUM_WORKERS"
    --encoder-pretrained "$ENCODER_PRETRAINED"
  )
  if [ "$FORCE_RETRAIN" = "1" ]; then
    args+=(--force-retrain)
  fi
  echo ">>> Basic dataset=$DATASET model=$model device=$device"
  if [ "${#DEVICE_LIST[@]}" -gt 1 ]; then
    wait_for_basic_slot
    python run_basic_model.py "${args[@]}" &
    BASIC_PIDS+=("$!")
  else
    python run_basic_model.py "${args[@]}"
  fi
}

for model in "${MODELS[@]}"; do
  run_basic_model_once "$model"
done

wait_for_all_basic_jobs

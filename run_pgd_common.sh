#!/bin/bash

set -euo pipefail

mkdir -p logs

echo "=============================="
echo "Job ID    : ${SLURM_JOB_ID:-local}"
echo "Node      : $(hostname)"
echo "Dataset   : ${DATASET_KEY}"
echo "Root path : ${ROOT_PATH}"
echo "Start     : $(date)"
echo "=============================="

if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE:-cuda-11.8.0-gcc-11.4.0-cuusula}" || true
fi

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-pgdunet}"
fi

PROJECT_DIR="${PROJECT_DIR:-$HOME/PGD-UNet}"
if [ -d "$PROJECT_DIR" ]; then
  cd "$PROJECT_DIR"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

SPLIT_SEED="${SPLIT_SEED:-1337}"
PREPARE_SPLITS="${PREPARE_SPLITS:-1}"
PREPARE_SPLITS_OVERWRITE="${PREPARE_SPLITS_OVERWRITE:-1}"
if [ "$PREPARE_SPLITS" = "1" ]; then
  SPLIT_ARGS=(--dataset "$DATASET_KEY" --seed "$SPLIT_SEED" --extract)
  if [ "$PREPARE_SPLITS_OVERWRITE" = "1" ]; then
    SPLIT_ARGS+=(--overwrite)
  fi
  python analysis_data/generate_splits.py "${SPLIT_ARGS[@]}"
fi

PRUNE_STRATEGY="${PRUNE_STRATEGY:-S1}"
PRUNE_STRATEGY="$(echo "$PRUNE_STRATEGY" | tr '[:lower:]' '[:upper:]')"
PRUNE_ARGS=()
STEP3_ARGS=()

STEP3_PRUNING="${STEP3_PRUNING:-1}"
STEP3_PRUNING="$(echo "$STEP3_PRUNING" | tr '[:upper:]' '[:lower:]')"
STEP3_PRUNING_EPOCHS="${STEP3_PRUNING_EPOCHS:-4}"
TEACHER_OUTPUT_ROOT="${TEACHER_OUTPUT_ROOT:-outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$TEACHER_OUTPUT_ROOT}"

TEACHER_MODEL="${TEACHER_MODEL:-unet_resnet152}"
MAX_EPOCHS_TEACHER="${MAX_EPOCHS_TEACHER:-50}"
MAX_EPOCHS_STUDENT="${MAX_EPOCHS_STUDENT:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PATCH_SIZE_H="${PATCH_SIZE_H:-256}"
PATCH_SIZE_W="${PATCH_SIZE_W:-256}"

USE_KD_OUTPUT="${USE_KD_OUTPUT:-1}"
USE_SPARSITY="${USE_SPARSITY:-1}"
USE_FEATURE_DISTILL="${USE_FEATURE_DISTILL:-0}"
USE_AUX_LOSS="${USE_AUX_LOSS:-0}"

LAMBDA_DISTILL="${LAMBDA_DISTILL:-0.3}"
LAMBDA_SPARSITY="${LAMBDA_SPARSITY:-0.3}"
LAMBDA_FEAT="${LAMBDA_FEAT:-0.1}"
LAMBDA_AUX="${LAMBDA_AUX:-0.2}"
FEATURE_LAYERS="${FEATURE_LAYERS:-bottleneck}"

case "$STEP3_PRUNING" in
  1|true|yes|y|on)
    STEP3_PRUNING_ENABLED=1
    if ! STEP3_PRUNING_EPOCHS_TAG=$(python -c 'import sys; v = int(sys.argv[1]); sys.exit("STEP3_PRUNING_EPOCHS must be > 0 when STEP3_PRUNING is enabled.") if v <= 0 else None; print(v)' "$STEP3_PRUNING_EPOCHS"); then
      echo "Invalid STEP3_PRUNING_EPOCHS=$STEP3_PRUNING_EPOCHS"
      exit 1
    fi
    STEP3_TAG="$STEP3_PRUNING_EPOCHS_TAG"
    STEP3_ARGS=(--enable_step3_pruning 1 --step3_pruning_epochs "$STEP3_PRUNING_EPOCHS_TAG" --warmup_pruning_epochs "$STEP3_PRUNING_EPOCHS_TAG")
    ;;
  0|false|no|n|off)
    STEP3_PRUNING_ENABLED=0
    STEP3_TAG="no"
    STEP3_ARGS=(--enable_step3_pruning 0 --step3_pruning_epochs 0 --warmup_pruning_epochs 0)
    ;;
  *)
    echo "Unsupported STEP3_PRUNING=$STEP3_PRUNING. Use 1/0, true/false, yes/no, or on/off."
    exit 1
    ;;
esac

case "$PRUNE_STRATEGY" in
  S1)
    PRUNE_METHOD="static"
    PRUNE_RATE="${PRUNE_RATE:-0.5}"
    if ! PRUNE_RATE_TAG=$(python -c 'import sys; v = float(sys.argv[1]); sys.exit("PRUNE_RATE must be in [0, 1).") if not (0.0 <= v < 1.0) else None; print(f"{v:.12g}")' "$PRUNE_RATE"); then
      echo "Invalid PRUNE_RATE=$PRUNE_RATE for S1 static pruning"
      exit 1
    fi
    RATE_TAG="$PRUNE_RATE_TAG"
    PRUNE_ARGS=(--prune_strategy "$PRUNE_STRATEGY" --prune_method "$PRUNE_METHOD" --static_prune_ratio "$PRUNE_RATE" --prune_ratio "$PRUNE_RATE")
    ;;
  S2)
    PRUNE_METHOD="kneedle"
    RATE_TAG="auto"
    PRUNE_ARGS=(--prune_strategy "$PRUNE_STRATEGY" --prune_method "$PRUNE_METHOD")
    ;;
  S3)
    PRUNE_METHOD="otsu"
    RATE_TAG="auto"
    PRUNE_ARGS=(--prune_strategy "$PRUNE_STRATEGY" --prune_method "$PRUNE_METHOD")
    ;;
  S4)
    PRUNE_METHOD="gmm"
    RATE_TAG="auto"
    PRUNE_ARGS=(--prune_strategy "$PRUNE_STRATEGY" --prune_method "$PRUNE_METHOD")
    ;;
  S5)
    PRUNE_METHOD="middle_static"
    PRUNE_RATE="${PRUNE_RATE:-0.5}"
    if ! PRUNE_RATE_TAG=$(python -c 'import sys; v = float(sys.argv[1]); sys.exit("PRUNE_RATE must be in [0, 1).") if not (0.0 <= v < 1.0) else None; print(f"{v:.12g}")' "$PRUNE_RATE"); then
      echo "Invalid PRUNE_RATE=$PRUNE_RATE for S5 middle-static pruning"
      exit 1
    fi
    RATE_TAG="$PRUNE_RATE_TAG"
    PRUNE_ARGS=(--prune_strategy "$PRUNE_STRATEGY" --prune_method "$PRUNE_METHOD" --static_prune_ratio "$PRUNE_RATE" --prune_ratio "$PRUNE_RATE")
    ;;
  S6)
    PRUNE_METHOD="middle_kneedle"
    RATE_TAG="auto"
    PRUNE_ARGS=(--prune_strategy "$PRUNE_STRATEGY" --prune_method "$PRUNE_METHOD")
    ;;
  S7)
    PRUNE_METHOD="middle_otsu"
    RATE_TAG="auto"
    PRUNE_ARGS=(--prune_strategy "$PRUNE_STRATEGY" --prune_method "$PRUNE_METHOD")
    ;;
  S8)
    PRUNE_METHOD="middle_gmm"
    RATE_TAG="auto"
    PRUNE_ARGS=(--prune_strategy "$PRUNE_STRATEGY" --prune_method "$PRUNE_METHOD")
    ;;
  *)
    echo "Unsupported PRUNE_STRATEGY=$PRUNE_STRATEGY. Use S1, S2, S3, S4, S5, S6, S7, or S8."
    exit 1
    ;;
esac

OUTPUT_DIR="output_${PRUNE_METHOD}_${RATE_TAG}_${STEP3_TAG}"

echo "Teacher model: $TEACHER_MODEL"
echo "Pruning strategy: $PRUNE_METHOD"
if [ "$PRUNE_METHOD" = "static" ] || [ "$PRUNE_METHOD" = "middle_static" ]; then
  echo "Static prune ratio: $PRUNE_RATE_TAG"
else
  echo "Static prune ratio: not used"
fi
if [ "$STEP3_PRUNING_ENABLED" = "1" ]; then
  echo "Step-3 pruning: enabled"
  echo "Step-3 pruning epochs: $STEP3_PRUNING_EPOCHS_TAG"
else
  echo "Step-3 pruning: disabled"
  echo "Step-3 pruning epochs: no"
fi
echo "Experiment folder: $OUTPUT_DIR"
echo "Output root: $OUTPUT_ROOT"
echo "Teacher output root: $TEACHER_OUTPUT_ROOT"
echo "Loss ablation: kd=$USE_KD_OUTPUT sparsity=$USE_SPARSITY feat=$USE_FEATURE_DISTILL aux=$USE_AUX_LOSS"
echo "Loss weights: kd=$LAMBDA_DISTILL sparsity=$LAMBDA_SPARSITY feat=$LAMBDA_FEAT aux=$LAMBDA_AUX"

python run_pgd.py \
  --dataset "$DATASET_KEY" \
  --root_path "$ROOT_PATH" \
  --teacher_model "$TEACHER_MODEL" \
  --exp "$EXP_NAME" \
  --max_epochs_teacher "$MAX_EPOCHS_TEACHER" \
  --max_epochs_student "$MAX_EPOCHS_STUDENT" \
  --output_root "$OUTPUT_ROOT" \
  --teacher_output_root "$TEACHER_OUTPUT_ROOT" \
  "${PRUNE_ARGS[@]}" \
  "${STEP3_ARGS[@]}" \
  --use_kd_output "$USE_KD_OUTPUT" \
  --use_sparsity "$USE_SPARSITY" \
  --use_feature_distill "$USE_FEATURE_DISTILL" \
  --use_aux_loss "$USE_AUX_LOSS" \
  --lambda_distill "$LAMBDA_DISTILL" \
  --lambda_sparsity "$LAMBDA_SPARSITY" \
  --lambda_feat "$LAMBDA_FEAT" \
  --lambda_aux "$LAMBDA_AUX" \
  --feature_layers $FEATURE_LAYERS \
  --batch_size "$BATCH_SIZE" \
  --patch_size "$PATCH_SIZE_H" "$PATCH_SIZE_W"

echo "=============================="
echo "End       : $(date)"
echo "=============================="

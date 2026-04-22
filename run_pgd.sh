#!/bin/bash
#SBATCH --job-name=pgd_cvc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

mkdir -p logs

echo "=============================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "Start     : $(date)"
echo "=============================="

# Load CUDA (quan trọng cho GPU)
module load cuda-11.8.0-gcc-11.4.0-cuusula

# Load conda env
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pgdunet

# Move to project
cd $HOME/PGD-UNet

# Check GPU
nvidia-smi

PRUNE_STRATEGY="${PRUNE_STRATEGY:-S1}"
PRUNE_STRATEGY="$(echo "$PRUNE_STRATEGY" | tr '[:lower:]' '[:upper:]')"
PRUNE_ARGS=()
STEP3_ARGS=()

STEP3_PRUNING="${STEP3_PRUNING:-1}"
STEP3_PRUNING="$(echo "$STEP3_PRUNING" | tr '[:upper:]' '[:lower:]')"
STEP3_PRUNING_EPOCHS="${STEP3_PRUNING_EPOCHS:-4}"
TEACHER_OUTPUT_ROOT="${TEACHER_OUTPUT_ROOT:-outputs}"

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
  *)
    echo "Unsupported PRUNE_STRATEGY=$PRUNE_STRATEGY. Use S1, S2, S3, S4, or S5."
    exit 1
    ;;
esac

OUTPUT_DIR="output_${PRUNE_METHOD}_${RATE_TAG}_${STEP3_TAG}"

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
echo "Teacher output root: $TEACHER_OUTPUT_ROOT"

# Run training
python train_pgd.py \
  --dataset cvc \
  --root_path data/CVC-ClinicDB \
  --teacher_model unet_resnet152 \
  --exp pgd_cvc \
  --max_epochs_teacher 50 \
  --max_epochs_student 50 \
  --output_root "$TEACHER_OUTPUT_ROOT" \
  --teacher_output_root "$TEACHER_OUTPUT_ROOT" \
  "${PRUNE_ARGS[@]}" \
  "${STEP3_ARGS[@]}" \
  --lambda_distill 0.3 \
  --lambda_sparsity 0.3 \
  --batch_size 8 \
  --patch_size 256 256

echo "=============================="
echo "End       : $(date)"
echo "=============================="

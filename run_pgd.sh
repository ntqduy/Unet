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

case "$PRUNE_STRATEGY" in
  S1)
    PRUNE_METHOD="static"
    PRUNE_RATE="${PRUNE_RATE:-0.5}"
    if ! PRUNE_RATE_TAG=$(python -c 'import sys; v = float(sys.argv[1]); sys.exit("PRUNE_RATE must be in [0, 1).") if not (0.0 <= v < 1.0) else None; print(f"{v:.12g}")' "$PRUNE_RATE"); then
      echo "Invalid PRUNE_RATE=$PRUNE_RATE for S1 static pruning"
      exit 1
    fi
    OUTPUT_DIR="output_static_${PRUNE_RATE_TAG}"
    PRUNE_ARGS=(--prune_strategy "$PRUNE_STRATEGY" --prune_method "$PRUNE_METHOD" --static_prune_ratio "$PRUNE_RATE" --prune_ratio "$PRUNE_RATE")
    ;;
  S2)
    PRUNE_METHOD="kneedle"
    OUTPUT_DIR="output_kneedle"
    PRUNE_ARGS=(--prune_strategy "$PRUNE_STRATEGY" --prune_method "$PRUNE_METHOD")
    ;;
  S3)
    PRUNE_METHOD="otsu"
    OUTPUT_DIR="output_otsu"
    PRUNE_ARGS=(--prune_strategy "$PRUNE_STRATEGY" --prune_method "$PRUNE_METHOD")
    ;;
  S4)
    PRUNE_METHOD="gmm"
    OUTPUT_DIR="output_gmm"
    PRUNE_ARGS=(--prune_strategy "$PRUNE_STRATEGY" --prune_method "$PRUNE_METHOD")
    ;;
  *)
    echo "Unsupported PRUNE_STRATEGY=$PRUNE_STRATEGY. Use S1, S2, S3, or S4."
    exit 1
    ;;
esac

echo "Pruning strategy: $PRUNE_METHOD"
if [ "$PRUNE_METHOD" = "static" ]; then
  echo "Static prune ratio: $PRUNE_RATE_TAG"
else
  echo "Static prune ratio: not used"
fi
echo "Output dir: $OUTPUT_DIR"

# Run training
python train_pgd.py \
  --dataset cvc \
  --root_path data/CVC-ClinicDB \
  --teacher_model unet_resnet152 \
  --exp pgd_cvc \
  --max_epochs_teacher 50 \
  --max_epochs_student 50 \
  --output_root "$OUTPUT_DIR" \
  "${PRUNE_ARGS[@]}" \
  --lambda_distill 0.3 \
  --lambda_sparsity 0.3 \
  --batch_size 8 \
  --patch_size 256 256

echo "=============================="
echo "End       : $(date)"
echo "=============================="

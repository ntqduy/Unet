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

# Run training
python train_pgd.py \
  --dataset cvc \
  --root_path data/CVC-ClinicDB \
  --teacher_model unet_resnet152 \
  --exp pgd_cvc \
  --max_epochs_teacher 50 \
  --max_epochs_student 50 \
  --prune_ratio 0.5 \
  --lambda_distill 0.3 \
  --lambda_sparsity 0.3 \
  --batch_size 8 \
  --patch_size 256 256

echo "=============================="
echo "End       : $(date)"
echo "=============================="
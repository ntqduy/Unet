#!/bin/bash
#SBATCH --job-name=pgd_kvasir_seg
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

DATASET_KEY="kvasir_seg"
ROOT_PATH="data/Kvasir-SEG"
EXP_NAME="pgd_kvasir_seg"

source "$(dirname "$0")/run_pgd_common.sh"

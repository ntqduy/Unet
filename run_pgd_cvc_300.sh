#!/bin/bash
#SBATCH --job-name=pgd_cvc_300
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

DATASET_KEY="cvc_300"
ROOT_PATH="data/CVC-300"
EXP_NAME="pgd_cvc_300"

source "$(dirname "$0")/run_pgd_common.sh"

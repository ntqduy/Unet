#!/bin/bash
#SBATCH --job-name=pgd_cvc_colondb
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

DATASET_KEY="cvc_colondb"
ROOT_PATH="data/CVC-ColonDB"
EXP_NAME="pgd_cvc_colondb"

source "$(dirname "$0")/run_pgd_common.sh"

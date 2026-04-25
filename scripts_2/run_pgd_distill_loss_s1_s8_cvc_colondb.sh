#!/bin/bash

DATASET_KEY="cvc_colondb"
ROOT_PATH="data/CVC-ColonDB"
EXP_NAME="pgd_cvc_colondb"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pgd_distill_loss_s1_s8_runner.inc"

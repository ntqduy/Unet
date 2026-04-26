#!/bin/bash

set -euo pipefail

OUTPUTS_ROOT="outputs"
SAVE_ROOT="statistics/outputs"
DATASET_MAIN="cvc_300"
CONDA_ENV="${CONDA_ENV:-pgdunet}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outputs-root)
      OUTPUTS_ROOT="$2"
      shift 2
      ;;
    --save-root)
      SAVE_ROOT="$2"
      shift 2
      ;;
    --dataset-main)
      DATASET_MAIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

echo "[RUN] Step 1/3: generate tables -> $SAVE_ROOT"
python statistics/src/generate_tables.py \
  --outputs-root "$OUTPUTS_ROOT" \
  --save-root "$SAVE_ROOT"

echo "[RUN] Step 2/3: generate figures -> $SAVE_ROOT"
python statistics/src/generate_figures.py \
  --outputs-root "$OUTPUTS_ROOT" \
  --save-root "$SAVE_ROOT"

echo "[RUN] Step 3/3: collect paper-ready artifacts -> statistics/paper_ready"
python statistics/src/collect_paper_artifacts.py \
  --outputs-root "$OUTPUTS_ROOT" \
  --statistics-root "$SAVE_ROOT" \
  --save-root "statistics/paper_ready" \
  --dataset-main "$DATASET_MAIN"

echo "[RUN] Done."

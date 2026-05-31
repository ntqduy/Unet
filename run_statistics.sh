#!/bin/bash

set -euo pipefail

OUTPUTS_ROOT="outputs"
SAVE_ROOT="statistics/outputs"
DATASET_MAIN="cvc_300"
PGD_TEACHER_DIR=""

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
    --pgd-teacher-dir)
      PGD_TEACHER_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

teacher_exists() {
  local teacher_dir="$1"
  compgen -G "$OUTPUTS_ROOT/pgd_unet/*/$teacher_dir" >/dev/null
}

run_statistics_for_teacher() {
  local teacher_dir="$1"
  local target_save_root="$2"
  local target_paper_root="$3"

  echo "[RUN] Teacher: $teacher_dir"
  echo "[RUN] Step 1/3: generate tables -> $target_save_root"
  python statistics/src/generate_tables.py \
    --outputs-root "$OUTPUTS_ROOT" \
    --save-root "$target_save_root" \
    --pgd-teacher-dir "$teacher_dir"

  echo "[RUN] Step 2/3: generate figures -> $target_save_root"
  python statistics/src/generate_figures.py \
    --outputs-root "$OUTPUTS_ROOT" \
    --save-root "$target_save_root" \
    --pgd-teacher-dir "$teacher_dir"

  echo "[RUN] Step 3/3: collect paper-ready artifacts -> $target_paper_root"
  python statistics/src/collect_paper_artifacts.py \
    --outputs-root "$OUTPUTS_ROOT" \
    --statistics-root "$target_save_root" \
    --save-root "$target_paper_root" \
    --dataset-main "$DATASET_MAIN"
}

if [ -n "$PGD_TEACHER_DIR" ]; then
  run_statistics_for_teacher "$PGD_TEACHER_DIR" "$SAVE_ROOT" "statistics/paper_ready"
  echo "[RUN] Done."
  exit 0
fi

UNET_TEACHER_DIR=""
if teacher_exists "unet_resnet152_teacher"; then
  UNET_TEACHER_DIR="unet_resnet152_teacher"
elif teacher_exists "unet_teacher"; then
  UNET_TEACHER_DIR="unet_teacher"
fi

UNET_PLUS_PLUS_EXISTS=0
if teacher_exists "unet_plus_plus_teacher"; then
  UNET_PLUS_PLUS_EXISTS=1
fi

if [ "$UNET_PLUS_PLUS_EXISTS" = "1" ] && [ -n "$UNET_TEACHER_DIR" ]; then
  run_statistics_for_teacher "unet_plus_plus_teacher" "$SAVE_ROOT/Unet_plus_plus" "statistics/paper_ready/Unet_plus_plus"
  run_statistics_for_teacher "$UNET_TEACHER_DIR" "$SAVE_ROOT/Unet" "statistics/paper_ready/Unet"
elif [ "$UNET_PLUS_PLUS_EXISTS" = "1" ]; then
  run_statistics_for_teacher "unet_plus_plus_teacher" "$SAVE_ROOT" "statistics/paper_ready"
elif [ -n "$UNET_TEACHER_DIR" ]; then
  run_statistics_for_teacher "$UNET_TEACHER_DIR" "$SAVE_ROOT" "statistics/paper_ready"
else
  run_statistics_for_teacher "unet_resnet152_teacher" "$SAVE_ROOT" "statistics/paper_ready"
fi

echo "[RUN] Done."

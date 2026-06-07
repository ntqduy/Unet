#!/bin/bash

set -euo pipefail

OUTPUTS_ROOT="outputs"
SAVE_ROOT="statistics/outputs"
PAPER_ROOT="statistics/paper_ready"
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
  local root
  root="$(resolve_teacher_outputs_root "$teacher_dir")"
  compgen -G "$root/pgd_unet/*/$teacher_dir" >/dev/null
}

teacher_save_tag() {
  case "$(printf "%s" "$1" | tr '[:upper:]' '[:lower:]')" in
    unet_resnet152_teacher|unet_resnet152)
      printf "Unet_resnet152"
      ;;
    unet_plus_plus_teacher|unet_plus_plus)
      printf "Unet_plus_plus"
      ;;
    unet_teacher|unet)
      printf "Unet"
      ;;
    *)
      printf "%s" "$1" | sed -E 's/_teacher$//; s/[^A-Za-z0-9._-]+/_/g; s/^[._-]+//; s/[._-]+$//'
      ;;
  esac
}

append_teacher_tag() {
  local root="$1"
  local tag="$2"
  if [ "$(basename "$root")" = "$tag" ]; then
    printf "%s" "$root"
  else
    printf "%s/%s" "$root" "$tag"
  fi
}

resolve_teacher_outputs_root() {
  local teacher_dir="$1"
  local tag
  tag="$(teacher_save_tag "$teacher_dir")"
  if compgen -G "$OUTPUTS_ROOT/$tag/pgd_unet/*/$teacher_dir" >/dev/null; then
    printf "%s/%s" "$OUTPUTS_ROOT" "$tag"
  else
    printf "%s" "$OUTPUTS_ROOT"
  fi
}

run_statistics_for_teacher() {
  local teacher_dir="$1"
  local target_save_root="$2"
  local target_paper_root="$3"
  local teacher_outputs_root
  teacher_outputs_root="$(resolve_teacher_outputs_root "$teacher_dir")"

  echo "[RUN] Teacher: $teacher_dir"
  echo "[RUN] Teacher outputs root: $teacher_outputs_root"
  echo "[RUN] Step 1/3: generate tables -> $target_save_root"
  python statistics/src/generate_tables.py \
    --outputs-root "$teacher_outputs_root" \
    --save-root "$target_save_root" \
    --pgd-teacher-dir "$teacher_dir"

  echo "[RUN] Step 2/3: generate figures -> $target_save_root"
  python statistics/src/generate_figures.py \
    --outputs-root "$teacher_outputs_root" \
    --save-root "$target_save_root" \
    --pgd-teacher-dir "$teacher_dir"

  echo "[RUN] Step 3/3: collect paper-ready artifacts -> $target_paper_root"
  python statistics/src/collect_paper_artifacts.py \
    --outputs-root "$teacher_outputs_root" \
    --statistics-root "$target_save_root" \
    --save-root "$target_paper_root" \
    --dataset-main "$DATASET_MAIN"
}

if [ -n "$PGD_TEACHER_DIR" ]; then
  TEACHER_TAG="$(teacher_save_tag "$PGD_TEACHER_DIR")"
  run_statistics_for_teacher \
    "$PGD_TEACHER_DIR" \
    "$(append_teacher_tag "$SAVE_ROOT" "$TEACHER_TAG")" \
    "$(append_teacher_tag "$PAPER_ROOT" "$TEACHER_TAG")"
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
  run_statistics_for_teacher \
    "unet_plus_plus_teacher" \
    "$(append_teacher_tag "$SAVE_ROOT" "$(teacher_save_tag "unet_plus_plus_teacher")")" \
    "$(append_teacher_tag "$PAPER_ROOT" "$(teacher_save_tag "unet_plus_plus_teacher")")"
  run_statistics_for_teacher \
    "$UNET_TEACHER_DIR" \
    "$(append_teacher_tag "$SAVE_ROOT" "$(teacher_save_tag "$UNET_TEACHER_DIR")")" \
    "$(append_teacher_tag "$PAPER_ROOT" "$(teacher_save_tag "$UNET_TEACHER_DIR")")"
elif [ "$UNET_PLUS_PLUS_EXISTS" = "1" ]; then
  run_statistics_for_teacher \
    "unet_plus_plus_teacher" \
    "$(append_teacher_tag "$SAVE_ROOT" "$(teacher_save_tag "unet_plus_plus_teacher")")" \
    "$(append_teacher_tag "$PAPER_ROOT" "$(teacher_save_tag "unet_plus_plus_teacher")")"
elif [ -n "$UNET_TEACHER_DIR" ]; then
  run_statistics_for_teacher \
    "$UNET_TEACHER_DIR" \
    "$(append_teacher_tag "$SAVE_ROOT" "$(teacher_save_tag "$UNET_TEACHER_DIR")")" \
    "$(append_teacher_tag "$PAPER_ROOT" "$(teacher_save_tag "$UNET_TEACHER_DIR")")"
else
  run_statistics_for_teacher \
    "unet_resnet152_teacher" \
    "$(append_teacher_tag "$SAVE_ROOT" "$(teacher_save_tag "unet_resnet152_teacher")")" \
    "$(append_teacher_tag "$PAPER_ROOT" "$(teacher_save_tag "unet_resnet152_teacher")")"
fi

echo "[RUN] Done."

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def _build_help_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wrapper for train_pgd.py")
    parser.add_argument("--root_path", type=str, default="data/Kvasir-SEG")
    parser.add_argument("--dataset", type=str, default="kvasir_seg")
    parser.add_argument("--exp", type=str, default="pdg_pipeline")
    parser.add_argument("--teacher_model", type=str, default="unet_resnet152")
    parser.add_argument("--teacher_checkpoint", type=str, default="")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--max_epochs_teacher", type=int, default=20)
    parser.add_argument("--max_epochs_student", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--teacher_lr", type=float, default=0.01)
    parser.add_argument("--student_lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--encoder_pretrained", type=int, default=1)
    parser.add_argument("--prune_ratio", type=float, default=0.5)
    parser.add_argument("--prune_strategy", type=str, default="")
    parser.add_argument("--prune_method", type=str, default="")
    parser.add_argument("--static_prune_ratio", type=float, default=None)
    parser.add_argument("--lambda_distill", type=float, default=0.3)
    parser.add_argument("--lambda_sparsity", type=float, default=0.3)
    parser.add_argument("--use_kd_output", type=int, default=1)
    parser.add_argument("--use_sparsity", type=int, default=1)
    parser.add_argument("--use_feature_distill", type=int, default=0)
    parser.add_argument("--use_aux_loss", type=int, default=0)
    parser.add_argument("--lambda_feat", type=float, default=0.1)
    parser.add_argument("--lambda_aux", type=float, default=0.2)
    parser.add_argument("--seg_loss_method", type=str, default="hybrid")
    parser.add_argument("--distill_loss_method", type=str, default="mse")
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--loss_method", type=str, default="")
    parser.add_argument("--feature_layers", nargs="*", default=["bottleneck"])
    parser.add_argument("--student_variant", type=str, default="full")
    parser.add_argument("--warmup_pruning_epochs", type=int, default=4)
    parser.add_argument("--enable_step3_pruning", type=int, default=1)
    parser.add_argument("--step3_pruning_epochs", type=int, default=None)
    parser.add_argument("--student_gate_near_off_threshold", type=float, default=0.10)
    parser.add_argument("--student_gate_open_value", type=float, default=0.999)
    parser.add_argument("--student_hard_gate_threshold", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_visualizations", type=int, default=1)
    parser.add_argument("--vis_num_samples", type=int, default=8)
    parser.add_argument("--final_eval_splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--force_retrain_teacher", type=int, default=0)
    parser.add_argument("--force_reprune", type=int, default=0)
    parser.add_argument("--force_retrain_student", type=int, default=0)
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--teacher_output_root", type=str, default="")
    parser.add_argument("--save_history_checkpoints", type=int, default=0)
    parser.add_argument("--save_last_checkpoint", type=int, default=1)
    parser.add_argument("--save_optimizer_state", type=int, default=0)
    return parser


if __name__ == "__main__":
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        _build_help_parser().parse_args()
        raise SystemExit(0)
    runpy.run_path(str(Path(__file__).with_name("train_pgd.py")), run_name="__main__")

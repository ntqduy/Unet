from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import Normalize, RandomGenerator, ToTensor, build_dataset, list_available_datasets
from networks.PGD_Unet.gated_unet import PDGUNet
from networks.PGD_Unet.middle_pruned_resnet_unet import (
    build_middle_pruned_resnet_unet,
    build_middle_pruned_resnet_unet_from_teacher,
)
from networks.PGD_Unet.pruning import PRUNE_METHODS, extract_pruned_blueprint, load_blueprint_artifact, save_blueprint_artifact
from networks.PGD_Unet.pruning_algorithms.pruning_smart import uses_static_prune_ratio
from networks.net_factory import get_model_metadata, list_models, net_factory
from utils.channel_analysis import (
    build_analysis_comparison,
    extract_channel_analysis,
    save_channel_analysis_artifacts,
    save_comparison_artifacts,
    save_gating_analysis_artifacts,
    save_pruning_analysis_artifacts,
)
from utils.checkpoints import (
    build_checkpoint_metadata,
    load_checkpoint,
    load_checkpoint_into_model,
    resolve_phase_checkpoint,
    save_checkpoint,
    save_checkpoint_payload_atomic,
)
from utils.checkpoint_resolver import (
    build_expected_signature,
    find_compatible_checkpoint,
    register_reused_checkpoint,
    resolve_basic_checkpoint,
    resolve_run_checkpoint,
)
from utils.compression_loss import CompressionLoss
from utils.evaluation import build_evaluation_output_dir, evaluate_segmentation_dataset, save_evaluation_artifacts
from utils.experiment import (
    PDG_PHASE_DIRS,
    build_pdg_phase_dir,
    build_pdg_root_dir,
    ensure_run_layout,
    normalize_path_string,
    project_relative_path,
    sanitize_tag,
    write_model_config,
    write_run_config,
)
from utils.losses import DiceLoss
from utils.model_output import extract_logits, extract_model_info
from utils.profiling import benchmark_inference, count_parameters, maybe_compute_flops
from utils.reporting import (
    save_channel_analysis_pdf,
    save_loss_pdf,
    save_performance_pdf,
    save_visualization_overview_image,
    save_visualization_pdf,
    write_metrics_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "Kvasir-SEG"
PRUNE_STRATEGY_TO_METHOD = {
    "S1": "static",
    "S2": "kneedle",
    "S3": "otsu",
    "S4": "gmm",
    "S5": "middle_static",
    "S6": "middle_kneedle",
    "S7": "middle_otsu",
    "S8": "middle_gmm",
}
PRUNE_METHOD_TO_STRATEGY = {method: strategy for strategy, method in PRUNE_STRATEGY_TO_METHOD.items()}
MIDDLE_PRUNED_RESNET_METHODS = {"middle_static", "middle_kneedle", "middle_otsu", "middle_gmm"}


def _format_float_for_path(value: float) -> str:
    text = f"{float(value):.12g}"
    return "0" if text == "-0" else text


def build_pruning_output_dir_name(prune_method: str, static_prune_ratio: float | None = None) -> str:
    prune_method = str(prune_method).lower()
    if uses_static_prune_ratio(prune_method):
        if static_prune_ratio is None:
            raise ValueError(f"static_prune_ratio is required for {prune_method} output directory naming.")
        return f"output_{prune_method}_{_format_float_for_path(static_prune_ratio)}"
    if prune_method in {"kneedle", "otsu", "gmm", "middle_kneedle", "middle_otsu", "middle_gmm"}:
        return f"output_{prune_method}"
    raise ValueError(f"Unsupported prune_method: {prune_method}")


def build_step3_output_dir_name(
    prune_method: str,
    static_prune_ratio: float | None,
    step3_pruning_enabled: bool,
    step3_pruning_epochs: int,
) -> str:
    prune_method = str(prune_method).lower()
    rate_tag = _format_float_for_path(static_prune_ratio) if uses_static_prune_ratio(prune_method) else "auto"
    step3_tag = str(int(step3_pruning_epochs)) if step3_pruning_enabled else "no"
    return f"output_{prune_method}_{rate_tag}_{step3_tag}"


def _uses_middle_pruned_resnet_student() -> bool:
    return str(args.prune_method).lower() in MIDDLE_PRUNED_RESNET_METHODS


def _student_model_name() -> str:
    return "middle_pruned_resnet_unet" if _uses_middle_pruned_resnet_student() else "pdg_unet"


def _middle_student_name(prefix: str = "student") -> str:
    return f"{args.prune_method}_{prefix}"


def _strategy_label() -> str:
    return args.prune_strategy or PRUNE_METHOD_TO_STRATEGY.get(args.prune_method, args.prune_method)


def _normalize_pruning_args(parsed_args: argparse.Namespace, active_parser: argparse.ArgumentParser) -> argparse.Namespace:
    strategy = str(parsed_args.prune_strategy or "").strip().upper()
    method_arg = str(parsed_args.prune_method or "").strip().lower()

    if strategy:
        if strategy not in PRUNE_STRATEGY_TO_METHOD:
            active_parser.error(f"--prune_strategy must be one of {', '.join(PRUNE_STRATEGY_TO_METHOD)}")
        mapped_method = PRUNE_STRATEGY_TO_METHOD[strategy]
        if method_arg and method_arg != mapped_method:
            active_parser.error(f"--prune_strategy {strategy} maps to '{mapped_method}', but --prune_method was '{method_arg}'.")
        prune_method = mapped_method
    else:
        if method_arg.upper() in PRUNE_STRATEGY_TO_METHOD:
            strategy = method_arg.upper()
            prune_method = PRUNE_STRATEGY_TO_METHOD[strategy]
        else:
            prune_method = method_arg or "static"
            strategy = PRUNE_METHOD_TO_STRATEGY.get(prune_method, "")

    if prune_method not in PRUNE_METHODS:
        active_parser.error(f"--prune_method must be one of {', '.join(PRUNE_METHODS)}")

    parsed_args.prune_strategy = strategy or PRUNE_METHOD_TO_STRATEGY[prune_method]
    parsed_args.prune_method = prune_method

    if uses_static_prune_ratio(prune_method):
        ratio = parsed_args.static_prune_ratio
        if ratio is None:
            ratio = parsed_args.prune_ratio
        if ratio is None:
            active_parser.error("--static_prune_ratio or --prune_ratio is required for static pruning methods.")
        ratio = float(ratio)
        if not 0.0 <= ratio < 1.0:
            active_parser.error("--static_prune_ratio must be in [0, 1).")
        parsed_args.static_prune_ratio = ratio
        parsed_args.prune_ratio = ratio
    elif parsed_args.static_prune_ratio is not None:
        parsed_args.static_prune_ratio = float(parsed_args.static_prune_ratio)

    parsed_args.pruning_output_dir_name = build_pruning_output_dir_name(
        parsed_args.prune_method,
        parsed_args.static_prune_ratio,
    )
    return parsed_args


def _normalize_step3_pruning_args(parsed_args: argparse.Namespace, active_parser: argparse.ArgumentParser) -> argparse.Namespace:
    if int(parsed_args.enable_step3_pruning) not in (0, 1):
        active_parser.error("--enable_step3_pruning must be 0 or 1.")

    enabled = bool(int(parsed_args.enable_step3_pruning))
    requested_epochs = parsed_args.step3_pruning_epochs
    if requested_epochs is None:
        requested_epochs = parsed_args.warmup_pruning_epochs
    requested_epochs = int(requested_epochs)

    if requested_epochs < 0:
        active_parser.error("--step3_pruning_epochs must be >= 0.")
    if enabled and requested_epochs <= 0:
        active_parser.error("--step3_pruning_epochs must be > 0 when --enable_step3_pruning 1.")

    parsed_args.enable_step3_pruning = int(enabled)
    parsed_args.step3_pruning_epochs = requested_epochs if enabled else 0
    parsed_args.warmup_pruning_epochs = requested_epochs if enabled else 0
    parsed_args.step3_output_dir_name = build_step3_output_dir_name(
        parsed_args.prune_method,
        parsed_args.static_prune_ratio,
        enabled,
        parsed_args.step3_pruning_epochs,
    )
    return parsed_args


parser = argparse.ArgumentParser(description="Teacher -> Pruning -> Student training for PDG-UNet")
parser.add_argument("--root_path", type=str, default=str(DEFAULT_DATA_ROOT))
parser.add_argument("--dataset", type=str, default="kvasir_seg", choices=list_available_datasets())
parser.add_argument("--exp", type=str, default="pdg_pipeline")
parser.add_argument("--teacher_model", type=str, default="unet_resnet152", choices=list_models())
parser.add_argument("--teacher_checkpoint", type=str, default="")
parser.add_argument("--train_split", type=str, default="train", choices=["train", "val", "test"])
parser.add_argument("--val_split", type=str, default="val", choices=["train", "val", "test"])
parser.add_argument("--max_epochs_teacher", type=int, default=20)
parser.add_argument("--max_epochs_student", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--teacher_lr", type=float, default=0.01)
parser.add_argument("--student_lr", type=float, default=1e-4)
parser.add_argument("--patch_size", nargs=2, type=int, default=[256, 256])
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--encoder_pretrained", type=int, default=1, help="defaults to 1 for unet_resnet152 teacher builds")
parser.add_argument("--prune_ratio", type=float, default=0.5, help="backward-compatible fixed ratio used by static pruning methods if --static_prune_ratio is omitted")
parser.add_argument("--prune_strategy", type=str, default="", help="external pruning strategy code: S1=static, S2=kneedle, S3=otsu, S4=gmm, S5=middle_static, S6=middle_kneedle, S7=middle_otsu, S8=middle_gmm")
parser.add_argument("--prune_method", type=str, default="", help="internal pruning method: static, kneedle, otsu, gmm, middle_static, middle_kneedle, middle_otsu, or middle_gmm")
parser.add_argument("--static_prune_ratio", type=float, default=None, help="fixed channel prune ratio for static and middle_static pruning; dynamic strategies ignore it")
parser.add_argument("--lambda_distill", type=float, default=0.3)
parser.add_argument("--lambda_sparsity", type=float, default=0.3)
parser.add_argument("--use_kd_output", type=int, default=1)
parser.add_argument("--use_sparsity", type=int, default=1)
parser.add_argument("--use_feature_distill", type=int, default=0)
parser.add_argument("--use_aux_loss", type=int, default=0)
parser.add_argument("--lambda_feat", type=float, default=0.1)
parser.add_argument("--lambda_aux", type=float, default=0.2)
parser.add_argument(
    "--feature_layers",
    nargs="*",
    default=["bottleneck"],
    help="feature distillation layers; default compares only the bottleneck tensor at the encoder/decoder interface",
)
parser.add_argument(
    "--student_variant",
    type=str,
    default="full",
    choices=["pruned_no_gate", "pruned_distill", "pruned_gate_sparsity", "full"],
    help="student compression variant for step 3",
)
parser.add_argument(
    "--warmup_pruning_epochs",
    type=int,
    default=4,
    help="number of final step-3 epochs reserved for late hard pruning and compact-student distillation",
)
parser.add_argument("--enable_step3_pruning", type=int, default=1, help="set to 1 to enable step-3 soft/late hard pruning, or 0 to train without step-3 pruning")
parser.add_argument("--step3_pruning_epochs", type=int, default=None, help="number of final student epochs reserved for step-3 late hard pruning; alias for --warmup_pruning_epochs")
parser.add_argument("--student_gate_near_off_threshold", type=float, default=0.10, help="gate value threshold used to flag channels as nearly switched off")
parser.add_argument("--student_gate_open_value", type=float, default=0.999, help="gate probability used when student_variant disables gating")
parser.add_argument(
    "--student_hard_gate_threshold",
    type=float,
    default=-1.0,
    help="optional threshold for late structural hard pruning; <=0 falls back to student_gate_near_off_threshold",
)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--save_visualizations", type=int, default=1)
parser.add_argument("--vis_num_samples", type=int, default=8)
parser.add_argument("--final_eval_splits", nargs="*", default=["train", "val", "test"])
parser.add_argument("--force_retrain_teacher", type=int, default=0)
parser.add_argument("--force_reprune", type=int, default=0)
parser.add_argument("--force_retrain_student", type=int, default=0)
parser.add_argument("--output_root", type=str, default="", help="root directory for all exported outputs; defaults to PROJECT_ROOT/outputs")
parser.add_argument("--teacher_output_root", type=str, default="", help="shared output root for the teacher phase; defaults to --output_root when omitted")
parser.add_argument("--save_history_checkpoints", type=int, default=0, help="set to 1 to keep per-epoch checkpoint history in addition to best.pth")
parser.add_argument("--save_last_checkpoint", type=int, default=1, help="set to 1 to keep an overwritten last.pth checkpoint in addition to best.pth")
parser.add_argument("--save_optimizer_state", type=int, default=0, help="set to 1 to include optimizer/scheduler/scaler states in saved checkpoints")
args = parser.parse_args()
args = _normalize_pruning_args(args, parser)
args = _normalize_step3_pruning_args(args, parser)
for _flag_name in ("use_kd_output", "use_sparsity", "use_feature_distill", "use_aux_loss", "save_history_checkpoints", "save_last_checkpoint", "save_optimizer_state"):
    if int(getattr(args, _flag_name)) not in (0, 1):
        parser.error(f"--{_flag_name} must be 0 or 1.")
    setattr(args, _flag_name, int(getattr(args, _flag_name)))
args.lambda_sparsity = float(args.lambda_sparsity) if bool(args.use_sparsity) else 0.0
args.loss_tag = None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _loss_tag() -> str:
    parts = ["loss", "seg"]
    if args.use_kd_output:
        parts.append("kd")
    if args.use_feature_distill:
        parts.append("feat")
    if args.use_aux_loss:
        parts.append("aux")
    if args.use_sparsity:
        parts.append("sparsity")
    if len(parts) == 2:
        parts.append("only")
    return "_".join(parts)


args.loss_tag = _loss_tag()


def _proposal_root_dir() -> Path:
    return _teacher_root_dir() / sanitize_tag(_loss_tag()) / sanitize_tag(args.step3_output_dir_name)


def _teacher_output_root() -> str | None:
    return args.teacher_output_root or args.output_root or None


def _teacher_root_dir() -> Path:
    return build_pdg_root_dir(
        project_root=PROJECT_ROOT,
        dataset=args.dataset,
        teacher_name=args.teacher_model,
        output_root=_teacher_output_root(),
    )


def _phase_dir(phase: str) -> Path:
    phase_key = sanitize_tag(phase)
    if phase_key == "teacher":
        return build_pdg_phase_dir(
            project_root=PROJECT_ROOT,
            dataset=args.dataset,
            teacher_name=args.teacher_model,
            phase=phase,
            output_root=_teacher_output_root(),
        )

    phase_dir_name = PDG_PHASE_DIRS.get(phase_key)
    if phase_dir_name is None:
        available = ", ".join(sorted(PDG_PHASE_DIRS))
        raise KeyError(f"Unknown PDG phase '{phase}'. Available phases: {available}.")
    return _proposal_root_dir() / phase_dir_name


def _pruning_metadata() -> dict:
    static_ratio = args.static_prune_ratio if uses_static_prune_ratio(args.prune_method) else None
    return {
        "prune_strategy": args.prune_strategy,
        "prune_method": args.prune_method,
        "static_prune_ratio": static_ratio,
        "prune_ratio": static_ratio,
        "pruning_output_dir_name": args.pruning_output_dir_name,
        "step3_output_dir_name": args.step3_output_dir_name,
        "loss_tag": args.loss_tag,
        "use_kd_output": int(args.use_kd_output),
        "use_sparsity": int(args.use_sparsity),
        "use_feature_distill": int(args.use_feature_distill),
        "use_aux_loss": int(args.use_aux_loss),
        "lambda_feat": float(args.lambda_feat),
        "lambda_aux": float(args.lambda_aux),
        "feature_layers": list(args.feature_layers),
        "step3_pruning_enabled": bool(args.enable_step3_pruning),
        "step3_pruning_epochs": int(args.step3_pruning_epochs),
        "step3_pruning_tag": str(int(args.step3_pruning_epochs)) if bool(args.enable_step3_pruning) else "no",
    }


def _blueprint_matches_current_pruning_config(candidate_blueprint: dict) -> bool:
    candidate_method = str(candidate_blueprint.get("prune_method", "static")).lower()
    if candidate_method != args.prune_method:
        return False
    if args.prune_method in MIDDLE_PRUNED_RESNET_METHODS and str(candidate_blueprint.get("student_architecture", "")).lower() != "middle_pruned_resnet_unet":
        return False
    if uses_static_prune_ratio(args.prune_method):
        candidate_ratio = candidate_blueprint.get("static_prune_ratio", candidate_blueprint.get("prune_ratio"))
        try:
            return float(candidate_ratio) == float(args.static_prune_ratio)
        except (TypeError, ValueError):
            return False
    return True


def _configure_logging(log_path: Path) -> None:
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def _teacher_signature(in_channels: int) -> dict:
    return build_expected_signature(
        dataset=args.dataset,
        model_name=args.teacher_model,
        num_classes=args.num_classes,
        in_channels=in_channels,
        patch_size=args.patch_size,
        encoder_pretrained=bool(args.encoder_pretrained) if args.teacher_model == "unet_resnet152" else None,
    )


def _student_signature(channel_config, in_channels: int) -> dict:
    return build_expected_signature(
        dataset=args.dataset,
        model_name=_student_model_name(),
        num_classes=args.num_classes,
        in_channels=in_channels,
        patch_size=args.patch_size,
        teacher_model=args.teacher_model,
        channel_config=channel_config,
        student_variant=args.student_variant,
        step3_pruning_enabled=bool(args.enable_step3_pruning),
        step3_pruning_epochs=int(args.step3_pruning_epochs),
    )


def _restore_loss_pdf_if_possible(history: dict | None, pdf_path: Path, title: str) -> None:
    if not history:
        return
    if any(values for values in history.values() if isinstance(values, list)):
        save_loss_pdf(history, pdf_path, title=title)


def _student_variant_policy() -> dict:
    policies = {
        "pruned_no_gate": {
            "variant_name": "pruned_no_gate",
            "use_distill": False,
            "use_gating": False,
            "use_sparsity": False,
            "description": "Pruned student baseline without distillation or learnable gating.",
        },
        "pruned_distill": {
            "variant_name": "pruned_distill",
            "use_distill": True,
            "use_gating": False,
            "use_sparsity": False,
            "description": "Pruned student with teacher distillation but fixed-open gates.",
        },
        "pruned_gate_sparsity": {
            "variant_name": "pruned_gate_sparsity",
            "use_distill": False,
            "use_gating": True,
            "use_sparsity": True,
            "description": "Pruned student with learnable gating and sparsity, without teacher distillation.",
        },
        "full": {
            "variant_name": "full",
            "use_distill": True,
            "use_gating": True,
            "use_sparsity": True,
            "description": "Full proposal step 3: blueprint initialization + teacher distillation + gating + sparsity.",
        },
    }
    policy = dict(policies[args.student_variant])
    policy["use_distill"] = bool(policy["use_distill"] and args.use_kd_output)
    policy["use_sparsity"] = bool(policy["use_sparsity"] and args.use_sparsity)
    if _uses_middle_pruned_resnet_student():
        policy["use_gating"] = False
        policy["use_sparsity"] = False
        policy["description"] = (
            f"{_strategy_label()} uses structural middle-channel pruning inside ResNet bottlenecks. "
            "The ResNet boundary layers stay full, so PDG gate-based late pruning is disabled for this student."
        )
    return policy


def _build_pruning_warmup_schedule(total_epochs: int, requested_warmup_epochs: int, variant_policy: dict) -> dict:
    requested = max(0, int(requested_warmup_epochs))
    total_epochs = max(1, int(total_epochs))
    hard_pruning_threshold = (
        float(args.student_hard_gate_threshold)
        if float(args.student_hard_gate_threshold) > 0
        else float(args.student_gate_near_off_threshold)
    )

    if not bool(args.enable_step3_pruning):
        return {
            "step3_pruning_enabled": False,
            "requested_warmup_pruning_epochs": 0,
            "effective_warmup_pruning_epochs": 0,
            "active_soft_pruning_epochs": 0,
            "late_hard_pruning_epochs": 0,
            "hard_pruning_start_epoch": None,
            "hard_pruning_threshold": None,
            "hard_pruning_enabled": False,
            "policy_note": (
                "Step-3 pruning is disabled for this run. The student is trained without gate sparsity pressure "
                "and without late structural hard pruning; distillation stays active whenever the selected variant enables it."
            ),
        }

    if not variant_policy["use_gating"] or not variant_policy["use_sparsity"]:
        return {
            "step3_pruning_enabled": bool(args.enable_step3_pruning),
            "requested_warmup_pruning_epochs": requested,
            "effective_warmup_pruning_epochs": 0,
            "active_soft_pruning_epochs": 0,
            "late_hard_pruning_epochs": 0,
            "hard_pruning_start_epoch": None,
            "hard_pruning_threshold": None,
            "hard_pruning_enabled": False,
            "policy_note": (
                "Soft pruning is disabled because gating or sparsity is disabled for the selected student variant. "
                "If the variant enables distillation, teacher distillation still stays active for the whole step-3 run, "
                "but no late structural hard pruning is applied."
            ),
        }

    effective_late_pruning_epochs = min(requested, total_epochs)
    if effective_late_pruning_epochs <= 0:
        return {
            "step3_pruning_enabled": bool(args.enable_step3_pruning),
            "requested_warmup_pruning_epochs": requested,
            "effective_warmup_pruning_epochs": 0,
            "active_soft_pruning_epochs": total_epochs,
            "late_hard_pruning_epochs": 0,
            "hard_pruning_start_epoch": None,
            "hard_pruning_threshold": None,
            "hard_pruning_enabled": False,
            "policy_note": (
                "Late hard pruning is disabled for this run. The student keeps learnable gating and sparsity pressure for the whole step-3 run, "
                "and distillation stays active whenever the selected variant enables it."
            ),
        }

    hard_pruning_start_epoch = max(1, total_epochs - effective_late_pruning_epochs + 1)
    active_soft_pruning_epochs = max(0, hard_pruning_start_epoch - 1)

    return {
        "step3_pruning_enabled": True,
        "requested_warmup_pruning_epochs": requested,
        "effective_warmup_pruning_epochs": effective_late_pruning_epochs,
        "active_soft_pruning_epochs": active_soft_pruning_epochs,
        "late_hard_pruning_epochs": effective_late_pruning_epochs,
        "hard_pruning_start_epoch": hard_pruning_start_epoch,
        "hard_pruning_threshold": hard_pruning_threshold,
        "hard_pruning_enabled": True,
        "policy_note": (
            "Learnable gating + sparsity stay active during the early search epochs. "
            "At the beginning of the final pruning window, the student is structurally hard-pruned using gate values, "
            "then the compact student continues training with frozen gates and ongoing distillation."
        ),
    }


def _student_epoch_policy(epoch: int, schedule: dict, variant_policy: dict) -> dict:
    if not variant_policy["use_gating"]:
        return {
            "phase_name": "distillation_only" if variant_policy["use_distill"] else "plain_student_training",
            "gate_trainable": False,
            "lambda_distill": args.lambda_distill if variant_policy["use_distill"] else 0.0,
            "lambda_sparsity": 0.0,
            "lambda_feat": args.lambda_feat if bool(args.use_feature_distill) else 0.0,
            "lambda_aux": args.lambda_aux if bool(args.use_aux_loss) else 0.0,
            "needs_teacher_output": bool((args.lambda_distill if variant_policy["use_distill"] else 0.0) > 0 or (bool(args.use_feature_distill) and args.lambda_feat > 0)),
            "soft_pruning_active": False,
            "hard_pruning_active": False,
        }

    active_soft_pruning_epochs = int(schedule.get("active_soft_pruning_epochs", 0))
    hard_pruning_start_epoch = schedule.get("hard_pruning_start_epoch")
    in_soft_pruning = variant_policy["use_sparsity"] and epoch <= active_soft_pruning_epochs
    hard_pruning_active = bool(schedule.get("hard_pruning_enabled")) and hard_pruning_start_epoch is not None and epoch == int(hard_pruning_start_epoch)

    if hard_pruning_active:
        phase_name = "late_hard_pruning_distillation" if variant_policy["use_distill"] else "late_hard_pruning"
    elif hard_pruning_start_epoch is not None and epoch > int(hard_pruning_start_epoch):
        phase_name = "post_hard_pruning_distillation" if variant_policy["use_distill"] else "post_hard_pruning_stabilization"
    elif in_soft_pruning:
        phase_name = "pre_hard_pruning_gate_search_distillation" if variant_policy["use_distill"] else "pre_hard_pruning_gate_search"
    elif variant_policy["use_distill"]:
        phase_name = "distillation_with_frozen_gates"
    else:
        phase_name = "gate_stabilization"

    lambda_distill = args.lambda_distill if variant_policy["use_distill"] else 0.0
    lambda_sparsity = args.lambda_sparsity if in_soft_pruning else 0.0
    lambda_feat = args.lambda_feat if bool(args.use_feature_distill) else 0.0
    lambda_aux = args.lambda_aux if bool(args.use_aux_loss) else 0.0
    return {
        "phase_name": phase_name,
        "gate_trainable": bool(in_soft_pruning),
        "lambda_distill": lambda_distill,
        "lambda_sparsity": lambda_sparsity,
        "lambda_feat": lambda_feat,
        "lambda_aux": lambda_aux,
        "needs_teacher_output": bool(lambda_distill > 0 or lambda_feat > 0),
        "soft_pruning_active": bool(in_soft_pruning),
        "hard_pruning_active": hard_pruning_active,
    }


def _build_student_pruning_epoch_config(total_epochs: int, pruning_schedule: dict) -> dict:
    total_epochs = max(1, int(total_epochs))
    hard_pruning_start_epoch = pruning_schedule.get("hard_pruning_start_epoch")
    if hard_pruning_start_epoch is not None:
        hard_pruning_start_epoch = int(hard_pruning_start_epoch)
    hard_pruning_apply_epoch_0based = None if hard_pruning_start_epoch is None else max(0, hard_pruning_start_epoch - 1)

    if hard_pruning_start_epoch is None:
        late_window_epochs: list[int] = []
        gate_search_epochs = list(range(1, total_epochs + 1))
        late_window_epochs_0based: list[int] = []
        gate_search_epochs_0based = list(range(0, total_epochs))
    else:
        late_window_epochs = list(range(hard_pruning_start_epoch, total_epochs + 1))
        gate_search_epochs = list(range(1, hard_pruning_start_epoch))
        late_window_epochs_0based = [max(0, epoch - 1) for epoch in late_window_epochs]
        gate_search_epochs_0based = [max(0, epoch - 1) for epoch in gate_search_epochs]

    return {
        "total_student_epochs": int(total_epochs),
        "total_student_epochs_0based_last_index": int(total_epochs - 1),
        "step3_pruning_enabled": bool(pruning_schedule.get("step3_pruning_enabled", False)),
        "step3_pruning_epochs": int(pruning_schedule.get("effective_warmup_pruning_epochs", 0) or 0),
        "requested_warmup_pruning_epochs": int(pruning_schedule.get("requested_warmup_pruning_epochs", 0) or 0),
        "warmup_pruning_epochs": int(pruning_schedule.get("effective_warmup_pruning_epochs", 0) or 0),
        "requested_late_pruning_epochs": int(pruning_schedule.get("requested_warmup_pruning_epochs", 0) or 0),
        "effective_late_pruning_epochs": int(pruning_schedule.get("effective_warmup_pruning_epochs", 0) or 0),
        "active_soft_pruning_epochs": int(pruning_schedule.get("active_soft_pruning_epochs", 0) or 0),
        "hard_pruning_enabled": bool(pruning_schedule.get("hard_pruning_enabled", False)),
        "hard_pruning_threshold": pruning_schedule.get("hard_pruning_threshold"),
        "hard_pruning_start_epoch": hard_pruning_start_epoch,
        "hard_pruning_apply_epoch": hard_pruning_apply_epoch_0based,
        "hard_pruning_apply_epoch_0based": hard_pruning_apply_epoch_0based,
        "hard_pruning_apply_epoch_1based": hard_pruning_start_epoch,
        "late_pruning_epoch_window": late_window_epochs,
        "late_pruning_epoch_window_0based": late_window_epochs_0based,
        "gate_search_epoch_window": gate_search_epochs,
        "gate_search_epoch_window_0based": gate_search_epochs_0based,
    }


def _freeze_teacher_model(teacher_model) -> None:
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad = False


def _build_student_from_blueprint(db_train, pruning_artifact: dict) -> nn.Module:
    blueprint = pruning_artifact["blueprint"]
    if str(blueprint.get("student_architecture", "")).lower() == "middle_pruned_resnet_unet":
        return build_middle_pruned_resnet_unet(
            in_channels=db_train.in_channels,
            num_classes=args.num_classes,
            blueprint=blueprint,
        )
    return PDGUNet(
        in_channels=db_train.in_channels,
        num_classes=args.num_classes,
        channel_config=tuple(blueprint["channel_config"]),
    )


def _configure_student_variant(student: PDGUNet, variant_policy: dict) -> None:
    if not variant_policy["use_gating"]:
        student.force_gates_open(args.student_gate_open_value)
        student.set_gate_trainable(False)


def _summarize_student_gates(student: PDGUNet, threshold: float) -> dict:
    layer_rows = []
    all_values = []
    for module_name, module in student.named_modules():
        if hasattr(module, "gate_values") and callable(module.gate_values):
            gate_tensor = module.gate_values().detach().cpu().float()
            values = gate_tensor.tolist()
            if not values:
                continue
            layer_rows.append(
                {
                    "layer_name": module_name,
                    "gate_mean": float(gate_tensor.mean().item()),
                    "gate_min": float(gate_tensor.min().item()),
                    "gate_max": float(gate_tensor.max().item()),
                    "gate_std": float(gate_tensor.std(unbiased=False).item()) if gate_tensor.numel() > 1 else 0.0,
                    "near_off_channels": int((gate_tensor < threshold).sum().item()),
                    "total_channels": int(gate_tensor.numel()),
                    "near_off_ratio": float((gate_tensor < threshold).float().mean().item()),
                }
            )
            all_values.extend(float(value) for value in values)

    if not all_values:
        return {
            "global": {
                "gate_mean": 0.0,
                "gate_min": 0.0,
                "gate_max": 0.0,
                "gate_std": 0.0,
                "near_off_channels": 0,
                "total_gate_channels": 0,
                "near_off_ratio": 0.0,
            },
            "layer_rows": [],
        }

    gate_tensor = torch.tensor(all_values, dtype=torch.float32)
    return {
        "global": {
            "gate_mean": float(gate_tensor.mean().item()),
            "gate_min": float(gate_tensor.min().item()),
            "gate_max": float(gate_tensor.max().item()),
            "gate_std": float(gate_tensor.std(unbiased=False).item()) if gate_tensor.numel() > 1 else 0.0,
            "near_off_channels": int((gate_tensor < threshold).sum().item()),
            "total_gate_channels": int(gate_tensor.numel()),
            "near_off_ratio": float((gate_tensor < threshold).float().mean().item()),
        },
        "layer_rows": layer_rows,
    }


def _resolve_student_from_pruning_checkpoint(student: PDGUNet, pruning_artifact: dict, device: torch.device) -> dict:
    pruning_checkpoint_path = Path(pruning_artifact["checkpoint_path"])
    payload = load_checkpoint_into_model(pruning_checkpoint_path, student, device=device)
    return {
        "checkpoint_path": pruning_checkpoint_path,
        "payload": payload,
    }


def _save_student_epoch_diagnostics(run_dir: Path, diagnostics_rows: list[dict]) -> None:
    if diagnostics_rows:
        write_metrics_rows(diagnostics_rows, run_dir / "metrics" / "student_epoch_diagnostics.csv")


def _copy_exact_matching_weights(source_model, target_model) -> dict:
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    copied_keys = []
    skipped_keys = []

    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None or source_tensor.shape != target_tensor.shape:
            skipped_keys.append(key)
            continue
        target_state[key] = source_tensor.detach().clone()
        copied_keys.append(key)

    target_model.load_state_dict(target_state, strict=False)
    total_target_tensors = len(target_state)
    return {
        "copied_tensor_keys": int(len(copied_keys)),
        "skipped_tensor_keys": int(len(skipped_keys)),
        "total_target_tensors": int(total_target_tensors),
        "copy_ratio": float(len(copied_keys) / max(1, total_target_tensors)),
        "copied_key_examples": copied_keys[:20],
        "skipped_key_examples": skipped_keys[:20],
    }


NORM_LAYER_TYPES = (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)


def _safe_copy_norm_state(source_norm, target_norm, out_indices: list[int]) -> bool:
    if source_norm is None or target_norm is None or not out_indices:
        return False
    if not hasattr(source_norm, "weight") or getattr(source_norm, "weight", None) is None:
        return False
    if not hasattr(target_norm, "weight") or getattr(target_norm, "weight", None) is None:
        return False
    if max(out_indices) >= int(source_norm.weight.shape[0]):
        return False
    if int(target_norm.weight.shape[0]) != len(out_indices):
        return False
    _copy_norm_state(source_norm, target_norm, out_indices)
    return True


def _safe_copy_conv2d_state(source_conv, target_conv, out_indices: list[int], in_indices: list[int]) -> dict:
    if source_conv is None or target_conv is None or not out_indices or not in_indices:
        return {"copied": False, "copy_mode": "unmapped", "reason": "missing_source_or_target"}
    if not isinstance(source_conv, nn.Conv2d) or not isinstance(target_conv, nn.Conv2d):
        return {"copied": False, "copy_mode": "unmapped", "reason": "unsupported_conv_type"}
    if max(out_indices) >= int(source_conv.weight.shape[0]):
        return {"copied": False, "copy_mode": "unmapped", "reason": "out_indices_exceed_source_channels"}
    if max(in_indices) >= int(source_conv.weight.shape[1]):
        return {"copied": False, "copy_mode": "unmapped", "reason": "in_indices_exceed_source_channels"}
    if int(target_conv.weight.shape[0]) != len(out_indices):
        return {"copied": False, "copy_mode": "unmapped", "reason": "target_out_channels_mismatch"}
    if int(target_conv.weight.shape[1]) != len(in_indices):
        return {"copied": False, "copy_mode": "unmapped", "reason": "target_in_channels_mismatch"}
    source_kernel = tuple(int(value) for value in source_conv.weight.shape[-2:])
    target_kernel = tuple(int(value) for value in target_conv.weight.shape[-2:])
    _copy_conv2d_state(source_conv, target_conv, out_indices, in_indices)
    return {
        "copied": True,
        "copy_mode": "resized_kernel" if source_kernel != target_kernel else "direct",
        "reason": "",
        "source_kernel_size": list(source_kernel),
        "target_kernel_size": list(target_kernel),
    }


def _collect_conv2d_layers(module: nn.Module) -> list[nn.Conv2d]:
    return [child for child in module.modules() if isinstance(child, nn.Conv2d)]


def _collect_norm_layers(module: nn.Module) -> list[nn.Module]:
    return [child for child in module.modules() if isinstance(child, NORM_LAYER_TYPES)]


def _copy_source_convnorm_to_target(component_prefix: str, source_conv, source_norm, target_convnormact, out_indices: list[int], in_indices: list[int]) -> dict:
    copied_components = []
    direct_components = []
    resized_components = []
    unmapped_components = []
    target_conv = target_convnormact.block[0]
    target_norm = target_convnormact.block[1] if len(target_convnormact.block) > 1 else None
    conv_result = _safe_copy_conv2d_state(source_conv, target_conv, out_indices, in_indices)
    if conv_result["copied"]:
        conv_label = f"{component_prefix}_conv"
        copied_components.append(conv_label)
        if conv_result["copy_mode"] == "resized_kernel":
            resized_components.append(conv_label)
        else:
            direct_components.append(conv_label)
    else:
        unmapped_components.append(f"{component_prefix}_conv")
    if _safe_copy_norm_state(source_norm, target_norm, out_indices):
        norm_label = f"{component_prefix}_norm"
        copied_components.append(norm_label)
        direct_components.append(norm_label)
    else:
        unmapped_components.append(f"{component_prefix}_norm")
    return {
        "copied_components": copied_components,
        "direct_components": direct_components,
        "resized_components": resized_components,
        "unmapped_components": unmapped_components,
        "conv_result": conv_result,
    }


def _transfer_teacher_stage_to_student_block(source_module: nn.Module, target_block, out_indices: list[int], in_indices: list[int]) -> dict:
    copied_components: list[str] = []
    direct_components: list[str] = []
    resized_components: list[str] = []
    unmapped_components: list[str] = []

    if hasattr(source_module, "conv") and hasattr(source_module.conv, "block"):
        source_module = source_module.conv

    if hasattr(source_module, "block") and isinstance(source_module.block, nn.Sequential) and len(source_module.block) >= 2:
        first_result = _copy_source_convnorm_to_target(
            "first",
            source_module.block[0].block[0] if len(source_module.block[0].block) > 0 else None,
            source_module.block[0].block[1] if len(source_module.block[0].block) > 1 else None,
            target_block.conv.block[0],
            out_indices,
            in_indices,
        )
        copied_components.extend(first_result["copied_components"])
        direct_components.extend(first_result["direct_components"])
        resized_components.extend(first_result["resized_components"])
        unmapped_components.extend(first_result["unmapped_components"])
        second_result = _copy_source_convnorm_to_target(
            "second",
            source_module.block[1].block[0] if len(source_module.block[1].block) > 0 else None,
            source_module.block[1].block[1] if len(source_module.block[1].block) > 1 else None,
            target_block.conv.block[1],
            out_indices,
            out_indices,
        )
        copied_components.extend(second_result["copied_components"])
        direct_components.extend(second_result["direct_components"])
        resized_components.extend(second_result["resized_components"])
        unmapped_components.extend(second_result["unmapped_components"])
    elif hasattr(source_module, "conv1") and hasattr(source_module, "conv2"):
        first_components = _copy_source_convnorm_to_target(
            "first",
            getattr(source_module, "conv1", None),
            getattr(source_module, "bn1", None),
            target_block.conv.block[0],
            out_indices,
            in_indices,
        )
        copied_components.extend(first_components["copied_components"])
        direct_components.extend(first_components["direct_components"])
        resized_components.extend(first_components["resized_components"])
        unmapped_components.extend(first_components["unmapped_components"])
        source_conv2 = None
        source_norm2 = None
        if isinstance(getattr(source_module, "conv2", None), nn.Sequential):
            source_conv2 = source_module.conv2[0] if len(source_module.conv2) > 0 else None
            source_norm2 = source_module.conv2[1] if len(source_module.conv2) > 1 else None
        elif isinstance(getattr(source_module, "conv2", None), nn.Conv2d):
            source_conv2 = source_module.conv2
            source_norm2 = getattr(source_module, "bn2", None)
        second_components = _copy_source_convnorm_to_target(
            "second",
            source_conv2,
            source_norm2,
            target_block.conv.block[1],
            out_indices,
            out_indices,
        )
        copied_components.extend(second_components["copied_components"])
        direct_components.extend(second_components["direct_components"])
        resized_components.extend(second_components["resized_components"])
        unmapped_components.extend(second_components["unmapped_components"])
    else:
        source_convs = _collect_conv2d_layers(source_module)
        source_norms = _collect_norm_layers(source_module)
        first_components = _copy_source_convnorm_to_target(
            "first",
            source_convs[0] if len(source_convs) >= 1 else None,
            source_norms[0] if len(source_norms) >= 1 else None,
            target_block.conv.block[0],
            out_indices,
            in_indices,
        )
        copied_components.extend(first_components["copied_components"])
        direct_components.extend(first_components["direct_components"])
        resized_components.extend(first_components["resized_components"])
        unmapped_components.extend(first_components["unmapped_components"])
        second_components = _copy_source_convnorm_to_target(
            "second",
            source_convs[1] if len(source_convs) >= 2 else None,
            source_norms[1] if len(source_norms) >= 2 else None,
            target_block.conv.block[1],
            out_indices,
            out_indices,
        )
        copied_components.extend(second_components["copied_components"])
        direct_components.extend(second_components["direct_components"])
        resized_components.extend(second_components["resized_components"])
        unmapped_components.extend(second_components["unmapped_components"])

    return {
        "copied_components": copied_components,
        "direct_components": direct_components,
        "resized_components": resized_components,
        "unmapped_components": unmapped_components,
        "copied": bool(copied_components),
    }


def _copy_teacher_head_to_student_head(teacher_model, student_model: PDGUNet, stem_indices: list[int]) -> dict:
    source_head = getattr(teacher_model, "head", None)
    target_head = getattr(student_model, "head", None)
    if not isinstance(source_head, nn.Conv2d) or not isinstance(target_head, nn.Conv2d):
        return {"copied": False, "copy_mode": "unmapped", "reason": "missing_teacher_or_student_head"}
    out_indices = list(range(int(target_head.out_channels)))
    if int(source_head.out_channels) != int(target_head.out_channels):
        return {"copied": False, "copy_mode": "unmapped", "reason": "head_out_channels_mismatch"}
    if not stem_indices:
        return {"copied": False, "copy_mode": "unmapped", "reason": "empty_stem_indices"}
    if max(stem_indices) >= int(source_head.in_channels):
        return {"copied": False, "copy_mode": "unmapped", "reason": "stem_indices_exceed_teacher_head_channels"}
    if len(stem_indices) != int(target_head.in_channels):
        return {"copied": False, "copy_mode": "unmapped", "reason": "student_head_in_channels_mismatch"}
    return _safe_copy_conv2d_state(source_head, target_head, out_indices, stem_indices)


def _initialize_pruned_student_from_teacher(teacher_model, student_model: PDGUNet, blueprint: dict, *, input_channels: int) -> dict:
    teacher_modules = dict(teacher_model.named_modules())
    stage_names = ("stem", "down1", "down2", "down3", "down4")
    student_stage_map = {
        "stem": student_model.stem,
        "down1": student_model.down1.conv,
        "down2": student_model.down2.conv,
        "down3": student_model.down3.conv,
        "down4": student_model.down4.conv,
    }
    blueprint_modules = list(blueprint.get("modules", []))
    previous_indices = list(range(int(input_channels)))
    stem_indices: list[int] = []
    stage_transfer_rows = []
    transferred_stages = 0

    for stage_index, stage_name in enumerate(stage_names):
        module_row = blueprint_modules[stage_index] if stage_index < len(blueprint_modules) else {}
        teacher_module_name = module_row.get("module_name") or module_row.get("layer_name")
        kept_indices = [int(index) for index in module_row.get("kept_channel_indices", [])]
        target_block = student_stage_map[stage_name]
        target_channels = int(target_block.gate.alpha.numel())
        if not kept_indices:
            kept_indices = list(range(target_channels))
        elif len(kept_indices) != target_channels:
            kept_indices = kept_indices[:target_channels]

        row = {
            "student_stage": stage_name,
            "teacher_module": teacher_module_name,
            "teacher_kept_channels": int(len(kept_indices)),
            "student_out_channels": int(target_channels),
            "copied_components": [],
            "direct_components": [],
            "resized_components": [],
            "unmapped_components": [],
            "status": "skipped",
        }
        source_module = teacher_modules.get(teacher_module_name) if teacher_module_name else None
        if source_module is not None:
            transfer_result = _transfer_teacher_stage_to_student_block(source_module, target_block, kept_indices, previous_indices)
            row["copied_components"] = list(transfer_result["copied_components"])
            row["direct_components"] = list(transfer_result["direct_components"])
            row["resized_components"] = list(transfer_result["resized_components"])
            row["unmapped_components"] = list(transfer_result["unmapped_components"])
            if transfer_result["copied"]:
                row["status"] = "channel_subset_reused"
                transferred_stages += 1
            else:
                row["status"] = "found_teacher_stage_but_no_compatible_subset"
        else:
            row["status"] = "teacher_stage_not_found"
        stage_transfer_rows.append(row)
        if stage_name == "stem":
            stem_indices = list(kept_indices)
        previous_indices = list(kept_indices)

    head_transfer = _copy_teacher_head_to_student_head(
        teacher_model,
        student_model,
        stem_indices if stem_indices else list(range(int(student_model.channel_config[0]))),
    )
    student_model.force_gates_open(args.student_gate_open_value)

    return {
        "strategy": "channel_subset_teacher_weight_reuse",
        "teacher_target_modules": [row.get("module_name") or row.get("layer_name") for row in blueprint_modules],
        "transferred_stages": int(transferred_stages),
        "requested_stages": int(len(stage_names)),
        "stage_transfer_ratio": float(transferred_stages / max(1, len(stage_names))),
        "stage_transfer_rows": stage_transfer_rows,
        "head_transfer_applied": bool(head_transfer.get("copied")),
        "head_transfer": head_transfer,
        "gate_initialization": {
            "strategy": "force_open_after_teacher_subset_transfer",
            "open_probability": float(args.student_gate_open_value),
        },
    }


def _copy_norm_state(source_norm, target_norm, out_indices: list[int]) -> None:
    if not out_indices:
        return
    index = torch.as_tensor(out_indices, dtype=torch.long, device=next(source_norm.parameters(), torch.tensor([], device="cpu")).device if any(True for _ in source_norm.parameters()) else torch.device("cpu"))
    if hasattr(source_norm, "weight") and getattr(source_norm, "weight", None) is not None and hasattr(target_norm, "weight") and getattr(target_norm, "weight", None) is not None:
        target_norm.weight.data.copy_(source_norm.weight.data.index_select(0, index).to(target_norm.weight.device))
    if hasattr(source_norm, "bias") and getattr(source_norm, "bias", None) is not None and hasattr(target_norm, "bias") and getattr(target_norm, "bias", None) is not None:
        target_norm.bias.data.copy_(source_norm.bias.data.index_select(0, index).to(target_norm.bias.device))
    if hasattr(source_norm, "running_mean") and getattr(source_norm, "running_mean", None) is not None and hasattr(target_norm, "running_mean") and getattr(target_norm, "running_mean", None) is not None:
        target_norm.running_mean.data.copy_(source_norm.running_mean.data.index_select(0, index).to(target_norm.running_mean.device))
    if hasattr(source_norm, "running_var") and getattr(source_norm, "running_var", None) is not None and hasattr(target_norm, "running_var") and getattr(target_norm, "running_var", None) is not None:
        target_norm.running_var.data.copy_(source_norm.running_var.data.index_select(0, index).to(target_norm.running_var.device))


def _resize_conv_weight_spatial(weight: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if tuple(int(value) for value in weight.shape[-2:]) == tuple(int(value) for value in target_hw):
        return weight
    out_channels, in_channels, kernel_h, kernel_w = weight.shape
    resized = F.interpolate(
        weight.reshape(out_channels * in_channels, 1, kernel_h, kernel_w),
        size=tuple(int(value) for value in target_hw),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(out_channels, in_channels, int(target_hw[0]), int(target_hw[1]))


def _copy_conv2d_state(source_conv, target_conv, out_indices: list[int], in_indices: list[int]) -> None:
    out_index = torch.as_tensor(out_indices, dtype=torch.long, device=source_conv.weight.device)
    in_index = torch.as_tensor(in_indices, dtype=torch.long, device=source_conv.weight.device)
    weight = source_conv.weight.data.index_select(0, out_index).index_select(1, in_index)
    if tuple(int(value) for value in weight.shape[-2:]) != tuple(int(value) for value in target_conv.weight.shape[-2:]):
        weight = _resize_conv_weight_spatial(weight, tuple(int(value) for value in target_conv.weight.shape[-2:]))
    target_conv.weight.data.copy_(weight.to(target_conv.weight.device))
    if source_conv.bias is not None and target_conv.bias is not None:
        target_conv.bias.data.copy_(source_conv.bias.data.index_select(0, out_index).to(target_conv.bias.device))


def _copy_convtranspose2d_state(source_conv, target_conv, in_indices: list[int], out_indices: list[int]) -> None:
    in_index = torch.as_tensor(in_indices, dtype=torch.long, device=source_conv.weight.device)
    out_index = torch.as_tensor(out_indices, dtype=torch.long, device=source_conv.weight.device)
    weight = source_conv.weight.data.index_select(0, in_index).index_select(1, out_index)
    target_conv.weight.data.copy_(weight.to(target_conv.weight.device))
    if source_conv.bias is not None and target_conv.bias is not None:
        target_conv.bias.data.copy_(source_conv.bias.data.index_select(0, out_index).to(target_conv.bias.device))


def _copy_conv_norm_act_state(source_block, target_block, out_indices: list[int], in_indices: list[int]) -> None:
    _copy_conv2d_state(source_block.block[0], target_block.block[0], out_indices, in_indices)
    _copy_norm_state(source_block.block[1], target_block.block[1], out_indices)


def _copy_double_conv_state(source_block, target_block, out_indices: list[int], in_indices: list[int]) -> None:
    _copy_conv_norm_act_state(source_block.block[0], target_block.block[0], out_indices, in_indices)
    _copy_conv_norm_act_state(source_block.block[1], target_block.block[1], out_indices, out_indices)


def _copy_gated_double_conv_state(source_block, target_block, out_indices: list[int], in_indices: list[int]) -> None:
    _copy_double_conv_state(source_block.conv, target_block.conv, out_indices, in_indices)
    out_index = torch.as_tensor(out_indices, dtype=torch.long, device=source_block.gate.alpha.device)
    target_block.gate.alpha.data.copy_(source_block.gate.alpha.data.index_select(0, out_index).to(target_block.gate.alpha.device))


def _concat_skip_and_up_indices(stage_indices: list[int], stage_channels: int) -> list[int]:
    return list(stage_indices) + [int(stage_channels) + int(index) for index in stage_indices]


def _build_late_hard_pruning_plan(student: PDGUNet, threshold: float) -> dict:
    stage_modules = (
        ("stem", student.stem),
        ("down1", student.down1.conv),
        ("down2", student.down2.conv),
        ("down3", student.down3.conv),
        ("down4", student.down4.conv),
    )
    stage_indices = {}
    rows = []
    total_before = 0
    total_after = 0
    total_pruned = 0
    for stage_name, module in stage_modules:
        gate_values = module.gate_values().detach().cpu().float()
        original_channels = int(gate_values.numel())
        kept_indices = [int(index) for index, value in enumerate(gate_values.tolist()) if float(value) >= float(threshold)]
        selection_policy = "threshold_keep"
        if len(kept_indices) == original_channels and original_channels > 1:
            weakest_index = int(torch.argmin(gate_values).item())
            kept_indices = [index for index in range(original_channels) if index != weakest_index]
            selection_policy = "force_prune_weakest_when_threshold_keeps_all"
        if not kept_indices:
            kept_indices = sorted(int(index) for index in torch.argsort(gate_values, descending=True)[:1].tolist())
            selection_policy = "fallback_keep_top1"
        pruned_indices = sorted(index for index in range(original_channels) if index not in set(kept_indices))
        stage_indices[stage_name] = kept_indices
        total_before += original_channels
        total_after += len(kept_indices)
        total_pruned += len(pruned_indices)
        rows.append(
            {
                "layer_name": stage_name,
                "original_out_channels": original_channels,
                "pruned_out_channels": int(len(kept_indices)),
                "channels_pruned": int(len(pruned_indices)),
                "hard_pruning_threshold": float(threshold),
                "selection_policy": selection_policy,
                "kept_channel_indices": kept_indices,
                "pruned_channel_indices": pruned_indices,
            }
        )
    return {
        "threshold": float(threshold),
        "channel_config": tuple(int(len(stage_indices[stage_name])) for stage_name, _ in stage_modules),
        "stage_indices": stage_indices,
        "rows": rows,
        "hard_pruning_applied": any(int(row["channels_pruned"]) > 0 for row in rows),
        "global_summary": {
            "total_channels_before": int(total_before),
            "total_channels_after": int(total_after),
            "total_channels_pruned": int(total_pruned),
            "global_prune_ratio": float(total_pruned / max(1, total_before)),
            "hard_pruning_threshold": float(threshold),
        },
    }


def _build_hard_pruned_student_from_plan(source_student: PDGUNet, db_train, hard_pruning_plan: dict) -> PDGUNet:
    stage_indices = dict(hard_pruning_plan["stage_indices"])
    stage0 = list(stage_indices["stem"])
    stage1 = list(stage_indices["down1"])
    stage2 = list(stage_indices["down2"])
    stage3 = list(stage_indices["down3"])
    stage4 = list(stage_indices["down4"])
    source_channels = tuple(int(channel) for channel in source_student.channel_config)
    input_indices = list(range(int(db_train.in_channels)))

    target_student = PDGUNet(
        in_channels=db_train.in_channels,
        num_classes=args.num_classes,
        channel_config=tuple(hard_pruning_plan["channel_config"]),
    ).to(next(source_student.parameters()).device)

    _copy_gated_double_conv_state(source_student.stem, target_student.stem, stage0, input_indices)
    _copy_gated_double_conv_state(source_student.down1.conv, target_student.down1.conv, stage1, stage0)
    _copy_gated_double_conv_state(source_student.down2.conv, target_student.down2.conv, stage2, stage1)
    _copy_gated_double_conv_state(source_student.down3.conv, target_student.down3.conv, stage3, stage2)
    _copy_gated_double_conv_state(source_student.down4.conv, target_student.down4.conv, stage4, stage3)

    _copy_convtranspose2d_state(source_student.up1.up, target_student.up1.up, stage4, stage3)
    _copy_gated_double_conv_state(source_student.up1.conv, target_student.up1.conv, stage3, _concat_skip_and_up_indices(stage3, source_channels[3]))
    _copy_convtranspose2d_state(source_student.up2.up, target_student.up2.up, stage3, stage2)
    _copy_gated_double_conv_state(source_student.up2.conv, target_student.up2.conv, stage2, _concat_skip_and_up_indices(stage2, source_channels[2]))
    _copy_convtranspose2d_state(source_student.up3.up, target_student.up3.up, stage2, stage1)
    _copy_gated_double_conv_state(source_student.up3.conv, target_student.up3.conv, stage1, _concat_skip_and_up_indices(stage1, source_channels[1]))
    _copy_convtranspose2d_state(source_student.up4.up, target_student.up4.up, stage1, stage0)
    _copy_gated_double_conv_state(source_student.up4.conv, target_student.up4.conv, stage0, _concat_skip_and_up_indices(stage0, source_channels[0]))

    _copy_conv2d_state(
        source_student.head,
        target_student.head,
        list(range(int(source_student.head.out_channels))),
        stage0,
    )
    return target_student


def _summarize_hard_pruning_weight_transfer(hard_pruning_plan: dict) -> dict:
    rows = list(hard_pruning_plan.get("rows", []))
    total_before = int(sum(int(row.get("original_out_channels", 0) or 0) for row in rows))
    total_after = int(sum(int(row.get("pruned_out_channels", 0) or 0) for row in rows))
    total_pruned = int(sum(int(row.get("channels_pruned", 0) or 0) for row in rows))
    return {
        "strategy": "subset_weight_reuse_from_pre_pruning_student",
        "source": "step3_student_before_late_hard_pruning",
        "num_stages": int(len(rows)),
        "total_channels_before": total_before,
        "total_channels_after": total_after,
        "total_channels_pruned": total_pruned,
        "channel_reuse_ratio": float(total_after / max(1, total_before)),
        "per_stage": [
            {
                "layer_name": row.get("layer_name"),
                "original_out_channels": row.get("original_out_channels"),
                "pruned_out_channels": row.get("pruned_out_channels"),
                "channels_pruned": row.get("channels_pruned"),
                "kept_channel_indices": row.get("kept_channel_indices", []),
                "selection_policy": row.get("selection_policy"),
            }
            for row in rows
        ],
    }


def _aggregate_phase_metrics(*phase_exports: dict) -> list[dict]:
    rows = []
    for export_bundle in phase_exports:
        rows.extend(list(export_bundle.get("metrics_rows", [])))
    return rows


def _build_pipeline_overview_rows(*phase_artifacts: dict) -> list[dict]:
    rows = []
    for artifact in phase_artifacts:
        phase = artifact.get("phase")
        export_bundle = artifact.get("evaluation_bundle", {})
        metrics_rows = list(export_bundle.get("metrics_rows", []))
        split_lookup = {row.get("split"): row for row in metrics_rows}
        row = {
            "phase": phase,
            "run_dir": project_relative_path(artifact.get("run_dir"), PROJECT_ROOT),
            "checkpoint_path": project_relative_path(artifact.get("checkpoint_path"), PROJECT_ROOT) if artifact.get("checkpoint_path") else "",
            "model_name": artifact.get("metadata", {}).get("model_name"),
            "backbone_name": artifact.get("metadata", {}).get("backbone_name"),
            "student_name": artifact.get("metadata", {}).get("student_name"),
            "train_dice": split_lookup.get("train", {}).get("dice"),
            "val_dice": split_lookup.get("val", {}).get("dice"),
            "test_dice": split_lookup.get("test", {}).get("dice"),
            "params": next((metric.get("params") for metric in metrics_rows if metric.get("params") is not None), None),
            "flops": next((metric.get("flops") for metric in metrics_rows if metric.get("flops") is not None), None),
            "fps": next((metric.get("fps") for metric in metrics_rows if metric.get("fps") is not None), None),
        }
        if phase == "pruning":
            row["evaluation_note"] = artifact.get("metadata", {}).get("evaluation_note")
            row["weight_transfer_copy_ratio"] = artifact.get("metadata", {}).get("weight_transfer", {}).get("copy_ratio")
        rows.append(row)
    return rows


def _build_pipeline_compression_rows(*phase_artifacts: dict) -> list[dict]:
    rows = []
    reference_teacher = next((artifact for artifact in phase_artifacts if artifact.get("phase") == "teacher"), None)
    teacher_metrics = reference_teacher.get("evaluation_bundle", {}).get("metrics_rows", []) if reference_teacher else []
    teacher_row = next((row for row in teacher_metrics if row.get("split") == "test"), teacher_metrics[0] if teacher_metrics else {})
    teacher_params = teacher_row.get("params")
    teacher_flops = teacher_row.get("flops")
    teacher_dice = teacher_row.get("dice")

    for artifact in phase_artifacts:
        phase = artifact.get("phase")
        metrics_rows = artifact.get("evaluation_bundle", {}).get("metrics_rows", [])
        test_row = next((row for row in metrics_rows if row.get("split") == "test"), metrics_rows[0] if metrics_rows else {})
        params = test_row.get("params")
        flops = test_row.get("flops")
        dice = test_row.get("dice")
        row = {
            "phase": phase,
            "model_name": artifact.get("metadata", {}).get("model_name"),
            "student_name": artifact.get("metadata", {}).get("student_name"),
            "params": params,
            "flops": flops,
            "fps": test_row.get("fps"),
            "test_dice": dice,
            "vs_teacher_param_ratio": (float(params) / float(teacher_params)) if teacher_params not in (None, "", 0) and params is not None else None,
            "vs_teacher_flops_ratio": (float(flops) / float(teacher_flops)) if teacher_flops not in (None, "", 0) and flops is not None else None,
            "vs_teacher_test_dice_delta": (float(dice) - float(teacher_dice)) if teacher_dice is not None and dice is not None else None,
        }
        if phase == "pruning":
            row["global_prune_ratio"] = artifact.get("blueprint", {}).get("global_pruning_summary", {}).get("global_prune_ratio")
        if phase == "student":
            row["student_variant"] = artifact.get("student_variant", {}).get("variant_name")
            row["active_soft_pruning_epochs"] = artifact.get("pruning_schedule", {}).get("active_soft_pruning_epochs")
            row["effective_warmup_pruning_epochs"] = artifact.get("pruning_schedule", {}).get("effective_warmup_pruning_epochs")
        rows.append(row)
    return rows


def _save_pipeline_outputs(pipeline_dir: Path, *phase_artifacts: dict) -> dict:
    layout = ensure_run_layout(pipeline_dir)
    pipeline_metrics_rows = _aggregate_phase_metrics(*phase_artifacts)
    pipeline_overview_rows = _build_pipeline_overview_rows(*phase_artifacts)
    pipeline_compression_rows = _build_pipeline_compression_rows(*phase_artifacts)

    if pipeline_overview_rows:
        write_metrics_rows(pipeline_overview_rows, layout["metrics_dir"] / "pipeline_stage_overview.csv")
    if pipeline_metrics_rows:
        write_metrics_rows(pipeline_metrics_rows, layout["metrics_dir"] / "pipeline_metrics.csv")
        report_rows = []
        for row in pipeline_metrics_rows:
            report_row = dict(row)
            report_row["split"] = f"{row.get('phase', 'phase')}:{row.get('split', 'split')}"
            report_rows.append(report_row)
        save_performance_pdf(
            report_rows,
            layout["reports_dir"] / "pipeline_performance.pdf",
            title="Pipeline stage performance",
        )
    if pipeline_compression_rows:
        write_metrics_rows(pipeline_compression_rows, layout["metrics_dir"] / "pipeline_compression_summary.csv")

    pipeline_visual_index = {}
    for artifact in phase_artifacts:
        phase = artifact.get("phase")
        export_bundle = artifact.get("evaluation_bundle", {})
        sample_map = export_bundle.get("visualization_samples_by_split", {})
        pipeline_visual_index[phase] = {}
        for split, samples in sample_map.items():
            if not samples:
                continue
            prefixed_samples = []
            for sample in samples:
                cloned = dict(sample)
                cloned["case"] = f"{phase}:{sample.get('case', 'unknown')}"
                prefixed_samples.append(cloned)
            save_visualization_pdf(
                prefixed_samples,
                layout["reports_dir"] / f"pipeline_{phase}_{split}_visualizations.pdf",
                title=f"pipeline | {phase} | {split}",
            )
            save_visualization_overview_image(
                prefixed_samples,
                layout["visualization_dir"] / f"pipeline_{phase}_{split}_visualizations.png",
            )
            pipeline_visual_index[phase][split] = {
                "pdf": project_relative_path(layout["reports_dir"] / f"pipeline_{phase}_{split}_visualizations.pdf", PROJECT_ROOT),
                "image": project_relative_path(layout["visualization_dir"] / f"pipeline_{phase}_{split}_visualizations.png", PROJECT_ROOT),
            }

    summary = {
        "dataset": args.dataset,
        "teacher_model": args.teacher_model,
        **_pruning_metadata(),
        "stages": {
            artifact.get("phase"): {
                "run_dir": project_relative_path(artifact.get("run_dir"), PROJECT_ROOT),
                "checkpoint_path": project_relative_path(artifact.get("checkpoint_path"), PROJECT_ROOT) if artifact.get("checkpoint_path") else "",
                "model_info": artifact.get("metadata", {}),
                "split_summaries": artifact.get("evaluation_bundle", {}).get("split_summaries", {}),
            }
            for artifact in phase_artifacts
        },
        "pipeline_stage_overview_csv": project_relative_path(layout["metrics_dir"] / "pipeline_stage_overview.csv", PROJECT_ROOT) if pipeline_overview_rows else "",
        "pipeline_metrics_csv": project_relative_path(layout["metrics_dir"] / "pipeline_metrics.csv", PROJECT_ROOT) if pipeline_metrics_rows else "",
        "pipeline_compression_summary_csv": project_relative_path(layout["metrics_dir"] / "pipeline_compression_summary.csv", PROJECT_ROOT) if pipeline_compression_rows else "",
        "pipeline_performance_pdf": project_relative_path(layout["reports_dir"] / "pipeline_performance.pdf", PROJECT_ROOT) if pipeline_metrics_rows else "",
        "pipeline_visualizations": pipeline_visual_index,
    }
    _write_json(layout["evaluation_dir"] / "pipeline_summary.json", summary)
    markdown_lines = [
        f"# Pipeline Evaluation Summary | {args.dataset} | {args.teacher_model}",
        "",
        f"- Teacher model: `{args.teacher_model}`",
        f"- Pruning strategy: `{args.prune_method}`",
        f"- Static prune ratio: `{args.static_prune_ratio}`" if uses_static_prune_ratio(args.prune_method) else "- Static prune ratio: `not used`",
        f"- Step-3 pruning: `{'enabled' if bool(args.enable_step3_pruning) else 'disabled'}`",
        f"- Step-3 pruning epochs: `{int(args.step3_pruning_epochs) if bool(args.enable_step3_pruning) else 'no'}`",
        "",
        "## Stages",
        "",
    ]
    for row in pipeline_overview_rows:
        markdown_lines.append(
            f"- `{row.get('phase')}` | train dice `{row.get('train_dice')}` | val dice `{row.get('val_dice')}` | test dice `{row.get('test_dice')}` | checkpoint `{row.get('checkpoint_path')}`"
        )
    with (layout["evaluation_dir"] / "pipeline_summary.md").open("w", encoding="utf-8") as file:
        file.write("\n".join(markdown_lines) + "\n")
    return {
        "overview_rows": pipeline_overview_rows,
        "metrics_rows": pipeline_metrics_rows,
        "compression_rows": pipeline_compression_rows,
        "summary": summary,
    }


def _export_student_final_shortcuts(student_run_dir: Path, proposal_root_dir: Path) -> dict:
    student_final_dir = proposal_root_dir / "student_final"
    student_final_dir.mkdir(parents=True, exist_ok=True)
    source_dir = student_run_dir / "checkpoints"
    mapping = {
        "best.pth": "best_student.pth",
    }
    exported = {}
    for source_name, target_name in mapping.items():
        source_path = source_dir / source_name
        if not source_path.is_file():
            continue
        target_path = student_final_dir / target_name
        try:
            payload = load_checkpoint(source_path)
        except (EOFError, RuntimeError, OSError, pickle.UnpicklingError) as error:
            logging.warning(
                "Skip exporting student final shortcut because checkpoint is not readable: %s | %s",
                project_relative_path(source_path, PROJECT_ROOT),
                error,
            )
            continue
        save_checkpoint_payload_atomic(payload, target_path)
        metadata = build_checkpoint_metadata(
            payload,
            target_path,
            project_root=PROJECT_ROOT,
            extra_fields={
                "shortcut_role": target_name.replace(".pth", ""),
                "source_checkpoint_path": project_relative_path(source_path, PROJECT_ROOT),
                "source_run_dir": project_relative_path(student_run_dir, PROJECT_ROOT),
            },
        )
        metadata_name = target_name.replace(".pth", ".json")
        _write_json(student_final_dir / metadata_name, metadata)
        exported[target_name] = project_relative_path(target_path, PROJECT_ROOT)
    return exported


def _validate_label_batch(label_batch, num_classes, sampled_batch):
    label_batch = label_batch.long()
    invalid_mask = (label_batch < 0) | (label_batch >= num_classes)
    if not invalid_mask.any():
        return
    raise ValueError(
        f"Invalid mask values detected. "
        f"Case={sampled_batch.get('case')} invalid={torch.unique(label_batch[invalid_mask].cpu()).tolist()[:20]}"
    )


def _build_loaders(device: torch.device, image_mode: str):
    train_transform = transforms.Compose([Normalize(), RandomGenerator(args.patch_size)])
    eval_transform = transforms.Compose([Normalize(), ToTensor()])
    db_train = build_dataset(args.dataset, args.root_path, split=args.train_split, transform=train_transform, image_mode=image_mode)
    db_val = build_dataset(args.dataset, args.root_path, split=args.val_split, transform=eval_transform, image_mode=image_mode)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1, pin_memory=device.type == "cuda")
    return db_train, db_val, trainloader, valloader


def _supervised_loss(logits: torch.Tensor, target: torch.Tensor, ce_loss, dice_loss) -> torch.Tensor:
    return 0.5 * (ce_loss(logits, target.long()) + dice_loss(torch.softmax(logits, dim=1), target.unsqueeze(1)))


def _compute_supervised_val_loss(model, dataloader, device, ce_loss, dice_loss) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            _validate_label_batch(batch["label"], args.num_classes, batch)
            image = F.interpolate(batch["image"].to(device), size=args.patch_size, mode="bilinear", align_corners=False)
            label = F.interpolate(batch["label"].unsqueeze(1).float().to(device), size=args.patch_size, mode="nearest").squeeze(1).long()
            losses.append(float(_supervised_loss(extract_logits(model(image)), label, ce_loss, dice_loss).item()))
    model.train()
    return float(np.mean(losses)) if losses else 0.0


def _compute_profile(model, device: torch.device) -> dict:
    input_shape = (1, args.in_channels, args.patch_size[0], args.patch_size[1])
    profile = {
        "params": int(count_parameters(model)),
        "trainable_params": int(count_parameters(model, trainable_only=True)),
    }
    profile.update(benchmark_inference(model, input_shape=input_shape, device=device))
    profile["flops"] = maybe_compute_flops(model, input_shape=input_shape, device=device)
    return profile


def _export_model_channel_analysis(
    run_dir: Path,
    model,
    *,
    checkpoint_path: Path | None,
    device: torch.device,
    prefix: str | None,
    title: str,
) -> dict:
    if checkpoint_path is not None:
        load_checkpoint_into_model(checkpoint_path, model, device=device)
    analysis = extract_channel_analysis(model, importance_name="filter_l1")
    save_channel_analysis_artifacts(run_dir / "artifacts" / "channel_analysis", analysis, prefix=prefix, title=title)
    logging.info(
        "%s | analyzed_layers=%d | total_channels=%d | gate_layers=%d",
        title,
        analysis["global_summary"]["num_channel_layers"],
        analysis["global_summary"]["total_output_channels"],
        analysis["global_summary"]["num_gate_layers"],
    )
    return analysis


def _export_phase_outputs(run_dir: Path, model, checkpoint_path: Path, phase: str, profile: dict, extra: dict, device: torch.device, image_mode: str) -> dict:
    load_checkpoint_into_model(checkpoint_path, model, device=device)
    layout = ensure_run_layout(run_dir)
    metrics_rows = []
    split_summaries = {}
    visualization_samples_by_split = {}
    eval_transform = transforms.Compose([Normalize(), ToTensor()])
    model_info = dict(extract_model_info(model))
    model_info.update(
        {
            key: value
            for key, value in extra.items()
            if key
            in {
                "model_name",
                "backbone_name",
                "student_name",
                "teacher_model",
                "prune_strategy",
                "prune_method",
                "static_prune_ratio",
                "prune_ratio",
                "pruning_output_dir_name",
                "step3_output_dir_name",
                "step3_pruning_enabled",
                "step3_pruning_epochs",
                "step3_pruning_tag",
            }
        }
    )
    for split in dict.fromkeys(args.final_eval_splits):
        try:
            dataset = build_dataset(args.dataset, args.root_path, split=split, transform=eval_transform, image_mode=image_mode)
        except Exception as error:
            logging.warning("Skip %s/%s: %s", phase, split, error)
            continue
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=device.type == "cuda")
        output_dir = build_evaluation_output_dir(run_dir, args.dataset, extra["model_name"], checkpoint_path, split)
        start = time.perf_counter()
        result = evaluate_segmentation_dataset(
            model,
            dataloader,
            device=device,
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            output_dir=output_dir,
            save_visualizations=bool(args.save_visualizations),
            vis_limit=args.vis_num_samples,
            sample_validator=_validate_label_batch,
            progress_desc=f"{phase}-{split}",
        )
        elapsed = time.perf_counter() - start
        summary = save_evaluation_artifacts(
            output_dir,
            {
                "experiment": args.exp,
                "dataset": args.dataset,
                "dataset_root": normalize_path_string(args.root_path),
                "split": split,
                "model": extra["model_name"],
                "architecture": extra["model_name"],
                "backbone_name": extra.get("backbone_name"),
                "student_name": extra.get("student_name"),
                "phase": phase,
                "model_info": model_info,
                "num_classes": args.num_classes,
                "in_channels": args.in_channels,
                "patch_size": list(args.patch_size),
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_path": project_relative_path(checkpoint_path, PROJECT_ROOT),
            },
            result["average_metric"],
            result["case_metrics"],
            project_root=PROJECT_ROOT,
        )
        split_summaries[split] = summary
        visualization_samples_by_split[split] = list(result["visualization_samples"])
        macro = summary["metrics"]["macro_mean"]
        metrics_rows.append(
            {
                "experiment": args.exp,
                "dataset": args.dataset,
                "split": split,
                "phase": phase,
                "model_name": extra["model_name"],
                "backbone_name": extra.get("backbone_name"),
                "student_name": extra.get("student_name"),
                "dice": macro["dice"],
                "iou": macro["iou"],
                "hd95": macro["hd95"],
                "params": profile.get("params"),
                "trainable_params": profile.get("trainable_params"),
                "flops": profile.get("flops"),
                "fps": profile.get("fps"),
                "inference_time_seconds": profile.get("inference_time_seconds"),
                "evaluation_time_seconds": elapsed,
                "teacher_model": args.teacher_model,
                "prune_strategy": extra.get("prune_strategy"),
                "prune_method": extra.get("prune_method"),
                "static_prune_ratio": extra.get("static_prune_ratio"),
                "prune_ratio": extra.get("prune_ratio"),
                "step3_pruning_enabled": extra.get("step3_pruning_enabled"),
                "step3_pruning_epochs": extra.get("step3_pruning_epochs"),
                "step3_pruning_tag": extra.get("step3_pruning_tag"),
                "step3_output_dir_name": extra.get("step3_output_dir_name"),
                "checkpoint_path": project_relative_path(checkpoint_path, PROJECT_ROOT),
            }
        )
        if result["visualization_samples"]:
            save_visualization_pdf(
                result["visualization_samples"],
                layout["reports_dir"] / f"{phase}_{split}_visualizations.pdf",
                title=f"{phase} | {split}",
            )
            save_visualization_overview_image(
                result["visualization_samples"],
                layout["visualization_dir"] / f"{phase}_{split}_visualizations.png",
            )
    if metrics_rows:
        write_metrics_rows(metrics_rows, layout["metrics_dir"] / f"{phase}_metrics.csv")
        save_performance_pdf(metrics_rows, layout["reports_dir"] / f"{phase}_performance.pdf", title=f"{phase} performance")
    return {
        "metrics_rows": metrics_rows,
        "split_summaries": split_summaries,
        "visualization_samples_by_split": visualization_samples_by_split,
        "phase": phase,
    }


def _build_teacher(in_channels: int):
    kwargs = {"mode": "train"}
    if args.teacher_model == "unetr":
        kwargs["image_size"] = tuple(args.patch_size)
    if args.teacher_model == "unet_resnet152":
        kwargs["encoder_pretrained"] = bool(args.encoder_pretrained)
    return net_factory(net_type=args.teacher_model, in_chns=in_channels, class_num=args.num_classes, **kwargs)


def _run_teacher(device: torch.device, image_mode: str, db_train, trainloader, valloader):
    run_dir = _phase_dir("teacher")
    layout = ensure_run_layout(run_dir)
    write_run_config(run_dir, vars(args))
    metadata = get_model_metadata(args.teacher_model)
    model = _build_teacher(db_train.in_channels).to(device)
    model_info = dict(metadata)
    model_info.update(extract_model_info(model))
    model_info["build_kwargs"] = {
        "net_type": args.teacher_model,
        "in_channels": db_train.in_channels,
        "num_classes": args.num_classes,
        "image_size": list(args.patch_size) if args.teacher_model == "unetr" else None,
        "encoder_pretrained": bool(args.encoder_pretrained) if args.teacher_model == "unet_resnet152" else None,
    }
    write_model_config(run_dir, model_info)
    teacher_signature = _teacher_signature(db_train.in_channels)
    history = {}
    best_path = None
    reused_from = None
    source_checkpoint_path = None
    registered_checkpoint_path = None

    if not bool(args.force_retrain_teacher):
        explicit_match = None
        if args.teacher_checkpoint:
            explicit_checkpoint_path = Path(args.teacher_checkpoint).expanduser()
            if not explicit_checkpoint_path.is_absolute():
                explicit_checkpoint_path = (PROJECT_ROOT / explicit_checkpoint_path).resolve()
            else:
                explicit_checkpoint_path = explicit_checkpoint_path.resolve()

            if not explicit_checkpoint_path.is_file():
                logging.warning("Ignore explicit teacher checkpoint because the file does not exist: %s", args.teacher_checkpoint)
            else:
                explicit_match = find_compatible_checkpoint(
                    [explicit_checkpoint_path],
                    expected_signature=teacher_signature,
                )
                if explicit_match is None:
                    logging.warning("Ignore explicit teacher checkpoint because it is not compatible: %s", args.teacher_checkpoint)

        if explicit_match is not None:
            explicit_checkpoint_path = Path(explicit_match["checkpoint_path"])
            source_checkpoint_path = explicit_checkpoint_path
            try:
                explicit_checkpoint_path.resolve().relative_to(run_dir.resolve())
                best_path = explicit_checkpoint_path
            except ValueError:
                copied = register_reused_checkpoint(
                    source_checkpoint_path=explicit_checkpoint_path,
                    target_run_dir=run_dir,
                    project_root=PROJECT_ROOT,
                    source_branch="explicit",
                    source_run_dir=explicit_checkpoint_path.parent.parent if explicit_checkpoint_path.parent.name == "checkpoints" else explicit_checkpoint_path.parent,
                    payload_updates={"phase": "teacher"},
                )
                best_path = Path(copied["best"]["checkpoint_path"])
                registered_checkpoint_path = best_path
            payload = load_checkpoint_into_model(explicit_checkpoint_path, model, device=device)
            history = payload.get("extra_state", {}).get("history", {})
            model_info.update(payload.get("model_info", {}))
            reused_from = "explicit"
            if registered_checkpoint_path is not None:
                logging.info(
                    "Loaded teacher checkpoint from explicit --teacher_checkpoint source: %s | registered at: %s",
                    project_relative_path(source_checkpoint_path, PROJECT_ROOT),
                    project_relative_path(registered_checkpoint_path, PROJECT_ROOT),
                )
            else:
                logging.info(
                    "Loaded teacher checkpoint from explicit --teacher_checkpoint: %s",
                    project_relative_path(source_checkpoint_path, PROJECT_ROOT),
                )
        else:
            proposal_match = resolve_run_checkpoint(run_dir, expected_signature=teacher_signature)
            proposal_match_source = "proposal"
            if proposal_match is None:
                teacher_candidate_dirs = []
                if args.teacher_output_root and args.output_root:
                    teacher_candidate_dirs.append(
                        build_pdg_phase_dir(
                            project_root=PROJECT_ROOT,
                            dataset=args.dataset,
                            teacher_name=args.teacher_model,
                            phase="teacher",
                            output_root=args.output_root or None,
                        )
                    )
                teacher_candidate_dirs.append(
                    build_pdg_phase_dir(
                        project_root=PROJECT_ROOT,
                        dataset=args.dataset,
                        teacher_name=args.teacher_model,
                        phase="teacher",
                        output_root=args.step3_output_dir_name,
                    )
                )
                for variant_teacher_run_dir in dict.fromkeys(teacher_candidate_dirs):
                    if variant_teacher_run_dir.resolve() == run_dir.resolve():
                        continue
                    proposal_match = resolve_run_checkpoint(variant_teacher_run_dir, expected_signature=teacher_signature)
                    if proposal_match is not None:
                        proposal_match["run_dir"] = variant_teacher_run_dir
                        proposal_match_source = "proposal_variant"
                        break
            if proposal_match is not None:
                proposal_checkpoint_path = Path(proposal_match["checkpoint_path"])
                source_checkpoint_path = proposal_checkpoint_path
                try:
                    proposal_checkpoint_path.resolve().relative_to(run_dir.resolve())
                    best_path = proposal_checkpoint_path
                except ValueError:
                    copied = register_reused_checkpoint(
                        source_checkpoint_path=proposal_checkpoint_path,
                        target_run_dir=run_dir,
                        project_root=PROJECT_ROOT,
                        source_branch=proposal_match_source,
                        source_run_dir=proposal_match.get("run_dir") or (proposal_checkpoint_path.parent.parent if proposal_checkpoint_path.parent.name == "checkpoints" else proposal_checkpoint_path.parent),
                        payload_updates={"phase": "teacher"},
                    )
                    best_path = Path(copied["best"]["checkpoint_path"])
                    reused_from = proposal_match_source
                    registered_checkpoint_path = best_path
                payload = load_checkpoint_into_model(best_path, model, device=device)
                history = payload.get("extra_state", {}).get("history", {})
                model_info.update(payload.get("model_info", {}))
                reused_from = reused_from or "proposal"
                logging.info("Loaded teacher checkpoint from %s: %s", reused_from, project_relative_path(best_path, PROJECT_ROOT))
            else:
                basic_match = resolve_basic_checkpoint(
                    project_root=PROJECT_ROOT,
                    output_root=_teacher_output_root(),
                    model_name=args.teacher_model,
                    dataset=args.dataset,
                    expected_signature=teacher_signature,
                )
                if basic_match is not None:
                    source_checkpoint_path = Path(basic_match["checkpoint_path"])
                    copied = register_reused_checkpoint(
                        source_checkpoint_path=source_checkpoint_path,
                        target_run_dir=run_dir,
                        project_root=PROJECT_ROOT,
                        source_branch="basic",
                        source_run_dir=basic_match.get("run_dir"),
                        payload_updates={"phase": "teacher"},
                    )
                    best_path = Path(copied["best"]["checkpoint_path"])
                    registered_checkpoint_path = best_path
                    payload = load_checkpoint_into_model(best_path, model, device=device)
                    history = payload.get("extra_state", {}).get("history", {})
                    model_info.update(payload.get("model_info", {}))
                    reused_from = "basic"
                    logging.info(
                        "Reused teacher checkpoint from basic branch and registered it under proposal outputs: %s",
                        project_relative_path(source_checkpoint_path, PROJECT_ROOT),
                    )

    if best_path is None:
        ce_loss = CrossEntropyLoss()
        dice_loss = DiceLoss(n_classes=args.num_classes)
        optimizer = optim.SGD(model.parameters(), lr=args.teacher_lr, momentum=0.9, weight_decay=1e-4)
        history = {"train_total_loss": [], "val_total_loss": [], "val_macro_dice": []}
        best_metric = float("-inf")
        best_path = None
        for epoch in tqdm(range(1, args.max_epochs_teacher + 1), desc="teacher", ncols=90):
            model.train()
            train_losses = []
            for batch in trainloader:
                _validate_label_batch(batch["label"], args.num_classes, batch)
                image = batch["image"].to(device)
                label = batch["label"].to(device)
                loss = _supervised_loss(extract_logits(model(image)), label, ce_loss, dice_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))
            val_result = evaluate_segmentation_dataset(
                model,
                valloader,
                device=device,
                num_classes=args.num_classes,
                patch_size=args.patch_size,
                output_dir=None,
                save_visualizations=False,
                vis_limit=0,
                sample_validator=_validate_label_batch,
            )
            val_dice = float(np.mean(val_result["average_metric"][:, 0]))
            val_loss = _compute_supervised_val_loss(model, valloader, device, ce_loss, dice_loss)
            history["train_total_loss"].append(float(np.mean(train_losses)) if train_losses else 0.0)
            history["val_total_loss"].append(val_loss)
            history["val_macro_dice"].append(val_dice)
            is_best = val_dice > best_metric
            if is_best:
                best_metric = val_dice
            checkpoint_path = save_checkpoint(
                run_dir,
                f"epoch_{epoch:03d}",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                metrics={"val_macro_dice": val_dice, "val_total_loss": val_loss},
                config=vars(args),
                model_info=model_info,
                phase="teacher",
                extra_state={"history": history},
                is_best=is_best,
                save_tagged_checkpoint=bool(args.save_history_checkpoints),
                save_last_checkpoint=bool(args.save_last_checkpoint),
                include_optimizer_state=bool(args.save_optimizer_state),
                project_root=PROJECT_ROOT,
            )
            if is_best:
                best_path = checkpoint_path
            logging.info(
                "Teacher epoch %d/%d | train_loss=%.6f | val_loss=%.6f | val_dice=%.6f",
                epoch,
                args.max_epochs_teacher,
                history["train_total_loss"][-1],
                val_loss,
                val_dice,
            )
        if best_path is None:
            best_path = resolve_phase_checkpoint(run_dir, "last")
        save_loss_pdf(history, layout["reports_dir"] / "teacher_loss.pdf", title="Teacher loss")
    else:
        _restore_loss_pdf_if_possible(history, layout["reports_dir"] / "teacher_loss.pdf", "Teacher loss")

    write_model_config(run_dir, model_info)
    profile = _compute_profile(model, device)
    evaluation_bundle = _export_phase_outputs(
        run_dir,
        model,
        best_path,
        "teacher",
        profile,
        {"model_name": metadata["model_name"], "backbone_name": metadata["backbone_name"], "student_name": None, "prune_ratio": None, "teacher_model": args.teacher_model},
        device,
        image_mode,
    )
    _export_model_channel_analysis(
        run_dir,
        model,
        checkpoint_path=best_path,
        device=device,
        prefix=None,
        title=f"Teacher Channel Analysis | {args.teacher_model}",
    )
    return {
        "model": model,
        "run_dir": run_dir,
        "checkpoint_path": best_path,
        "source_checkpoint_path": source_checkpoint_path,
        "registered_checkpoint_path": registered_checkpoint_path or best_path,
        "metadata": model_info,
        "history": history,
        "reused_from": reused_from,
        "evaluation_bundle": evaluation_bundle,
        "phase": "teacher",
    }


def _run_pruning(device: torch.device, image_mode: str, db_train, teacher_artifact: dict):
    run_dir = _phase_dir("pruning")
    layout = ensure_run_layout(run_dir)
    write_run_config(run_dir, vars(args))
    blueprint_path = layout["artifacts_dir"] / "blueprint.json"
    reusable_blueprint = None
    if blueprint_path.is_file() and not bool(args.force_reprune):
        candidate_blueprint = load_blueprint_artifact(blueprint_path)
        if (
            candidate_blueprint.get("teacher_model") == args.teacher_model
            and _blueprint_matches_current_pruning_config(candidate_blueprint)
        ):
            reusable_blueprint = candidate_blueprint
    if reusable_blueprint is not None:
        blueprint = reusable_blueprint
        logging.info("Loaded blueprint: %s", project_relative_path(blueprint_path, PROJECT_ROOT))
    else:
        blueprint = extract_pruned_blueprint(
            teacher_artifact["model"],
            prune_ratio=args.prune_ratio,
            prune_method=args.prune_method,
            static_prune_ratio=args.static_prune_ratio,
        )
        blueprint.update(
            {
                "teacher_model": args.teacher_model,
                "teacher_checkpoint_path": project_relative_path(teacher_artifact["checkpoint_path"], PROJECT_ROOT),
                "teacher_run_dir": project_relative_path(teacher_artifact["run_dir"], PROJECT_ROOT),
                "student_name": _student_model_name(),
                "mapping_rule": (
                    f"teacher_resnet_bottleneck_conv2 -> middle_pruned_resnet_unet ({args.prune_method})"
                    if _uses_middle_pruned_resnet_student()
                    else "teacher_encoder -> student_channel_config"
                ),
                **_pruning_metadata(),
            }
        )
        save_blueprint_artifact(blueprint, layout["artifacts_dir"])
    save_pruning_analysis_artifacts(layout["artifacts_dir"] / "pruning_analysis", blueprint, title="Teacher -> Student Pruning Analysis")
    _write_json(layout["configs_dir"] / "pruning_config.json", blueprint)
    _write_json(layout["metrics_dir"] / "pruning_summary.json", blueprint)
    global_summary = blueprint.get("global_pruning_summary", {})
    logging.info(
        "Global pruning summary | method=%s | layers=%s | channels_before=%s | channels_after=%s | pruned=%s | ratio=%.4f",
        args.prune_method,
        global_summary.get("num_layers_analyzed"),
        global_summary.get("total_channels_before"),
        global_summary.get("total_channels_after"),
        global_summary.get("total_channels_pruned"),
        global_summary.get("global_prune_ratio", 0.0),
    )
    for row in blueprint.get("teacher_vs_student_rows", []):
        logging.info(
            "Pruning layer %s | pruning_layer=%s | policy=%s | %s -> %s | pruned=%s | ratio=%.4f",
            row["layer_name"],
            row.get("pruning_layer_name", row["layer_name"]),
            row.get("selection_policy", args.prune_method),
            row["teacher_out_channels"],
            row["student_out_channels"],
            row["channels_pruned"],
            row["actual_prune_ratio"],
        )

    if str(blueprint.get("student_architecture", "")).lower() == "middle_pruned_resnet_unet":
        pruned_student, weight_transfer = build_middle_pruned_resnet_unet_from_teacher(
            teacher_artifact["model"],
            in_channels=db_train.in_channels,
            num_classes=args.num_classes,
            blueprint=blueprint,
        )
        pruned_student = pruned_student.to(device)
        weight_transfer["copy_ratio"] = weight_transfer.get("block_transfer_ratio")
        weight_transfer["exact_match_copy_ratio"] = weight_transfer.get("exact_matching_full_weight_copy", {}).get("copy_ratio")
        weight_transfer["effective_note"] = (
            f"{_strategy_label()} initializes a middle-pruned ResNet-UNet from the teacher. In every ResNet bottleneck, conv1 output and conv3 output stay full; "
            "only conv2 output, bn2, and conv3 input are subset-copied by the selected pruning mask."
        )
        logging.info(
            "Step-2 %s middle reuse | copied_blocks=%s/%s | exact_match_copy_ratio=%.4f",
            _strategy_label(),
            weight_transfer.get("copied_blocks"),
            weight_transfer.get("requested_blocks"),
            float(weight_transfer.get("exact_match_copy_ratio", 0.0) or 0.0),
        )
        for row in weight_transfer.get("rows", [])[:20]:
            logging.info(
                "Step-2 %s block reuse | block=%s | middle=%s | status=%s | kept=%s/%s | protected=%s | pruned=%s",
                _strategy_label(),
                row.get("block_name"),
                row.get("middle_layer_name"),
                row.get("status"),
                row.get("kept_middle_channels"),
                row.get("original_middle_channels"),
                ",".join(row.get("protected_components", [])) if row.get("protected_components") else "none",
                ",".join(row.get("pruned_components", [])) if row.get("pruned_components") else "none",
            )
    else:
        pruned_student = PDGUNet(
            in_channels=db_train.in_channels,
            num_classes=args.num_classes,
            channel_config=tuple(blueprint["channel_config"]),
        ).to(device)
        weight_transfer = _initialize_pruned_student_from_teacher(
            teacher_artifact["model"],
            pruned_student,
            blueprint,
            input_channels=db_train.in_channels,
        )
        exact_match_fallback = _copy_exact_matching_weights(teacher_artifact["model"], pruned_student)
        pruned_student.force_gates_open(args.student_gate_open_value)
        weight_transfer["exact_match_fallback"] = exact_match_fallback
        weight_transfer["copy_ratio"] = weight_transfer.get("stage_transfer_ratio")
        weight_transfer["exact_match_copy_ratio"] = exact_match_fallback.get("copy_ratio")
        weight_transfer["effective_note"] = (
            "Step-2 pruned student is initialized from teacher channels kept by the pruning blueprint. "
            "Remaining compatible tensors are then copied via exact-key fallback, and gates are forced open for fair baseline evaluation."
        )
        logging.info(
            "Step-2 teacher reuse | transferred_stages=%s/%s | head_transfer=%s | exact_match_copy_ratio=%.4f",
            weight_transfer.get("transferred_stages"),
            weight_transfer.get("requested_stages"),
            int(bool(weight_transfer.get("head_transfer_applied"))),
            float(exact_match_fallback.get("copy_ratio", 0.0)),
        )
        for row in weight_transfer.get("stage_transfer_rows", []):
            logging.info(
                "Step-2 stage reuse | student_stage=%s | teacher_module=%s | status=%s | direct=%s | resized=%s | unmapped=%s",
                row.get("student_stage"),
                row.get("teacher_module"),
                row.get("status"),
                ",".join(row.get("direct_components", [])) if row.get("direct_components") else "none",
                ",".join(row.get("resized_components", [])) if row.get("resized_components") else "none",
                ",".join(row.get("unmapped_components", [])) if row.get("unmapped_components") else "none",
            )
        head_transfer = weight_transfer.get("head_transfer", {})
        logging.info(
            "Step-2 head reuse | status=%s | mode=%s | reason=%s",
            "copied" if head_transfer.get("copied") else "unmapped",
            head_transfer.get("copy_mode", "unmapped"),
            head_transfer.get("reason", ""),
        )
    middle_static_student = str(blueprint.get("student_architecture", "")).lower() == "middle_pruned_resnet_unet"
    pruning_model_info = extract_model_info(pruned_student)
    pruning_model_info.update(
        {
            "branch": "proposal",
            "phase_name": "pruning",
            "teacher_model": args.teacher_model,
            "teacher_backbone_name": teacher_artifact["metadata"].get("backbone_name"),
            "student_name": _middle_student_name("student_init") if middle_static_student else "pruned_student_init",
            "blueprint_path": project_relative_path(blueprint_path, PROJECT_ROOT),
            "blueprint": blueprint,
            **_pruning_metadata(),
            "weight_transfer": weight_transfer,
            "checkpoint_is_random_init": False,
            "checkpoint_weight_status": "teacher_middle_subset_reuse_then_saved" if middle_static_student else "teacher_subset_reuse_then_saved",
            "checkpoint_weight_source": "teacher_full_boundary_weights + middle_conv2_subset" if middle_static_student else "teacher_kept_channels + exact_match_fallback",
            "evaluation_note": (
                f"This phase evaluates the {_strategy_label()} middle-pruned ResNet-UNet immediately after structural pruning. "
                "Inside each ResNet bottleneck, conv1 output and conv3 output remain full; only conv2 output, bn2, and conv3 input are pruned."
                if middle_static_student
                else (
                    "This phase evaluates the pruned student immediately after structural pruning. "
                    "The student is initialized with the teacher weights of the kept channels whenever the stage-wise mapping is compatible, "
                    "instead of starting from a fresh random initialization."
                )
            ),
            "build_kwargs": {
                "in_channels": db_train.in_channels,
                "num_classes": args.num_classes,
                "channel_config": list(blueprint["channel_config"]),
                "stage_middle_channel_config": blueprint.get("stage_middle_channel_config"),
            },
        }
    )
    write_model_config(run_dir, pruning_model_info)

    checkpoint_path = save_checkpoint(
        run_dir,
        "pruning_init",
        model=pruned_student,
        epoch=0,
        global_step=0,
        best_metric=None,
        metrics={},
        config=vars(args),
        model_info=pruning_model_info,
        phase="pruning",
        extra_state={"blueprint": blueprint, "weight_transfer": weight_transfer},
        is_best=True,
        save_tagged_checkpoint=False,
        save_last_checkpoint=bool(args.save_last_checkpoint),
        include_optimizer_state=bool(args.save_optimizer_state),
        project_root=PROJECT_ROOT,
    )

    profile = _compute_profile(pruned_student, device)
    evaluation_bundle = _export_phase_outputs(
        run_dir,
        pruned_student,
        checkpoint_path,
        "pruning",
        profile,
        {
            "model_name": _student_model_name(),
            "backbone_name": teacher_artifact["metadata"].get("backbone_name"),
            "student_name": pruning_model_info.get("student_name"),
            "teacher_model": args.teacher_model,
            **_pruning_metadata(),
        },
        device,
        image_mode,
    )
    _export_model_channel_analysis(
        run_dir,
        pruned_student,
        checkpoint_path=checkpoint_path,
        device=device,
        prefix="pruned_student",
        title="Pruned Student Before Tuning",
    )
    return {
        "run_dir": run_dir,
        "blueprint": blueprint,
        "blueprint_path": blueprint_path,
        "model": pruned_student,
        "checkpoint_path": checkpoint_path,
        "metadata": pruning_model_info,
        "evaluation_bundle": evaluation_bundle,
        "phase": "pruning",
    }


def _compute_student_val_losses(student, teacher, valloader, device, criterion, *, epoch_policy: dict):
    student.eval()
    teacher.eval()
    tracked = {
        key: []
        for key in (
            "total_loss",
            "segmentation_loss",
            "distillation_loss",
            "sparsity_loss",
            "feature_distill_loss",
            "auxiliary_loss",
        )
    }
    with torch.no_grad():
        for batch in valloader:
            _validate_label_batch(batch["label"], args.num_classes, batch)
            image = F.interpolate(batch["image"].to(device), size=args.patch_size, mode="bilinear", align_corners=False)
            label = F.interpolate(batch["label"].unsqueeze(1).float().to(device), size=args.patch_size, mode="nearest").squeeze(1).long()
            teacher_output = teacher(image) if epoch_policy.get("needs_teacher_output", False) else None
            loss_dict = criterion(
                student(image),
                teacher_output,
                student.get_gate_tensors(),
                label,
                lambda_distill=epoch_policy.get("lambda_distill", 0.0),
                lambda_sparsity=epoch_policy.get("lambda_sparsity", 0.0),
                lambda_feat=epoch_policy.get("lambda_feat", 0.0),
                lambda_aux=epoch_policy.get("lambda_aux", 0.0),
            )
            for key in tracked:
                tracked[key].append(float(loss_dict[key].item()))
    student.train()
    return {key: float(np.mean(values)) if values else 0.0 for key, values in tracked.items()}


def _run_student(device: torch.device, image_mode: str, db_train, trainloader, valloader, teacher_artifact: dict, pruning_artifact: dict):
    run_dir = _phase_dir("student")
    layout = ensure_run_layout(run_dir)
    write_run_config(run_dir, vars(args))
    teacher = teacher_artifact["model"]
    _freeze_teacher_model(teacher)

    variant_policy = _student_variant_policy()
    pruning_schedule = _build_pruning_warmup_schedule(
        args.max_epochs_student,
        args.warmup_pruning_epochs,
        variant_policy,
    )
    student_pruning_epoch_config = _build_student_pruning_epoch_config(args.max_epochs_student, pruning_schedule)

    student = _build_student_from_blueprint(db_train, pruning_artifact).to(device)
    _configure_student_variant(student, variant_policy)
    pruning_initialization = _resolve_student_from_pruning_checkpoint(student, pruning_artifact, device)
    _configure_student_variant(student, variant_policy)

    student_model_info = extract_model_info(student)
    student_model_info.update(
        {
            "branch": "proposal",
            "teacher_model": args.teacher_model,
            "teacher_backbone_name": teacher_artifact["metadata"].get("backbone_name"),
            "student_name": (
                _middle_student_name(f"student_{variant_policy['variant_name']}")
                if _uses_middle_pruned_resnet_student()
                else f"gated_student_{variant_policy['variant_name']}"
            ),
            "blueprint_path": project_relative_path(pruning_artifact["blueprint_path"], PROJECT_ROOT),
            "blueprint": pruning_artifact["blueprint"],
            **_pruning_metadata(),
            "student_variant": variant_policy,
            "distillation_target": "logits",
            "soft_pruning_definition": (
                f"{_strategy_label()} already applies structural middle-channel pruning inside ResNet bottlenecks; gate-based soft pruning is disabled."
                if _uses_middle_pruned_resnet_student()
                else "Soft pruning is implemented through learnable channel gates plus sparsity regularization during the active pruning epochs."
            ),
            "hard_pruning_definition": (
                f"{_strategy_label()} keeps ResNet bottleneck boundary outputs full, so late PDG hard pruning is disabled for this architecture."
                if _uses_middle_pruned_resnet_student()
                else (
                    "Late hard pruning is applied at the beginning of the final pruning window. The student is rebuilt with fewer channels based on gate values, "
                    "then the compact student continues distillation for the remaining epochs."
                )
            ),
            "step3_distillation_definition": (
                "When the selected student variant enables distillation, the frozen teacher supervises the student across the whole step-3 run. "
                "The final pruning window controls when structural hard pruning happens."
            ),
            "pruning_schedule": pruning_schedule,
            "student_pruning_epoch_config": student_pruning_epoch_config,
            "student_initialization": {
                "source_phase": "2_pruning",
                "source_checkpoint_path": project_relative_path(pruning_initialization["checkpoint_path"], PROJECT_ROOT),
            },
            "checkpoint_is_random_init": False,
            "checkpoint_weight_status": "loaded_from_pruning_checkpoint_then_trained",
            "checkpoint_weight_source": project_relative_path(pruning_initialization["checkpoint_path"], PROJECT_ROOT),
            "build_kwargs": {
                "in_channels": db_train.in_channels,
                "num_classes": args.num_classes,
                "channel_config": list(pruning_artifact["blueprint"]["channel_config"]),
                "stage_middle_channel_config": pruning_artifact["blueprint"].get("stage_middle_channel_config"),
            },
        }
    )
    _write_json(layout["configs_dir"] / "student_pruning_config.json", student_pruning_epoch_config)
    write_model_config(run_dir, student_model_info)
    logging.info(
        "Student step-3 setup | variant=%s | step3_pruning_enabled=%s | step3_pruning_epochs=%s | use_distill=%s | use_gating=%s | use_sparsity=%s | gate_search_epochs=%s | hard_pruning_start_epoch=%s | hard_pruning_threshold=%s",
        variant_policy["variant_name"],
        int(bool(args.enable_step3_pruning)),
        int(args.step3_pruning_epochs),
        int(variant_policy["use_distill"]),
        int(variant_policy["use_gating"]),
        int(variant_policy["use_sparsity"]),
        pruning_schedule["active_soft_pruning_epochs"],
        pruning_schedule.get("hard_pruning_start_epoch"),
        pruning_schedule.get("hard_pruning_threshold"),
    )
    student_input_analysis = _export_model_channel_analysis(
        run_dir,
        student,
        checkpoint_path=None,
        device=device,
        prefix="student_input",
        title="Student Input Architecture Before Tuning",
    )
    input_gating_paths = save_gating_analysis_artifacts(
        run_dir / "artifacts" / "gating_analysis",
        student_input_analysis,
        prefix="student_input",
        title="Student Input Gating Analysis",
    )
    history = {}
    reusable_match = None
    diagnostics_rows = []
    hard_pruning_plan = None
    hard_pruning_applied = False
    hard_pruning_weight_transfer = None
    if not bool(args.force_retrain_student):
        reusable_signature = build_expected_signature(
            dataset=args.dataset,
            model_name=_student_model_name(),
            num_classes=args.num_classes,
            in_channels=db_train.in_channels,
            patch_size=args.patch_size,
            teacher_model=args.teacher_model,
            student_variant=args.student_variant,
            step3_pruning_enabled=bool(args.enable_step3_pruning),
            step3_pruning_epochs=int(args.step3_pruning_epochs),
        )
        reusable_match = resolve_run_checkpoint(
            run_dir,
            expected_signature=reusable_signature,
        )
    if reusable_match is not None:
        reusable_signature = reusable_match.get("compatibility", {}).get("actual_signature", {})
        reusable_channel_config = reusable_signature.get("channel_config")
        if reusable_channel_config:
            reusable_channel_config = tuple(int(channel) for channel in reusable_channel_config)
            if reusable_channel_config != tuple(student.channel_config) and not _uses_middle_pruned_resnet_student():
                student = PDGUNet(
                    in_channels=db_train.in_channels,
                    num_classes=args.num_classes,
                    channel_config=reusable_channel_config,
                ).to(device)
        best_path = Path(reusable_match["checkpoint_path"])
        payload = load_checkpoint_into_model(best_path, student, device=device)
        _configure_student_variant(student, variant_policy)
        history = payload.get("extra_state", {}).get("history", {})
        diagnostics_rows = list(payload.get("extra_state", {}).get("epoch_diagnostics", []))
        hard_pruning_plan = payload.get("extra_state", {}).get("hard_pruning_plan")
        hard_pruning_applied = bool(payload.get("extra_state", {}).get("hard_pruning_applied", False))
        hard_pruning_weight_transfer = payload.get("extra_state", {}).get("hard_pruning_weight_transfer")
        student_model_info.update(payload.get("model_info", {}))
        logging.info("Loaded student checkpoint: %s", project_relative_path(best_path, PROJECT_ROOT))
        _restore_loss_pdf_if_possible(history, layout["reports_dir"] / "student_loss.pdf", "Student loss")
    else:
        criterion = CompressionLoss(
            args.num_classes,
            args.lambda_distill,
            args.lambda_sparsity,
            use_kd_output=args.use_kd_output,
            use_sparsity=args.use_sparsity,
            use_feature_distill=args.use_feature_distill,
            use_aux_loss=args.use_aux_loss,
            lambda_feat=args.lambda_feat,
            lambda_aux=args.lambda_aux,
            feature_layers=args.feature_layers,
        )
        optimizer = optim.AdamW(student.parameters(), lr=args.student_lr, weight_decay=1e-4)
        history = {
            "train_total_loss": [],
            "train_segmentation_loss": [],
            "train_distillation_loss": [],
            "train_sparsity_loss": [],
            "train_feature_distill_loss": [],
            "train_auxiliary_loss": [],
            "val_total_loss": [],
            "val_segmentation_loss": [],
            "val_distillation_loss": [],
            "val_sparsity_loss": [],
            "val_feature_distill_loss": [],
            "val_auxiliary_loss": [],
            "val_macro_dice": [],
            "epoch_gate_mean": [],
            "epoch_gate_near_off_ratio": [],
            "epoch_lambda_distill": [],
            "epoch_lambda_sparsity": [],
            "epoch_lambda_feat": [],
            "epoch_lambda_aux": [],
        }
        best_metric = float("-inf")
        best_path = None
        for epoch in tqdm(range(1, args.max_epochs_student + 1), desc="student", ncols=90):
            epoch_policy = _student_epoch_policy(epoch, pruning_schedule, variant_policy)
            if epoch_policy["hard_pruning_active"] and not hard_pruning_applied:
                hard_pruning_plan = _build_late_hard_pruning_plan(student, float(pruning_schedule["hard_pruning_threshold"]))
                if hard_pruning_plan["hard_pruning_applied"]:
                    hard_pruning_weight_transfer = _summarize_hard_pruning_weight_transfer(hard_pruning_plan)
                    student = _build_hard_pruned_student_from_plan(student, db_train, hard_pruning_plan)
                    _configure_student_variant(student, variant_policy)
                    optimizer = optim.AdamW(student.parameters(), lr=args.student_lr, weight_decay=1e-4)
                    hard_pruning_applied = True
                    best_metric = float("-inf")
                    best_path = None
                    student_model_info = extract_model_info(student)
                    student_model_info.update(
                        {
                            "branch": "proposal",
                            "teacher_model": args.teacher_model,
                            "teacher_backbone_name": teacher_artifact["metadata"].get("backbone_name"),
                            "student_name": f"gated_student_{variant_policy['variant_name']}",
                            "blueprint_path": project_relative_path(pruning_artifact["blueprint_path"], PROJECT_ROOT),
                            "blueprint": pruning_artifact["blueprint"],
                            **_pruning_metadata(),
                            "student_variant": variant_policy,
                            "distillation_target": "logits",
                            "soft_pruning_definition": "Soft pruning is implemented through learnable channel gates plus sparsity regularization during the active gate-search epochs.",
                            "hard_pruning_definition": (
                                "Late hard pruning was applied in step 3 by rebuilding the student with channels kept above the configured gate threshold, "
                                "then continuing distillation on the compact student."
                            ),
                            "step3_distillation_definition": (
                                "When the selected student variant enables distillation, the frozen teacher supervises the student across the whole step-3 run, "
                                "including the post-hard-pruning epochs."
                            ),
                            "pruning_schedule": pruning_schedule,
                            "student_pruning_epoch_config": student_pruning_epoch_config,
                            "student_initialization": {
                                "source_phase": "2_pruning",
                                "source_checkpoint_path": project_relative_path(pruning_initialization["checkpoint_path"], PROJECT_ROOT),
                            },
                            "hard_pruning_applied": True,
                            "hard_pruning_plan": hard_pruning_plan,
                            "hard_pruning_weight_transfer": hard_pruning_weight_transfer,
                            "checkpoint_is_random_init": False,
                            "checkpoint_weight_status": "subset_reused_from_pre_pruning_student_then_trained",
                            "checkpoint_weight_source": "step3_pre_hard_pruning_student",
                            "build_kwargs": {
                                "in_channels": db_train.in_channels,
                                "num_classes": args.num_classes,
                                "channel_config": list(hard_pruning_plan["channel_config"]),
                            },
                        }
                    )
                    write_model_config(run_dir, student_model_info)
                    logging.info(
                        "Applied late hard pruning at epoch %d | threshold=%.4f | channel_config=%s | global_prune_ratio=%.4f | weight_init=%s | channel_reuse_ratio=%.4f",
                        epoch,
                        hard_pruning_plan["threshold"],
                        list(hard_pruning_plan["channel_config"]),
                        hard_pruning_plan["global_summary"]["global_prune_ratio"],
                        hard_pruning_weight_transfer["strategy"],
                        hard_pruning_weight_transfer["channel_reuse_ratio"],
                    )
                else:
                    logging.info(
                        "Late hard pruning checkpoint reached at epoch %d, but no channel satisfied the structural prune condition. Training continues without rebuild.",
                        epoch,
                    )
            if variant_policy["use_gating"]:
                student.set_gate_trainable(epoch_policy["gate_trainable"])
            else:
                student.force_gates_open(args.student_gate_open_value)
                student.set_gate_trainable(False)

            student.train()
            tracked = {
                key: []
                for key in (
                    "total_loss",
                    "segmentation_loss",
                    "distillation_loss",
                    "sparsity_loss",
                    "feature_distill_loss",
                    "auxiliary_loss",
                )
            }
            for batch in trainloader:
                _validate_label_batch(batch["label"], args.num_classes, batch)
                image = batch["image"].to(device)
                label = batch["label"].to(device)
                teacher_output = None
                if epoch_policy["needs_teacher_output"]:
                    with torch.no_grad():
                        teacher_output = teacher(image)
                loss_dict = criterion(
                    student(image),
                    teacher_output,
                    student.get_gate_tensors(),
                    label,
                    lambda_distill=epoch_policy["lambda_distill"],
                    lambda_sparsity=epoch_policy["lambda_sparsity"],
                    lambda_feat=epoch_policy["lambda_feat"],
                    lambda_aux=epoch_policy["lambda_aux"],
                )
                optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                optimizer.step()
                for key in tracked:
                    tracked[key].append(float(loss_dict[key].item()))
            val_metrics = evaluate_segmentation_dataset(
                student,
                valloader,
                device=device,
                num_classes=args.num_classes,
                patch_size=args.patch_size,
                output_dir=None,
                save_visualizations=False,
                vis_limit=0,
                sample_validator=_validate_label_batch,
            )
            val_losses = _compute_student_val_losses(student, teacher, valloader, device, criterion, epoch_policy=epoch_policy)
            val_dice = float(np.mean(val_metrics["average_metric"][:, 0]))
            gate_stats = _summarize_student_gates(student, args.student_gate_near_off_threshold)
            history["train_total_loss"].append(float(np.mean(tracked["total_loss"])) if tracked["total_loss"] else 0.0)
            history["train_segmentation_loss"].append(float(np.mean(tracked["segmentation_loss"])) if tracked["segmentation_loss"] else 0.0)
            history["train_distillation_loss"].append(float(np.mean(tracked["distillation_loss"])) if tracked["distillation_loss"] else 0.0)
            history["train_sparsity_loss"].append(float(np.mean(tracked["sparsity_loss"])) if tracked["sparsity_loss"] else 0.0)
            history["train_feature_distill_loss"].append(float(np.mean(tracked["feature_distill_loss"])) if tracked["feature_distill_loss"] else 0.0)
            history["train_auxiliary_loss"].append(float(np.mean(tracked["auxiliary_loss"])) if tracked["auxiliary_loss"] else 0.0)
            history["val_total_loss"].append(val_losses["total_loss"])
            history["val_segmentation_loss"].append(val_losses["segmentation_loss"])
            history["val_distillation_loss"].append(val_losses["distillation_loss"])
            history["val_sparsity_loss"].append(val_losses["sparsity_loss"])
            history["val_feature_distill_loss"].append(val_losses["feature_distill_loss"])
            history["val_auxiliary_loss"].append(val_losses["auxiliary_loss"])
            history["val_macro_dice"].append(val_dice)
            history["epoch_gate_mean"].append(gate_stats["global"]["gate_mean"])
            history["epoch_gate_near_off_ratio"].append(gate_stats["global"]["near_off_ratio"])
            history["epoch_lambda_distill"].append(epoch_policy["lambda_distill"])
            history["epoch_lambda_sparsity"].append(epoch_policy["lambda_sparsity"])
            history["epoch_lambda_feat"].append(epoch_policy["lambda_feat"])
            history["epoch_lambda_aux"].append(epoch_policy["lambda_aux"])
            diagnostics_rows.append(
                {
                    "epoch": epoch,
                    "phase_name": epoch_policy["phase_name"],
                    "soft_pruning_active": int(epoch_policy["soft_pruning_active"]),
                    "hard_pruning_active": int(epoch_policy["hard_pruning_active"]),
                    "distillation_active": int(epoch_policy["lambda_distill"] > 0),
                    "feature_distill_active": int(epoch_policy["lambda_feat"] > 0),
                    "auxiliary_loss_active": int(epoch_policy["lambda_aux"] > 0),
                    "gate_trainable": int(epoch_policy["gate_trainable"]),
                    "lambda_distill": epoch_policy["lambda_distill"],
                    "lambda_sparsity": epoch_policy["lambda_sparsity"],
                    "lambda_feat": epoch_policy["lambda_feat"],
                    "lambda_aux": epoch_policy["lambda_aux"],
                    "train_total_loss": history["train_total_loss"][-1],
                    "train_segmentation_loss": history["train_segmentation_loss"][-1],
                    "train_distillation_loss": history["train_distillation_loss"][-1],
                    "train_sparsity_loss": history["train_sparsity_loss"][-1],
                    "train_feature_distill_loss": history["train_feature_distill_loss"][-1],
                    "train_auxiliary_loss": history["train_auxiliary_loss"][-1],
                    "val_total_loss": val_losses["total_loss"],
                    "val_segmentation_loss": val_losses["segmentation_loss"],
                    "val_distillation_loss": val_losses["distillation_loss"],
                    "val_sparsity_loss": val_losses["sparsity_loss"],
                    "val_feature_distill_loss": val_losses["feature_distill_loss"],
                    "val_auxiliary_loss": val_losses["auxiliary_loss"],
                    "val_macro_dice": val_dice,
                    "gate_mean": gate_stats["global"]["gate_mean"],
                    "gate_std": gate_stats["global"]["gate_std"],
                    "near_off_channels": gate_stats["global"]["near_off_channels"],
                    "total_gate_channels": gate_stats["global"]["total_gate_channels"],
                    "near_off_ratio": gate_stats["global"]["near_off_ratio"],
                }
            )
            is_best = val_dice > best_metric
            if is_best:
                best_metric = val_dice
            checkpoint_path = save_checkpoint(
                run_dir,
                f"epoch_{epoch:03d}",
                model=student,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                metrics={
                    "val_macro_dice": val_dice,
                    "val_total_loss": val_losses["total_loss"],
                    "val_segmentation_loss": val_losses["segmentation_loss"],
                    "val_distillation_loss": val_losses["distillation_loss"],
                    "val_sparsity_loss": val_losses["sparsity_loss"],
                    "val_feature_distill_loss": val_losses["feature_distill_loss"],
                    "val_auxiliary_loss": val_losses["auxiliary_loss"],
                },
                config=vars(args),
                model_info=student_model_info,
                phase="student",
                extra_state={
                    "history": history,
                    "blueprint": pruning_artifact["blueprint"],
                    "epoch_diagnostics": diagnostics_rows,
                    "student_variant": variant_policy,
                    "pruning_schedule": pruning_schedule,
                    "hard_pruning_plan": hard_pruning_plan,
                    "hard_pruning_applied": hard_pruning_applied,
                    "hard_pruning_weight_transfer": hard_pruning_weight_transfer,
                },
                is_best=is_best,
                save_tagged_checkpoint=bool(args.save_history_checkpoints),
                save_last_checkpoint=bool(args.save_last_checkpoint),
                include_optimizer_state=bool(args.save_optimizer_state),
                project_root=PROJECT_ROOT,
            )
            if is_best:
                best_path = checkpoint_path
            logging.info(
                "Student epoch %d/%d | phase=%s | train_total=%.6f | val_total=%.6f | val_dice=%.6f | gate_mean=%.4f | near_off_ratio=%.4f | lambda_distill=%.4f | lambda_sparsity=%.4f",
                epoch,
                args.max_epochs_student,
                epoch_policy["phase_name"],
                history["train_total_loss"][-1],
                val_losses["total_loss"],
                val_dice,
                gate_stats["global"]["gate_mean"],
                gate_stats["global"]["near_off_ratio"],
                epoch_policy["lambda_distill"],
                epoch_policy["lambda_sparsity"],
            )
        if best_path is None:
            best_path = resolve_phase_checkpoint(run_dir, "last")
        save_loss_pdf(history, layout["reports_dir"] / "student_loss.pdf", title="Student loss")
    _save_student_epoch_diagnostics(run_dir, diagnostics_rows)
    write_model_config(run_dir, student_model_info)
    profile = _compute_profile(student, device)
    evaluation_bundle = _export_phase_outputs(
        run_dir,
        student,
        best_path,
        "student",
        profile,
        {
            "model_name": _student_model_name(),
            "backbone_name": teacher_artifact["metadata"].get("backbone_name"),
            "student_name": student_model_info.get("student_name"),
            "teacher_model": args.teacher_model,
            **_pruning_metadata(),
        },
        device,
        image_mode,
    )
    student_final_analysis = _export_model_channel_analysis(
        run_dir,
        student,
        checkpoint_path=best_path,
        device=device,
        prefix="student_final",
        title="Student Final Architecture After Tuning",
    )
    final_gating_paths = save_gating_analysis_artifacts(
        run_dir / "artifacts" / "gating_analysis",
        student_final_analysis,
        prefix="student_final",
        title="Student Final Gating Analysis",
    )
    student_comparison_rows = build_analysis_comparison(
        student_input_analysis,
        student_final_analysis,
        before_label="input",
        after_label="final",
    )
    save_comparison_artifacts(
        run_dir / "artifacts" / "student_tuning_analysis",
        student_comparison_rows,
        prefix="student_tuning_comparison",
        title="Student Tuning Comparison",
        extra_report={
            "global_summary": {
                "input_channel_layers": student_input_analysis["global_summary"]["num_channel_layers"],
                "final_channel_layers": student_final_analysis["global_summary"]["num_channel_layers"],
                "input_total_channels": student_input_analysis["global_summary"]["total_output_channels"],
                "final_total_channels": student_final_analysis["global_summary"]["total_output_channels"],
                "input_gate_layers": student_input_analysis["global_summary"]["num_gate_layers"],
                "final_gate_layers": student_final_analysis["global_summary"]["num_gate_layers"],
                "step3_pruning_enabled": bool(args.enable_step3_pruning),
                "step3_pruning_epochs": int(args.step3_pruning_epochs),
                "soft_pruning_threshold": args.student_gate_near_off_threshold,
                "hard_pruning_threshold": pruning_schedule.get("hard_pruning_threshold"),
                "requested_warmup_pruning_epochs": pruning_schedule["requested_warmup_pruning_epochs"],
                "effective_warmup_pruning_epochs": pruning_schedule["effective_warmup_pruning_epochs"],
                "hard_pruning_applied": int(hard_pruning_applied),
                "hard_pruning_start_epoch": pruning_schedule.get("hard_pruning_start_epoch"),
            }
        },
    )
    if hard_pruning_plan is not None:
        _write_json(run_dir / "artifacts" / "student_tuning_analysis" / "late_hard_pruning_plan.json", hard_pruning_plan)
        if hard_pruning_plan.get("rows"):
            write_metrics_rows(
                hard_pruning_plan["rows"],
                run_dir / "artifacts" / "student_tuning_analysis" / "late_hard_pruning_plan.csv",
            )
    gating_report_payload = {
        "global_summary": {
            "student_variant": variant_policy["variant_name"],
            "student_variant_description": variant_policy["description"],
            "step3_pruning_enabled": bool(args.enable_step3_pruning),
            "step3_pruning_epochs": int(args.step3_pruning_epochs),
            "soft_pruning_threshold": args.student_gate_near_off_threshold,
            "hard_pruning_threshold": pruning_schedule.get("hard_pruning_threshold"),
            "hard_pruning_applied": bool(hard_pruning_applied),
            "hard_pruning_stage": "late_structural_pruning" if hard_pruning_applied else "not_triggered",
            "hard_pruning_weight_init": (
                hard_pruning_weight_transfer.get("strategy")
                if isinstance(hard_pruning_weight_transfer, dict)
                else "not_applied"
            ),
            "hard_pruning_channel_reuse_ratio": (
                hard_pruning_weight_transfer.get("channel_reuse_ratio")
                if isinstance(hard_pruning_weight_transfer, dict)
                else None
            ),
            "requested_warmup_pruning_epochs": pruning_schedule["requested_warmup_pruning_epochs"],
            "effective_warmup_pruning_epochs": pruning_schedule["effective_warmup_pruning_epochs"],
            "active_soft_pruning_epochs": pruning_schedule["active_soft_pruning_epochs"],
            "late_hard_pruning_epochs": pruning_schedule.get("late_hard_pruning_epochs"),
            "hard_pruning_start_epoch": pruning_schedule.get("hard_pruning_start_epoch"),
            "policy_note": pruning_schedule["policy_note"],
        },
        "gate_summary_rows": list(student_final_analysis.get("gate_summary_rows", [])),
        "comparison_rows": list(student_comparison_rows),
        "pruning_summary_rows": list(hard_pruning_plan.get("rows", [])) if hard_pruning_plan else [],
    }
    save_channel_analysis_pdf(
        gating_report_payload,
        layout["reports_dir"] / "student_channel_gating_report.pdf",
        title="Student Channel Gating Report",
    )
    student_final_exports = _export_student_final_shortcuts(run_dir, _proposal_root_dir())
    return {
        "model": student,
        "run_dir": run_dir,
        "checkpoint_path": best_path,
        "metadata": student_model_info,
        "student_input_analysis": student_input_analysis,
        "student_final_analysis": student_final_analysis,
        "student_final_exports": student_final_exports,
        "evaluation_bundle": evaluation_bundle,
        "phase": "student",
        "input_gating_paths": input_gating_paths,
        "final_gating_paths": final_gating_paths,
        "pruning_schedule": pruning_schedule,
        "student_variant": variant_policy,
        "hard_pruning_plan": hard_pruning_plan,
        "hard_pruning_applied": hard_pruning_applied,
        "hard_pruning_weight_transfer": hard_pruning_weight_transfer,
        "student_initialization": {
            "source_checkpoint_path": project_relative_path(pruning_initialization["checkpoint_path"], PROJECT_ROOT),
            "source_phase": "2_pruning",
        },
    }


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_mode = "grayscale" if args.in_channels == 1 else "rgb"

    proposal_root_dir = _proposal_root_dir()
    proposal_root_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir = _phase_dir("pipeline")
    ensure_run_layout(pipeline_dir)
    write_run_config(pipeline_dir, vars(args))
    _configure_logging(pipeline_dir / "run.log")
    logging.info("PDG pipeline args: %s", args)
    logging.info("Pruning strategy: %s", args.prune_method)
    if uses_static_prune_ratio(args.prune_method):
        logging.info("Static prune ratio: %s", _format_float_for_path(args.static_prune_ratio))
    logging.info("Step-3 pruning: %s", "enabled" if bool(args.enable_step3_pruning) else "disabled")
    logging.info("Step-3 pruning epochs: %s", int(args.step3_pruning_epochs) if bool(args.enable_step3_pruning) else "no")
    logging.info("Output dir: %s", project_relative_path(proposal_root_dir, PROJECT_ROOT))
    logging.info("Teacher output root: %s", args.teacher_output_root or args.output_root or "outputs")
    logging.info("Teacher run dir: %s", project_relative_path(_phase_dir("teacher"), PROJECT_ROOT))

    db_train, db_val, trainloader, valloader = _build_loaders(device, image_mode)
    logging.info("Dataset summary | train=%d | val=%d", len(db_train), len(db_val))

    teacher_artifact = _run_teacher(device, image_mode, db_train, trainloader, valloader)
    pruning_artifact = _run_pruning(device, image_mode, db_train, teacher_artifact)
    student_artifact = _run_student(device, image_mode, db_train, trainloader, valloader, teacher_artifact, pruning_artifact)
    pipeline_export = _save_pipeline_outputs(pipeline_dir, teacher_artifact, pruning_artifact, student_artifact)

    _write_json(
        pipeline_dir / "pipeline_summary.json",
        {
            "proposal_root_dir": project_relative_path(proposal_root_dir, PROJECT_ROOT),
            "teacher_root_dir": project_relative_path(_teacher_root_dir(), PROJECT_ROOT),
            **_pruning_metadata(),
            "teacher_checkpoint": project_relative_path(teacher_artifact["checkpoint_path"], PROJECT_ROOT),
            "teacher_model_info": teacher_artifact["metadata"],
            "pruning_blueprint_path": project_relative_path(pruning_artifact["blueprint_path"], PROJECT_ROOT),
            "pruning_blueprint": pruning_artifact["blueprint"],
            "pruning_checkpoint": project_relative_path(pruning_artifact["checkpoint_path"], PROJECT_ROOT),
            "pruning_model_info": pruning_artifact["metadata"],
            "student_checkpoint": project_relative_path(student_artifact["checkpoint_path"], PROJECT_ROOT),
            "student_model_info": student_artifact["metadata"],
            "teacher_run_dir": project_relative_path(teacher_artifact["run_dir"], PROJECT_ROOT),
            "pruning_run_dir": project_relative_path(pruning_artifact["run_dir"], PROJECT_ROOT),
            "student_run_dir": project_relative_path(student_artifact["run_dir"], PROJECT_ROOT),
            "student_final_exports": student_artifact.get("student_final_exports", {}),
            "teacher_reused_from": teacher_artifact.get("reused_from"),
            "pipeline_evaluation": pipeline_export.get("summary", {}),
        },
    )
    logging.info("PDG pipeline completed.")

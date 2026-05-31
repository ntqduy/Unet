from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch

from networks.PGD_Unet.blueprint_unet_plus_plus import build_blueprint_unet_plus_plus
from networks.PGD_Unet.full_pruning_unet_plus_plus import build_full_pruning_unet_plus_plus
from networks.PGD_Unet.gated_unet import PDGUNet
from networks.PGD_Unet.middle_pruned_unet_plus_plus import build_middle_pruned_unet_plus_plus
from networks.net_factory import list_models, net_factory
from utils.checkpoints import load_checkpoint
from utils.experiment import sanitize_tag
from utils.profiling import count_parameters, maybe_compute_flops


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = Path(
    "/data2/Medical/StrokeCT-MultiModal/Multi_Modal/Stroke_MultiModal_Predict/segmentation/Unet/outputs"
)
DEFAULT_FLOPS_DIR = PROJECT_ROOT / "flops"
DEFAULT_PATCH_SIZE = (256, 256)
EXCLUDED_FOLDERS = {"pgd_unet_v1"}
BASIC_MODEL_NAMES = set(list_models())
PGD_MODEL_NAMES = {
    "pdg_unet",
    "pgd_unet",
    "gated_unet",
    "blueprint_unet_plus_plus",
    "middle_pruned_resnet_unet",
    "middle_pruned_unet_plus_plus",
    "full_pruning_resnet_unet",
    "full_pruning_unet_plus_plus",
}
MIDDLE_PRUNED_MODEL_NAMES = {"middle_pruned_resnet_unet", "middle_pruned_unet_plus_plus"}
FULL_PRUNED_MODEL_NAMES = {"full_pruning_resnet_unet", "full_pruning_unet_plus_plus"}
CSV_FIELDS = [
    "branch",
    "dataset",
    "model",
    "architecture",
    "loss",
    "method",
    "phase",
    "checkpoint_path",
    "input_shape",
    "params",
    "trainable_params",
    "flops",
    "flops_g",
    "backbone_name",
    "student_name",
    "status",
    "error",
]


@dataclass(frozen=True)
class CheckpointPathInfo:
    branch: str
    dataset: str
    model: str
    loss: str
    method: str
    phase: str
    csv_group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute FLOPs for every best.pth under an outputs directory and export "
            "flops/basic_model.csv plus one flops/loss_*.csv per PGD loss folder."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="root outputs directory to scan")
    parser.add_argument("--flops-dir", type=Path, default=DEFAULT_FLOPS_DIR, help="directory for generated CSV files")
    parser.add_argument("--checkpoint-name", type=str, default="best.pth", help="checkpoint filename to scan")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--patch-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="override input size used for FLOPs; default comes from checkpoint config or 256 256",
    )
    parser.add_argument("--strict-load", type=int, default=1, choices=[0, 1], help="load checkpoint weights with strict=True")
    parser.add_argument("--load-weights", type=int, default=1, choices=[0, 1], help="load best.pth weights before profiling")
    parser.add_argument(
        "--cache-architectures",
        type=int,
        default=1,
        choices=[0, 1],
        help="reuse FLOPs for checkpoints with identical state_dict tensor shapes",
    )
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1], help="print progress for each checkpoint")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def path_has_excluded_folder(path: Path) -> bool:
    return any(part in EXCLUDED_FOLDERS for part in path.parts)


def iter_checkpoint_paths(output_root: Path, checkpoint_name: str) -> Iterable[Path]:
    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")
    for checkpoint_path in sorted(output_root.rglob(checkpoint_name)):
        if checkpoint_path.is_file() and not path_has_excluded_folder(checkpoint_path):
            yield checkpoint_path


def strip_teacher_suffix(value: str) -> str:
    return value[: -len("_teacher")] if value.endswith("_teacher") else value


def parse_checkpoint_path(checkpoint_path: Path, output_root: Path) -> CheckpointPathInfo:
    try:
        parts = checkpoint_path.relative_to(output_root).parts
    except ValueError:
        parts = checkpoint_path.parts

    if len(parts) >= 2 and parts[0] == "pgd_unet":
        dataset = parts[1] if len(parts) > 1 else ""
        teacher_dir = parts[2] if len(parts) > 2 else ""
        model = strip_teacher_suffix(teacher_dir)
        loss = ""
        method = ""
        phase = ""
        if len(parts) > 3:
            if parts[3] in {"1_teacher", "2_pruning", "3_student", "pipeline", "student_final"}:
                phase = parts[3]
            else:
                loss = parts[3]
                method = parts[4] if len(parts) > 4 else ""
                phase = parts[5] if len(parts) > 5 else ""
        csv_group = loss if loss else "pgd_teacher"
        return CheckpointPathInfo(
            branch="pgd",
            dataset=dataset,
            model=model,
            loss=loss,
            method=method,
            phase=phase,
            csv_group=csv_group,
        )

    model = parts[0] if len(parts) > 0 else ""
    dataset = parts[1] if len(parts) > 1 else ""
    return CheckpointPathInfo(
        branch="basic",
        dataset=dataset,
        model=model,
        loss="basic_model",
        method="basic",
        phase="basic",
        csv_group="basic_model",
    )


def as_dict(value: Any) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def get_nested(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def first_value(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None and not (isinstance(value, str) and value == ""):
            return value
    return default


def to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def to_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def to_int_tuple(value: Any, *, length: int | None = None) -> tuple[int, ...] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        values = [item for item in value.replace(",", " ").split() if item]
    else:
        try:
            values = list(value)
        except TypeError:
            return None
    try:
        result = tuple(int(item) for item in values)
    except (TypeError, ValueError):
        return None
    if length is not None and len(result) != length:
        return None
    return result


def clean_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in dict(state_dict).items():
        suffix = str(key).rsplit(".", 1)[-1]
        if suffix in {"total_ops", "total_params"}:
            continue
        cleaned[str(key)] = value
    return cleaned


def checkpoint_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        return clean_state_dict(as_dict(payload["model_state_dict"]))
    return clean_state_dict(as_dict(payload))


def tensor_shape_from_state(state_dict: Mapping[str, Any], key: str) -> tuple[int, ...] | None:
    value = state_dict.get(key)
    if torch.is_tensor(value):
        return tuple(int(dim) for dim in value.shape)
    return None


def infer_num_classes(state_dict: Mapping[str, Any], default: int = 2) -> int:
    for key in (
        "head.weight",
        "base_model.head.weight",
        "model.segmentation_head.0.weight",
        "out_conv.weight",
        "outc.conv.weight",
    ):
        shape = tensor_shape_from_state(state_dict, key)
        if shape:
            return int(shape[0])
    for key, value in state_dict.items():
        if key.endswith(".weight") and torch.is_tensor(value) and value.ndim == 4 and value.shape[0] <= 16:
            if any(token in key for token in ("head", "out", "segmentation_head", "final")):
                return int(value.shape[0])
    return int(default)


def infer_in_channels(state_dict: Mapping[str, Any], default: int = 3) -> int:
    for key in (
        "stem.conv.block.0.block.0.weight",
        "stem.block.0.block.0.weight",
        "stem.0.weight",
        "base_model.stem.0.weight",
        "model.encoder.conv1.weight",
    ):
        shape = tensor_shape_from_state(state_dict, key)
        if shape and len(shape) >= 2:
            return int(shape[1])
    return int(default)


def infer_channel_config_from_state(state_dict: Mapping[str, Any]) -> tuple[int, int, int, int, int] | None:
    keys = (
        "stem.conv.block.1.block.0.weight",
        "down1.conv.conv.block.1.block.0.weight",
        "down2.conv.conv.block.1.block.0.weight",
        "down3.conv.conv.block.1.block.0.weight",
        "down4.conv.conv.block.1.block.0.weight",
    )
    values: list[int] = []
    for key in keys:
        shape = tensor_shape_from_state(state_dict, key)
        if not shape:
            return None
        values.append(int(shape[0]))
    return tuple(values)  # type: ignore[return-value]


def architecture_cache_key(payload: Any, input_shape: tuple[int, ...]) -> str:
    state_dict = checkpoint_state_dict(payload)
    shape_rows = []
    for key, value in sorted(state_dict.items()):
        if torch.is_tensor(value):
            shape_rows.append((key, tuple(int(dim) for dim in value.shape)))
    raw = repr((tuple(input_shape), shape_rows)).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def find_json_file(candidates: Iterable[Path]) -> dict:
    for path in candidates:
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as file:
                    return json.load(file)
            except (json.JSONDecodeError, OSError):
                continue
    return {}


def find_blueprint_near_checkpoint(checkpoint_path: Path) -> dict:
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    proposal_root = run_dir.parent if run_dir.name in {"2_pruning", "3_student"} else run_dir
    return find_json_file(
        [
            run_dir / "artifacts" / "blueprint.json",
            run_dir / "configs" / "pruning_config.json",
            run_dir / "metrics" / "pruning_summary.json",
            proposal_root / "2_pruning" / "artifacts" / "blueprint.json",
            proposal_root / "2_pruning" / "configs" / "pruning_config.json",
            proposal_root / "2_pruning" / "metrics" / "pruning_summary.json",
        ]
    )


def extract_blueprint(payload: Mapping[str, Any], checkpoint_path: Path, model_info: Mapping[str, Any]) -> dict:
    for blueprint in (
        get_nested(payload, "extra_state", "blueprint"),
        model_info.get("blueprint"),
        get_nested(model_info, "build_kwargs", "blueprint"),
    ):
        if isinstance(blueprint, Mapping) and blueprint:
            return dict(blueprint)
    return find_blueprint_near_checkpoint(checkpoint_path)


def checkpoint_config(payload: Mapping[str, Any]) -> dict:
    return as_dict(payload.get("config", {}))


def checkpoint_model_info(payload: Mapping[str, Any]) -> dict:
    return as_dict(payload.get("model_info", {}))


def resolve_patch_size(config: Mapping[str, Any], model_info: Mapping[str, Any], override: Optional[tuple[int, int]]) -> tuple[int, int]:
    if override is not None:
        return override
    architecture_config = as_dict(model_info.get("architecture_config", {}))
    value = first_value(
        config.get("patch_size"),
        architecture_config.get("patch_size"),
        architecture_config.get("image_size"),
        get_nested(model_info, "build_kwargs", "patch_size"),
        default=DEFAULT_PATCH_SIZE,
    )
    resolved = to_int_tuple(value, length=2)
    return resolved if resolved is not None else DEFAULT_PATCH_SIZE


def resolve_model_name(path_info: CheckpointPathInfo, model_info: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    return str(
        first_value(
            model_info.get("model_name"),
            model_info.get("architecture"),
            config.get("model"),
            config.get("teacher_model") if path_info.phase == "1_teacher" else None,
            path_info.model,
            default="",
        )
    ).lower()


def resolve_teacher_model(path_info: CheckpointPathInfo, model_info: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    return str(first_value(config.get("teacher_model"), model_info.get("teacher_model"), path_info.model, default="")).lower()


def resolve_in_channels(
    config: Mapping[str, Any],
    model_info: Mapping[str, Any],
    state_dict: Mapping[str, Any],
) -> int:
    architecture_config = as_dict(model_info.get("architecture_config", {}))
    return to_int(
        first_value(
            config.get("in_channels"),
            architecture_config.get("in_channels"),
            get_nested(model_info, "build_kwargs", "in_channels"),
            default=None,
        ),
        infer_in_channels(state_dict, default=3),
    )


def resolve_num_classes(
    config: Mapping[str, Any],
    model_info: Mapping[str, Any],
    state_dict: Mapping[str, Any],
) -> int:
    architecture_config = as_dict(model_info.get("architecture_config", {}))
    return to_int(
        first_value(
            config.get("num_classes"),
            architecture_config.get("num_classes"),
            get_nested(model_info, "build_kwargs", "num_classes"),
            default=None,
        ),
        infer_num_classes(state_dict, default=2),
    )


def resolve_channel_config(model_info: Mapping[str, Any], blueprint: Mapping[str, Any], state_dict: Mapping[str, Any]) -> tuple[int, ...]:
    architecture_config = as_dict(model_info.get("architecture_config", {}))
    hard_pruning_plan = first_value(
        model_info.get("hard_pruning_plan"),
        get_nested(model_info, "extra_state", "hard_pruning_plan"),
        default=None,
    )
    value = first_value(
        get_nested(hard_pruning_plan or {}, "channel_config"),
        get_nested(model_info, "build_kwargs", "channel_config"),
        architecture_config.get("channel_config"),
        blueprint.get("channel_config") if isinstance(blueprint, Mapping) else None,
        infer_channel_config_from_state(state_dict),
        default=(32, 64, 128, 256, 512),
    )
    resolved = to_int_tuple(value)
    return resolved if resolved is not None else (32, 64, 128, 256, 512)


def blueprint_student_architecture(blueprint: Mapping[str, Any], teacher_model: str, fallback: str = "pdg_unet") -> str:
    architecture = str(blueprint.get("student_architecture", "") or "").lower()
    if architecture:
        return architecture
    prune_method = str(blueprint.get("prune_method", "") or "").lower()
    if prune_method in {"middle_static", "middle_kneedle", "middle_otsu", "middle_gmm"}:
        return "middle_pruned_unet_plus_plus" if teacher_model == "unet_plus_plus" else "middle_pruned_resnet_unet"
    if prune_method in {"full_static", "full_kneedle", "full_otsu", "full_gmm"}:
        return "full_pruning_unet_plus_plus" if teacher_model == "unet_plus_plus" else "full_pruning_resnet_unet"
    if teacher_model == "unet_plus_plus":
        return "blueprint_unet_plus_plus"
    return fallback


def build_basic_model(
    model_name: str,
    *,
    in_channels: int,
    num_classes: int,
    patch_size: tuple[int, int],
    config: Mapping[str, Any],
    model_info: Mapping[str, Any],
):
    architecture_config = as_dict(model_info.get("architecture_config", {}))
    kwargs: dict[str, Any] = {"mode": "test"}
    if model_name == "unetr":
        kwargs["image_size"] = tuple(patch_size)
    if model_name in {"unet_resnet152", "unet_plus_plus"}:
        kwargs["encoder_pretrained"] = False
        if model_name == "unet_plus_plus":
            kwargs["encoder_weights"] = None
    if model_name == "vnet":
        kwargs["normalization"] = first_value(
            config.get("vnet_normalization"),
            architecture_config.get("normalization"),
            default="groupnorm",
        )
        kwargs["has_dropout"] = to_bool(first_value(config.get("vnet_has_dropout"), architecture_config.get("has_dropout")), False)
        kwargs["has_residual"] = to_bool(first_value(config.get("vnet_has_residual"), architecture_config.get("has_residual")), True)
    return net_factory(net_type=model_name, in_chns=in_channels, class_num=num_classes, **kwargs)


def build_pgd_model(
    architecture: str,
    *,
    in_channels: int,
    num_classes: int,
    channel_config: tuple[int, ...],
    blueprint: Mapping[str, Any],
):
    if architecture in MIDDLE_PRUNED_MODEL_NAMES:
        if not blueprint:
            raise ValueError(f"{architecture} requires a pruning blueprint.")
        return build_middle_pruned_unet_plus_plus(
            in_channels=in_channels,
            num_classes=num_classes,
            blueprint=blueprint,
        )
    if architecture in FULL_PRUNED_MODEL_NAMES:
        if not blueprint:
            raise ValueError(f"{architecture} requires a pruning blueprint.")
        return build_full_pruning_unet_plus_plus(
            in_channels=in_channels,
            num_classes=num_classes,
            blueprint=blueprint,
        )
    if architecture == "blueprint_unet_plus_plus":
        return build_blueprint_unet_plus_plus(
            in_channels=in_channels,
            num_classes=num_classes,
            channel_config=channel_config,
        )
    return PDGUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        channel_config=channel_config,
    )


def build_model_for_checkpoint(
    checkpoint_path: Path,
    path_info: CheckpointPathInfo,
    payload: Mapping[str, Any],
    patch_size_override: Optional[tuple[int, int]],
):
    config = checkpoint_config(payload)
    model_info = checkpoint_model_info(payload)
    state_dict = checkpoint_state_dict(payload)
    patch_size = resolve_patch_size(config, model_info, patch_size_override)
    in_channels = resolve_in_channels(config, model_info, state_dict)
    num_classes = resolve_num_classes(config, model_info, state_dict)
    model_name = resolve_model_name(path_info, model_info, config)
    teacher_model = resolve_teacher_model(path_info, model_info, config)

    if path_info.phase == "1_teacher":
        model_name = teacher_model or model_name

    if model_name in BASIC_MODEL_NAMES:
        model = build_basic_model(
            model_name,
            in_channels=in_channels,
            num_classes=num_classes,
            patch_size=patch_size,
            config=config,
            model_info=model_info,
        )
        return model, model_name, patch_size

    if path_info.branch == "pgd" or model_name in PGD_MODEL_NAMES:
        blueprint = extract_blueprint(payload, checkpoint_path, model_info)
        architecture = model_name if model_name in PGD_MODEL_NAMES else blueprint_student_architecture(blueprint, teacher_model)
        channel_config = resolve_channel_config(model_info, blueprint, state_dict)
        model = build_pgd_model(
            architecture,
            in_channels=in_channels,
            num_classes=num_classes,
            channel_config=channel_config,
            blueprint=blueprint,
        )
        return model, architecture, patch_size

    if path_info.model in BASIC_MODEL_NAMES:
        model = build_basic_model(
            path_info.model,
            in_channels=in_channels,
            num_classes=num_classes,
            patch_size=patch_size,
            config=config,
            model_info=model_info,
        )
        return model, path_info.model, patch_size

    raise ValueError(f"Cannot infer model architecture for checkpoint: {checkpoint_path}")


def load_weights_into_model(model, payload: Mapping[str, Any], *, strict: bool) -> None:
    state_dict = checkpoint_state_dict(payload)
    model.load_state_dict(state_dict, strict=strict)


def profile_model(model, *, input_shape: tuple[int, ...], device: torch.device) -> dict[str, Any]:
    model = model.to(device)
    model.eval()
    profile = {
        "params": int(count_parameters(model)),
        "trainable_params": int(count_parameters(model, trainable_only=True)),
        "flops": maybe_compute_flops(model, input_shape=input_shape, device=device),
    }
    flops = profile["flops"]
    profile["flops_g"] = (float(flops) / 1e9) if flops is not None else None
    return profile


def row_from_checkpoint(
    checkpoint_path: Path,
    output_root: Path,
    payload: Mapping[str, Any],
    path_info: CheckpointPathInfo,
    architecture: str,
    input_shape: tuple[int, ...],
    profile: Mapping[str, Any],
    *,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    model_info = checkpoint_model_info(payload)
    return {
        "branch": path_info.branch,
        "dataset": path_info.dataset,
        "model": path_info.model,
        "architecture": architecture,
        "loss": path_info.loss,
        "method": path_info.method,
        "phase": path_info.phase,
        "checkpoint_path": str(checkpoint_path),
        "input_shape": "x".join(str(dim) for dim in input_shape),
        "params": profile.get("params"),
        "trainable_params": profile.get("trainable_params"),
        "flops": profile.get("flops"),
        "flops_g": profile.get("flops_g"),
        "backbone_name": model_info.get("backbone_name"),
        "student_name": model_info.get("student_name"),
        "status": status,
        "error": error,
    }


def error_row(checkpoint_path: Path, path_info: CheckpointPathInfo, error: BaseException) -> dict[str, Any]:
    return {
        "branch": path_info.branch,
        "dataset": path_info.dataset,
        "model": path_info.model,
        "architecture": "",
        "loss": path_info.loss,
        "method": path_info.method,
        "phase": path_info.phase,
        "checkpoint_path": str(checkpoint_path),
        "input_shape": "",
        "params": "",
        "trainable_params": "",
        "flops": "",
        "flops_g": "",
        "backbone_name": "",
        "student_name": "",
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def csv_path_for_group(flops_dir: Path, group: str) -> Path:
    if group == "basic_model":
        return flops_dir / "basic_model.csv"
    return flops_dir / f"{sanitize_tag(group)}.csv"


def compute_all_flops(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = args.output_root.expanduser()
    flops_dir = args.flops_dir.expanduser()
    device = resolve_device(args.device)
    patch_size_override = tuple(args.patch_size) if args.patch_size else None
    profile_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    checkpoint_paths = list(iter_checkpoint_paths(output_root, args.checkpoint_name))
    if args.verbose:
        print(f"Found {len(checkpoint_paths)} checkpoints under {output_root}")
        print(f"Device: {device}")

    for index, checkpoint_path in enumerate(checkpoint_paths, start=1):
        path_info = parse_checkpoint_path(checkpoint_path, output_root)
        if args.verbose:
            print(f"[{index}/{len(checkpoint_paths)}] {checkpoint_path}")
        try:
            payload = load_checkpoint(checkpoint_path, device=torch.device("cpu"))
            if not isinstance(payload, Mapping):
                payload = {"model_state_dict": payload}
            model, architecture, patch_size = build_model_for_checkpoint(
                checkpoint_path,
                path_info,
                payload,
                patch_size_override,
            )
            state_dict = checkpoint_state_dict(payload)
            in_channels = resolve_in_channels(checkpoint_config(payload), checkpoint_model_info(payload), state_dict)
            input_shape = (1, int(in_channels), int(patch_size[0]), int(patch_size[1]))

            if bool(args.load_weights):
                load_weights_into_model(model, payload, strict=bool(args.strict_load))

            cache_key = architecture_cache_key(payload, input_shape)
            if bool(args.cache_architectures) and cache_key in profile_cache:
                profile = dict(profile_cache[cache_key])
            else:
                profile = profile_model(model, input_shape=input_shape, device=device)
                if bool(args.cache_architectures):
                    profile_cache[cache_key] = dict(profile)

            rows.append(
                row_from_checkpoint(
                    checkpoint_path,
                    output_root,
                    payload,
                    path_info,
                    architecture,
                    input_shape,
                    profile,
                    status="ok",
                )
            )
        except (EOFError, RuntimeError, OSError, pickle.UnpicklingError, ImportError, ValueError, KeyError) as error:
            logging.exception("Failed to profile %s", checkpoint_path)
            rows.append(error_row(checkpoint_path, path_info, error))
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        path_info = parse_checkpoint_path(Path(row["checkpoint_path"]), output_root)
        grouped[path_info.csv_group].append(row)

    for group, group_rows in sorted(grouped.items()):
        write_csv(csv_path_for_group(flops_dir, group), group_rows)
    write_csv(flops_dir / "all_flops.csv", rows)

    if args.verbose:
        print(f"Wrote {len(grouped)} grouped CSV file(s) and {flops_dir / 'all_flops.csv'}")
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    compute_all_flops(args)


if __name__ == "__main__":
    main()

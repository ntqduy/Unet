from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def project_pretrain_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "pretrain"


def ensure_pretrain_cache() -> Path:
    pretrain_dir = project_pretrain_dir()
    checkpoints_dir = pretrain_dir / "hub" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(pretrain_dir)
    os.environ["TORCH_MODEL_ZOO"] = str(checkpoints_dir)
    return pretrain_dir


def torch_checkpoints_dir() -> Path:
    return ensure_pretrain_cache() / "hub" / "checkpoints"


def checkpoint_name_from_url(url: str) -> str:
    return Path(urlparse(str(url)).path).name


def cached_checkpoint_path(url: str) -> Path:
    return torch_checkpoints_dir() / checkpoint_name_from_url(url)


def find_cached_checkpoint(url: str, *, min_bytes: int = 1024) -> Path | None:
    path = cached_checkpoint_path(url)
    if path.is_file() and path.stat().st_size >= int(min_bytes):
        return path
    return None

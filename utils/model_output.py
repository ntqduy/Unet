from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn


@dataclass
class SegmentationModelOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    preds: torch.Tensor
    features: Any = None
    aux: Dict[str, Any] = field(default_factory=dict)
    model_name: str = "unknown_model"
    backbone_name: Optional[str] = None
    student_name: Optional[str] = None
    phase_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logits": self.logits,
            "probs": self.probs,
            "preds": self.preds,
            "features": self.features,
            "aux": self.aux,
            "model_name": self.model_name,
            "backbone_name": self.backbone_name,
            "student_name": self.student_name,
            "phase_name": self.phase_name,
        }


def build_segmentation_output(
    logits: torch.Tensor,
    *,
    features: Any = None,
    aux: Optional[Dict[str, Any]] = None,
    model_name: str,
    backbone_name: Optional[str] = None,
    student_name: Optional[str] = None,
    phase_name: Optional[str] = None,
    threshold: float = 0.5,
) -> SegmentationModelOutput:
    if logits.ndim < 4:
        raise ValueError("Segmentation logits must be a 4D or higher tensor.")

    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).long()
    else:
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1, keepdim=True).long()

    return SegmentationModelOutput(
        logits=logits,
        probs=probs,
        preds=preds,
        features=features,
        aux=aux or {},
        model_name=model_name,
        backbone_name=backbone_name,
        student_name=student_name,
        phase_name=phase_name,
    )


class BaseSegmentationModel(nn.Module):
    model_name = "segmentation_model"
    backbone_name: Optional[str] = None
    student_name: Optional[str] = None
    phase_name: Optional[str] = None
    prediction_threshold = 0.5
    architecture_config: Dict[str, Any]

    def set_architecture_config(self, **kwargs: Any) -> None:
        self.architecture_config = {key: _to_serializable(value) for key, value in kwargs.items()}

    def get_architecture_config(self) -> Dict[str, Any]:
        return dict(getattr(self, "architecture_config", {}))

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backbone_name": self.backbone_name,
            "student_name": self.student_name,
            "phase_name": self.phase_name,
            "architecture_config": self.get_architecture_config(),
        }

    def build_output(self, logits: torch.Tensor, *, features: Any = None, aux: Optional[Dict[str, Any]] = None) -> SegmentationModelOutput:
        return build_segmentation_output(
            logits,
            features=features,
            aux=aux,
            model_name=self.model_name,
            backbone_name=self.backbone_name,
            student_name=self.student_name,
            phase_name=self.phase_name,
            threshold=self.prediction_threshold,
        )


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    return str(value)


def extract_model_info(model: Any) -> Dict[str, Any]:
    if hasattr(model, "get_model_info") and callable(model.get_model_info):
        return _to_serializable(model.get_model_info())
    return {
        "model_name": getattr(model, "model_name", model.__class__.__name__),
        "backbone_name": getattr(model, "backbone_name", None),
        "student_name": getattr(model, "student_name", None),
        "phase_name": getattr(model, "phase_name", None),
        "architecture_config": _to_serializable(getattr(model, "architecture_config", {})),
    }


def _first_tensor_candidate(items: Any) -> Optional[torch.Tensor]:
    for item in items:
        if torch.is_tensor(item) and item.ndim >= 4:
            return item
    return None


def extract_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, SegmentationModelOutput):
        return model_output.logits
    if isinstance(model_output, Mapping):
        logits = model_output.get("logits")
        if logits is None:
            raise ValueError("Mapping model output does not contain 'logits'.")
        return logits
    if isinstance(model_output, (list, tuple)):
        tensor_candidate = _first_tensor_candidate(model_output)
        if tensor_candidate is None:
            raise ValueError("Model output does not contain segmentation logits.")
        return tensor_candidate
    if torch.is_tensor(model_output):
        return model_output
    raise TypeError(f"Unsupported model output type: {type(model_output)!r}")


def extract_probs(model_output: Any) -> Optional[torch.Tensor]:
    if isinstance(model_output, SegmentationModelOutput):
        return model_output.probs
    if isinstance(model_output, Mapping):
        return model_output.get("probs")
    return None


def extract_preds(model_output: Any) -> Optional[torch.Tensor]:
    if isinstance(model_output, SegmentationModelOutput):
        return model_output.preds
    if isinstance(model_output, Mapping):
        return model_output.get("preds")
    return None


def extract_features(model_output: Any) -> Any:
    if isinstance(model_output, SegmentationModelOutput):
        return model_output.features
    if isinstance(model_output, Mapping):
        return model_output.get("features")
    if isinstance(model_output, (list, tuple)) and len(model_output) > 1:
        return model_output[1]
    return None


def extract_aux(model_output: Any) -> Dict[str, Any]:
    if isinstance(model_output, SegmentationModelOutput):
        return model_output.aux
    if isinstance(model_output, Mapping):
        return dict(model_output.get("aux", {}))
    return {}


def extract_metadata(model_output: Any) -> Dict[str, Any]:
    if isinstance(model_output, SegmentationModelOutput):
        return {
            "model_name": model_output.model_name,
            "backbone_name": model_output.backbone_name,
            "student_name": model_output.student_name,
            "phase_name": model_output.phase_name,
        }
    if isinstance(model_output, Mapping):
        return {
            "model_name": model_output.get("model_name"),
            "backbone_name": model_output.get("backbone_name"),
            "student_name": model_output.get("student_name"),
            "phase_name": model_output.get("phase_name"),
        }
    return {}

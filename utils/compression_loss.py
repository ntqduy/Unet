import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.losses import DiceLoss
from utils.model_output import extract_aux, extract_features, extract_logits


def _flatten_feature_dict(features):
    if not isinstance(features, dict):
        return {}
    flattened = {}
    for key, value in features.items():
        if torch.is_tensor(value):
            flattened[str(key)] = value
        elif isinstance(value, dict):
            for child_key, child_value in _flatten_feature_dict(value).items():
                flattened[str(child_key)] = child_value
                flattened[f"{key}.{child_key}"] = child_value
    return flattened


class CompressionLoss(nn.Module):
    def __init__(
        self,
        num_classes=2,
        lambda_distill=0.3,
        lambda_sparsity=0.3,
        *,
        use_kd_output=1,
        use_sparsity=1,
        use_feature_distill=0,
        use_aux_loss=0,
        lambda_feat=0.1,
        lambda_aux=0.2,
        feature_layers=None,
    ):
        super().__init__()
        self.dice_loss = DiceLoss(n_classes=num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_distill = lambda_distill
        self.lambda_sparsity = lambda_sparsity
        self.use_kd_output = bool(int(use_kd_output))
        self.use_sparsity = bool(int(use_sparsity))
        self.use_feature_distill = bool(int(use_feature_distill))
        self.use_aux_loss = bool(int(use_aux_loss))
        self.lambda_feat = float(lambda_feat)
        self.lambda_aux = float(lambda_aux)
        self.feature_layers = list(feature_layers or [])

    def _feature_distill_loss(self, student_output, teacher_output, reference_logits):
        if teacher_output is None or not self.use_feature_distill or self.lambda_feat <= 0:
            return reference_logits.new_zeros(())

        student_features = _flatten_feature_dict(extract_features(student_output))
        teacher_features = _flatten_feature_dict(extract_features(teacher_output))
        losses = []
        selected_layers = self.feature_layers or sorted(set(student_features) & set(teacher_features))
        for layer_name in selected_layers:
            student_feature = student_features.get(layer_name)
            teacher_feature = teacher_features.get(layer_name)
            if student_feature is None or teacher_feature is None:
                continue
            if student_feature.shape[-2:] != teacher_feature.shape[-2:]:
                teacher_feature = F.interpolate(teacher_feature, size=student_feature.shape[-2:], mode="bilinear", align_corners=False)
            if student_feature.shape[1] != teacher_feature.shape[1]:
                student_feature = student_feature.mean(dim=1, keepdim=True)
                teacher_feature = teacher_feature.mean(dim=1, keepdim=True)
            losses.append(F.mse_loss(student_feature, teacher_feature.detach()))
        if not losses:
            return reference_logits.new_zeros(())
        return torch.stack(losses).mean()

    def _auxiliary_loss(self, student_output, target, reference_logits):
        if not self.use_aux_loss or self.lambda_aux <= 0:
            return reference_logits.new_zeros(())
        aux = extract_aux(student_output)
        aux_logits = aux.get("aux_logits") if isinstance(aux, dict) else None
        if torch.is_tensor(aux_logits):
            aux_logits = {"aux": aux_logits}
        if not isinstance(aux_logits, dict):
            return reference_logits.new_zeros(())

        losses = []
        for logits in aux_logits.values():
            if not torch.is_tensor(logits):
                continue
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
            probs = torch.softmax(logits, dim=1)
            losses.append(0.5 * self.ce_loss(logits, target) + 0.5 * self.dice_loss(probs, target.unsqueeze(1)))
        if not losses:
            return reference_logits.new_zeros(())
        return torch.stack(losses).mean()

    def forward(
        self,
        student_output,
        teacher_output,
        student_gates,
        target,
        *,
        lambda_distill: float | None = None,
        lambda_sparsity: float | None = None,
        lambda_feat: float | None = None,
        lambda_aux: float | None = None,
    ):
        student_logits = extract_logits(student_output)
        student_probs = torch.softmax(student_logits, dim=1)
        effective_lambda_distill = (self.lambda_distill if lambda_distill is None else float(lambda_distill)) if self.use_kd_output else 0.0
        effective_lambda_sparsity = (self.lambda_sparsity if lambda_sparsity is None else float(lambda_sparsity)) if self.use_sparsity else 0.0
        effective_lambda_feat = (self.lambda_feat if lambda_feat is None else float(lambda_feat)) if self.use_feature_distill else 0.0
        effective_lambda_aux = (self.lambda_aux if lambda_aux is None else float(lambda_aux)) if self.use_aux_loss else 0.0

        l_seg = 0.5 * self.ce_loss(student_logits, target) + 0.5 * self.dice_loss(student_probs, target.unsqueeze(1))

        if teacher_output is None or effective_lambda_distill <= 0:
            l_distill = student_logits.new_zeros(())
        else:
            teacher_logits = extract_logits(teacher_output)
            l_distill = F.mse_loss(student_logits, teacher_logits)

        gate_tensors = [gate for gate in student_gates if gate is not None]
        if not gate_tensors or effective_lambda_sparsity <= 0:
            l_sparsity = student_logits.new_zeros(())
        else:
            l_sparsity = torch.stack([gate.mean() for gate in gate_tensors]).mean()

        l_feature = self._feature_distill_loss(student_output, teacher_output, student_logits)
        l_aux = self._auxiliary_loss(student_output, target, student_logits)

        total_loss = (
            l_seg
            + effective_lambda_distill * l_distill
            + effective_lambda_feat * l_feature
            + effective_lambda_aux * l_aux
            + effective_lambda_sparsity * l_sparsity
        )
        return {
            "total_loss": total_loss,
            "segmentation_loss": l_seg,
            "distillation_loss": l_distill,
            "sparsity_loss": l_sparsity,
            "feature_distill_loss": l_feature,
            "auxiliary_loss": l_aux,
            "lambda_distill": effective_lambda_distill,
            "lambda_sparsity": effective_lambda_sparsity,
            "lambda_feat": effective_lambda_feat,
            "lambda_aux": effective_lambda_aux,
        }

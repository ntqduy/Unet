import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.losses import DiceLoss
from utils.model_output import extract_logits


class CompressionLoss(nn.Module):
    def __init__(self, num_classes=2, lambda_distill=0.3, lambda_sparsity=0.3):
        super().__init__()
        self.dice_loss = DiceLoss(n_classes=num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_distill = lambda_distill
        self.lambda_sparsity = lambda_sparsity

    def forward(
        self,
        student_output,
        teacher_output,
        student_gates,
        target,
        *,
        lambda_distill: float | None = None,
        lambda_sparsity: float | None = None,
    ):
        student_logits = extract_logits(student_output)
        student_probs = torch.softmax(student_logits, dim=1)
        effective_lambda_distill = self.lambda_distill if lambda_distill is None else float(lambda_distill)
        effective_lambda_sparsity = self.lambda_sparsity if lambda_sparsity is None else float(lambda_sparsity)

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

        total_loss = l_seg + effective_lambda_distill * l_distill + effective_lambda_sparsity * l_sparsity
        return {
            "total_loss": total_loss,
            "segmentation_loss": l_seg,
            "distillation_loss": l_distill,
            "sparsity_loss": l_sparsity,
            "lambda_distill": effective_lambda_distill,
            "lambda_sparsity": effective_lambda_sparsity,
        }

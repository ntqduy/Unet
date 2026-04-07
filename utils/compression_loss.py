# utils/compression_loss.py
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

    def forward(self, student_output, teacher_output, student_gates, target):
        student_logits = extract_logits(student_output)
        teacher_logits = extract_logits(teacher_output)
        student_probs = torch.softmax(student_logits, dim=1)

        # L_seg
        L_seg = 0.5 * self.ce_loss(student_logits, target) + \
                0.5 * self.dice_loss(student_probs, target.unsqueeze(1))

        # L_distill (feature-level)
        L_distill = F.mse_loss(student_logits, teacher_logits)

        # L_sparsity
        L_sparsity = torch.stack([g.mean() for g in student_gates]).mean()

        total_loss = L_seg + self.lambda_distill * L_distill + self.lambda_sparsity * L_sparsity
        return {
            "total_loss": total_loss,
            "segmentation_loss": L_seg,
            "distillation_loss": L_distill,
            "sparsity_loss": L_sparsity,
        }

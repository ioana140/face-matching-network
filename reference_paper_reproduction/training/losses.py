"""Contrastive loss.

  L = (1 - y) * D^2  +  y * [max(m - D, 0)]^2

y = 0 (same person): loss = D^2, minimizes distance.
y = 1 (different):   loss = [max(m-D, 0)]^2, pushes distance beyond margin m.
"""

import torch
import torch.nn as nn

from config import CONTRASTIVE_MARGIN


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = CONTRASTIVE_MARGIN):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        distances: torch.Tensor,
        labels:    torch.Tensor,
    ) -> torch.Tensor:
        d = distances                          # (B,)
        y = labels.float()                     # (B,)

        loss_sim  = (1.0 - y) * d.pow(2)
        loss_dis  = y * torch.clamp(self.margin - d, min=0.0).pow(2)

        return torch.mean(loss_sim + loss_dis)

    def extra_repr(self) -> str:
        return f"margin={self.margin}"



import torch
import torch.nn as nn

from config import DECISION_THRESHOLD


class EuclideanDistance(nn.Module):
    def forward(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
    ) -> torch.Tensor:
        diff   = embedding_a - embedding_b           # (B, D)
        sum_sq = torch.sum(diff ** 2, dim=1)          # (B,)
        dist   = torch.sqrt(torch.clamp(sum_sq, min=1e-9))  # (B,)
        return dist


def predict_labels(
    distances: torch.Tensor,
    threshold: float = DECISION_THRESHOLD,
) -> torch.Tensor:
    return (distances >= threshold).long()

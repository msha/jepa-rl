from __future__ import annotations

import torch
import torch.nn.functional as F


def normalized_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return 2.0 - 2.0 * (pred * target).sum(dim=-1).mean()


def variance_loss(z: torch.Tensor, variance_floor: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(variance_floor - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    batch, dim = z.shape
    if batch <= 1:
        return z.new_zeros(())
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (batch - 1)
    off_diag = cov.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].flatten()
    return (off_diag.pow(2).sum() / dim)


def effective_rank(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    z = z - z.mean(dim=0)
    singular_values = torch.linalg.svdvals(z.float())
    probs = singular_values / (singular_values.sum() + eps)
    entropy = -(probs * torch.log(probs + eps)).sum()
    return torch.exp(entropy)


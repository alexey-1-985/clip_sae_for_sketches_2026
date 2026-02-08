import torch
from sklearn.metrics import r2_score
import numpy as np


def compute_l0(latents, threshold=1e-3):

    return (latents.abs() > threshold).float().sum(dim=-1).mean().item()


def explained_variance_ratio(recon, target, eps=1e-8):

    with torch.no_grad():
        mse = torch.sum((target - recon) ** 2)
        var = torch.sum((target - target.mean(dim=0, keepdim=True)) ** 2)

        if var < eps:
            return float("nan")

        r2 = 1.0 - mse / (var + eps)
        return r2.item()
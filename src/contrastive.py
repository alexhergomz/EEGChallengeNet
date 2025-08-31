from __future__ import annotations

import torch
from torch import nn, Tensor


class SupConLoss(nn.Module):
    """
    Supervised contrastive loss (single-view per sample).
    Positives are defined by label equality in the provided label vector.
    Input: features (B, D) L2-normalized or not (we will normalize), labels (B,)
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        device = features.device
        b, d = features.shape
        z = torch.nn.functional.normalize(features, dim=-1)
        logits = z @ z.t()  # (B, B)
        logits = logits / max(self.temperature, self.eps)
        # mask out self-comparisons
        logits = logits - torch.eye(b, device=device) * 1e9

        labels = labels.view(-1)
        assert labels.shape[0] == b
        # positives mask
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(z.dtype)
        pos_mask = pos_mask - torch.eye(b, device=device, dtype=z.dtype)

        # log-softmax denominator over all non-self
        # subtract max for numerical stability
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits_stable = logits - logits_max
        exp_logits = torch.exp(logits_stable)
        exp_logits = exp_logits + 1e-30
        # denominator excludes self due to masking above
        denom = exp_logits.sum(dim=1, keepdim=True)

        # For each anchor i: sum over positives j of -log( exp(logit_ij)/denom_i ) / num_pos_i
        log_prob_pos = logits_stable - torch.log(denom)
        # multiply by positive mask and average over positives per anchor
        pos_count = pos_mask.sum(dim=1)
        # avoid div by zero: only anchors with positives contribute
        valid = pos_count > 0
        if not torch.any(valid):
            return torch.zeros((), device=device, dtype=z.dtype)
        loss_i = -(pos_mask * log_prob_pos).sum(dim=1) / (pos_count + 1e-12)
        loss = loss_i[valid].mean()
        return loss



import torch
from torch import nn, Tensor


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) simplified for embeddings.
    Here we only use the normalization step (no denormalization), since we
    train encoders to produce invariant embeddings.

    Expected input shape: (B, L, C) — normalize per instance over time (L) per channel (C).
    If a 4D tensor is provided (B, L, C, r), it will normalize over time (L) per channel (C)
    while preserving the feature vector r.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            # x: (B, L, C)
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, unbiased=False, keepdim=True)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                x_hat = x_hat * self.weight.view(1, 1, self.num_channels) + self.bias.view(1, 1, self.num_channels)
            return x_hat
        elif x.dim() == 4:
            # x: (B, L, C, r) — normalize over time per channel
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, unbiased=False, keepdim=True)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                x_hat = x_hat * self.weight.view(1, 1, self.num_channels, 1) + self.bias.view(1, 1, self.num_channels, 1)
            return x_hat
        else:
            raise ValueError(f"RevIN expects 3D or 4D input, got shape {tuple(x.shape)}")



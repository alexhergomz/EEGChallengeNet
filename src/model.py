from typing import Optional

import torch
from torch import nn, Tensor

from .blocks import (
    PerVariateLift,
    SplitAxisBlock,
)


class SplitAxisModel(nn.Module):
    """
    End-to-end model:
      (B, L, C) -> PerVariateLift -> SplitAxisBlock -> (B, L, C, r)
    """

    def __init__(
        self,
        num_channels: int,
        r: int,
        r_v: int = 16,
        d_qk: int = 16,
        z: int = 16,
        ffn_expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.r = r

        self.lift = PerVariateLift(num_channels, r)
        self.block = SplitAxisBlock(
            num_channels=num_channels,
            r=r,
            r_v=r_v,
            d_qk=d_qk,
            z=z,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C)
        y = self.lift(x)
        y = self.block(y)
        return y


def tiny_model(num_channels: int = 4, r: int = 8) -> SplitAxisModel:
    """A tiny instantiation suitable for 2GB VRAM tests."""
    return SplitAxisModel(
        num_channels=num_channels,
        r=r,
        r_v=max(4, r // 2),
        d_qk=max(8, r),
        z=max(8, r),
        ffn_expansion=2,
        dropout=0.0,
    )



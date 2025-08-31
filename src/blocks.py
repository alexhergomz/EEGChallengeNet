import math
from typing import Optional

import torch
from torch import nn, Tensor


class ChannelWiseNorm(nn.Module):
    """
    Normalize only along the r dimension per variate (channel).
    Parameters are per-channel (C, r) to avoid cross-channel mixing.
    Input/Output shape: (B, L, C, r)
    """

    def __init__(self, num_channels: int, r: int, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.r = r
        self.eps = eps
        # Per-channel affine
        self.weight = nn.Parameter(torch.ones(num_channels, r))
        self.bias = nn.Parameter(torch.zeros(num_channels, r))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C, r)
        assert x.dim() == 4 and x.size(-1) == self.r and x.size(-2) == self.num_channels
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        # Apply per-channel affine (broadcast over B,L)
        return x_hat * self.weight.unsqueeze(0).unsqueeze(0) + self.bias.unsqueeze(0).unsqueeze(0)


class ChannelWiseScaleNorm(nn.Module):
    """
    Per-channel ScaleNorm along r dimension.
    y = g_c * x / (||x|| / sqrt(r) + eps), where g_c is a learnable scalar per channel.
    No mixing across channels or time. Input/Output: (B, L, C, r)
    """

    def __init__(self, num_channels: int, r: int, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.r = r
        self.eps = eps
        # One scale per channel
        self.gain = nn.Parameter(torch.ones(num_channels))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C, r)
        assert x.dim() == 4 and x.size(-1) == self.r and x.size(-2) == self.num_channels
        # L2 norm over r
        l2 = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
        denom = (l2 / math.sqrt(self.r)) + self.eps
        y = x / denom
        return y * self.gain.view(1, 1, self.num_channels, 1)


class BlockWiseLinear(nn.Module):
    """
    Per-vari ate linear: (B, L, C, r_in) -> (B, L, C, r_out)
    Implements separate weights for each channel to avoid mixing.
    """

    def __init__(self, num_channels: int, r_in: int, r_out: int, bias: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.r_in = r_in
        self.r_out = r_out
        self.weight = nn.Parameter(torch.empty(num_channels, r_in, r_out))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_channels, r_out))
        else:
            self.bias = None
        # Kaiming uniform similar to nn.Linear default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = r_in
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C, r_in)
        assert x.dim() == 4 and x.size(-1) == self.r_in and x.size(-2) == self.num_channels
        y = torch.einsum('blcr,cro->blco', x, self.weight)
        if self.bias is not None:
            y = y + self.bias.unsqueeze(0).unsqueeze(0)
        return y


class BlockWiseFFN(nn.Module):
    """
    Per-channel MLP: r -> e*r -> r with activation. No cross-channel mixing.
    """

    def __init__(self, num_channels: int, r: int, expansion: int = 4, activation: Optional[nn.Module] = None, dropout: float = 0.0):
        super().__init__()
        hidden = expansion * r
        self.fc1 = BlockWiseLinear(num_channels, r, hidden, bias=True)
        self.fc2 = BlockWiseLinear(num_channels, hidden, r, bias=True)
        self.act = activation if activation is not None else nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        y = self.fc1(x)
        y = self.act(y)
        y = self.dropout(y)
        y = self.fc2(y)
        return y


class BlockWiseSwiGLU(nn.Module):
    """
    Per-channel SwiGLU MLP: x -> (SiLU(Gate(x)) * Up(x)) -> Down -> r
    Hidden dim = expansion * r. No cross-channel mixing.
    """

    def __init__(self, num_channels: int, r: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = expansion * r
        self.up = BlockWiseLinear(num_channels, r, hidden, bias=True)
        self.gate = BlockWiseLinear(num_channels, r, hidden, bias=True)
        self.down = BlockWiseLinear(num_channels, hidden, r, bias=True)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        u = self.up(x)
        g = self.act(self.gate(x))
        h = self.dropout(u * g)
        return self.down(h)


class PerVariateLift(nn.Module):
    """
    Map (B, L, C) to (B, L, C, r) via channel-specific vectors.
    """

    def __init__(self, num_channels: int, r: int):
        super().__init__()
        self.num_channels = num_channels
        self.r = r
        self.lift = nn.Parameter(torch.empty(num_channels, r))
        nn.init.xavier_uniform_(self.lift)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C)
        assert x.dim() == 3 and x.size(-1) == self.num_channels
        # Broadcast: (B,L,C,1) * (C,r) -> (B,L,C,r)
        return x.unsqueeze(-1) * self.lift.unsqueeze(0).unsqueeze(0)


class TemporalAttention(nn.Module):
    """
    Attention over time. QK mix channels by flattening (C,r) -> (C*r) for projections.
    Multi-Query variant: H query heads, single shared K and single shared V head.
    V is block-wise per channel.
    Input/Output: (B, L, C, r)
    """

    def __init__(self, num_channels: int, r: int, r_v: int, d_qk: int, num_q_heads: int = 1, num_sink_q: int = 0):
        super().__init__()
        self.num_channels = num_channels
        self.r = r
        self.r_v = r_v
        self.d_qk = d_qk
        self.num_q_heads = max(1, num_q_heads)
        self.num_sink_q = max(0, num_sink_q)

        # Q: H heads, each of size d_qk
        self.to_q = nn.Linear(num_channels * r, self.num_q_heads * d_qk, bias=True)
        # Shared K head
        self.to_k = nn.Linear(num_channels * r, d_qk, bias=True)
        # V: block-wise per channel (shared for all heads)
        self.to_v = BlockWiseLinear(num_channels, r, r_v, bias=True)
        self.out = BlockWiseLinear(num_channels, r_v, r, bias=True)
        # Learned sink queries per head (H, S, d)
        if self.num_sink_q > 0:
            self.sink_q = nn.Parameter(torch.empty(self.num_q_heads, self.num_sink_q, d_qk))
            nn.init.xavier_uniform_(self.sink_q)
        else:
            self.register_parameter('sink_q', None)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C, r)
        b, l, c, r = x.shape
        assert c == self.num_channels and r == self.r
        residual = x

        # Flatten channels for QK
        x_flat = x.reshape(b, l, c * r)
        q = self.to_q(x_flat).reshape(b, l, self.num_q_heads, self.d_qk)  # (B,L,H,d)
        k = self.to_k(x_flat)  # (B,L,d)

        # Compute temporal attention weights per head: (B,H,L,L)
        scale = 1.0 / math.sqrt(self.d_qk)
        attn_logits = torch.einsum('blhd,bsd->bhls', q, k).float() * scale
        attn = attn_logits.softmax(dim=-1).to(x.dtype)

        # V is per-channel (B,L,C,r_v). Apply attention independently for each channel.
        v = self.to_v(x)  # (B, L, C, r_v)
        y_heads = torch.einsum('bhls, bscv -> bhlcv', attn, v)  # (B,H,L,C,r_v)
        # Merge heads by mean (could be learned combine, but keep simple and light)
        y = y_heads.mean(dim=1)  # (B,L,C,r_v)
        # Sink queries: global memory added across time
        if self.num_sink_q > 0:
            # logits_s: (B,H,S,L)
            logits_s = torch.einsum('hsd, bld->bhsl', self.sink_q, k).float() * scale
            attn_s = logits_s.softmax(dim=-1).to(x.dtype)
            # sink output: (B,H,S,C,r_v)
            y_sink = torch.einsum('bhsl, blcv -> bhscv', attn_s, v)
            y_sink = y_sink.mean(dim=(1, 2))  # (B,C,r_v)
            y = y + y_sink.unsqueeze(1)
        y = self.out(y)  # (B, L, C, r)
        return residual + y


class ChannelAttention(nn.Module):
    """
    Attention over channels per time step. QK may mix channels; V is block-wise per channel.
    Multi-Query variant: H query heads, single shared K and V head.
    Input/Output: (B, L, C, r)
    """

    def __init__(self, num_channels: int, r: int, r_v: int, z: int, num_q_heads: int = 1, num_sink_q: int = 0):
        super().__init__()
        self.num_channels = num_channels
        self.r = r
        self.r_v = r_v
        self.z = z
        self.num_q_heads = max(1, num_q_heads)
        self.num_sink_q = max(0, num_sink_q)

        # Q: per-channel, H heads of size z
        self.q_proj = nn.Linear(num_channels * r, num_channels * self.num_q_heads * z, bias=True)
        # Shared K: per-channel, single head of size z
        self.k_proj = nn.Linear(num_channels * r, num_channels * z, bias=True)

        # V: block-wise projection and output projection (shared for all heads)
        self.to_v = BlockWiseLinear(num_channels, r, r_v, bias=True)
        self.out = BlockWiseLinear(num_channels, r_v, r, bias=True)
        if self.num_sink_q > 0:
            self.sink_q = nn.Parameter(torch.empty(self.num_q_heads, self.num_sink_q, self.z))
            nn.init.xavier_uniform_(self.sink_q)
        else:
            self.register_parameter('sink_q', None)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C, r)
        b, l, c, r = x.shape
        assert c == self.num_channels and r == self.r
        residual = x

        # Build Q, K per time step with channel mixing
        x_flat = x.reshape(b, l, c * r)
        q_all = self.q_proj(x_flat).reshape(b, l, c, self.num_q_heads, self.z)  # (B,L,C,H,Z)
        k_all = self.k_proj(x_flat).reshape(b, l, c, self.z)  # (B,L,C,Z)

        scale = 1.0 / math.sqrt(self.z)
        # logits: (B,L,H,C,C) = q[b,l,i,h,:] · k[b,l,j,:]
        logits = torch.einsum('blchz,bljz->blhcj', q_all, k_all).float() * scale
        attn = logits.softmax(dim=-1).to(x.dtype)

        v = self.to_v(x)  # (B,L,C,r_v)
        # Aggregate over channels per head
        y_heads = torch.einsum('blhcj,bljv->blhcv', attn, v)  # (B,L,H,C,r_v)
        y = y_heads.mean(dim=2)  # (B,L,C,r_v)
        # Sink queries across channels: produce (B,L,r_v) memory broadcast to all channels
        if self.num_sink_q > 0:
            # logits_s: (B,L,H,S,C)
            logits_s = torch.einsum('hsz, bljz->blhsj', self.sink_q, k_all).float() * scale
            attn_s = logits_s.softmax(dim=-1).to(x.dtype)
            # (B,L,H,S,r_v)
            y_sink = torch.einsum('blhsj, bljv->blhsv', attn_s, v)
            y_sink = y_sink.mean(dim=(2, 3))  # (B,L,r_v)
            y = y + y_sink.unsqueeze(2)
        y = self.out(y)
        return residual + y


class SplitAxisBlock(nn.Module):
    """
    Order:
      1) Norm -> TemporalAttention -> Residual
      2) Norm -> BlockWiseFFN -> Residual
      3) Norm -> ChannelAttention -> Residual
      4) Norm -> BlockWiseFFN -> Residual
    PreNorm everywhere is implemented in submodules by explicit Norm before each op.
    """

    def __init__(
        self,
        num_channels: int,
        r: int,
        r_v: int = 32,
        d_qk: int = 32,
        z: int = 32,
        ffn_expansion: int = 2,
        dropout: float = 0.0,
        use_scale_norm: bool = True,
        num_q_heads_time: int = 1,
        num_q_heads_channel: int = 1,
    ):
        super().__init__()
        Norm = ChannelWiseScaleNorm if use_scale_norm else ChannelWiseNorm
        self.norm1 = Norm(num_channels, r)
        self.temporal_attn = TemporalAttention(num_channels, r, r_v=r_v, d_qk=d_qk, num_q_heads=num_q_heads_time)
        self.norm2 = Norm(num_channels, r)
        self.ffn1 = BlockWiseSwiGLU(num_channels, r, expansion=ffn_expansion, dropout=dropout)
        self.norm3 = Norm(num_channels, r)
        self.channel_attn = ChannelAttention(num_channels, r, r_v=r_v, z=z, num_q_heads=num_q_heads_channel)
        self.norm4 = Norm(num_channels, r)
        self.ffn2 = BlockWiseSwiGLU(num_channels, r, expansion=ffn_expansion, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # All residuals handled inside modules; we do PreNorm here
        y = self.temporal_attn(self.norm1(x))
        y = self.ffn1(self.norm2(y)) + y
        y = self.channel_attn(self.norm3(y))
        y = self.ffn2(self.norm4(y)) + y
        return y


class TwoStageCLSPool(nn.Module):
    """
    CLS pooling in two stages:
      1) Time-wise: append a learnable CLS token (per channel) along time, run TemporalAttention, take CLS output -> (B, C, r)
      2) Channel-wise: append a learnable CLS channel, run ChannelAttention over (C+1), take CLS channel output -> (B, r)
    """

    def __init__(
        self,
        num_channels: int,
        r: int,
        r_v: int = 16,
        d_qk: int = 16,
        z: int = 16,
        num_q_heads_time: int = 1,
        num_q_heads_channel: int = 1,
        num_sink_q_time: int = 0,
        num_sink_q_channel: int = 0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.r = r
        # CLS params
        self.cls_time = nn.Parameter(torch.empty(num_channels, r))
        nn.init.xavier_uniform_(self.cls_time)

        # Attention modules for pooling
        self.temporal = TemporalAttention(
            num_channels=num_channels,
            r=r,
            r_v=r_v,
            d_qk=d_qk,
            num_q_heads=num_q_heads_time,
            num_sink_q=num_sink_q_time,
        )
        # Channel attention with C+1 because of appended CLS channel
        self.channel = ChannelAttention(
            num_channels=num_channels + 1,
            r=r,
            r_v=r_v,
            z=z,
            num_q_heads=num_q_heads_channel,
            num_sink_q=num_sink_q_channel,
        )

    def forward(self, x: Tensor) -> Tensor:
        b, l, c, r = x.shape
        assert c == self.num_channels and r == self.r
        # Time-wise CLS
        cls_t = self.cls_time.unsqueeze(0).unsqueeze(0).expand(b, 1, c, r)
        xt = torch.cat([x, cls_t], dim=1)
        yt = self.temporal(xt)  # (B, L+1, C, r)
        t_cls = yt[:, -1]  # (B, C, r)

        # Channel-wise CLS
        t_cls = t_cls.unsqueeze(1)  # (B,1,C,r)
        # Derive channel-CLS token from time-wise CLS output (average over channels)
        cls_c_vec = t_cls.mean(dim=2)  # (B,1,r)
        cls_c = cls_c_vec.unsqueeze(2)  # (B,1,1,r)
        xc = torch.cat([t_cls, cls_c], dim=2)  # (B,1,C+1,r)
        yc = self.channel(xc)  # (B,1,C+1,r)
        c_cls = yc[:, :, -1, :]  # (B,1,r)
        return c_cls.squeeze(1)  # (B,r)



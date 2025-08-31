import os
import torch

from src.model import tiny_model
from src.blocks import ChannelWiseNorm, BlockWiseLinear, TemporalAttention, ChannelAttention


@torch.no_grad()
def test_shapes_cpu():
    B, L, C, r = 2, 8, 4, 6
    x = torch.randn(B, L, C)
    model = tiny_model(num_channels=C, r=r)
    y = model(x)
    assert y.shape == (B, L, C, r)


def test_blockwise_independence_no_attention():
    # Disable attention by zeroing QK projections
    B, L, C, r = 1, 4, 3, 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(B, L, C, device=device)
    from src.model import SplitAxisModel
    model = SplitAxisModel(num_channels=C, r=r, r_v=4, d_qk=4, z=4).to(device)

    # Zero out Q,K projections and also V/out projections so attention path = identity
    for m in model.block.temporal_attn.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    for m in model.block.channel_attn.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    with torch.no_grad():
        for bw in [
            model.block.temporal_attn.to_v,
            model.block.temporal_attn.out,
            model.block.channel_attn.to_v,
            model.block.channel_attn.out,
        ]:
            torch.nn.init.zeros_(bw.weight)
            if bw.bias is not None:
                torch.nn.init.zeros_(bw.bias)

    x1 = x.clone()
    y_before = model(x1)

    # Change block-wise weights for channel 0 only
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'fc1.weight' in name or 'fc2.weight' in name:
                # Param shapes have leading channel dimension
                param[0].add_(0.123)

    x2 = x.clone()
    y_after = model(x2)

    # Only channel 0 should change when attention is disabled
    diff = (y_after - y_before).abs().mean(dim=-1)  # (B, L, C)
    changed = diff.mean(dim=(0, 1))  # (C,)
    # Channel 0 changed noticeably
    assert changed[0] > 1e-6
    # Other channels nearly unchanged
    assert torch.all(changed[1:] < 1e-6)


def test_temporal_attention_identity_when_v_and_out_zero():
    B, L, C, r = 2, 5, 3, 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(B, L, C, r, device=device)
    attn = TemporalAttention(num_channels=C, r=r, r_v=3, d_qk=6).to(device)
    with torch.no_grad():
        # Zero V and out so module reduces to residual
        torch.nn.init.zeros_(attn.to_v.weight)
        if attn.to_v.bias is not None:
            torch.nn.init.zeros_(attn.to_v.bias)
        torch.nn.init.zeros_(attn.out.weight)
        if attn.out.bias is not None:
            torch.nn.init.zeros_(attn.out.bias)
    y = attn(x)
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)


def test_channel_attention_identity_when_v_and_out_zero():
    B, L, C, r = 2, 5, 3, 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(B, L, C, r, device=device)
    attn = ChannelAttention(num_channels=C, r=r, r_v=3, z=6).to(device)
    with torch.no_grad():
        torch.nn.init.zeros_(attn.to_v.weight)
        if attn.to_v.bias is not None:
            torch.nn.init.zeros_(attn.to_v.bias)
        torch.nn.init.zeros_(attn.out.weight)
        if attn.out.bias is not None:
            torch.nn.init.zeros_(attn.out.bias)
    y = attn(x)
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)


def test_forward_backward_gradients_tiny():
    torch.manual_seed(0)
    B, L, C, r = 2, 8, 3, 6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(B, L, C, device=device)
    target = torch.randn(B, L, C, r, device=device)
    model = tiny_model(num_channels=C, r=r).to(device)
    model.train()

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    y = model(x)
    loss = (y - target).pow(2).mean()
    loss.backward()

    total_nonzero = 0
    total_params = 0
    for p in model.parameters():
        total_params += 1
        assert p.grad is not None, "Parameter has no gradient"
        assert torch.isfinite(p.grad).all(), "Gradient contains NaN/Inf"
        if p.grad.abs().sum().item() > 0:
            total_nonzero += 1

    assert total_nonzero > 0, "No parameter received a non-zero gradient"



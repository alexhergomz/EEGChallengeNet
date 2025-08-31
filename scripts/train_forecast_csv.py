import argparse
import math
import os
import random
import sys
from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

# Ensure project root is importable when running this script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import tiny_model
from src.blocks import BlockWiseLinear


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_matrix(path: str) -> np.ndarray:
    try:
        arr = np.loadtxt(path, delimiter=',')
    except Exception:
        arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr.astype(np.float32)


def train_val_test_split_time(T: int, train_ratio: float, val_ratio: float) -> Tuple[int, int]:
    train_T = int(T * train_ratio)
    val_T = int(T * (train_ratio + val_ratio))
    return train_T, val_T


class SlidingWindowDataset(Dataset):
    def __init__(self, ts: np.ndarray, L: int, H: int, stride: int, start: int, end: int):
        # ts: (T, C)
        self.ts = ts
        self.L = L
        self.H = H
        self.stride = stride
        self.start = start
        self.end = end
        self.indices = list(range(start, max(start, end - (L + H)) + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t0 = self.indices[idx]
        x = self.ts[t0:t0 + self.L]  # (L, C)
        y = self.ts[t0 + self.L:t0 + self.L + self.H]  # (H, C)
        # to torch
        x = torch.from_numpy(x).contiguous()  # (L, C)
        y = torch.from_numpy(y).contiguous()  # (H, C)
        return x, y


class ForecastModel(nn.Module):
    def __init__(self, num_channels: int, r: int, horizon: int):
        super().__init__()
        self.backbone = tiny_model(num_channels=num_channels, r=r)
        self.head = BlockWiseLinear(num_channels=num_channels, r_in=r, r_out=horizon, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C)
        feats = self.backbone(x)  # (B, L, C, r)
        last = feats[:, -1]  # (B, C, r)
        # BlockWiseLinear expects (B, L, C, r). Add a singleton L dimension.
        y = self.head(last.unsqueeze(1))  # (B, 1, C, H)
        y = y.squeeze(1)  # (B, C, H)
        return y.permute(0, 2, 1).contiguous()  # (B, H, C)


def zscore_fit(ts_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = ts_train.mean(axis=0, keepdims=True)
    std = ts_train.std(axis=0, keepdims=True) + 1e-8
    return mean, std


def zscore_apply(ts: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (ts - mean) / std


def masked_mae(pred: Tensor, true: Tensor) -> Tensor:
    return (pred - true).abs().mean()


def masked_rmse(pred: Tensor, true: Tensor) -> Tensor:
    return torch.sqrt(((pred - true) ** 2).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description='Train forecasting on multivariate time series CSV/TXT')
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV/TXT with shape (T, C)')
    parser.add_argument('--channels', type=int, default=16, help='Use first C channels')
    parser.add_argument('--L', type=int, default=96, help='Input window length')
    parser.add_argument('--H', type=int, default=24, help='Forecast horizon')
    parser.add_argument('--stride', type=int, default=1, help='Stride between windows')
    parser.add_argument('--r', type=int, default=8, help='Backbone per-channel width')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda')
    print(f'Device: {device}')

    ts = load_matrix(args.data_path)  # (T, C_all)
    C_all = ts.shape[1]
    C = min(args.channels, C_all)
    ts = ts[:, :C]

    # Split by time
    T = ts.shape[0]
    train_T, val_T = train_val_test_split_time(T, args.train_ratio, args.val_ratio)
    ts_train = ts[:train_T]
    ts_val = ts[train_T:val_T]
    ts_test = ts[val_T:]

    # Normalize using train stats only
    mean, std = zscore_fit(ts_train)
    ts_train = zscore_apply(ts_train, mean, std)
    ts_val = zscore_apply(ts_val, mean, std)
    ts_test = zscore_apply(ts_test, mean, std)

    # Datasets
    train_ds = SlidingWindowDataset(ts_train, L=args.L, H=args.H, stride=args.stride, start=0, end=ts_train.shape[0])
    val_ds = SlidingWindowDataset(ts_val, L=args.L, H=args.H, stride=args.stride, start=0, end=ts_val.shape[0])
    test_ds = SlidingWindowDataset(ts_test, L=args.L, H=args.H, stride=args.stride, start=0, end=ts_test.shape[0])

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=pin)

    model = ForecastModel(num_channels=C, r=args.r, horizon=args.H).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    def run_epoch(loader: DataLoader, train: bool) -> Tuple[float, float]:
        if train:
            model.train()
        else:
            model.eval()
        total_mae = 0.0
        total_rmse = 0.0
        total_n = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)  # (B,L,C)
            yb = yb.to(device, non_blocking=True)  # (B,H,C)
            if train:
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    pred = model(xb)
                    loss = ((pred - yb) ** 2).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                mae = masked_mae(pred.detach(), yb)
                rmse = masked_rmse(pred.detach(), yb)
            else:
                with torch.no_grad():
                    pred = model(xb)
                    mae = masked_mae(pred, yb)
                    rmse = masked_rmse(pred, yb)
            n = xb.size(0)
            total_mae += mae.item() * n
            total_rmse += rmse.item() * n
            total_n += n
        return total_mae / max(1, total_n), total_rmse / max(1, total_n)

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        train_mae, train_rmse = run_epoch(train_loader, train=True)
        val_mae, val_rmse = run_epoch(val_loader, train=False)
        print(f'Epoch {epoch:02d}/{args.epochs} | train MAE {train_mae:.4f} RMSE {train_rmse:.4f} | val MAE {val_mae:.4f} RMSE {val_rmse:.4f}')
        best_val = min(best_val, val_mae)

    test_mae, test_rmse = run_epoch(test_loader, train=False)
    print(f'Test  MAE {test_mae:.4f} RMSE {test_rmse:.4f}')


if __name__ == '__main__':
    main()



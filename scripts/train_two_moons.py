import argparse
import math
import os
import random
import sys
from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Ensure project root is importable when running this script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import tiny_model


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CrossFormerClassifier(nn.Module):
    def __init__(self, num_channels: int, r: int, num_classes: int = 2):
        super().__init__()
        self.backbone = tiny_model(num_channels=num_channels, r=r)
        self.head = nn.Linear(r, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C)
        feats = self.backbone(x)  # (B, L, C, r)
        pooled = feats.mean(dim=(1, 2))  # (B, r)
        return self.head(pooled)


def make_two_moons_tensors(n_samples: int = 1200, noise: float = 0.2, test_size: float = 0.2, seed: int = 42) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    # standardize
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - mean) / std
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    # reshape to (N, L=2, C=1)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).view(-1, 2, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).view(-1, 2, 1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    return X_train_t, y_train_t, X_test_t, y_test_t


def accuracy(logits: Tensor, targets: Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny SplitAxis model on Two Moons")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true', help='Force CPU even if CUDA is available')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda')
    print(f"Device: {device}")

    X_train, y_train, X_test, y_test = make_two_moons_tensors(n_samples=1500, noise=0.2, test_size=0.2, seed=args.seed)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=pin)

    model = CrossFormerClassifier(num_channels=1, r=args.r, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * xb.size(0)
            running_acc += accuracy(logits.detach(), yb) * xb.size(0)
            count += xb.size(0)

        train_loss = running_loss / max(1, count)
        train_acc = running_acc / max(1, count)

        # eval
        model.eval()
        eval_loss = 0.0
        eval_acc = 0.0
        eval_count = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                eval_loss += loss.item() * xb.size(0)
                eval_acc += accuracy(logits, yb) * xb.size(0)
                eval_count += xb.size(0)

        val_loss = eval_loss / max(1, eval_count)
        val_acc = eval_acc / max(1, eval_count)
        best_acc = max(best_acc, val_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss {train_loss:.4f} acc {train_acc:.3f} | val_loss {val_loss:.4f} acc {val_acc:.3f} | best {best_acc:.3f}")

    print("Training complete.")


if __name__ == '__main__':
    main()



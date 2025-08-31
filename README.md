SplitAxis Transformer (Core)

Run tests without venv (global environment):

PowerShell:

1) .\scripts\run_tests.ps1

CMD:

1) scripts\run_tests.bat

Or directly:

    python -m pytest -q

Notes:
- Tests and imports work without setting PYTHONPATH (handled via tests/conftest.py).
- Tiny model sizes keep memory small (<2GB VRAM). Use CPU if preferred.

Train on Two Moons (tiny example):

PowerShell:

    python scripts/train_two_moons.py --epochs 20 --batch_size 64 --r 8

CPU only:

    python scripts/train_two_moons.py --cpu

You should see accuracy > 0.9 after a few epochs.

Train on real multivariate time series (CSV/TXT matrix T x C):

Example with the Electricity dataset (after you download and preprocess to a numeric matrix):

    python scripts/train_forecast_csv.py --data_path path/to/electricity.txt --channels 64 --L 96 --H 24 --epochs 5 --r 8

Notes:
- The script expects a plain numeric matrix with shape (T, C). CSV is preferred; TXT whitespace-delimited also works.
- It uses sliding windows, z-score normalization (train split only), and reports MAE/RMSE.
- Reduce `channels`, `L`, `H`, and increase `stride` to fit 2GB VRAM.

Generate a synthetic multivariate time series (AR + seasonal + jumps + piecewise + spikes + correlated):

    python scripts/generate_synthetic.py --out synthetic.csv --meta synthetic_meta.json --T 4000 --C 32

Then train on it:

    python scripts/train_forecast_csv.py --data_path synthetic.csv --channels 16 --L 96 --H 24 --epochs 5 --r 8

Contrastive pretraining (subject/task invariances) on synthetic EEG:

    python scripts/train_contrastive_eeg.py --subjects 8 --tasks 3 --T 2000 --C 16 --L 256 --stride 128 --epochs 5 --r 8 --proj_dim 64

This trains two heads with supervised contrastive loss: one using subject ids as positives, the other using task ids as positives.

Self-distillation (EMA teacher) pretraining on synthetic EEG (dual invariances):

    python scripts/train_selfdistill_eeg.py --subjects 8 --tasks 3 --T 2000 --C 16 --L 256 --stride 128 --epochs 5 --r 8 --proj_dim 64 --ema_tau 0.99 --sink_time 2 --sink_channel 2

This uses a shared student encoder with two outputs (subject/task) and an EMA teacher; losses are cosine matching between student and opposite-view teacher embeddings for each head.

Install and usage (as a package):

Local install:

    pip install -e .

Python usage:

```python
import torch
from src import tiny_model

B,L,C,r = 2, 128, 8, 16
x = torch.randn(B,L,C)
model = tiny_model(num_channels=C, r=r)
emb = model(x)  # (B,L,C,r)
```

CLI training scripts (after clone): see scripts/ section above for forecasting, contrastive, and self-distillation.


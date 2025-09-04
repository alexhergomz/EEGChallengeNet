#!/usr/bin/env bash
set -euo pipefail

# Download EEG2025 mini (no credentials required)
aws s3 cp --recursive s3://nmdatasets/NeurIPS25/R1_mini_L100_bdf ./local_directory --no-sign-request

# Optional dependency for EEG file formats
python -m pip install -q mne

# Train contrastive (SupCon) on local directory
python scripts/train_contrastive_eeg.py \
  --local_dir ./local_directory \
  --C 16 --L 256 --stride 128 \
  --batch_size 64 --epochs 5 \
  --r 16 --proj_dim 64 \
  --loss supcon --no_aug



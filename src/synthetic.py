from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class SyntheticConfig:
    T: int = 2000
    C: int = 16
    seed: int = 42
    # Correlation and noise
    base_correlation: float = 0.3  # average off-diagonal correlation in innovations
    noise_scale: float = 1.0       # std scale for innovations per channel
    # AR(1) coefficients per channel will be sampled in [ar_min, ar_max]
    ar_min: float = 0.4
    ar_max: float = 0.95
    # Seasonality (number of sin components per channel)
    num_seasonal: int = 2
    # Piecewise linear segments (trend)
    num_segments: int = 4
    # Jumps (level shifts)
    num_jumps: int = 8
    jump_scale: float = 2.0
    # Spiking (impulse) outliers
    spike_prob: float = 0.001
    spike_scale: float = 6.0


def _random_correlation_matrix(num_channels: int, base_rho: float, rng: np.random.RandomState) -> np.ndarray:
    base_rho = float(np.clip(base_rho, -0.9, 0.9))
    # Start with constant-correlation matrix
    R = np.full((num_channels, num_channels), base_rho, dtype=np.float64)
    np.fill_diagonal(R, 1.0)
    # Add small random SPD perturbation
    A = rng.normal(0.0, 0.05, size=(num_channels, num_channels))
    P = A @ A.T
    P = P / (np.sqrt(np.outer(np.diag(P), np.diag(P))) + 1e-12)
    R = 0.8 * R + 0.2 * P
    # Project to nearest correlation-like matrix by clipping eigenvalues
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.clip(eigvals, 1e-3, None)
    R = (eigvecs * eigvals) @ eigvecs.T
    D = np.sqrt(np.diag(R))
    R = R / (D[None, :] * D[:, None])
    return R.astype(np.float32)


def _piecewise_linear_trend(T: int, num_segments: int, rng: np.random.RandomState) -> np.ndarray:
    if num_segments <= 1:
        return np.zeros(T, dtype=np.float32)
    # Choose breakpoints
    bps = sorted(rng.choice(np.arange(T // 10, T - T // 10), size=num_segments - 1, replace=False).tolist())
    bps = [0] + bps + [T]
    trend = np.zeros(T, dtype=np.float32)
    current = 0.0
    for i in range(len(bps) - 1):
        start, end = bps[i], bps[i + 1]
        slope = rng.uniform(-0.01, 0.01)
        length = end - start
        segment = current + slope * np.arange(length, dtype=np.float32)
        trend[start:end] = segment
        current = segment[-1]
    return trend


def _seasonal_component(T: int, num_terms: int, rng: np.random.RandomState) -> np.ndarray:
    t = np.arange(T, dtype=np.float32)
    comp = np.zeros(T, dtype=np.float32)
    for _ in range(max(0, num_terms)):
        period = rng.uniform(24.0, 400.0)  # generic seasonal periods
        omega = 2.0 * math.pi / period
        amp = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        comp += amp * np.sin(omega * t + phase)
    return comp


def _apply_jumps(signal: np.ndarray, num_jumps: int, scale: float, rng: np.random.RandomState) -> None:
    T = signal.shape[0]
    if num_jumps <= 0:
        return
    times = rng.choice(np.arange(T // 10, T - T // 10), size=num_jumps, replace=False)
    for tj in times:
        shift = rng.normal(0.0, scale)
        signal[tj:] += shift


def _apply_spikes(X: np.ndarray, prob: float, scale: float, rng: np.random.RandomState) -> None:
    if prob <= 0:
        return
    T, C = X.shape
    mask = rng.binomial(1, p=min(1.0, prob), size=(T, C)).astype(bool)
    spikes = rng.laplace(0.0, scale, size=(T, C)).astype(np.float32)
    X[mask] += spikes[mask]


def generate_synthetic_mts(cfg: SyntheticConfig) -> Tuple[np.ndarray, Dict]:
    rng = np.random.RandomState(cfg.seed)

    T, C = cfg.T, cfg.C

    # Correlated innovation covariance
    R = _random_correlation_matrix(C, cfg.base_correlation, rng)
    # Per-channel noise scales
    scales = rng.uniform(0.5 * cfg.noise_scale, 1.5 * cfg.noise_scale, size=C).astype(np.float32)
    Sigma = (scales[:, None] * R * scales[None, :]).astype(np.float32)
    L_cov = np.linalg.cholesky(Sigma + 1e-6 * np.eye(C, dtype=np.float32)).astype(np.float32)

    # AR(1) coefficients per channel
    phi = rng.uniform(cfg.ar_min, cfg.ar_max, size=C).astype(np.float32)

    # Base containers
    X = np.zeros((T, C), dtype=np.float32)

    # Seasonality and trend per channel
    seasonal = np.stack([_seasonal_component(T, cfg.num_seasonal, rng) for _ in range(C)], axis=1)
    trends = np.stack([_piecewise_linear_trend(T, cfg.num_segments, rng) for _ in range(C)], axis=1)

    # Jumps per channel (level shifts)
    jumps = np.zeros_like(X)
    for c in range(C):
        jc = jumps[:, c]
        _apply_jumps(jc, cfg.num_jumps, cfg.jump_scale, rng)
        jumps[:, c] = jc

    # AR(1) process with correlated noise
    eps = rng.normal(size=(T, C)).astype(np.float32)
    eps = eps @ L_cov.T  # correlated innovations

    for t in range(1, T):
        X[t] = phi * X[t - 1] + eps[t]

    # Compose components
    X = X + seasonal + trends + jumps

    # Spiking outliers
    _apply_spikes(X, cfg.spike_prob, cfg.spike_scale, rng)

    meta = {
        'T': T,
        'C': C,
        'base_correlation': cfg.base_correlation,
        'noise_scale_mean': float(np.mean(scales)),
        'ar_phi_min': float(phi.min()),
        'ar_phi_max': float(phi.max()),
        'num_seasonal': cfg.num_seasonal,
        'num_segments': cfg.num_segments,
        'num_jumps': cfg.num_jumps,
        'spike_prob': cfg.spike_prob,
    }
    return X.astype(np.float32), meta


@dataclass
class EEGConfig:
    T: int = 2000
    C: int = 16
    num_subjects: int = 8
    num_tasks: int = 3
    seed: int = 123
    # subject-level variations
    subj_freq_jitter_hz: float = 1.5
    subj_noise_mean_std: float = 0.3
    subj_noise_scale_jitter: float = 0.4
    subj_slope_jitter: float = 0.01
    # shared task template
    task_base_periods = (64.0, 128.0)
    task_spike_times = (200, 700, 1400)
    task_spike_amp = 5.0
    # AR/correlation
    ar_min: float = 0.4
    ar_max: float = 0.95
    base_correlation: float = 0.3


def generate_synthetic_eeg(cfg: EEGConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: (N, L, C) stacked windows for all subjects and tasks
      subj_ids: (N,) subject id per window
      task_ids: (N,) task id per window
    Each task has the same macro-pattern: shared spikes at fixed times and shared seasonal components,
    but each subject jitters frequencies, slopes, noise mean/scale.
    """
    rng = np.random.RandomState(cfg.seed)

    T, C = cfg.T, cfg.C
    ar_cfg = SyntheticConfig(T=T, C=C, seed=cfg.seed, base_correlation=cfg.base_correlation, ar_min=cfg.ar_min, ar_max=cfg.ar_max,
                              num_seasonal=0, num_segments=2, num_jumps=2, jump_scale=1.5, spike_prob=0.0)

    # Shared task templates
    t = np.arange(T, dtype=np.float32)
    task_templates = []
    for k in range(cfg.num_tasks):
        comp = np.zeros((T, C), dtype=np.float32)
        # shared seasonal across channels (same phases per task), two periods
        for period in cfg.task_base_periods:
            omega = 2.0 * math.pi / period
            comp += np.sin(omega * t + k)[:, None]
        # shared spikes at specific times
        for st in cfg.task_spike_times:
            if 0 <= st < T:
                comp[st:st + 1] += cfg.task_spike_amp
        task_templates.append(comp)

    X_all = []
    subj_ids = []
    task_ids = []
    for sid in range(cfg.num_subjects):
        # subject jitter parameters
        freq_delta = rng.uniform(-cfg.subj_freq_jitter_hz, cfg.subj_freq_jitter_hz)
        noise_mu = rng.normal(0.0, cfg.subj_noise_mean_std)
        noise_scale = 1.0 + rng.uniform(-cfg.subj_noise_scale_jitter, cfg.subj_noise_scale_jitter)
        slope_delta = rng.uniform(-cfg.subj_slope_jitter, cfg.subj_slope_jitter)

        # subject AR base series
        base, _ = generate_synthetic_mts(ar_cfg)
        # per-channel slope tweak
        slopes = rng.uniform(-slope_delta, slope_delta, size=(1, C)).astype(np.float32)
        base = base + slopes * t[:, None]

        # subject noise mean shift
        base = base + noise_mu

        # build tasks from subject base + jittered templates
        for tid, tmpl in enumerate(task_templates):
            # apply a slight frequency jitter by resampling via phase perturbation
            comp = np.zeros_like(tmpl)
            for period in cfg.task_base_periods:
                omega = 2.0 * math.pi / (period + freq_delta)
                comp += np.sin(omega * t + tid)[:, None]
            # spikes are shared; keep from template
            comp += (tmpl
                     - np.sin(2.0 * math.pi / cfg.task_base_periods[0] * t + tid)[:, None]
                     - np.sin(2.0 * math.pi / cfg.task_base_periods[1] * t + tid)[:, None])

            x = base + comp
            # scale noise
            x = x + rng.normal(0.0, noise_scale, size=x.shape).astype(np.float32)
            X_all.append(x)
            subj_ids.append(sid)
            task_ids.append(tid)

    X = np.stack(X_all, axis=0)  # (num_subjects*num_tasks, T, C)
    subj_ids = np.asarray(subj_ids, dtype=np.int64)
    task_ids = np.asarray(task_ids, dtype=np.int64)
    return X, subj_ids, task_ids




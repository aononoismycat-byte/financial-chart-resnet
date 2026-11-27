"""Probabilistic classification metrics and dependence-aware uncertainty (P1 Stage 1).

All functions take binary labels (1 = Up, 0 = Non-up) and scores interpreted as
P(Up). Hard 0/1 predictions are accepted as degenerate scores.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binomtest, rankdata

EPS = 1e-15


def _labels_scores(labels, scores):
    y = np.asarray(labels)
    p = np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or y.size == 0:
        raise ValueError("labels and scores must be nonempty equally sized 1-D arrays")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary 0/1")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("scores must be finite probabilities in [0, 1]")
    return y.astype(np.int8), p


def accuracy(labels, predictions):
    y, d = _labels_scores(labels, predictions)
    return float(np.mean(y == d))


def balanced_accuracy(labels, predictions):
    """Mean of per-class recall; NaN if a class is absent from labels."""
    y, d = _labels_scores(labels, predictions)
    if y.min() == y.max():
        return math.nan
    return float(0.5 * (np.mean(d[y == 1] == 1) + np.mean(d[y == 0] == 0)))


def roc_auc(labels, scores):
    """Mann-Whitney ROC-AUC with average ranks for ties; NaN for one class."""
    y, p = _labels_scores(labels, scores)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return math.nan
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(labels, probabilities):
    y, p = _labels_scores(labels, probabilities)
    return float(np.mean((p - y) ** 2))


def log_loss(labels, probabilities):
    y, p = _labels_scores(labels, probabilities)
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_skill_score(labels, probabilities, reference_probability):
    """1 - Brier(model) / Brier(constant reference probability)."""
    y, _ = _labels_scores(labels, probabilities)
    reference = brier(y, np.full(y.size, float(reference_probability)))
    return math.nan if reference == 0 else 1.0 - brier(y, probabilities) / reference


def calibration_bins(labels, probabilities, n_bins=10):
    """Equal-frequency reliability bins and expected calibration error.

    Samples are sorted by probability (stable) and split into n_bins groups of
    near-equal size, which keeps bins populated when scores cluster near 0.5.
    """
    y, p = _labels_scores(labels, probabilities)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    order = np.argsort(p, kind="stable")
    bins, ece = [], 0.0
    for chunk in np.array_split(order, min(n_bins, y.size)):
        mean_p, observed = float(p[chunk].mean()), float(y[chunk].mean())
        bins.append({"n": int(chunk.size), "mean_probability": mean_p, "observed_up_rate": observed,
                     "min_probability": float(p[chunk].min()), "max_probability": float(p[chunk].max())})
        ece += chunk.size / y.size * abs(observed - mean_p)
    return {"ece": float(ece), "bins": bins}


def binomial_test(labels, predictions, null_accuracy=0.5):
    """Exact two-sided binomial test of accuracy; assumes independent samples."""
    y, d = _labels_scores(labels, predictions)
    correct = int(np.sum(y == d))
    return {"correct": correct, "n": int(y.size), "null_accuracy": float(null_accuracy),
            "p_value": float(binomtest(correct, y.size, null_accuracy).pvalue)}


def summarize(labels, probabilities, predictions, reference_probability, hard_scores=False):
    """Point metrics for one model on one sample group."""
    y, p = _labels_scores(labels, probabilities)
    _, d = _labels_scores(y, predictions)
    calibration = calibration_bins(y, p)
    return {"n": int(y.size), "up_ratio": float(y.mean()), "accuracy": accuracy(y, d),
            "balanced_accuracy": balanced_accuracy(y, d), "roc_auc": roc_auc(y, p),
            "brier": brier(y, p), "brier_skill_score": brier_skill_score(y, p, reference_probability),
            "log_loss": log_loss(y, p), "ece": calibration["ece"], "hard_scores": bool(hard_scores),
            "binomial_vs_50": binomial_test(y, d)}


def date_block_bootstrap(dates, block_length, n_boot, seed, chunk_size=500):
    """Yield arrays of row indices from a circular moving-block bootstrap over dates.

    Blocks are drawn over the sorted unique dates; every row sharing a resampled
    date is kept, preserving cross-sectional dependence. Requires the same number
    of rows on every date (true for the balanced asset calendar used here). Each
    replicate contains exactly as many rows as the input.
    """
    dates = np.asarray(dates)
    unique, inverse, counts = np.unique(dates, return_inverse=True, return_counts=True)
    if counts.min() != counts.max():
        raise ValueError("date_block_bootstrap requires the same number of rows per date")
    if block_length < 1 or n_boot < 1:
        raise ValueError("block_length and n_boot must be positive")
    n_dates, per_date = unique.size, int(counts[0])
    rows_by_date = np.argsort(inverse, kind="stable").reshape(n_dates, per_date)
    n_blocks = math.ceil(n_dates / block_length)
    rng = np.random.default_rng(seed)
    offsets = np.arange(block_length)
    produced = 0
    while produced < n_boot:
        size = min(chunk_size, n_boot - produced)
        starts = rng.integers(0, n_dates, size=(size, n_blocks))
        date_index = ((starts[:, :, None] + offsets) % n_dates).reshape(size, -1)[:, :n_dates]
        for row in rows_by_date[date_index].reshape(size, -1):
            yield row
        produced += size


def percentile_interval(values, level=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [math.nan, math.nan]
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return [float(low), float(high)]


def bootstrap_p_value(differences):
    """Two-sided bootstrap sign p-value for a paired difference (min 1/(B+1))."""
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return math.nan
    tail = min(np.mean(d <= 0), np.mean(d >= 0))
    return float(min(1.0, max(2 * tail, 1 / (d.size + 1))))


def holm(p_values) -> list[float]:
    """Holm step-down adjusted p-values in the original input order."""
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return []
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("p_values must be a 1-D sequence of finite numbers in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, float(p[idx]) * (m - rank))
        adjusted[idx] = min(1.0, running_max)
    return [float(v) for v in adjusted]

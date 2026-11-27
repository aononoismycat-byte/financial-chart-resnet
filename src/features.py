"""Tabular feature engineering from 20-day Close windows and OHLCV history (P1 Stage 3)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import skew

PRIMARY_FEATURES = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_19",
    "vol_5",
    "vol_19",
    "rsi_14",
    "ma_dev_5",
    "ma_dev_10",
    "ma_dev_20",
    "max_dd_20",
    "range_pos_20",
    "skew_19",
    "up_frac_19",
]

EXTENDED_FEATURES = [
    "hl_range_5",
    "volume_z_20",
    "mom_60",
    "vol_60",
]

ASSET_TICKERS = ("SPY", "QQQ", "IWM", "TLT", "GLD")
ASSET_ONEHOT = [f"asset_{t}" for t in ASSET_TICKERS]


def window_features(closes20) -> dict[str, float]:
    """Compute primary price-only features from 20 chronological Close values."""
    c = np.asarray(closes20, dtype=float)
    if c.shape != (20,) or not np.isfinite(c).all() or (c <= 0).any():
        raise ValueError("window_features requires exactly 20 finite positive Close values")

    c19 = float(c[19])
    diffs = np.diff(c)
    lr = np.diff(np.log(c))

    diffs_14 = diffs[-14:]
    mean_gain = float(np.mean(np.maximum(diffs_14, 0.0)))
    mean_loss = float(np.mean(np.maximum(-diffs_14, 0.0)))
    denom = mean_gain + mean_loss
    rsi_14 = 50.0 if denom == 0.0 else 100.0 * mean_gain / denom

    c_min, c_max = float(np.min(c)), float(np.max(c))
    range_pos_20 = 0.5 if c_max == c_min else (c19 - c_min) / (c_max - c_min)

    cummax = np.maximum.accumulate(c)
    max_dd_20 = float(np.min(c / cummax) - 1.0)

    lr_std = float(np.std(lr))
    if lr_std <= 1e-15:
        skew_19 = 0.0
    else:
        val = float(skew(lr, bias=True))
        skew_19 = val if math.isfinite(val) else 0.0

    return {
        "ret_1": float(c19 / c[18] - 1.0),
        "ret_5": float(c19 / c[14] - 1.0),
        "ret_10": float(c19 / c[9] - 1.0),
        "ret_19": float(c19 / c[0] - 1.0),
        "vol_5": float(np.std(lr[-5:], ddof=1)),
        "vol_19": float(np.std(lr, ddof=1)),
        "rsi_14": float(rsi_14),
        "ma_dev_5": float(c19 / np.mean(c[-5:]) - 1.0),
        "ma_dev_10": float(c19 / np.mean(c[-10:]) - 1.0),
        "ma_dev_20": float(c19 / np.mean(c) - 1.0),
        "max_dd_20": max_dd_20,
        "range_pos_20": float(range_pos_20),
        "skew_19": skew_19,
        "up_frac_19": float(np.mean(diffs > 0.0)),
    }


def extended_features(frame_ohlcv, end_idx: int) -> dict[str, float]:
    """Compute exploratory OHLCV + 60-session features ending at end_idx."""
    if end_idx < 59 or end_idx >= len(frame_ohlcv):
        raise ValueError(f"end_idx must be in [59, {len(frame_ohlcv) - 1}], got {end_idx}")

    w5 = frame_ohlcv.iloc[end_idx - 4 : end_idx + 1]
    w20 = frame_ohlcv.iloc[end_idx - 19 : end_idx + 1]
    w60 = frame_ohlcv.iloc[end_idx - 59 : end_idx + 1]

    high5 = w5["High"].to_numpy(dtype=float)
    low5 = w5["Low"].to_numpy(dtype=float)
    close5 = w5["Close"].to_numpy(dtype=float)
    hl_range_5 = float(np.mean((high5 - low5) / close5))

    vol20 = w20["Volume"].to_numpy(dtype=float)
    vol_std = float(np.std(vol20, ddof=1))
    volume_z_20 = 0.0 if vol_std <= 1e-15 else float((vol20[-1] - np.mean(vol20)) / vol_std)

    close60 = w60["Close"].to_numpy(dtype=float)
    if not np.isfinite(close60).all() or (close60 <= 0).any():
        raise ValueError("Close values in 60-session window must be finite and positive")
    mom_60 = float(close60[-1] / close60[0] - 1.0)
    vol_60 = float(np.std(np.diff(np.log(close60)), ddof=1))

    return {
        "hl_range_5": hl_range_5,
        "volume_z_20": volume_z_20,
        "mom_60": mom_60,
        "vol_60": vol_60,
    }


def save_logreg_json(pipeline, feature_names, path) -> None:
    """Serialize a StandardScaler + LogisticRegression pipeline to portable JSON."""
    scaler = pipeline.named_steps["standardscaler"]
    clf = pipeline.named_steps["logisticregression"]
    payload = {
        "feature_names": list(feature_names),
        "classes": [int(c) for c in clf.classes_],
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def predict_logreg_json(path, frame_or_array):
    """Run exact StandardScaler + LogisticRegression inference from saved JSON."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if hasattr(frame_or_array, "columns"):
        x = frame_or_array[spec["feature_names"]].to_numpy(dtype=float)
    else:
        x = np.asarray(frame_or_array, dtype=float)
    z = (x - np.asarray(spec["scaler_mean"], dtype=float)) / np.asarray(spec["scaler_scale"], dtype=float)
    logits = np.asarray(spec["intercept"], dtype=float) + z @ np.asarray(spec["coef"], dtype=float)
    prob_pos = 1.0 / (1.0 + np.exp(-np.clip(logits, -500.0, 500.0)))
    if spec["classes"] == [0, 1]:
        prob_up = prob_pos
    elif spec["classes"] == [1, 0]:
        prob_up = 1.0 - prob_pos
    else:
        raise ValueError(f"Unsupported binary classes: {spec['classes']}")
    pred = (prob_up > 0.5).astype(int)
    return prob_up, pred


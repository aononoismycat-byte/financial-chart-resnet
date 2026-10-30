"""D1 SPY baselines: train/validation only; never evaluate the test split."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.samples import load_checked_spy


def majority_class(train_labels):
    labels = np.asarray(train_labels)
    if labels.size == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("training labels must be nonempty and binary")
    return int(np.count_nonzero(labels == 1) > np.count_nonzero(labels == 0))


def trend_predictions(first_prices, last_prices):
    first, last = np.asarray(first_prices, dtype=float), np.asarray(last_prices, dtype=float)
    if (first.ndim != 1 or last.shape != first.shape or first.size == 0
            or not np.isfinite(first).all() or not np.isfinite(last).all()
            or (first <= 0).any() or (last <= 0).any()):
        raise ValueError("input endpoint prices must be nonempty, finite positive vectors")
    return (last > first).astype(int)


def calculate_metrics(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if (labels.ndim != 1 or predictions.shape != labels.shape or labels.size == 0
            or not np.isin(labels, [0, 1]).all() or not np.isin(predictions, [0, 1]).all()):
        raise ValueError("labels and predictions must be nonempty equally sized binary vectors")
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": int(labels.size), "up_ratio": float(labels.mean()),
            "accuracy": (tp + tn) / labels.size, "precision_up": precision,
            "recall_up": recall, "f1_up": f1, "confusion_matrix": [[tn, fp], [fn, tp]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional new JSON report; existing files are never overwritten")
    args = parser.parse_args()
    raw = load_checked_spy(ROOT / "data/raw/SPY.csv", ROOT / "data/raw/SPY.metadata.json")
    prices = raw.assign(Date=raw.Date.dt.strftime("%Y-%m-%d")).set_index("Date")["Close"]
    rows = pd.read_csv(ROOT / "data/processed/spy_samples.csv")
    rows = rows.loc[rows["split"].isin(["train", "val"])].copy()
    if rows.empty or rows.label_date.duplicated().any():
        raise ValueError("train/validation samples are empty or duplicate")
    first_prices = rows.input_start_date.map(prices)
    end_prices = rows.input_end_date.map(prices)
    if first_prices.isna().any() or end_prices.isna().any():
        raise ValueError("sample input date missing from raw data")
    if not np.allclose(end_prices, rows.close_t, rtol=1e-12, atol=0):
        raise ValueError("sample close_t does not match the input end date")
    # Only the 20-day input's first and last closes are used for predictions.
    rows["trend_prediction"] = trend_predictions(first_prices, end_prices)
    selected = majority_class(rows.loc[rows.split == "train", "label"])
    result = {"majority_class_from_train": selected,
              "confusion_matrix_order": "rows actual [0,1], columns predicted [0,1]",
              "zero_division_convention": 0,
              "results": {}}
    for split in ("train", "val"):
        subset = rows.loc[rows.split == split]
        result["results"][split] = {
            "majority": calculate_metrics(subset.label, np.full(len(subset), selected)),
            "trend_20d": calculate_metrics(subset.label, subset.trend_prediction)}
    report = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            file.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()

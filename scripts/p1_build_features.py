"""P1 Stage 3: build 20-day primary and exploratory OHLCV features for all samples."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.charts import normalized_prices
from src.features import ASSET_ONEHOT, EXTENDED_FEATURES, PRIMARY_FEATURES, extended_features, window_features
from src.samples import iter_samples, load_checked_asset

TICKERS = ("SPY", "QQQ", "IWM", "TLT", "GLD")
NORM_COLS = [f"norm_close_{i:02d}" for i in range(20)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "data/processed/p1_features.csv")
    parser.add_argument("--output-meta", type=Path, default=ROOT / "results/p1/stage3_features_meta.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.overwrite and (args.output_csv.exists() or args.output_meta.exists()):
        raise FileExistsError("Feature outputs exist; pass --overwrite to regenerate")

    summary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
    manifest_path = ROOT / "data/processed/multi_asset_samples.csv"
    if sha256(manifest_path) != summary["freeze"]["samples_sha256"]:
        raise ValueError("Manifest hash mismatch against frozen results/summary.json")
    manifest = pd.read_csv(manifest_path)

    rows = []
    raw_hashes = {}
    for ticker in TICKERS:
        raw_path = ROOT / "data/raw" / f"{ticker}.csv"
        raw_hash = sha256(raw_path)
        if raw_hash != summary["asset_audit"][ticker]["raw_sha256"]:
            raise ValueError(f"{ticker} raw CSV hash mismatch")
        raw_hashes[ticker] = raw_hash
        checked = load_checked_asset(raw_path, raw_path.with_suffix(".metadata.json"), ticker)
        ohlcv = pd.read_csv(raw_path, dtype={"Date": "string"})
        ohlcv["Date_ts"] = pd.to_datetime(ohlcv["Date"], format="%Y-%m-%d", errors="raise")
        if not ohlcv["Date_ts"].equals(checked["Date"]):
            raise ValueError(f"{ticker} OHLCV Date alignment mismatch")
        date_to_idx = {d: i for i, d in enumerate(ohlcv["Date"].tolist())}

        onehot = {f"asset_{t}": int(t == ticker) for t in TICKERS}
        for meta_row, closes in iter_samples(checked, ticker):
            end_idx = date_to_idx[meta_row["input_end_date"]]
            norm = normalized_prices(closes)
            record = {
                "ticker": meta_row["ticker"],
                "label_date": meta_row["label_date"],
                "split": meta_row["split"],
                "label": int(meta_row["label"]),
            }
            for i, val in enumerate(norm):
                record[f"norm_close_{i:02d}"] = float(val)
            record.update(window_features(closes))
            record.update(extended_features(ohlcv, end_idx))
            record.update(onehot)
            rows.append(record)

    feature_df = pd.DataFrame(rows)
    key_cols = ["ticker", "label_date", "split", "label"]
    pd.testing.assert_frame_equal(feature_df[key_cols], manifest[key_cols], check_dtype=False)

    # Gate: (ret_19 > 0) must match Stage 1 trend_pred on every val/test row.
    stage1_preds = pd.read_csv(ROOT / "results/p1/predictions.csv")
    vt = feature_df[feature_df["split"].isin(["val", "test"])].reset_index(drop=True)
    merged = vt[["split", "ticker", "label_date", "ret_19"]].merge(
        stage1_preds[["split", "ticker", "label_date", "trend_pred"]],
        on=["split", "ticker", "label_date"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(stage1_preds):
        raise ValueError("Val/test row count mismatch against results/p1/predictions.csv")
    if not ((merged["ret_19"] > 0).astype(int).to_numpy() == merged["trend_pred"].to_numpy(dtype=int)).all():
        raise ValueError("Gate failed: (ret_19 > 0) does not equal trend_pred")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(args.output_csv, index=False, float_format="%.12g", lineterminator="\n")

    meta = {
        "stage": "P1 Stage 3 feature engineering",
        "manifest_sha256": summary["freeze"]["samples_sha256"],
        "raw_sha256": raw_hashes,
        "features_csv": args.output_csv.relative_to(ROOT).as_posix(),
        "features_sha256": sha256(args.output_csv),
        "rows_total": int(len(feature_df)),
        "rows_by_split": {s: int((feature_df["split"] == s).sum()) for s in ("train", "val", "test")},
        "norm_close_columns": NORM_COLS,
        "primary_features": PRIMARY_FEATURES,
        "extended_features": EXTENDED_FEATURES,
        "asset_onehot": ASSET_ONEHOT,
        "gates": {
            "manifest_identity_passed": True,
            "trend_pred_equality_passed": True,
            "val_test_checked_rows": int(len(merged)),
        },
    }
    args.output_meta.parent.mkdir(parents=True, exist_ok=True)
    args.output_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Gates passed ({len(feature_df)} rows). Saved {args.output_csv} and {args.output_meta}")


if __name__ == "__main__":
    main()

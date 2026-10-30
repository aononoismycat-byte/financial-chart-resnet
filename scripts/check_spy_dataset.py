"""Read-only audit of generated SPY samples, images, and DataLoader batches."""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from PIL import Image, ImageChops
from torch.utils.data import DataLoader

from src.charts import normalized_prices, render_chart
from src.dataset import ChartDataset
from src.samples import load_checked_spy, split_for


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    raw = load_checked_spy(ROOT / "data/raw/SPY.csv", ROOT / "data/raw/SPY.metadata.json")
    raw_dates = raw.Date.dt.strftime("%Y-%m-%d").tolist()
    positions = {date: i for i, date in enumerate(raw_dates)}
    closes = raw.Close.to_numpy()
    metadata = pd.read_csv(ROOT / "data/processed/spy_samples.csv")
    summary = json.loads((ROOT / "data/processed/spy_summary.json").read_text(encoding="utf-8"))
    check(not metadata.label_date.duplicated().any(), "duplicate labels")
    check(len(metadata) == summary["image_count"], "metadata count mismatch")
    expected_labels = {date for date in raw_dates[20:] if split_for(date) is not None}
    check(set(metadata.label_date) == expected_labels, "missing or unexpected label dates")
    check(hashlib.sha256((ROOT / "data/raw/SPY.csv").read_bytes()).hexdigest() == summary["raw_sha256"],
          "raw CSV no longer matches the dataset's recorded checksum")
    clipped = 0
    real_ties = []
    boundaries = {}
    selected = {"2010-01-04", "2022-01-03", "2024-01-02", "2025-12-31", "2020-03-13"}
    for row in metadata.itertuples(index=False):
        check(row.ticker == "SPY", "unexpected ticker")
        i = positions[row.input_end_date]
        dates = row.input_dates.split("|")
        check(i >= 19 and i + 1 < len(raw_dates), f"invalid range for {row.label_date}")
        check(dates == raw_dates[i-19:i+1] and row.input_start_date == dates[0]
              and row.label_date == raw_dates[i+1], f"invalid input/label dates for {row.label_date}")
        check(math.isclose(row.close_t, closes[i], rel_tol=1e-12)
              and math.isclose(row.close_t1, closes[i+1], rel_tol=1e-12)
              and row.label == int(closes[i+1] > closes[i]), f"price/label mismatch for {row.label_date}")
        check(row.split == split_for(row.label_date), f"split mismatch for {row.label_date}")
        values = normalized_prices(closes[i-19:i+1])
        clipped += int((values < 0.8).any() or (values > 1.2).any())
        path = (ROOT / row.image_path).resolve()
        check(path.is_relative_to(ROOT / "data/images/SPY" / row.split) and path.is_file(),
              f"missing or misplaced image for {row.label_date}")
        if closes[i] == closes[i+1]:
            real_ties.append(row.label_date)
        if row.label_date in selected:
            with Image.open(path) as image:
                check(image.mode == "RGB" and image.size == (224, 224), "incorrect image format")
                check(ImageChops.difference(image, render_chart(closes[i-19:i+1])).getbbox() is None,
                      f"image differs from input-window rendering: {row.label_date}")
            boundaries[row.label_date] = {"input_dates": dates, "close_t": float(closes[i]),
                "close_t1": float(closes[i+1]), "hand_checked_label": int(closes[i+1] > closes[i]),
                "split": row.split, "image_path": row.image_path}
    check(set(boundaries) == selected, "a required audit image is missing")
    check(clipped == summary["clipped_windows_total"], "clipped-window count mismatch")
    batch_info = {}
    for split in ("train", "val", "test"):
        rows = metadata[metadata.split == split]
        images = list((ROOT / "data/images/SPY" / split).glob("*.png"))
        check(len(images) == len(rows) and {x.name for x in images} ==
              {Path(x).name for x in rows.image_path}, f"image count mismatch: {split}")
        image_batch, labels = next(iter(DataLoader(ChartDataset(ROOT / "data/processed/spy_samples.csv", split),
                                                   batch_size=4, shuffle=False)))
        check(tuple(image_batch.shape) == (4, 3, 224, 224) and str(image_batch.dtype) == "torch.float32"
              and tuple(labels.shape) == (4,) and str(labels.dtype) == "torch.int64", "batch contract mismatch")
        batch_info[split] = {"images": len(images), "image_shape": list(image_batch.shape),
                             "image_dtype": str(image_batch.dtype), "label_shape": list(labels.shape),
                             "label_dtype": str(labels.dtype)}
    print(json.dumps({"all_sample_rows_checked": len(metadata), "clipped_windows": clipped,
                      "real_tie_dates": real_ties, "boundary_and_image_checks": boundaries,
                      "batches": batch_info}, indent=2))


if __name__ == "__main__":
    main()

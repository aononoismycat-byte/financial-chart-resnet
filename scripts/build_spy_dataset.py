"""Build the fixed SPY dataset once; never download or overwrite existing outputs."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.charts import Y_MAX, Y_MIN, normalized_prices, render_chart
from src.samples import iter_samples, load_checked_spy

RAW = ROOT / "data" / "raw" / "SPY.csv"
METADATA = ROOT / "data" / "raw" / "SPY.metadata.json"
OUTPUT = ROOT / "data" / "processed" / "spy_samples.csv"
SUMMARY = ROOT / "data" / "processed" / "spy_summary.json"
IMAGES = ROOT / "data" / "images" / "SPY"


def main():
    frame = load_checked_spy(RAW, METADATA)
    samples = list(iter_samples(frame))
    if not samples or {row["split"] for row, _ in samples} != {"train", "val", "test"}:
        raise ValueError("all three splits must contain samples")
    for path in (OUTPUT, SUMMARY, IMAGES):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val", "test"):
        (IMAGES / split).mkdir()
    rows = []
    clipped_by_split = {split: 0 for split in ("train", "val", "test")}
    min_normalized, max_normalized = float("inf"), float("-inf")
    for row, closes in samples:
        values = normalized_prices(closes)
        min_normalized = min(min_normalized, float(values.min()))
        max_normalized = max(max_normalized, float(values.max()))
        if (values < Y_MIN).any() or (values > Y_MAX).any():
            clipped_by_split[row["split"]] += 1
        path = IMAGES / row["split"] / f"{row['label_date']}.png"
        render_chart(closes).save(path, format="PNG")
        row["image_path"] = path.relative_to(ROOT).as_posix()
        rows.append(row)
    result = pd.DataFrame(rows)
    with OUTPUT.open("x", encoding="utf-8", newline="") as output:
        result.to_csv(output, index=False)
    split_stats = {split: {"samples": int(len(group)), "up": int(group["label"].sum()),
                           "up_ratio": float(group["label"].mean()),
                           "clipped_windows": clipped_by_split[split]}
                   for split, group in result.groupby("split", sort=False)}
    summary = {"raw_file": RAW.relative_to(ROOT).as_posix(),
               "raw_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
               "raw_rows": len(frame), "raw_first_date": frame["Date"].iloc[0].strftime("%Y-%m-%d"),
               "raw_last_date": frame["Date"].iloc[-1].strftime("%Y-%m-%d"),
               "price_column": "Close", "session_calendar": "NYSE/NYSE Arca, bounded 2009-2025",
               "calendar_sources": ["https://www.nyse.com/publicdocs/nyse/regulation/nyse/NYSE_Rules.pdf",
                                    "https://datashop.cboe.com/faqs"],
               "image_size": [224, 224], "y_min": Y_MIN, "y_max": Y_MAX,
               "image_count": len(rows), "split_statistics": split_stats,
               "clipped_windows_total": sum(clipped_by_split.values()),
               "minimum_normalized_price": min_normalized,
               "maximum_normalized_price": max_normalized}
    with SUMMARY.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Samples: {OUTPUT}\nImages: {IMAGES}\nSummary: {SUMMARY}")


if __name__ == "__main__":
    main()

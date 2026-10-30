"""Stage-C checks; real SPY is read-only and small synthetic data tests rules."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import ImageChops
from torch.utils.data import DataLoader

from src.charts import render_chart
from src.dataset import ChartDataset
from src.samples import (expected_sessions, iter_samples, label_for,
                         load_checked_spy, split_for)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "SPY.csv"
METADATA = ROOT / "data" / "raw" / "SPY.metadata.json"


class SpySamplesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_checked_spy(RAW, METADATA)

    def test_sessions_and_missing_session_rejected(self):
        self.assertEqual(len(self.frame), len(expected_sessions()))
        with tempfile.TemporaryDirectory() as tmp:
            incomplete = Path(tmp) / "SPY.csv"
            pd.read_csv(RAW).drop(index=100).to_csv(incomplete, index=False)
            with self.assertRaisesRegex(ValueError, "session mismatch"):
                load_checked_spy(incomplete, METADATA)

    def test_boundaries_labels_and_no_future_in_image(self):
        self.assertEqual(label_for(100.0, 100.0), 0)  # synthetic tie
        self.assertEqual(label_for(100.0, 100.01), 1)
        self.assertIsNone(split_for("2026-01-02"))
        selected = {row["label_date"]: (row, closes) for row, closes in iter_samples(self.frame)
                    if row["label_date"] in ("2010-01-04", "2022-01-03", "2024-01-02", "2025-12-31")}
        self.assertEqual(set(selected), {"2010-01-04", "2022-01-03", "2024-01-02", "2025-12-31"})
        for date, split in (("2010-01-04", "train"), ("2022-01-03", "val"),
                            ("2024-01-02", "test"), ("2025-12-31", "test")):
            row, closes = selected[date]
            self.assertEqual(len(closes), 20)
            self.assertEqual(len(row["input_dates"].split("|")), 20)
            self.assertEqual(row["split"], split)
            self.assertEqual(row["label"], int(row["close_t1"] > row["close_t"]))
            self.assertLess(row["input_end_date"], date)
        self.assertEqual(selected["2022-01-03"][0]["input_end_date"], "2021-12-31")
        self.assertEqual(selected["2024-01-02"][0]["input_end_date"], "2023-12-29")
        row, closes = selected["2022-01-03"]
        altered = self.frame.copy()
        altered.loc[altered["Date"] == pd.Timestamp(row["label_date"]), "Close"] *= 1.5
        changed_row, changed_closes = next((r, c) for r, c in iter_samples(altered)
                                           if r["label_date"] == row["label_date"])
        self.assertNotEqual(changed_row["close_t1"], row["close_t1"])
        self.assertIsNone(ImageChops.difference(render_chart(closes), render_chart(changed_closes)).getbbox())

    def test_dataset_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = render_chart([100.0 + i for i in range(20)])
            image.save(root / "sample.png")
            csv = root / "samples.csv"
            pd.DataFrame([{"split": "train", "image_path": "sample.png", "label": 0},
                          {"split": "train", "image_path": "sample.png", "label": 1}]).to_csv(csv, index=False)
            images, labels = next(iter(DataLoader(ChartDataset(csv, "train", root), batch_size=2)))
            self.assertEqual(tuple(images.shape), (2, 3, 224, 224))
            self.assertEqual(images.dtype, torch.float32)
            self.assertEqual(tuple(labels.shape), (2,))
            self.assertEqual(labels.dtype, torch.long)
            self.assertTrue(bool(torch.all((images >= 0) & (images <= 1))))


if __name__ == "__main__":
    unittest.main(verbosity=2)

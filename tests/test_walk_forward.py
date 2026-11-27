"""Synthetic unit tests for P1 Stage 4 walk-forward fold construction."""
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.p1_walk_forward import build_walk_forward_folds

ROOT = Path(__file__).resolve().parent.parent
S4_PREDS = ROOT / "results/p1/stage4_walk_forward_predictions.csv"


class WalkForwardTest(unittest.TestCase):
    def test_fold_construction_no_overlap(self):
        years = np.repeat(np.arange(2010, 2026), 10)
        folds = build_walk_forward_folds(years, start_test_year=2014, end_test_year=2025, min_train_year=2010)
        self.assertEqual(len(folds), 12)
        for fold in folds:
            ty = fold["test_year"]
            tr, va, te = fold["train"], fold["val"], fold["test"]
            self.assertTrue((years[tr] <= ty - 2).all())
            self.assertTrue((years[tr] >= 2010).all())
            self.assertTrue((years[va] == ty - 1).all())
            self.assertTrue((years[te] == ty).all())
            self.assertFalse((tr & va).any())
            self.assertFalse((tr & te).any())
            self.assertFalse((va & te).any())


@unittest.skipUnless(S4_PREDS.is_file(), "Stage 4 predictions not generated yet")
class CommittedStage4Test(unittest.TestCase):
    def test_walk_forward_predictions_hash_and_no_prices(self):
        meta = json.loads((S4_PREDS.parent / "stage4_walk_forward_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(S4_PREDS.read_bytes()).hexdigest(), meta["walk_forward_predictions_sha256"])
        df = pd.read_csv(S4_PREDS)
        self.assertFalse({"close_t", "close_t1", "Close", "Adj Close"} & set(df.columns))
        self.assertEqual(sorted(df["test_year"].unique().tolist()), list(range(2014, 2026)))


if __name__ == "__main__":
    unittest.main(verbosity=2)

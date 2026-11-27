"""Synthetic unit tests for Stage 8 holdout calendar and overlap verification."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import scripts.p1_download_holdout as hd


class HoldoutVerificationTest(unittest.TestCase):
    def test_expected_2026_sessions_excludes_holidays_and_bounds(self):
        sessions = hd.expected_2026_sessions()
        s_set = set(sessions.strftime("%Y-%m-%d"))
        self.assertIn("2026-01-02", s_set)
        self.assertIn("2026-11-25", s_set)
        self.assertNotIn("2026-01-01", s_set)
        self.assertNotIn("2026-11-26", s_set)
        for h in hd.HOLIDAYS_2026:
            self.assertNotIn(h, s_set)

    def test_verify_holdout_file_overlap_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/raw").mkdir(parents=True)
            overlap_dates = pd.bdate_range("2025-10-01", "2025-12-31").strftime("%Y-%m-%d")
            orig_df = pd.DataFrame(
                {
                    "Date": overlap_dates,
                    "Close": np.linspace(100.0, 110.0, len(overlap_dates)),
                    "Adj Close": np.linspace(100.0, 110.0, len(overlap_dates)),
                }
            )
            orig_df.to_csv(root / "data/raw/SPY.csv", index=False)

            s2026 = hd.expected_2026_sessions().strftime("%Y-%m-%d")
            hold_dates = list(overlap_dates) + list(s2026)
            hold_df = pd.DataFrame(
                {
                    "Date": hold_dates,
                    "Close": np.r_[orig_df["Close"].to_numpy(), np.linspace(111.0, 120.0, len(s2026))],
                    "Adj Close": np.r_[orig_df["Adj Close"].to_numpy(), np.linspace(111.0, 120.0, len(s2026))],
                }
            )
            with patch.object(hd, "ROOT", root):
                info = hd.verify_holdout_file("SPY", hold_df)
                self.assertEqual(info["overlap_rows"], len(overlap_dates))
                self.assertEqual(info["sessions_2026"], len(s2026))

                bad_overlap = hold_df.copy()
                bad_overlap.loc[0, "Close"] *= 1.01
                with self.assertRaisesRegex(ValueError, "overlap Close values differ"):
                    hd.verify_holdout_file("SPY", bad_overlap)

                bad_cal = hold_df.iloc[:-1].copy()
                with self.assertRaisesRegex(ValueError, "2026 session mismatch"):
                    hd.verify_holdout_file("SPY", bad_cal)


S8_PREDS = Path(__file__).resolve().parent.parent / "results/p1/stage8_holdout_predictions.csv"


@unittest.skipUnless(S8_PREDS.is_file(), "Stage 8 holdout predictions not generated yet")
class CommittedStage8Test(unittest.TestCase):
    def test_stage8_predictions_hash_and_no_prices(self):
        meta = json.loads((S8_PREDS.parent / "stage8_holdout_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(S8_PREDS.read_bytes()).hexdigest(), meta["predictions_sha256"])
        df = pd.read_csv(S8_PREDS)
        self.assertFalse({"close_t", "close_t1", "Close", "Adj Close"} & set(df.columns))
        self.assertEqual(len(df), 1135)
        self.assertTrue(df["label_date"].min() >= "2026-01-02")
        self.assertTrue(df["label_date"].max() <= "2026-11-25")


if __name__ == "__main__":
    unittest.main(verbosity=2)

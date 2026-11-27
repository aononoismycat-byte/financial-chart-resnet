"""Synthetic unit tests for src.features."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (
    EXTENDED_FEATURES,
    PRIMARY_FEATURES,
    extended_features,
    predict_logreg_json,
    save_logreg_json,
    window_features,
)


class FeaturesTest(unittest.TestCase):
    def test_linear_ramp_and_constant_series(self):
        ramp = np.linspace(100.0, 119.0, 20)
        f = window_features(ramp)
        self.assertEqual(set(f), set(PRIMARY_FEATURES))
        self.assertEqual(f["up_frac_19"], 1.0)
        self.assertEqual(f["rsi_14"], 100.0)
        self.assertEqual(f["max_dd_20"], 0.0)
        self.assertAlmostEqual(f["range_pos_20"], 1.0, places=12)
        self.assertAlmostEqual(f["ret_19"], 0.19, places=12)
        self.assertTrue(f["ret_19"] > 0)

        flat = np.full(20, 100.0)
        ff = window_features(flat)
        self.assertEqual(ff["range_pos_20"], 0.5)
        self.assertEqual(ff["rsi_14"], 50.0)
        self.assertEqual(ff["skew_19"], 0.0)
        self.assertEqual(ff["max_dd_20"], 0.0)
        self.assertEqual(ff["vol_5"], 0.0)

    def test_ret19_sign_matches_trend_and_extended_features(self):
        rng = np.random.default_rng(7)
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 70)))
        frame = pd.DataFrame(
            {
                "Open": closes * 0.995,
                "High": closes * 1.01,
                "Low": closes * 0.99,
                "Close": closes,
                "Volume": np.linspace(1e6, 2e6, 70),
            }
        )
        wf = window_features(closes[-20:])
        self.assertEqual(int(wf["ret_19"] > 0), int(closes[-1] > closes[-20]))
        ef = extended_features(frame, 69)
        self.assertEqual(set(ef), set(EXTENDED_FEATURES))
        self.assertAlmostEqual(ef["hl_range_5"], 0.02, places=12)
        self.assertAlmostEqual(ef["mom_60"], closes[-1] / closes[10] - 1.0, places=12)
        self.assertTrue(all(np.isfinite(v) for v in ef.values()))

    def test_logreg_json_roundtrip(self):
        rng = np.random.default_rng(11)
        x = pd.DataFrame(rng.normal(size=(80, 3)), columns=["f1", "f2", "f3"])
        y = (x["f1"] - 0.5 * x["f2"] > 0).astype(int).to_numpy()
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42))
        pipe.fit(x, y)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "lr.json"
            save_logreg_json(pipe, ["f1", "f2", "f3"], p)
            prob, pred = predict_logreg_json(p, x)
        np.testing.assert_allclose(prob, pipe.predict_proba(x)[:, 1], atol=1e-12)
        np.testing.assert_array_equal(pred, pipe.predict(x))


S3_PREDS = Path(__file__).resolve().parent.parent / "results/p1/stage3_predictions.csv"


@unittest.skipUnless(S3_PREDS.is_file(), "Stage 3 predictions not generated yet")
class CommittedStage3Test(unittest.TestCase):
    def test_stage3_predictions_hash_and_no_prices(self):
        meta = json.loads((S3_PREDS.parent / "stage3_predictions_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(S3_PREDS.read_bytes()).hexdigest(), meta["stage3_predictions_sha256"])
        self.assertTrue(meta["gates"]["logreg_closes_exact_pred_match"])
        df = pd.read_csv(S3_PREDS)
        self.assertFalse({"close_t", "close_t1", "Close", "Adj Close"} & set(df.columns))
        self.assertEqual(df.groupby("split").size().to_dict(), {"test": 2510, "val": 2505})


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Tests for src.evaluation shared evaluation, auc_vs_half and markdown export."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import auc_vs_half, evaluate_predictions, write_metrics_markdown


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        dates = pd.bdate_range("2024-01-02", periods=60).strftime("%Y-%m-%d")
        rows = []
        for date in dates:
            for ticker in ("AAA", "BBB", "CCC"):
                label = int(rng.random() < 0.55)
                p1 = float(np.clip(0.5 + 0.15 * (label - 0.5) + rng.normal(0, 0.1), 0.05, 0.95))
                p2 = float(np.clip(0.48 + 0.05 * (label - 0.5) + rng.normal(0, 0.1), 0.05, 0.95))
                rows.append(
                    {
                        "label_date": date,
                        "ticker": ticker,
                        "label": label,
                        "clim": 0.52,
                        "m1_prob": p1,
                        "m1_pred": int(p1 > 0.5),
                        "m2_prob": p2,
                        "m2_pred": int(p2 > 0.5),
                    }
                )
        self.frame = pd.DataFrame(rows)
        self.models = {
            "m1": {"prob": "m1_prob", "pred": "m1_pred", "hard": False, "label": "Model One"},
            "m2": {"prob": "m2_prob", "pred": "m2_pred", "hard": False, "label": "Model Two"},
        }
        self.settings = {"block_length_sessions": 5, "replicates": 50, "seed": 42, "interval": "95% percentile"}

    def test_evaluate_predictions_end_to_end_and_markdown(self):
        res = evaluate_predictions(
            self.frame,
            self.models,
            reference_probability=0.52,
            settings=self.settings,
            groups=("overall", "AAA", "BBB", "CCC"),
            comparisons=[("m1", "m2")],
            bootstrap=True,
        )
        self.assertEqual(set(res["groups"]), {"overall", "AAA", "BBB", "CCC"})
        self.assertEqual(res["groups"]["overall"]["m1"]["metrics"]["n"], 180)
        self.assertIn("roc_auc", res["groups"]["overall"]["m1"]["ci"])
        self.assertEqual(len(res["comparisons"]["overall"]), 5)
        self.assertGreater(res["groups"]["overall"]["m1"]["metrics"]["roc_auc"], 0.6)

        # Also verify column-based reference probability and auc_vs_half
        res_col = evaluate_predictions(
            self.frame,
            self.models,
            reference_probability="clim",
            settings=self.settings,
            groups=("overall",),
            comparisons=[],
            bootstrap=False,
        )
        self.assertAlmostEqual(
            res_col["groups"]["overall"]["m1"]["metrics"]["brier_skill_score"],
            res["groups"]["overall"]["m1"]["metrics"]["brier_skill_score"],
            places=12,
        )

        half = auc_vs_half(self.frame, self.models["m1"], settings=self.settings)
        self.assertAlmostEqual(half["difference"], half["roc_auc"] - 0.5, places=12)
        self.assertGreater(half["roc_auc"], 0.5)
        self.assertGreater(half["p_boot"], 0.0)
        self.assertLessEqual(half["p_boot"], 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "summary.md"
            write_metrics_markdown(res, md_path, "Synthetic Eval", ["Intro line."], validation_result=res_col)
            text = md_path.read_text(encoding="utf-8")
            self.assertIn("Model One", text)
            self.assertIn("Model One − Model Two", text)
            self.assertIn("Validation context", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

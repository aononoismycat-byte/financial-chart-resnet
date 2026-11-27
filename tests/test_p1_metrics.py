"""Tests for P1 Stage 1 metrics, bootstrap and committed prediction artifacts."""
import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss as sk_log_loss, roc_auc_score

from src.metrics import (balanced_accuracy, binomial_test, bootstrap_p_value, brier, brier_skill_score,
                         calibration_bins, date_block_bootstrap, holm, log_loss, percentile_interval, roc_auc,
                         summarize)

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / 'results/p1/predictions.csv'


class MetricsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = rng.integers(0, 2, 400)
        self.p = np.clip(0.5 + 0.2 * (self.y - 0.5) + rng.normal(0, 0.2, 400), 0, 1)
        self.d = (self.p > 0.5).astype(int)

    def test_matches_sklearn(self):
        self.assertAlmostEqual(roc_auc(self.y, self.p), roc_auc_score(self.y, self.p), places=12)
        self.assertAlmostEqual(balanced_accuracy(self.y, self.d), balanced_accuracy_score(self.y, self.d), places=12)
        self.assertAlmostEqual(brier(self.y, self.p), brier_score_loss(self.y, self.p), places=12)
        self.assertAlmostEqual(log_loss(self.y, np.clip(self.p, 0.01, 0.99)),
                               sk_log_loss(self.y, np.clip(self.p, 0.01, 0.99)), places=10)

    def test_degenerate_scores(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(roc_auc(y, [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(roc_auc(y, [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertEqual(roc_auc(y, [0.55] * 4), 0.5)  # constant forecasts carry no ranking
        self.assertEqual(balanced_accuracy(y, [1, 1, 1, 1]), 0.5)  # majority has no skill here
        self.assertTrue(math.isnan(roc_auc([1, 1], [0.2, 0.7])))
        self.assertAlmostEqual(brier_skill_score(y, [0.6] * 4, 0.6), 0.0)
        with self.assertRaises(ValueError):
            brier(y, [0.1, 0.2, 1.2, 0.5])
        with self.assertRaises(ValueError):
            roc_auc([0, 2, 1, 1], [0.1, 0.2, 0.3, 0.4])

    def test_calibration_and_binomial(self):
        bins = calibration_bins(self.y, self.p, n_bins=10)
        self.assertEqual(sum(b['n'] for b in bins['bins']), self.y.size)
        y = np.array([1, 1, 1, 0])
        self.assertAlmostEqual(calibration_bins(y, [0.5] * 4, n_bins=1)['ece'], 0.25)
        self.assertAlmostEqual(calibration_bins(y, [0.5] * 4, n_bins=10)['ece'], 0.5)  # one sample per bin
        self.assertAlmostEqual(binomial_test([1] * 60 + [0] * 40, [1] * 100)['p_value'], 0.056887, places=5)
        summary = summarize(self.y, self.p, self.d, 0.5)
        self.assertEqual(summary['n'], 400)
        self.assertIn('binomial_vs_50', summary)


class BootstrapTest(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range('2024-01-01', periods=50).strftime('%Y-%m-%d')
        self.frame = pd.DataFrame({'date': np.repeat(dates, 3), 'asset': np.tile(['A', 'B', 'C'], 50)})

    def test_deterministic_and_keeps_dates_together(self):
        first = list(date_block_bootstrap(self.frame.date, 5, 20, seed=7, chunk_size=6))
        second = list(date_block_bootstrap(self.frame.date, 5, 20, seed=7))
        self.assertEqual(len(first), 20)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.size, len(self.frame))
            counts = self.frame.iloc[a].groupby('date').asset.agg(lambda s: tuple(sorted(s.value_counts().values)))
            self.assertTrue(all(len(set(c)) == 1 and len(c) == 3 for c in counts))

    def test_blocks_are_contiguous_and_circular(self):
        (rep,) = date_block_bootstrap(self.frame.date, 50, 1, seed=3)
        order = pd.Index(sorted(self.frame.date.unique()))
        positions = order.get_indexer(self.frame.date.iloc[rep][::3])
        self.assertTrue(all((b - a) % 50 == 1 for a, b in zip(positions, positions[1:])))

    def test_rejects_unbalanced_dates_and_summaries(self):
        with self.assertRaises(ValueError):
            next(date_block_bootstrap(['d1', 'd1', 'd2'], 1, 1, seed=0))
        self.assertEqual(bootstrap_p_value(np.ones(99)), 0.01)
        self.assertEqual(bootstrap_p_value(np.r_[np.ones(50), -np.ones(50)]), 1.0)
        np.testing.assert_allclose(percentile_interval(np.arange(101)), [2.5, 97.5])
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        np.testing.assert_allclose(holm([0.8, 0.9]), [1.0, 1.0])
        self.assertEqual(holm([]), [])
        with self.assertRaises(ValueError):
            holm([0.1, 1.2])


@unittest.skipUnless(PREDICTIONS.is_file(), 'Stage 1 predictions not exported yet')
class CommittedPredictionsTest(unittest.TestCase):
    def test_predictions_match_metadata_and_contain_no_prices(self):
        meta = json.loads((PREDICTIONS.parent / 'predictions_meta.json').read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest(), meta['predictions_sha256'])
        self.assertTrue(meta['reproduction_gate']['passed'])
        frame = pd.read_csv(PREDICTIONS)
        self.assertFalse({'close_t', 'close_t1', 'input_dates'} & set(frame.columns))
        self.assertEqual(frame.groupby('split').size().to_dict(), {'test': 2510, 'val': 2505})
        self.assertTrue(frame.groupby(['split', 'label_date']).size().eq(5).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)

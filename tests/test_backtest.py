"""Synthetic unit tests for src.backtest."""
import math
import unittest

import numpy as np
import pandas as pd

from src.backtest import performance, portfolio, sleeve_returns


class BacktestTest(unittest.TestCase):
    def test_hand_computed_costs_and_turnover(self):
        pos = pd.Series([1, 1, 0, 1], index=["d1", "d2", "d3", "d4"])
        ret = pd.Series([0.01, -0.02, 0.03, 0.005], index=pos.index)
        df = sleeve_returns(pos, ret, cost_bps=10.0, delay=0)
        np.testing.assert_allclose(df["turnover"], [1.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(df["gross"], [0.01, -0.02, 0.0, 0.005])
        np.testing.assert_allclose(df["net"], [0.009, -0.02, -0.001, 0.004])

    def test_delay_shift(self):
        pos = [1, 1, 0, 1]
        ret = [0.02, 0.01, -0.03, 0.04]
        df = sleeve_returns(pos, ret, cost_bps=10.0, delay=1)
        np.testing.assert_allclose(df["position"], [0.0, 1.0, 1.0, 0.0])
        np.testing.assert_allclose(df["turnover"], [0.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(df["net"], [0.0, 0.009, -0.03, -0.001])

    def test_buy_and_hold_cost_and_drawdown_and_constant_sharpe(self):
        pos = [1, 1, 1]
        ret = [0.1, -0.2, 0.05]
        df = sleeve_returns(pos, ret, cost_bps=5.0, delay=0)
        np.testing.assert_allclose(df["turnover"], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(df["net"], [0.1 - 0.0005, -0.2, 0.05])

        perf = performance([0.1, -0.2, 0.05], [1, 1, 1], [1, 0, 0])
        self.assertAlmostEqual(perf["max_drawdown"], -0.20, places=12)
        self.assertAlmostEqual(perf["hit_rate"], 2.0 / 3.0, places=12)
        self.assertAlmostEqual(perf["exposure"], 1.0, places=12)

        const_perf = performance([0.001, 0.001, 0.001, 0.001], [1, 1, 1, 1], [1, 0, 0, 0])
        self.assertTrue(math.isnan(const_perf["sharpe"]))
        self.assertTrue(math.isnan(const_perf["max_drawdown"]) is False)
        self.assertEqual(const_perf["max_drawdown"], 0.0)

    def test_equal_weight_portfolio(self):
        idx = pd.Index(["d1", "d2"])
        s1 = sleeve_returns(pd.Series([1, 1], index=idx), pd.Series([0.02, -0.01], index=idx), cost_bps=0.0)
        s2 = sleeve_returns(pd.Series([0, 1], index=idx), pd.Series([0.04, 0.03], index=idx), cost_bps=0.0)
        port = portfolio({"A": s1, "B": s2})
        np.testing.assert_allclose(port["net"], [0.01, 0.01])
        np.testing.assert_allclose(port["position"], [0.5, 1.0])
        np.testing.assert_allclose(port["turnover"], [0.5, 0.5])


if __name__ == "__main__":
    unittest.main(verbosity=2)

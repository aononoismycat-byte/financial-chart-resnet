import unittest

from scripts.evaluate_spy_baselines import calculate_metrics, majority_class, trend_predictions


class BaselineTest(unittest.TestCase):
    def test_majority_ties_choose_zero(self):
        self.assertEqual(majority_class([0, 1]), 0)
        self.assertEqual(majority_class([0, 1, 1]), 1)
        with self.assertRaises(ValueError):
            majority_class([])

    def test_confusion_matrix_and_metrics(self):
        result = calculate_metrics([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        for name in ("accuracy", "precision_up", "recall_up", "f1_up"):
            self.assertEqual(result[name], 0.5)
        no_positive = calculate_metrics([0, 1], [0, 0])
        self.assertEqual(no_positive["precision_up"], 0.0)
        self.assertEqual(no_positive["f1_up"], 0.0)

    def test_trend_only_compares_input_endpoints(self):
        self.assertEqual(trend_predictions([100, 100, 100], [101, 100, 99]).tolist(), [1, 0, 0])
        with self.assertRaises(ValueError):
            trend_predictions([100], [float("nan")])


if __name__ == "__main__":
    unittest.main(verbosity=2)

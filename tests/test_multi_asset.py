"""Checks for independent assets and pooled metric alignment."""
import unittest
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.samples import iter_samples
from scripts.train_multi_asset import detailed_inference, add_asset_losses

class MultiAssetTest(unittest.TestCase):
    def test_each_ticker_uses_its_own_window_and_next_close(self):
        dates = pd.bdate_range('2021-12-01', periods=30)
        a = pd.DataFrame({'Date': dates, 'Close': list(range(100,130))})
        b = pd.DataFrame({'Date': dates, 'Close': list(range(200,170,-1))})
        ar, ac = next(iter_samples(a, 'AAA'))
        br, bc = next(iter_samples(b, 'BBB'))
        self.assertEqual((ar['ticker'], br['ticker']), ('AAA','BBB'))
        self.assertEqual((ar['label'], br['label']), (1,0))
        self.assertEqual((len(ac), len(bc)), (20,20))
        self.assertEqual((ac[-1], bc[-1]), (ar['close_t'], br['close_t']))
        self.assertLess(ar['input_end_date'], ar['label_date'])

    def test_inference_and_asset_losses_keep_sample_order(self):
        logits = torch.tensor([[2., 0.], [0., 2.], [1., 0.], [0., 1.], [0., 3.]])
        labels = torch.tensor([0, 1, 1, 0, 1])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2, shuffle=False)
        result = detailed_inference(torch.nn.Identity(), loader, torch.device('cpu'))
        self.assertEqual(result['labels'], labels.tolist())
        self.assertEqual(result['predictions'], [0, 1, 0, 1, 1])
        self.assertAlmostEqual(result['loss'], torch.nn.functional.cross_entropy(logits, labels).item(), places=6)
        rows = pd.DataFrame({'ticker': ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']})
        metrics = {ticker: {'resnet18': {}} for ticker in ['overall', *rows.ticker]}
        add_asset_losses(metrics, rows, result['losses'])
        self.assertAlmostEqual(metrics['overall']['resnet18']['loss'], result['loss'])
        self.assertAlmostEqual(metrics['IWM']['resnet18']['loss'], result['losses'][2])

if __name__ == '__main__':
    unittest.main()

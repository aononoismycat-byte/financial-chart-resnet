"""Tests for inference inputs and the fixed numeric baseline."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from scripts.predict import read_closes, predict
from scripts.numeric_baseline_posthoc import fit_model, evaluate
from src.charts import render_chart
from src.dataset import ChartDataset
from scripts.train_spy_resnet import NormalizedDataset
from torchvision.transforms import Normalize
from scripts.train_spy_resnet import MEAN, STD

class UpgradeTest(unittest.TestCase):
    def test_prediction_class_zero_is_non_up_including_score_tie(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / 'model.pt'
            checkpoint.touch()
            saved = {'model_config': {'architecture': 'resnet18', 'num_classes': 2,
                                      'training_mode': 'classifier_only'}, 'model_state_dict': {}}
            for logits, expected in (([2., 1.], 'Non-up'), ([1., 2.], 'Up'), ([1., 1.], 'Non-up')):
                with patch('scripts.predict.torch.load', return_value=saved), patch('scripts.predict.ResNet18Classifier') as model:
                    model.return_value.return_value = torch.tensor([logits])
                    result = predict([100.] * 20, checkpoint)
                self.assertEqual(result['prediction'], expected)
                self.assertIn('probability_non_up', result)
                self.assertNotIn('probability_down', result)

    def test_input_validation_and_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'input.csv'
            p.write_text('Close\n'+'\n'.join(['100']*20)+'\n')
            self.assertEqual(len(read_closes(p)), 20)
            with self.assertRaisesRegex(FileNotFoundError, 'Trained checkpoint not found'):
                predict(read_closes(p), Path(temp)/'missing.pt')
            for invalid in (['100']*19, ['100']*19+['nan'], ['100']*19+['0'], ['100']*19+['inf']):
                p.write_text('Close\n'+'\n'.join(invalid)+'\n')
                with self.assertRaises(ValueError):
                    read_closes(p)
            p.write_text('price\n'+'\n'.join(['100']*20)+'\n')
            with self.assertRaises(ValueError):
                read_closes(p)

    def test_chart_tensor_matches_training_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = render_chart(np.arange(100,120))
            image.save(root/'chart.png')
            pd.DataFrame([{'split':'train','image_path':'chart.png','label':1}]).to_csv(root/'rows.csv', index=False)
            training, _ = NormalizedDataset(ChartDataset(root/'rows.csv', 'train', root))[0]
            direct = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2,0,1).float().div_(255.)
            self.assertTrue(torch.equal(training, Normalize(MEAN, STD)(direct)))

    def test_numeric_fit_train_only_and_metric_grouping(self):
        x_train = np.array([[0.,1.],[1.,0.],[2.,1.],[3.,0.]])
        model = fit_model(x_train, np.array([0,0,1,1]))
        np.testing.assert_allclose(model.named_steps['standardscaler'].mean_, x_train.mean(axis=0))
        x_val = np.array([[100., 100.],[101., 101.],[102., 102.],[103., 103.],[104., 104.]])
        before = model.named_steps['standardscaler'].mean_.copy()
        predictions = model.predict(x_val)
        np.testing.assert_array_equal(model.named_steps['standardscaler'].mean_, before)
        rows = pd.DataFrame({'ticker':['SPY','QQQ','IWM','TLT','GLD'], 'label':[0,1,0,1,1]})
        metrics = evaluate(rows, predictions)
        self.assertEqual(metrics['overall']['n'], 5)
        self.assertEqual(sum(map(sum, metrics['overall']['confusion_matrix'])), 5)
        self.assertTrue(all(metrics[t]['n'] == 1 for t in rows.ticker))

if __name__ == '__main__':
    unittest.main()

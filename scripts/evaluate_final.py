"""Freeze the selected checkpoints and evaluate the held-out test split once."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.dataset import ChartDataset
from src.model import ResNet18Classifier
from src.training import load_checkpoint
from scripts.train_spy_resnet import NormalizedDataset, sha256
from scripts.train_multi_asset import metrics_by_asset, detailed_inference, add_asset_losses
from scripts.evaluate_spy_baselines import majority_class

OUT = ROOT/'experiments/final_evaluation'
SAMPLES = ROOT/'data/processed/multi_asset_samples.csv'


def main():
    d2 = ROOT/'experiments/spy_resnet18_d2_seed42'
    full = ROOT/'experiments/spy_resnet18_full_finetune_seed42'
    pooled = ROOT/'experiments/multi_asset_resnet18_seed42'
    protocol = json.loads((ROOT/'experiments/research_protocol.json').read_text(encoding='utf-8-sig'))
    pooled_config = json.loads((pooled/'config.json').read_text())
    mode = pooled_config['mode']
    spy = full if mode == 'full_finetune' else d2
    for folder in (spy, pooled):
        if not json.loads((folder/'validation_result.json').read_text())['reload_verified']:
            raise ValueError('Checkpoint validation reload not verified')
    OUT.mkdir(parents=True, exist_ok=False)
    freeze = {'protocol': protocol, 'selected_mode': mode,
              'spy_checkpoint': spy.relative_to(ROOT).as_posix()+'/best_model.pt',
              'spy_checkpoint_sha256': sha256(spy/'best_model.pt'),
              'pooled_checkpoint': pooled.relative_to(ROOT).as_posix()+'/best_model.pt',
              'pooled_checkpoint_sha256': sha256(pooled/'best_model.pt'),
              'samples_sha256': sha256(SAMPLES), 'pooled_config': pooled_config,
              'spy_validation_result': json.loads((spy/'validation_result.json').read_text()),
              'pooled_validation_best_epoch': json.loads((pooled/'validation_result.json').read_text())['best_epoch']}
    (OUT/'freeze.json').write_text(json.dumps(freeze, indent=2)+'\n', encoding='utf-8')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rows = pd.read_csv(SAMPLES)
    test = rows[rows.split == 'test'].reset_index(drop=True)
    training = rows[rows.split == 'train']
    pooled_majority = majority_class(training.label)
    spy_majority = majority_class(training.loc[training.ticker == 'SPY','label'])
    results = {}
    for name, folder, data, majority in [('spy', spy, test[test.ticker == 'SPY'], spy_majority),
                                          ('pooled', pooled, test, pooled_majority)]:
        data = data.reset_index(drop=True)
        manifest = OUT/(name+'_test_manifest.csv')
        data.to_csv(manifest, index=False)
        loader = DataLoader(NormalizedDataset(ChartDataset(manifest, 'test')), batch_size=32, shuffle=False, num_workers=0)
        model = ResNet18Classifier(pretrained=False, training_mode=mode).to(device)
        load_checkpoint(folder/'best_model.pt', model, device)
        val = detailed_inference(model, loader, device)
        if val['labels'] != data.label.tolist():
            raise ValueError('Prediction/sample order mismatch')
        # Only pooled result has all assets; for SPY use the same shared metric function's SPY/overall entries.
        if name == 'spy':
            from scripts.train_multi_asset import get_rows_with_trend
            from scripts.evaluate_spy_baselines import calculate_metrics
            data = get_rows_with_trend(data)
            metrics = {'resnet18': calculate_metrics(data.label, val['predictions']),
                       'majority': calculate_metrics(data.label, [majority]*len(data)),
                       'trend_20d': calculate_metrics(data.label, data.trend)}
        else:
            metrics = metrics_by_asset(data, val['predictions'], majority)
            add_asset_losses(metrics, data, val['losses'])
        results[name] = {'loss': val['loss'], 'majority_class_from_train': majority, 'metrics': metrics}
        print(name, 'loss', val['loss'], flush=True)
    (OUT/'test_results.json').write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
    print('Final test evaluation complete; frozen configurations were not changed.')

if __name__ == '__main__':
    main()

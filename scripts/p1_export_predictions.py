"""P1 Stage 1: export verified per-sample validation/test predictions.

Regenerates probabilities from the frozen checkpoints and the fixed post-hoc
logistic regression, then aborts unless every frozen confusion matrix in
results/summary.json and results/numeric_baseline_posthoc.json is reproduced
exactly. The exported CSV contains no prices, only dates, labels and scores, so
Stage 1 metrics can be recomputed without data downloads or a GPU.
"""
import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import sklearn
import torch
from torch.utils.data import DataLoader
from src.dataset import ChartDataset
from src.model import ResNet18Classifier
from src.training import load_checkpoint
from scripts.evaluate_spy_baselines import calculate_metrics, majority_class
from scripts.numeric_baseline_posthoc import fit_model, prepare_samples
from scripts.train_multi_asset import get_rows_with_trend
from scripts.train_spy_resnet import NormalizedDataset

TICKERS = ('SPY', 'QQQ', 'IWM', 'TLT', 'GLD')
GROUPS = ('overall', *TICKERS)
MANIFEST = ROOT / 'data/processed/multi_asset_samples.csv'
PROTOCOL = ROOT / 'experiments/p1_protocol.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def infer(checkpoint, rows, device, workdir):
    """Return P(Up), argmax prediction and per-sample cross entropy in row order."""
    manifest = workdir / 'manifest.csv'
    rows.to_csv(manifest, index=False)
    split = rows.split.iloc[0]
    loader = DataLoader(NormalizedDataset(ChartDataset(manifest, split)), batch_size=32, shuffle=False, num_workers=0)
    model = ResNet18Classifier(pretrained=False, training_mode='classifier_only').to(device)
    load_checkpoint(checkpoint, model, device)
    model.eval()
    probs, preds, losses, labels = [], [], [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            targets = targets.to(device)
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            losses.extend(torch.nn.functional.cross_entropy(logits, targets, reduction='none').cpu().tolist())
            labels.extend(targets.cpu().tolist())
    if labels != rows.label.tolist():
        raise ValueError('Inference order does not match manifest rows')
    return np.asarray(probs), np.asarray(preds, dtype=int), np.asarray(losses)


class Gate:
    """Collect exact reproduction checks; abort on the first mismatch."""
    def __init__(self):
        self.checks = []

    def matrix(self, name, labels, predictions, expected):
        actual = calculate_metrics(labels, predictions)['confusion_matrix']
        if actual != expected:
            raise ValueError(f'Reproduction gate failed for {name}: {actual} != {expected}')
        self.checks.append(name)

    def loss(self, name, actual, expected):
        if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError(f'Reproduction gate failed for {name}: loss {actual} != {expected}')
        self.checks.append(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/p1')
    args = parser.parse_args()
    out_csv, out_meta = args.output_dir / 'predictions.csv', args.output_dir / 'predictions_meta.json'
    for path in (out_csv, out_meta):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')

    summary = json.loads((ROOT / 'results/summary.json').read_text(encoding='utf-8'))
    numeric = json.loads((ROOT / 'results/numeric_baseline_posthoc.json').read_text(encoding='utf-8'))
    freeze = summary['freeze']
    if freeze['selected_mode'] != 'classifier_only':
        raise ValueError('Stage 1 export assumes the frozen classifier_only selection')
    pooled_ckpt, spy_ckpt = ROOT / freeze['pooled_checkpoint'], ROOT / freeze['spy_checkpoint']
    for path, expected in ((MANIFEST, freeze['samples_sha256']), (pooled_ckpt, freeze['pooled_checkpoint_sha256']),
                           (spy_ckpt, freeze['spy_checkpoint_sha256'])):
        if sha256(path) != expected:
            raise ValueError(f'{path.name} differs from the frozen artifact')

    # Logistic regression rebuild also verifies raw hashes and sample identity.
    rebuilt, x, _ = prepare_samples()
    rows = pd.read_csv(MANIFEST)
    if not (rebuilt.label_date.to_numpy() == rows.label_date.to_numpy()).all():
        raise ValueError('Rebuilt numeric samples are not aligned with the manifest')
    train = rows.split.to_numpy() == 'train'
    logreg = fit_model(x[train], rows.label.to_numpy()[train])
    rows['logreg_prob'] = logreg.predict_proba(x)[:, list(logreg.classes_).index(1)]
    rows['logreg_pred'] = logreg.predict(x)
    rows['trend_pred'] = get_rows_with_trend(rows).trend.astype(int)
    majority = majority_class(rows.label[train])
    climatology = float(rows.label[train].mean())
    rows['majority_pred'] = majority
    rows['climatology_prob'] = climatology

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gate = Gate()
    workdir = args.output_dir / '.tmp_manifests'
    workdir.mkdir(parents=True, exist_ok=True)
    frames = []
    try:
        for split, frozen, numeric_key in (('val', summary['pooled_validation'], 'validation'),
                                           ('test', summary['test']['pooled'], 'test')):
            part = rows[rows.split == split].reset_index(drop=True)
            prob, pred, loss = infer(pooled_ckpt, part, device, workdir)
            part['resnet_pooled_prob'], part['resnet_pooled_pred'] = prob, pred
            part['resnet_spy_prob'], part['resnet_spy_pred'] = np.nan, pd.array([pd.NA] * len(part), dtype='Int64')
            spy = (part.ticker == 'SPY').to_numpy()
            s_prob, s_pred, s_loss = infer(spy_ckpt, part[spy].reset_index(drop=True), device, workdir)
            part.loc[spy, 'resnet_spy_prob'] = s_prob
            part.loc[spy, 'resnet_spy_pred'] = s_pred
            for group in GROUPS:
                mask = np.ones(len(part), bool) if group == 'overall' else (part.ticker == group).to_numpy()
                labels = part.label[mask]
                expected = frozen['metrics'][group]
                gate.matrix(f'{split}/{group}/resnet_pooled', labels, part.resnet_pooled_pred[mask], expected['resnet18']['confusion_matrix'])
                gate.matrix(f'{split}/{group}/majority', labels, part.majority_pred[mask], expected['majority']['confusion_matrix'])
                gate.matrix(f'{split}/{group}/trend_20d', labels, part.trend_pred[mask], expected['trend_20d']['confusion_matrix'])
                gate.matrix(f'{split}/{group}/logreg_closes', labels, part.logreg_pred[mask], numeric[numeric_key][group]['confusion_matrix'])
                gate.loss(f'{split}/{group}/resnet_pooled_loss', float(loss[mask].mean()), expected['resnet18']['loss'])
            spy_expected = (summary['spy_validation']['classifier_only']['validation_metrics']['confusion_matrix']
                            if split == 'val' else summary['test']['spy']['metrics']['resnet18']['confusion_matrix'])
            gate.matrix(f'{split}/SPY/resnet_spy', part.label[spy], s_pred, spy_expected)
            if split == 'test':
                gate.loss('test/SPY/resnet_spy_loss', float(s_loss.mean()), summary['test']['spy']['loss'])
            else:
                gate.loss('val/SPY/resnet_spy_loss', float(s_loss.mean()), summary['spy_validation']['classifier_only']['best_val_loss'])
            frames.append(part)
    finally:
        for file in workdir.glob('*'):
            file.unlink()
        workdir.rmdir()

    columns = ['split', 'ticker', 'label_date', 'label', 'climatology_prob', 'majority_pred', 'trend_pred',
               'logreg_prob', 'logreg_pred', 'resnet_pooled_prob', 'resnet_pooled_pred', 'resnet_spy_prob', 'resnet_spy_pred']
    output = pd.concat(frames, ignore_index=True)[columns]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with out_csv.open('x', encoding='utf-8', newline='') as stream:
        output.to_csv(stream, index=False, float_format='%.12g', lineterminator='\n')
    meta = {'stage': 'P1 Stage 1 prediction export (post-hoc exploratory)',
            'protocol_file': 'experiments/p1_protocol.json', 'protocol_sha256': sha256(PROTOCOL),
            'manifest_sha256': freeze['samples_sha256'],
            'raw_sha256': {t: summary['asset_audit'][t]['raw_sha256'] for t in TICKERS},
            'pooled_checkpoint_sha256': freeze['pooled_checkpoint_sha256'],
            'spy_checkpoint_sha256': freeze['spy_checkpoint_sha256'],
            'majority_class_from_pooled_train': majority, 'climatology_probability': climatology,
            'reproduction_gate': {'passed': True, 'checks': len(gate.checks), 'names': gate.checks},
            'rows': {split: int((output.split == split).sum()) for split in ('val', 'test')},
            'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                            'scikit_learn': sklearn.__version__, 'device': str(device),
                            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
            'predictions_file': 'results/p1/predictions.csv', 'predictions_sha256': sha256(out_csv)}
    with out_meta.open('x', encoding='utf-8') as stream:
        json.dump(meta, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Reproduction gate passed: {len(gate.checks)} checks. Saved {out_csv} and {out_meta}')


if __name__ == '__main__':
    main()

"""Train one pooled experiment using the mode selected by SPY validation loss."""
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from src.dataset import ChartDataset
from src.model import ResNet18Classifier
from src.training import set_seed, train_one_epoch, validate, save_checkpoint, load_checkpoint
from scripts.train_spy_resnet import NormalizedDataset, sha256, plot_history, WEIGHTS
from scripts.evaluate_spy_baselines import calculate_metrics, majority_class, trend_predictions
from src.samples import load_checked_asset

TICKERS = ('SPY', 'QQQ', 'IWM', 'TLT', 'GLD')
SAMPLES = ROOT / 'data/processed/multi_asset_samples.csv'
OUT = ROOT / 'experiments/multi_asset_resnet18_seed42'


def get_rows_with_trend(rows):
    rows = rows.copy()
    preds = []
    for ticker, group in rows.groupby('ticker', sort=False):
        raw = ROOT / 'data/raw' / f'{ticker}.csv'
        f = load_checked_asset(raw, raw.with_suffix('.metadata.json'), ticker)
        prices = f.assign(Date=f.Date.dt.strftime('%Y-%m-%d')).set_index('Date').Close
        p = trend_predictions(group.input_start_date.map(prices), group.input_end_date.map(prices))
        preds.extend(zip(group.index, p.tolist()))
    rows['trend'] = pd.Series(dict(preds))
    return rows


def metrics_by_asset(rows, predictions, majority):
    rows = get_rows_with_trend(rows.reset_index(drop=True))
    rows['prediction'] = predictions
    results = {}
    for ticker in ('overall', *TICKERS):
        group = rows if ticker == 'overall' else rows[rows.ticker == ticker]
        results[ticker] = {name: calculate_metrics(group.label, pred) for name, pred in
            [('resnet18', group.prediction), ('majority', [majority] * len(group)), ('trend_20d', group.trend)]}
    return results


def detailed_inference(model, loader, device):
    """Collect predictions and per-sample cross entropy in a single pass."""
    model.eval()
    losses, labels, predictions = [], [], []
    criterion = nn.CrossEntropyLoss(reduction='none')
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            targets = targets.to(device)
            losses.extend(criterion(logits, targets).cpu().tolist())
            labels.extend(targets.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return {'loss': sum(losses) / len(losses), 'losses': losses,
            'labels': labels, 'predictions': predictions}


def add_asset_losses(metrics, rows, losses):
    rows = rows.reset_index(drop=True).copy()
    rows['loss'] = losses
    for ticker in ('overall', *TICKERS):
        group = rows if ticker == 'overall' else rows[rows.ticker == ticker]
        metrics[ticker]['resnet18']['loss'] = float(group['loss'].mean())


def main():
    # Selection is fully specified in research_protocol.json, before observing this comparison.
    d2 = json.loads((ROOT/'experiments/spy_resnet18_d2_seed42/validation_result.json').read_text())
    full = json.loads((ROOT/'experiments/spy_resnet18_full_finetune_seed42/validation_result.json').read_text())
    mode = 'full_finetune' if full['best_val_loss'] < d2['best_val_loss'] else 'classifier_only'
    if not WEIGHTS.is_file() or not sha256(WEIGHTS).startswith('f37072fd'):
        raise ValueError('Pretrained weight missing or incorrect')
    rows = pd.read_csv(SAMPLES)
    train_val = rows[rows.split.isin(['train', 'val'])].copy()
    if set(train_val.ticker) != set(TICKERS):
        raise ValueError('Missing asset')
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = OUT / 'train_val_samples.csv'
    train_val.to_csv(manifest, index=False)
    set_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.hub.set_dir(str(ROOT/'data/model_weights/torch_hub'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train = NormalizedDataset(ChartDataset(manifest, 'train'))
    val = NormalizedDataset(ChartDataset(manifest, 'val'))
    tl = DataLoader(train, batch_size=32, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(42))
    vl = DataLoader(val, batch_size=32, shuffle=False, num_workers=0)
    model = ResNet18Classifier(pretrained=True, training_mode=mode).to(device)
    if mode == 'classifier_only':
        opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    else:
        opt = torch.optim.Adam([{'params': [p for n,p in model.named_parameters() if not n.startswith('backbone.fc.')], 'lr': 1e-5},
                                {'params': model.backbone.fc.parameters(), 'lr': 1e-3}])
    criterion = nn.CrossEntropyLoss()
    config = {'mode': mode, 'selection': 'lower SPY minimum validation loss; classifier_only on tie',
              'spy_validation_losses': {'classifier_only': d2['best_val_loss'], 'full_finetune': full['best_val_loss']},
              'seed': 42, 'batch_size': 32, 'epochs': 10, 'optimizer': 'Adam',
              'backbone_learning_rate': 1e-5 if mode == 'full_finetune' else None,
              'classifier_learning_rate': 1e-3, 'loss': 'CrossEntropyLoss', 'device': str(device),
              'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
              'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
              'weight_sha256': sha256(WEIGHTS), 'samples_sha256': sha256(SAMPLES),
              'selection_rule': 'minimum overall validation loss; earliest epoch on tie',
              'preprocessing': 'stage C chart, ImageNet normalization, no augmentation',
              'train_samples': len(train), 'validation_samples': len(val)}
    (OUT/'config.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    history, best, best_epoch, best_result = [], float('inf'), None, None
    with (OUT/'training.log').open('x', encoding='utf-8', buffering=1) as log:
        def report(s):
            print(s, flush=True); print(s, file=log, flush=True)
        report('Mode: '+mode+'; device: '+str(device)+'; train: '+str(len(train))+'; val: '+str(len(val)))
        for epoch in range(1,11):
            start = time.perf_counter()
            tr = train_one_epoch(model, tl, opt, criterion, device)
            v = validate(model, vl, criterion, device)
            if not all(math.isfinite(x) for x in (tr['loss'], tr['accuracy'], v['loss'], v['accuracy'])):
                raise ValueError('Non-finite training metrics')
            history.append({'epoch':epoch, 'train_loss':tr['loss'], 'train_accuracy':tr['accuracy'], 'val_loss':v['loss'], 'val_accuracy':v['accuracy']})
            improved = v['loss'] < best
            if improved:
                best, best_epoch, best_result = v['loss'], epoch, v
                save_checkpoint(OUT/'best_model.pt', model, seed=42, training_parameters={**config, 'best_epoch':epoch, 'best_val_loss':best}, history=history)
            (OUT/'history.json').write_text(json.dumps(history, indent=2)+'\n', encoding='utf-8')
            report(f"Epoch {epoch}/10 train_loss={tr['loss']:.6f} train_acc={tr['accuracy']:.6f} val_loss={v['loss']:.6f} val_acc={v['accuracy']:.6f} best={improved} seconds={time.perf_counter()-start:.2f}")
        restored = ResNet18Classifier(pretrained=False, training_mode=mode).to(device)
        metadata = load_checkpoint(OUT/'best_model.pt', restored, device)
        v = detailed_inference(restored, vl, device)
        if not math.isclose(v['loss'], best, rel_tol=1e-6, abs_tol=1e-7) or v['predictions'] != best_result['predictions'] or v['labels'] != best_result['labels']:
            raise ValueError('Reload validation mismatch')
        valrows = train_val[train_val.split == 'val'].reset_index(drop=True)
        maj = majority_class(train_val.loc[train_val.split == 'train','label'])
        metrics = metrics_by_asset(valrows, v['predictions'], maj)
        add_asset_losses(metrics, valrows, v['losses'])
        result = {'best_epoch':best_epoch, 'best_val_loss':best, 'reload_verified':True,
                  'majority_class_from_pooled_train':maj, 'metrics':metrics}
        (OUT/'validation_result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        plot_history(history, OUT/'loss_curve.png', 'loss', title='Five-asset ResNet-18')
        plot_history(history, OUT/'accuracy_curve.png', 'accuracy', title='Five-asset ResNet-18')
        report('Best epoch: '+str(best_epoch)+' validation loss: '+str(best))
        report('Reload verified; test set not evaluated.')

if __name__ == '__main__':
    main()

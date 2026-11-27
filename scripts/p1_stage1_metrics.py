"""P1 Stage 1: probabilistic metrics, calibration and date-block bootstrap intervals.

Reads only results/p1/predictions.csv (verified against its export metadata) and
the bootstrap settings in experiments/p1_protocol.json, so it runs without raw
prices, images, checkpoints or a GPU. Post-hoc exploratory analysis.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.metrics import (accuracy, balanced_accuracy, bootstrap_p_value, brier, calibration_bins,
                         date_block_bootstrap, log_loss, percentile_interval, roc_auc, summarize)

TICKERS = ('SPY', 'QQQ', 'IWM', 'TLT', 'GLD')
GROUPS = ('overall', *TICKERS)
P1 = ROOT / 'results/p1'
# name: (probability column, prediction column, hard scores, label)
MODELS = {
    'majority': ('climatology_prob', 'majority_pred', False, 'Majority / climatology'),
    'trend_20d': ('trend_pred', 'trend_pred', True, '20-day trend'),
    'logreg_closes': ('logreg_prob', 'logreg_pred', False, 'Logistic regression (closes)'),
    'resnet_pooled': ('resnet_pooled_prob', 'resnet_pooled_pred', False, 'Pooled ResNet-18'),
    'resnet_spy': ('resnet_spy_prob', 'resnet_spy_pred', False, 'SPY ResNet-18'),
}
COMPARISONS = [('resnet_pooled', 'majority'), ('resnet_pooled', 'logreg_closes'),
               ('logreg_closes', 'majority'), ('trend_20d', 'majority'), ('resnet_spy', 'majority')]
BOOT_METRICS = {'accuracy': lambda y, p, d: accuracy(y, d),
                'balanced_accuracy': lambda y, p, d: balanced_accuracy(y, d),
                'roc_auc': lambda y, p, d: roc_auc(y, p),
                'brier': lambda y, p, d: brier(y, p),
                'log_loss': lambda y, p, d: log_loss(y, p)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def models_for(group):
    return [m for m in MODELS if m != 'resnet_spy' or group == 'SPY']


def arrays(frame, model):
    prob_col, pred_col, _, _ = MODELS[model]
    return (frame.label.to_numpy(), frame[prob_col].to_numpy(dtype=float), frame[pred_col].to_numpy(dtype=int))


def group_frame(frame, group):
    return frame if group == 'overall' else frame[frame.ticker == group]


def bootstrap_group(frame, group, settings):
    """Return {model: {metric: replicate array}} for one test group."""
    models = models_for(group)
    data = {m: arrays(frame, m) for m in models}
    reps = {m: {k: [] for k in BOOT_METRICS} for m in models}
    for idx in date_block_bootstrap(frame.label_date.to_numpy(), settings['block_length_sessions'],
                                    settings['replicates'], settings['seed']):
        for m in models:
            y, p, d = (a[idx] for a in data[m])
            for key, fn in BOOT_METRICS.items():
                reps[m][key].append(fn(y, p, d))
    return {m: {k: np.asarray(v) for k, v in metrics.items()} for m, metrics in reps.items()}


def fmt_pct(value):
    return '—' if value is None or not math.isfinite(value) else f'{100 * value:.2f}%'


def fmt_num(value, digits=4):
    return '—' if value is None or not math.isfinite(value) else f'{value:.{digits}f}'


def fmt_ci(ci, pct=False, digits=4):
    f = fmt_pct if pct else (lambda v: fmt_num(v, digits))
    return f'[{f(ci[0])}, {f(ci[1])}]'


def plot_calibration(test, reference, output):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {'resnet_pooled': '#d62728', 'logreg_closes': '#1f77b4'}
    lo, hi = 1.0, 0.0
    for model, color in colors.items():
        y, p, _ = arrays(test, model)
        bins = calibration_bins(y, p)['bins']
        x = [b['mean_probability'] for b in bins]
        o = [b['observed_up_rate'] for b in bins]
        lo, hi = min(lo, *x, *o), max(hi, *x, *o)
        left.plot(x, o, marker='o', color=color, label=MODELS[model][3])
        right.hist(p, bins=40, alpha=0.55, color=color, label=MODELS[model][3])
    observed = float(test.label.mean())
    left.scatter([reference], [observed], marker='s', color='gray', zorder=3,
                 label='Climatology (train up-ratio) vs test up-rate')
    lo, hi = min(lo, reference, observed) - 0.02, max(hi, reference, observed) + 0.02
    left.plot([lo, hi], [lo, hi], linestyle='--', color='black', linewidth=1, label='Perfect calibration')
    left.set(xlim=(lo, hi), ylim=(lo, hi), xlabel='Mean predicted P(Up) per decile bin',
             ylabel='Observed Up rate', title='Reliability, pooled test 2024-2025 (equal-frequency bins)')
    left.legend(fontsize=8)
    right.axvline(0.5, color='black', linestyle='--', linewidth=1, label='Decision threshold 0.5')
    right.axvline(observed, color='gray', linestyle=':', linewidth=1.5, label=f'Test Up rate {observed:.3f}')
    right.set(xlabel='Predicted P(Up)', ylabel='Test samples', title='Distribution of predicted probabilities')
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def write_markdown(result, path):
    t, v, cmp_ = result['test'], result['validation'], result['comparisons']
    ref = result['reference_probability']
    lines = ['# P1 Stage 1 — Probabilistic evaluation and uncertainty', '',
             '> Post-hoc exploratory analysis of the frozen 2024–2025 test predictions (protocol: '
             '`experiments/p1_protocol.json`). Intervals are 95% circular moving-block bootstrap percentiles over '
             f"label dates (block {result['bootstrap']['block_length_sessions']} sessions, "
             f"{result['bootstrap']['replicates']} replicates, all assets on a resampled date kept together).", '',
             f'Climatology reference probability (pooled training up-ratio): **{ref:.4f}**. '
             'The 20-day trend rule outputs hard 0/1 scores, so its AUC equals balanced accuracy and probability '
             'metrics are not meaningful (shown as —).', '',
             '## Pooled test (2,510 samples)', '',
             '| Model | Accuracy [95% CI] | Balanced accuracy [95% CI] | ROC-AUC [95% CI] | Brier | Brier skill | Log loss | ECE | Binomial p (acc vs 50%) |',
             '|---|---|---|---|---:|---:|---:|---:|---:|']
    for m, entry in t['overall'].items():
        s, ci, hard = entry['metrics'], entry['ci'], entry['metrics']['hard_scores']
        lines.append(f"| {MODELS[m][3]} | {fmt_pct(s['accuracy'])} {fmt_ci(ci['accuracy'], True)} | "
                     f"{fmt_pct(s['balanced_accuracy'])} {fmt_ci(ci['balanced_accuracy'], True)} | "
                     f"{fmt_num(s['roc_auc'], 3)} {fmt_ci(ci['roc_auc'], digits=3)} | "
                     f"{'—' if hard else fmt_num(s['brier'])} | {'—' if hard else fmt_num(s['brier_skill_score'])} | "
                     f"{'—' if hard else fmt_num(s['log_loss'])} | {'—' if hard else fmt_num(s['ece'])} | "
                     f"{s['binomial_vs_50']['p_value']:.3g} |")
    lines += ['', '## Per-asset test metrics', '',
              '| Asset | Up rate | Model | Accuracy | Balanced accuracy [95% CI] | ROC-AUC [95% CI] | Brier skill |',
              '|---|---:|---|---:|---|---|---:|']
    for g in TICKERS:
        for m, entry in t[g].items():
            s, ci = entry['metrics'], entry['ci']
            lines.append(f"| {g} | {fmt_pct(s['up_ratio'])} | {MODELS[m][3]} | {fmt_pct(s['accuracy'])} | "
                         f"{fmt_pct(s['balanced_accuracy'])} {fmt_ci(ci['balanced_accuracy'], True)} | "
                         f"{fmt_num(s['roc_auc'], 3)} {fmt_ci(ci['roc_auc'], digits=3)} | "
                         f"{'—' if s['hard_scores'] else fmt_num(s['brier_skill_score'])} |")
    lines += ['', '## Paired differences on the pooled test (A − B)', '',
              'Positive differences favour A for accuracy, balanced accuracy and AUC; negative differences favour A '
              'for Brier and log loss. Bootstrap p is a two-sided sign p-value from the same paired replicates.', '',
              '| A − B | Metric | Difference | 95% CI | Bootstrap p |', '|---|---|---:|---|---:|']
    for row in cmp_['overall']:
        if MODELS[row['a']][2] and row['metric'] in ('brier', 'log_loss'):
            continue
        pct = row['metric'] in ('accuracy', 'balanced_accuracy')
        diff = f"{100 * row['difference']:+.2f} pp" if pct else f"{row['difference']:+.4f}"
        ci = (f"[{100 * row['ci'][0]:+.2f}, {100 * row['ci'][1]:+.2f}] pp" if pct
              else f"[{row['ci'][0]:+.4f}, {row['ci'][1]:+.4f}]")
        lines.append(f"| {MODELS[row['a']][3]} − {MODELS[row['b']][3]} | {row['metric']} | {diff} | {ci} | {row['p_boot']:.3g} |")
    spy_rows = [r for r in cmp_['SPY'] if r['a'] == 'resnet_spy']
    lines += ['', '## SPY-only ResNet-18 versus majority / climatology (502 SPY test samples)', '',
              '| Metric | Difference | 95% CI | Bootstrap p |', '|---|---:|---|---:|']
    for row in spy_rows:
        pct = row['metric'] in ('accuracy', 'balanced_accuracy')
        diff = f"{100 * row['difference']:+.2f} pp" if pct else f"{row['difference']:+.4f}"
        ci = (f"[{100 * row['ci'][0]:+.2f}, {100 * row['ci'][1]:+.2f}] pp" if pct
              else f"[{row['ci'][0]:+.4f}, {row['ci'][1]:+.4f}]")
        lines.append(f"| {row['metric']} | {diff} | {ci} | {row['p_boot']:.3g} |")
    lines += ['', '## Validation context (2022–2023, point estimates)', '',
              '| Model | Accuracy | Balanced accuracy | ROC-AUC | Brier skill | Log loss |', '|---|---:|---:|---:|---:|---:|']
    for m, s in v['overall'].items():
        lines.append(f"| {MODELS[m][3]} | {fmt_pct(s['accuracy'])} | {fmt_pct(s['balanced_accuracy'])} | "
                     f"{fmt_num(s['roc_auc'], 3)} | {'—' if s['hard_scores'] else fmt_num(s['brier_skill_score'])} | "
                     f"{'—' if s['hard_scores'] else fmt_num(s['log_loss'])} |")
    lines += ['', '![Calibration](stage1_calibration_test.png)', '',
              'Binomial p-values assume independent observations and are optimistic under overlapping windows and '
              'cross-asset correlation; the block-bootstrap intervals are the primary uncertainty measure.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=P1 / 'predictions.csv')
    parser.add_argument('--output-dir', type=Path, default=P1)
    parser.add_argument('--overwrite', action='store_true', help='replace existing Stage 1 metric outputs')
    args = parser.parse_args()
    outputs = [args.output_dir / n for n in ('stage1_metrics.json', 'stage1_metrics.md', 'stage1_calibration_test.png')]
    if not args.overwrite and any(p.exists() for p in outputs):
        raise FileExistsError('Stage 1 outputs exist; pass --overwrite to regenerate them')
    meta = json.loads((args.predictions.parent / 'predictions_meta.json').read_text(encoding='utf-8'))
    if sha256(args.predictions) != meta['predictions_sha256'] or not meta['reproduction_gate']['passed']:
        raise ValueError('predictions.csv does not match its verified export metadata')
    protocol = json.loads((ROOT / 'experiments/p1_protocol.json').read_text(encoding='utf-8'))
    settings = protocol['stages']['stage_1_probabilistic_evaluation']['uncertainty']
    frame = pd.read_csv(args.predictions)
    reference = float(meta['climatology_probability'])

    result = {'stage': 'P1 Stage 1 (post-hoc exploratory)', 'protocol_sha256': sha256(ROOT / 'experiments/p1_protocol.json'),
              'predictions_sha256': meta['predictions_sha256'], 'reference_probability': reference,
              'bootstrap': {k: settings[k] for k in ('block_length_sessions', 'replicates', 'seed', 'interval')},
              'validation': {}, 'test': {}, 'comparisons': {}, 'calibration': {}}
    for split in ('val', 'test'):
        part = frame[frame.split == split]
        key = 'validation' if split == 'val' else 'test'
        result['calibration'][key] = {m: calibration_bins(*arrays(part, m)[:2])['bins']
                                      for m in models_for('overall') if not MODELS[m][2]}
        for group in GROUPS:
            sub = group_frame(part, group)
            point = {m: summarize(*arrays(sub, m), reference, hard_scores=MODELS[m][2]) for m in models_for(group)}
            if split == 'val':
                result['validation'][group] = point
                continue
            reps = bootstrap_group(sub, group, settings)
            result['test'][group] = {m: {'metrics': point[m],
                                         'ci': {k: percentile_interval(reps[m][k]) for k in BOOT_METRICS}}
                                     for m in point}
            rows = []
            for a, b in COMPARISONS:
                if a not in point or b not in point:
                    continue
                for k in BOOT_METRICS:
                    y, pa, da = arrays(sub, a)
                    _, pb, db = arrays(sub, b)
                    diff = BOOT_METRICS[k](y, pa, da) - BOOT_METRICS[k](y, pb, db)
                    d = reps[a][k] - reps[b][k]
                    rows.append({'a': a, 'b': b, 'metric': k, 'difference': float(diff),
                                 'ci': percentile_interval(d), 'p_boot': bootstrap_p_value(d)})
            result['comparisons'][group] = rows
            print(f'test/{group}: bootstrap complete', flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(result, indent=2, allow_nan=True) + '\n', encoding='utf-8')
    write_markdown(result, outputs[1])
    plot_calibration(frame[frame.split == 'test'], reference, outputs[2])
    print('Saved', *(p.as_posix() for p in outputs))


if __name__ == '__main__':
    main()

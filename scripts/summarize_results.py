"""Create publication-ready aggregate results from completed local experiments."""
import json
import math
import os
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'experiments'/'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = ROOT/'experiments'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results', help='new results directory')
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(exist_ok=False)
    d2, full, pooled, final = (EXP/'spy_resnet18_d2_seed42', EXP/'spy_resnet18_full_finetune_seed42',
                                EXP/'multi_asset_resnet18_seed42', EXP/'final_evaluation')
    summary = {'constant_probability_cross_entropy': math.log(2),
               'spy_validation': {'classifier_only': read(d2/'validation_result.json'),
                                  'full_finetune': read(full/'validation_result.json'),
                                  'baselines': read(ROOT/'data/reports/spy_baselines_d1.json')['results']['val']},
               'pooled_validation': read(pooled/'validation_result.json'),
               'freeze': read(final/'freeze.json'),
               'test': read(final/'test_results.json'),
               'asset_audit': read(ROOT/'data/processed/multi_asset_audit.json')}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    def table_row(asset, method, metrics, loss=None):
        p = lambda key: f"{metrics[key] * 100:.2f}%"
        loss_text = f'{loss:.6f}' if loss is not None else '—'
        return f"| {asset} | {method} | {metrics['n']} | {loss_text} | {p('accuracy')} | {p('precision_up')} | {p('recall_up')} | {p('f1_up')} | {metrics['confusion_matrix']} |"
    header = '| Asset | Method | N | Loss | Accuracy | Precision ↑ | Recall ↑ | F1 ↑ | Confusion matrix |\n|---|---|---:|---:|---:|---:|---:|---:|---|'
    lines = ['# Results', '', 'Confusion matrices: rows actual [0,1], columns predicted [0,1]. Loss is cross entropy. Constant p=0.5: ln(2) = 0.693147.', '', '## SPY validation', '', header]
    for mode in ('classifier_only', 'full_finetune'):
        result = summary['spy_validation'][mode]
        lines.append(table_row('SPY', mode, result['validation_metrics'], result['best_val_loss']))
    for name, metrics in summary['spy_validation']['baselines'].items():
        lines.append(table_row('SPY', name, metrics))
    for title, groups in [('Five-asset validation', summary['pooled_validation']['metrics']),
                          ('Five-asset test', summary['test']['pooled']['metrics'])]:
        lines.extend(['', '## '+title, '', header])
        for asset in ('SPY','QQQ','IWM','TLT','GLD','overall'):
            for name in ('resnet18','majority','trend_20d'):
                metrics = groups[asset][name]
                lines.append(table_row(asset, name, metrics, metrics.get('loss')))
    lines.extend(['', '## Standalone selected SPY model test', '', header])
    for name, metrics in summary['test']['spy']['metrics'].items():
        lines.append(table_row('SPY', name, metrics, summary['test']['spy']['loss'] if name == 'resnet18' else None))
    (OUT/'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for name, source in [('spy_classifier', d2), ('spy_full_finetune', full), ('five_asset', pooled)]:
        for image in ('loss_curve.png', 'accuracy_curve.png'):
            shutil.copyfile(source/image, OUT/(name+'_'+image))
        shutil.copyfile(source/'history.json', OUT/(name+'_history.json'))
    models = summary['spy_validation']
    fig, ax = plt.subplots(figsize=(6,4))
    for label, path in [('classifier only', d2), ('full fine-tune', full)]:
        h = read(path/'history.json')
        ax.plot([r['epoch'] for r in h], [r['val_loss'] for r in h], marker='o', label=label)
    ax.axhline(math.log(2), color='grey', linestyle='--', label='constant p=0.5')
    ax.set(xlabel='Epoch', ylabel='Validation cross entropy', title='SPY validation loss')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT/'spy_validation_comparison.png', dpi=150); plt.close(fig)
    cm = summary['test']['pooled']['metrics']['overall']['resnet18']['confusion_matrix']
    fig, ax = plt.subplots(figsize=(4.5,4))
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center', color='black')
    ax.set(xticks=[0,1], yticks=[0,1], xlabel='Predicted', ylabel='Actual', title='Pooled test confusion matrix')
    fig.tight_layout(); fig.savefig(OUT/'pooled_test_confusion_matrix.png', dpi=150); plt.close(fig)
    data = summary['test']['pooled']['metrics']
    assets = ['SPY','QQQ','IWM','TLT','GLD','overall']
    x = range(len(assets)); width=.25
    fig, ax = plt.subplots(figsize=(8,4.5))
    for offset, (name, label) in enumerate([('majority','Majority'),('trend_20d','20-day trend'),('resnet18','ResNet-18')]):
        ax.bar([i+(offset-1)*width for i in x], [data[a][name]['accuracy'] for a in assets], width=width, label=label)
    ax.set(xticks=list(x), xticklabels=assets, ylabel='Accuracy', ylim=(0,1), title='Held-out test accuracy')
    ax.legend(); ax.grid(axis='y', alpha=.2); fig.tight_layout(); fig.savefig(OUT/'test_accuracy.png', dpi=150); plt.close(fig)
    print('Wrote aggregate JSON, histories, curves, confusion matrix and comparison figure to results/.')

if __name__ == '__main__':
    main()

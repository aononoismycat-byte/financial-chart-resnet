"""Fixed post-hoc numeric comparison on the original five-asset samples."""
import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.charts import normalized_prices
from src.samples import iter_samples, load_checked_asset
from scripts.evaluate_spy_baselines import calculate_metrics

TICKERS = ('SPY', 'QQQ', 'IWM', 'TLT', 'GLD')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_samples():
    summary = json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    manifest = ROOT/'data/processed/multi_asset_samples.csv'
    if sha256(manifest) != summary['freeze']['samples_sha256']:
        raise ValueError('Manifest differs from original frozen dataset; cannot compare identical samples')
    original = pd.read_csv(manifest)
    metadata, features = [], []
    for ticker in TICKERS:
        raw = ROOT/'data/raw'/f'{ticker}.csv'
        if sha256(raw) != summary['asset_audit'][ticker]['raw_sha256']:
            raise ValueError(f'{ticker} data differs from original snapshot')
        frame = load_checked_asset(raw, raw.with_suffix('.metadata.json'), ticker)
        for row, closes in iter_samples(frame, ticker):
            metadata.append(row)
            features.append(normalized_prices(closes))
    rebuilt = pd.DataFrame(metadata)
    cols = ['ticker', 'input_start_date', 'input_end_date', 'input_dates', 'label_date', 'close_t', 'close_t1', 'label', 'split']
    pd.testing.assert_frame_equal(rebuilt[cols], original[cols], check_dtype=False, rtol=1e-12, atol=0)
    return rebuilt, np.asarray(features), summary


def evaluate(rows, predictions):
    results = {}
    predictions = np.asarray(predictions)
    for asset in ('overall', *TICKERS):
        mask = np.ones(len(rows), dtype=bool) if asset == 'overall' else (rows.ticker.to_numpy() == asset)
        results[asset] = calculate_metrics(rows.label.to_numpy()[mask], predictions[mask])
    return results


def fit_model(x_train, y_train):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, random_state=42))
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x_train, y_train)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/numeric_baseline_posthoc.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    rows, x, summary = prepare_samples()
    train = rows.split.to_numpy() == 'train'
    val = rows.split.to_numpy() == 'val'
    test = rows.split.to_numpy() == 'test'
    y = rows.label.to_numpy()
    model = fit_model(x[train], y[train])
    validation = evaluate(rows.loc[val], model.predict(x[val]))
    # The fixed model is now frozen. No test-driven selection or refit.
    test_results = evaluate(rows.loc[test], model.predict(x[test]))
    result = {'study_type': 'Post-hoc exploratory comparison; ResNet test results previously observed',
              'protocol_file': 'results/numeric_baseline_protocol.json',
              'input_manifest_sha256': summary['freeze']['samples_sha256'],
              'raw_sha256': {t: summary['asset_audit'][t]['raw_sha256'] for t in TICKERS},
              'sample_identity_verified': True, 'scikit_learn_version': sklearn.__version__,
              'configuration': {'features': 'Close / first Close', 'scaler_fit': 'pooled train only',
                                'model': 'LogisticRegression', 'C': 1.0, 'solver': 'lbfgs', 'max_iter': 2000, 'seed': 42,
                                'validation_selection': 'none; fixed configuration'},
              'validation': validation, 'test': test_results,
              'original_test_comparators': summary['test']['pooled']['metrics']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'validation_overall': validation['overall'], 'test_overall': test_results['overall']}, indent=2))
    print(f'Saved {args.output}')

if __name__ == '__main__':
    main()

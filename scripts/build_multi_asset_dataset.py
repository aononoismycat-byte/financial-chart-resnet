"""Build five-asset samples from existing checked data without replacing SPY images."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.charts import Y_MIN, Y_MAX, normalized_prices, render_chart
from src.samples import iter_samples, load_checked_asset

TICKERS = ('SPY', 'QQQ', 'IWM', 'TLT', 'GLD')
OUTPUT = ROOT / 'data/processed/multi_asset_samples.csv'
REPORT = ROOT / 'data/processed/multi_asset_audit.json'


def main():
    destinations = [OUTPUT, REPORT] + [ROOT / 'data/images' / t for t in TICKERS[1:]]
    for path in destinations:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite: {path}')
    all_rows, audits = [], {}
    existing = pd.read_csv(ROOT / 'data/processed/spy_samples.csv')
    for ticker in TICKERS:
        raw = ROOT / 'data/raw' / f'{ticker}.csv'
        frame = load_checked_asset(raw, raw.with_suffix('.metadata.json'), ticker)
        samples = list(iter_samples(frame, ticker))
        rows, clipped = [], {'train': 0, 'val': 0, 'test': 0}
        if ticker != 'SPY':
            for split in clipped:
                (ROOT / 'data/images' / ticker / split).mkdir(parents=True, exist_ok=False)
        for row, closes in samples:
            values = normalized_prices(closes)
            clipped[row['split']] += int((values < Y_MIN).any() or (values > Y_MAX).any())
            path = ROOT / 'data/images' / ticker / row['split'] / (row['label_date'] + '.png')
            if ticker != 'SPY':
                render_chart(closes).save(path, format='PNG')
            elif not path.is_file():
                raise FileNotFoundError(path)
            row['image_path'] = path.relative_to(ROOT).as_posix()
            rows.append(row)
        result = pd.DataFrame(rows)
        if ticker == 'SPY':
            pd.testing.assert_frame_equal(result, existing, check_dtype=False, rtol=1e-12)
        if result.duplicated(['ticker', 'label_date']).any():
            raise ValueError(f'duplicate sample for {ticker}')
        all_rows.extend(rows)
        audits[ticker] = {'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
                          'raw_rows': len(frame), 'date_min': str(frame.Date.iloc[0].date()),
                          'date_max': str(frame.Date.iloc[-1].date()), 'duplicate_dates': 0,
                          'missing_dates_or_sessions': 0, 'invalid_close': 0,
                          'price_column': 'Close (auto_adjust=False)',
                          'split_statistics': {s: {'samples': len(g), 'up': int(g.label.sum()),
                                                    'up_ratio': float(g.label.mean()),
                                                    'clipped_windows': clipped[s]}
                                               for s, g in result.groupby('split', sort=False)}}
        print(ticker, audits[ticker]['split_statistics'], flush=True)
    with OUTPUT.open('x', encoding='utf-8', newline='') as f:
        pd.DataFrame(all_rows).to_csv(f, index=False)
    with REPORT.open('x', encoding='utf-8') as f:
        json.dump(audits, f, indent=2, allow_nan=False)
        f.write('\n')

if __name__ == '__main__':
    main()

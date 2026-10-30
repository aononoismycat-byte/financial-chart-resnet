"""Download and inspect daily ETF history from Yahoo Finance via yfinance.

Run from any directory with the project's .venv Python. Existing CSV and
metadata files are never overwritten. Failed requests are appended to an error log.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TICKERS = ("SPY", "QQQ", "IWM", "TLT", "GLD")
PARAMS = {
    "start": "2009-01-01",  # inclusive
    "end": "2026-01-01",  # exclusive
    "interval": "1d",
    "auto_adjust": False,
    "actions": False,
    "keepna": True,
    "timeout": 20,
    "raise_errors": True,
}
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_error(ticker: str, error: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with (RAW_DIR / "download_errors.log").open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] {ticker}\n{error.rstrip()}\n")


def inspect_data(frame: pd.DataFrame) -> dict:
    dates = pd.DatetimeIndex(frame.index)
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    dates = dates.normalize()
    frame.index = dates
    frame.index.name = "Date"
    missing = {str(k): int(v) for k, v in frame.isna().sum().items()}
    invalid_prices = {}
    for column in PRICE_COLUMNS:
        if column in frame:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            invalid_prices[column] = int((frame[column].notna() & (numeric.isna() | ~np.isfinite(numeric) | (numeric <= 0))).sum())
    # Weekdays are only candidates: exchange holidays and exceptional closures
    # need an actual NYSE calendar to distinguish them from missing sessions.
    observed = dates.dropna().unique()
    candidates = pd.bdate_range(observed.min(), observed.max()).difference(observed) if len(observed) else []
    return {
        "rows": len(frame),
        "date_min": dates.min().strftime("%Y-%m-%d") if dates.notna().any() else None,
        "date_max": dates.max().strftime("%Y-%m-%d") if dates.notna().any() else None,
        "missing_dates": int(dates.isna().sum()),
        "duplicate_dates": int(dates.duplicated().sum()),
        "missing_values_by_column": missing,
        "invalid_price_values_by_column": invalid_prices,
        "missing_required_columns": [c for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume") if c not in frame.columns],
        "dates_outside_requested_range": int(((dates < pd.Timestamp(PARAMS['start'])) | (dates >= pd.Timestamp(PARAMS['end']))).sum()),
        "weekday_gap_candidates": [day.strftime("%Y-%m-%d") for day in candidates],
        "gap_caveat": "Weekdays absent between the first and last rows; includes exchange holidays and closures, and is not an NYSE trading calendar.",
    }


def print_report(frame: pd.DataFrame, report: dict) -> None:
    print("First 5 rows:\n", frame.head(5).to_string())
    print("Last 5 rows:\n", frame.tail(5).to_string())
    summary = dict(report)
    gaps = summary.pop("weekday_gap_candidates")
    summary["weekday_gap_candidate_count"] = len(gaps)
    summary["weekday_gap_candidate_sample_first_20"] = gaps[:20]
    print("Validation summary:\n", json.dumps(summary, ensure_ascii=False, indent=2))
    print("Price basis: Close as returned by Yahoo, auto_adjust=False; Adj Close is separate. Missing weekdays include possible market holidays.", flush=True)


def download_one(ticker: str) -> bool:
    csv_path = RAW_DIR / f"{ticker}.csv"
    meta_path = RAW_DIR / f"{ticker}.metadata.json"
    if csv_path.exists() or meta_path.exists():
        try:
            if not csv_path.exists() or not meta_path.exists():
                raise ValueError("CSV and metadata must both exist to reuse existing data")
            with meta_path.open(encoding="utf-8") as source:
                metadata = json.load(source)
            if metadata.get("ticker") != ticker or metadata.get("parameters") != PARAMS:
                raise ValueError("Existing metadata does not match this ticker and download parameters")
            frame = pd.read_csv(csv_path, index_col="Date", parse_dates=["Date"])
            if frame.empty or "Close" not in frame:
                raise ValueError("Existing CSV is empty or lacks Close")
            report = inspect_data(frame)
            print(f"\n{ticker}: rechecking existing CSV (not downloading or overwriting): {csv_path}")
            print_report(frame, report)
            return True
        except Exception:
            error = "Existing output was not overwritten.\n" + traceback.format_exc()
            print(f"{ticker} failed:\n{error}", file=sys.stderr, flush=True)
            record_error(ticker, error)
            return False
    print(f"\nDownloading {ticker} ...", flush=True)
    try:
        frame = yf.Ticker(ticker).history(**PARAMS)
        if frame.empty:
            raise RuntimeError("Yahoo Finance returned an empty DataFrame; no price data saved.")
        report = inspect_data(frame)
        if "Close" not in frame:
            raise RuntimeError("The response does not contain the required Close column.")
        downloaded_at = utc_now()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects an existing file even if another process
        # creates it between the check above and writing.
        with csv_path.open("x", encoding="utf-8", newline="") as output:
            frame.to_csv(output, date_format="%Y-%m-%d")
        metadata = {
            "ticker": ticker,
            "source": "Yahoo Finance via yfinance",
            "source_url": f"https://finance.yahoo.com/quote/{ticker}/history/",
            "downloaded_at_utc": downloaded_at,
            "parameters": PARAMS,
            "end_date_exclusive": True,
            "prediction_price_column": "Close",
            "price_adjustment": "auto_adjust=False: Close is Yahoo Finance's returned Close, not yfinance's auto-adjusted OHLC. Adj Close is retained separately when provided. Yahoo historical prices may reflect split conventions; Close should not be assumed to be dividend-adjusted.",
            "validation": report,
        }
        with meta_path.open("x", encoding="utf-8") as output:
            json.dump(metadata, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        print(f"Saved: {csv_path}\nMetadata: {meta_path}")
        print_report(frame, report)
        return True
    except Exception:
        error = traceback.format_exc()
        print(f"{ticker} failed:\n{error}", file=sys.stderr, flush=True)
        try:
            record_error(ticker, error)
        except Exception as log_error:
            print(f"Could not save error log: {log_error}", file=sys.stderr, flush=True)
        return False


def main() -> int:
    # yfinance keeps SQLite connections open until process exit on Windows.
    # Use a persistent project-local cache instead of deleting an open database.
    yf.set_tz_cache_location(str(ROOT / "data" / ".yfinance_cache"))
    if not download_one("SPY"):
        print("SPY did not complete; remaining tickers were not attempted.", file=sys.stderr)
        return 1
    failures = [ticker for ticker in TICKERS[1:] if not download_one(ticker)]
    if failures:
        print(f"Failed tickers: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""P1 Stage 8d (step 1): download 2025-10-01..2026-11-25 holdout data and verify calendar + overlap."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ("SPY", "QQQ", "IWM", "TLT", "GLD")
OUT_DIR = ROOT / "data/raw_holdout_2026"
PARAMS = {
    "start": "2025-10-01",
    "end": "2026-11-26",
    "interval": "1d",
    "auto_adjust": False,
    "actions": False,
    "keepna": True,
    "timeout": 20,
    "raise_errors": True,
}
HOLIDAYS_2026 = (
    "2026-01-19",
    "2026-02-16",
    "2026-04-03",
    "2026-05-25",
    "2026-06-19",
    "2026-07-03",
    "2026-09-07",
)


def expected_2026_sessions():
    bdays = pd.bdate_range("2026-01-02", "2026-11-25")
    holidays = pd.to_datetime(list(HOLIDAYS_2026))
    return bdays.difference(holidays)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_holdout_file(ticker: str, frame: pd.DataFrame) -> dict:
    dates = pd.DatetimeIndex(frame["Date"])
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"{ticker} holdout dates invalid or non-increasing")
    for col in ("Close", "Adj Close"):
        vals = pd.to_numeric(frame[col], errors="coerce")
        if vals.isna().any() or not np.isfinite(vals).all() or (vals <= 0).any():
            raise ValueError(f"{ticker} holdout {col} contains invalid values")

    # Overlap check against original 2025-10-01..2025-12-31 Close
    orig = pd.read_csv(ROOT / "data/raw" / f"{ticker}.csv", dtype={"Date": "string"})
    orig_ov = orig[orig["Date"] >= "2025-10-01"].sort_values("Date").reset_index(drop=True)
    new_ov = frame[frame["Date"] <= "2025-12-31"].sort_values("Date").reset_index(drop=True)
    if orig_ov["Date"].astype(str).tolist() != new_ov["Date"].astype(str).tolist():
        raise ValueError(f"{ticker} overlap dates differ from original 2025-10-01..2025-12-31 raw CSV")
    if not np.allclose(orig_ov["Close"].to_numpy(dtype=float), new_ov["Close"].to_numpy(dtype=float), rtol=1e-6, atol=0.0):
        raise ValueError(f"{ticker} overlap Close values differ beyond rtol=1e-6")

    # Expected 2026 sessions check
    obs_2026 = pd.DatetimeIndex(pd.to_datetime(frame.loc[frame["Date"] >= "2026-01-01", "Date"]))
    exp_2026 = expected_2026_sessions()
    missing = exp_2026.difference(obs_2026)
    unexpected = obs_2026.difference(exp_2026)
    if len(missing) or len(unexpected):
        raise ValueError(
            f"{ticker} 2026 session mismatch: missing={missing.strftime('%Y-%m-%d').tolist()}, "
            f"unexpected={unexpected.strftime('%Y-%m-%d').tolist()}"
        )
    return {
        "rows_total": int(len(frame)),
        "overlap_rows": int(len(new_ov)),
        "sessions_2026": int(len(obs_2026)),
        "first_2026_session": obs_2026[0].strftime("%Y-%m-%d"),
        "last_2026_session": obs_2026[-1].strftime("%Y-%m-%d"),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_out = {}
    for ticker in TICKERS:
        csv_path = OUT_DIR / f"{ticker}.csv"
        meta_path = OUT_DIR / f"{ticker}.metadata.json"
        if not csv_path.exists() and not meta_path.exists():
            hist = yf.Ticker(ticker).history(**PARAMS)
            if hist.empty:
                raise RuntimeError(f"Empty response from Yahoo Finance for {ticker}")
            idx = pd.DatetimeIndex(hist.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            hist.index = idx.normalize()
            hist.index.name = "Date"
            with csv_path.open("x", encoding="utf-8", newline="") as stream:
                hist.to_csv(stream, date_format="%Y-%m-%d")
            frame = pd.read_csv(csv_path, dtype={"Date": "string"})
            check_info = verify_holdout_file(ticker, frame)
            meta = {
                "ticker": ticker,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "parameters": PARAMS,
                "verification": check_info,
                "csv_sha256": sha256(csv_path),
            }
            with meta_path.open("x", encoding="utf-8") as stream:
                json.dump(meta, stream, indent=2)
                stream.write("\n")
        else:
            if not csv_path.exists() or not meta_path.exists():
                raise ValueError(f"Partial holdout files for {ticker}")
            frame = pd.read_csv(csv_path, dtype={"Date": "string"})
            check_info = verify_holdout_file(ticker, frame)
        summary_out[ticker] = {**check_info, "csv_sha256": sha256(csv_path)}
        print(f"{ticker}: overlap={check_info['overlap_rows']}, 2026_sessions={check_info['sessions_2026']} OK")

    (ROOT / "results/p1/stage8_holdout_download_audit.json").write_text(
        json.dumps(summary_out, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

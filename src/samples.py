"""Per-asset sample construction and bounded 2009–2025 NYSE session validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday,
    USLaborDay, USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
    USThanksgivingDay, nearest_workday, sunday_to_monday)

START = pd.Timestamp("2009-01-02")
END = pd.Timestamp("2025-12-31")
WINDOW = 20


class _NyseHolidays(AbstractHolidayCalendar):
    # This calendar is explicitly scoped to 2009–2025; it must not be used to
    # infer future closures. New Year's Saturday is not observed on Dec 31.
    rules = [Holiday("New Year", month=1, day=1, observance=sunday_to_monday),
             USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
             Holiday("Juneteenth", month=6, day=19, observance=nearest_workday,
                     start_date="2022-01-01"),
             Holiday("Independence", month=7, day=4, observance=nearest_workday),
             USLaborDay, USThanksgivingDay,
             Holiday("Christmas", month=12, day=25, observance=nearest_workday)]


def expected_sessions():
    """Known NYSE/NYSE Arca full-day closures for the fixed data interval."""
    special = pd.to_datetime(["2012-10-29", "2012-10-30",  # Hurricane Sandy
                              "2018-12-05", "2025-01-09"])  # national mourning
    holidays = _NyseHolidays().holidays(start=START, end=END).union(special)
    return pd.bdate_range(START, END).difference(holidays)


def load_checked_asset(csv_path, metadata_path, ticker):
    """Check one asset's prices, metadata and sessions against the fixed calendar."""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if (metadata.get("ticker") != ticker or metadata.get("prediction_price_column") != "Close"
            or metadata.get("parameters", {}).get("auto_adjust") is not False):
        raise ValueError(f"{ticker} metadata or Close price basis is incompatible")
    frame = pd.read_csv(csv_path, dtype={"Date": "string"})
    if "Date" not in frame or "Close" not in frame or frame.empty:
        raise ValueError(f"{ticker} CSV must contain Date and Close and must not be empty")
    dates = pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"{ticker} dates must be valid, unique, and strictly increasing")
    if close.isna().any() or not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError(f"{ticker} Close must contain only finite positive values")
    observed = pd.DatetimeIndex(dates)
    expected = expected_sessions()
    missing, unexpected = expected.difference(observed), observed.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(f"NYSE session mismatch: missing={missing.strftime('%Y-%m-%d').tolist()}, "
                         f"unexpected={unexpected.strftime('%Y-%m-%d').tolist()}")
    report = metadata.get("validation", {})
    if (report.get("rows") != len(frame) or report.get("date_min") != START.strftime("%Y-%m-%d")
            or report.get("date_max") != END.strftime("%Y-%m-%d")
            or report.get("missing_values_by_column", {}).get("Close") != 0
            or report.get("invalid_price_values_by_column", {}).get("Close") != 0):
        raise ValueError(f"{ticker} metadata validation does not match the checked CSV")
    return pd.DataFrame({"Date": dates, "Close": close.astype(float)})


def load_checked_spy(csv_path, metadata_path):
    return load_checked_asset(csv_path, metadata_path, "SPY")


def label_for(close_t, close_t1):
    return int(close_t1 > close_t)


def split_for(date):
    date = pd.Timestamp(date)
    if pd.Timestamp("2010-01-01") <= date <= pd.Timestamp("2021-12-31"):
        return "train"
    if pd.Timestamp("2022-01-01") <= date <= pd.Timestamp("2023-12-31"):
        return "val"
    if pd.Timestamp("2024-01-01") <= date <= pd.Timestamp("2025-12-31"):
        return "test"
    return None


def iter_samples(frame, ticker="SPY"):
    """Yield (metadata, 20 input closes); caller must validate sessions first."""
    for end_idx in range(WINDOW - 1, len(frame) - 1):
        window = frame.iloc[end_idx - WINDOW + 1:end_idx + 1]
        next_row = frame.iloc[end_idx + 1]
        split = split_for(next_row["Date"])
        if split is None:
            continue
        current = window.iloc[-1]
        dates = window["Date"].dt.strftime("%Y-%m-%d").tolist()
        yield ({"ticker": ticker, "input_start_date": dates[0], "input_end_date": dates[-1],
                "input_dates": "|".join(dates), "label_date": next_row["Date"].strftime("%Y-%m-%d"),
                "close_t": float(current["Close"]), "close_t1": float(next_row["Close"]),
                "label": label_for(current["Close"], next_row["Close"]), "split": split},
               window["Close"].to_numpy(copy=True))

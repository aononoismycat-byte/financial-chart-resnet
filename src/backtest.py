"""Cost-aware long/flat backtesting primitives (P1 Stage 2)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def sleeve_returns(positions, asset_returns, cost_bps=0.0, delay=0):
    """Compute gross/net returns, effective position and one-way turnover for one sleeve."""
    pos = np.asarray(positions, dtype=float)
    ret = np.asarray(asset_returns, dtype=float)
    if pos.ndim != 1 or ret.shape != pos.shape or pos.size == 0:
        raise ValueError("positions and asset_returns must be nonempty equally sized 1-D arrays")
    if not np.isfinite(pos).all() or not np.isfinite(ret).all():
        raise ValueError("positions and asset_returns must be finite")
    if not np.isin(pos, [0.0, 1.0]).all():
        raise ValueError("sleeve positions must be binary 0/1")
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    if delay < 0 or not isinstance(delay, (int, np.integer)):
        raise ValueError("delay must be a non-negative integer")

    if delay == 0:
        eff_pos = pos.copy()
    elif delay >= pos.size:
        eff_pos = np.zeros_like(pos)
    else:
        eff_pos = np.concatenate([np.zeros(int(delay), dtype=float), pos[:-int(delay)]])

    prev_pos = np.concatenate([[0.0], eff_pos[:-1]])
    turnover = np.abs(eff_pos - prev_pos)
    gross = eff_pos * ret
    net = gross - (float(cost_bps) / 1e4) * turnover
    index = positions.index if isinstance(positions, pd.Series) else None
    return pd.DataFrame(
        {"gross": gross, "net": net, "position": eff_pos, "turnover": turnover},
        index=index,
    )


def performance(net_returns, positions, turnover, periods=252):
    """Compute annualized return, risk, drawdown, turnover and hit-rate metrics."""
    r = np.asarray(net_returns, dtype=float)
    pos = np.asarray(positions, dtype=float)
    to = np.asarray(turnover, dtype=float)
    if r.ndim != 1 or pos.shape != r.shape or to.shape != r.shape or r.size == 0:
        raise ValueError("net_returns, positions and turnover must be nonempty equally sized 1-D arrays")
    n = r.size
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    total_return = float(wealth[-1] - 1.0)
    cagr = float(wealth[-1] ** (periods / n) - 1.0) if wealth[-1] > 0 else -1.0
    std = float(np.std(r, ddof=1)) if n > 1 else 0.0
    mean_r = float(np.mean(r))
    ann_vol = float(std * math.sqrt(periods)) if n > 1 else math.nan
    sharpe = float(mean_r / std * math.sqrt(periods)) if (n > 1 and std > 1e-15) else math.nan
    downside_rms = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2)))
    sortino = float(mean_r / downside_rms * math.sqrt(periods)) if downside_rms > 1e-15 else math.nan
    peaks = np.maximum.accumulate(wealth)
    drawdowns = wealth / peaks - 1.0
    max_dd = float(np.min(drawdowns))
    calmar = float(cagr / abs(max_dd)) if max_dd < -1e-15 else math.nan
    annual_turnover = float(np.sum(to) / n * periods)
    exposure = float(np.mean(pos))
    in_market = pos > 0
    hit_rate = float(np.mean(r[in_market] > 0)) if in_market.any() else math.nan
    return {
        "n": int(n),
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "annual_turnover": annual_turnover,
        "exposure": exposure,
        "hit_rate": hit_rate,
    }


def portfolio(sleeves):
    """Equal-weight daily portfolio across sleeve DataFrames."""
    frames = list(sleeves.values()) if isinstance(sleeves, dict) else list(sleeves)
    if not frames:
        raise ValueError("sleeves must be nonempty")
    ref_index = frames[0].index
    for df in frames[1:]:
        if not df.index.equals(ref_index):
            raise ValueError("all sleeves must share identical date alignment")
    cols = ["gross", "net", "position", "turnover"]
    stacked = np.mean([df[cols].to_numpy(dtype=float) for df in frames], axis=0)
    return pd.DataFrame(stacked, columns=cols, index=ref_index)

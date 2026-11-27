"""P1 Stage 2 (and reusable): cost-aware long/flat backtest on Adj Close returns.

Evaluates directional signals (1 = long 1 unit, 0 = flat cash at 0% return)
across one-way transaction costs (0, 2, 5, 10 bps) and execution delays (0, 1).
Post-hoc exploratory evaluation; never used for model selection or tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import performance, portfolio, sleeve_returns
from src.metrics import bootstrap_p_value, date_block_bootstrap, percentile_interval

TICKERS = ("SPY", "QQQ", "IWM", "TLT", "GLD")
COSTS_BPS = (0, 2, 5, 10)
DELAYS = (0, 1)

MODEL_SPECS = {
    "buy_and_hold": {"col": None, "label": "Buy & hold (= Majority)", "portfolio": True},
    "trend_20d": {"col": "trend_pred", "label": "20-day trend", "portfolio": True},
    "logreg_closes": {"col": "logreg_pred", "alt_col": "logreg_closes_pred", "label": "Logistic regression (closes)", "portfolio": True},
    "resnet_pooled": {"col": "resnet_pooled_pred", "label": "Pooled ResNet-18", "portfolio": True},
    "resnet_spy": {"col": "resnet_spy_pred", "label": "SPY ResNet-18", "portfolio": False},
    "logreg_features": {"col": "logreg_features_pred", "label": "Logistic regression (features)", "portfolio": True},
    "lightgbm": {"col": "lightgbm_pred", "label": "LightGBM (features)", "portfolio": True},
    "lightgbm_ext": {"col": "lightgbm_ext_pred", "label": "LightGBM (extended, exploratory)", "portfolio": True},
    "resnet_embedding_head": {"col": "resnet_embedding_head_pred", "label": "ResNet-18 embedding + LR", "portfolio": True},
    "logreg_closes_wf2026": {"col": "logreg_closes_wf2026_pred", "label": "Logistic regression (closes, WF2026)", "portfolio": True},
    "logreg_features_wf2026": {"col": "logreg_features_wf2026_pred", "label": "Logistic regression (features, WF2026)", "portfolio": True},
    "lightgbm_wf2026": {"col": "lightgbm_wf2026_pred", "label": "LightGBM (features, WF2026)", "portfolio": True},
    "resnet_embedding_head_wf2026": {"col": "resnet_embedding_head_wf2026_pred", "label": "ResNet-18 embedding + LR (WF2026)", "portfolio": True},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_adj_returns(tickers, raw_dir, expected_hashes=None, manifest_path=None):
    """Load close-to-close Adj Close total returns and verify session alignment."""
    manifest_map = None
    if manifest_path is not None:
        manifest_df = pd.read_csv(manifest_path, usecols=["ticker", "label_date", "input_end_date"])
        manifest_map = manifest_df.set_index(["ticker", "label_date"])["input_end_date"].to_dict()

    returns_by_ticker = {}
    raw_hashes = {}
    for ticker in tickers:
        csv_path = Path(raw_dir) / f"{ticker}.csv"
        file_hash = sha256(csv_path)
        if expected_hashes is not None and file_hash != expected_hashes[ticker]:
            raise ValueError(f"{ticker} raw CSV hash mismatch: {file_hash} != {expected_hashes[ticker]}")
        raw_hashes[ticker] = file_hash
        df = pd.read_csv(csv_path, dtype={"Date": "string"})
        if "Date" not in df or "Adj Close" not in df:
            raise ValueError(f"{ticker} CSV must contain Date and Adj Close")
        dates = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
        adj = pd.to_numeric(df["Adj Close"], errors="coerce")
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError(f"{ticker} dates invalid or non-increasing")
        if adj.isna().any() or not np.isfinite(adj).all() or (adj <= 0).any():
            raise ValueError(f"{ticker} Adj Close contains invalid values")
        date_strs = dates.dt.strftime("%Y-%m-%d").tolist()
        adj_vals = adj.to_numpy(dtype=float)
        by_date = {}
        for i in range(1, len(date_strs)):
            by_date[date_strs[i]] = (date_strs[i - 1], float(adj_vals[i] / adj_vals[i - 1] - 1.0))
        returns_by_ticker[ticker] = by_date
    return returns_by_ticker, manifest_map, raw_hashes


def align_returns(frame, returns_by_ticker, manifest_map=None):
    """Attach Adj Close return for each (ticker, label_date) after session checks."""
    out = frame.copy()
    adj_rets = []
    for row in out.itertuples(index=False):
        t_map = returns_by_ticker[row.ticker]
        if row.label_date not in t_map:
            raise ValueError(f"{row.ticker} label_date {row.label_date} missing from raw Adj Close history")
        prev_date, ret = t_map[row.label_date]
        if manifest_map is not None:
            key = (row.ticker, row.label_date)
            if key not in manifest_map:
                raise ValueError(f"Missing {key} in manifest")
            if prev_date != manifest_map[key]:
                raise ValueError(
                    f"Session alignment mismatch for {key}: raw prior session {prev_date} != input_end_date {manifest_map[key]}"
                )
        adj_rets.append(ret)
    out["adj_return"] = adj_rets
    return out


def _sharpe(returns, periods=252):
    n = returns.size
    std = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    if n <= 1 or std <= 1e-15:
        return math.nan
    return float(np.mean(returns) / std * math.sqrt(periods))


def bootstrap_portfolio_sharpe(port_by_model, settings, baseline="buy_and_hold"):
    """Date-block bootstrap Sharpe and Sharpe minus buy-and-hold on daily portfolio returns."""
    models = list(port_by_model.keys())
    dates = port_by_model[models[0]].index.to_numpy()
    nets = {m: port_by_model[m]["net"].to_numpy(dtype=float) for m in models}
    reps = {m: [] for m in models}
    for idx in date_block_bootstrap(
        dates,
        settings["block_length_sessions"],
        settings["replicates"],
        settings["seed"],
    ):
        for m in models:
            reps[m].append(_sharpe(nets[m][idx]))
    reps = {m: np.asarray(v, dtype=float) for m, v in reps.items()}
    out = {}
    for m in models:
        point = _sharpe(nets[m])
        entry = {"sharpe": point, "sharpe_ci": percentile_interval(reps[m])}
        if baseline in reps and m != baseline:
            diff_point = point - _sharpe(nets[baseline])
            diff_rep = reps[m] - reps[baseline]
            entry["diff_vs_buy_and_hold"] = float(diff_point)
            entry["diff_ci"] = percentile_interval(diff_rep)
            entry["p_boot"] = bootstrap_p_value(diff_rep)
        out[m] = entry
    return out


def run_backtest(
    frame,
    models,
    returns_by_ticker,
    manifest_map,
    settings,
    costs=COSTS_BPS,
    delays=DELAYS,
    compute_bootstrap=True,
):
    """Run sleeve and equal-weight portfolio backtests across costs and delays."""
    aligned = align_returns(frame, returns_by_ticker, manifest_map)
    aligned = aligned.sort_values(["ticker", "label_date"]).reset_index(drop=True)
    tickers = tuple(t for t in TICKERS if t in set(aligned["ticker"]))

    results = {"models": {m: MODEL_SPECS[m]["label"] for m in models}, "portfolio": {}, "sleeves": {t: {} for t in tickers}}
    equity_5bp_d0 = {}
    port_5bp_d0 = {}

    for model in models:
        spec = MODEL_SPECS[model]
        col = spec["col"]
        if col is not None and col not in aligned.columns and spec.get("alt_col") in aligned.columns:
            col = spec["alt_col"]
        is_portfolio = spec["portfolio"]
        if is_portfolio:
            results["portfolio"][model] = {}
        for delay in delays:
            d_key = f"delay_{delay}"
            if is_portfolio:
                results["portfolio"][model][d_key] = {}
            for cost in costs:
                c_key = f"{cost}bp"
                sleeves = {}
                for ticker in tickers:
                    sub = aligned[aligned["ticker"] == ticker].sort_values("label_date")
                    if col is None:
                        pos = pd.Series(1, index=sub["label_date"].to_numpy(), dtype=float)
                    else:
                        if sub[col].isna().all():
                            continue
                        if sub[col].isna().any():
                            raise ValueError(f"Partial NaNs for {model} on {ticker}")
                        pos = pd.Series(sub[col].to_numpy(dtype=float), index=sub["label_date"].to_numpy())
                    ret = pd.Series(sub["adj_return"].to_numpy(dtype=float), index=sub["label_date"].to_numpy())
                    s_df = sleeve_returns(pos, ret, cost_bps=cost, delay=delay)
                    sleeves[ticker] = s_df
                    results["sleeves"][ticker].setdefault(model, {}).setdefault(d_key, {})[c_key] = performance(
                        s_df["net"], s_df["position"], s_df["turnover"]
                    )
                if is_portfolio:
                    port_df = portfolio(sleeves)
                    results["portfolio"][model][d_key][c_key] = performance(
                        port_df["net"], port_df["position"], port_df["turnover"]
                    )
                    if delay == 0 and cost == 5:
                        port_5bp_d0[model] = port_df
                        equity_5bp_d0[model] = (1.0 + port_df["net"]).cumprod()

    if compute_bootstrap and port_5bp_d0:
        results["bootstrap_5bp_delay0"] = bootstrap_portfolio_sharpe(port_5bp_d0, settings)
    else:
        results["bootstrap_5bp_delay0"] = {}
    return results, equity_5bp_d0


def _fmt_pct(v):
    return "—" if v is None or not math.isfinite(v) else f"{100 * v:.2f}%"


def _fmt_num(v, digits=3):
    return "—" if v is None or not math.isfinite(v) else f"{v:.{digits}f}"


def _fmt_ci(ci, digits=3):
    return f"[{_fmt_num(ci[0], digits)}, {_fmt_num(ci[1], digits)}]"


def plot_equity_curves(equity_dict, model_labels, output, title):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    palette = ["#222222", "#7f7f7f", "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for i, (model, series) in enumerate(equity_dict.items()):
        dates = pd.to_datetime(series.index)
        ax.plot(dates, series.to_numpy(), label=model_labels[model], color=palette[i % len(palette)], linewidth=1.7)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.9)
    ax.set(xlabel="Date", ylabel="Portfolio equity (start = 1.00)", title=title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_cost_sensitivity(results, model_labels, output, title):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    palette = ["#222222", "#7f7f7f", "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for i, model in enumerate(results["portfolio"]):
        sharpes = [results["portfolio"][model]["delay_0"][f"{c}bp"]["sharpe"] for c in COSTS_BPS]
        ax.plot(COSTS_BPS, sharpes, marker="o", label=model_labels[model], color=palette[i % len(palette)], linewidth=1.7)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    ax.set(xticks=list(COSTS_BPS), xlabel="One-way transaction cost (bps)", ylabel="Annualized Sharpe ratio (rf = 0)", title=title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def write_backtest_markdown(payload, path, title, equity_img=None, cost_img=None):
    primary = payload["primary"]
    boot = primary.get("bootstrap_5bp_delay0", {})
    labels = primary["models"]
    lines = [
        f"# {title}",
        "",
        f"> {payload['study_status']} Long/flat equal-weight five-asset portfolio (`SPY`, `QQQ`, `IWM`, `TLT`, `GLD`) "
        "on dividend-adjusted `Adj Close` returns, market-on-close execution (`delay=0`), zero risk-free rate. "
        "Because the training majority class is 1 (`Up`), the constant majority baseline is identical to **Buy & hold**.",
        "",
        f"## Equal-weight portfolio ({payload['primary_split']}, 0 bp & 5 bp one-way cost, delay 0)",
        "",
        "| Model | Cost | CAGR | Ann. vol | Sharpe [95% CI] | Sortino | Max DD | Calmar | Turnover/yr | Exposure | Hit rate |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, d_map in primary["portfolio"].items():
        for cost in (0, 5):
            m = d_map["delay_0"][f"{cost}bp"]
            s_str = _fmt_num(m["sharpe"])
            if cost == 5 and model in boot:
                s_str = f"{s_str} {_fmt_ci(boot[model]['sharpe_ci'])}"
            lines.append(
                f"| {labels[model]} | {cost} bp | {_fmt_pct(m['cagr'])} | {_fmt_pct(m['ann_vol'])} | {s_str} | "
                f"{_fmt_num(m['sortino'])} | {_fmt_pct(m['max_drawdown'])} | {_fmt_num(m['calmar'])} | "
                f"{_fmt_num(m['annual_turnover'], 1)}x | {_fmt_pct(m['exposure'])} | {_fmt_pct(m['hit_rate'])} |"
            )

    if boot:
        lines.extend(
            [
                "",
                "## Portfolio Sharpe difference vs. Buy & hold (5 bp, delay 0, date-block bootstrap)",
                "",
                "| Model − Buy & hold | Sharpe diff | 95% CI | Bootstrap p |",
                "|---|---:|---|---:|",
            ]
        )
        for model, b in boot.items():
            if "diff_vs_buy_and_hold" not in b:
                continue
            lines.append(
                f"| {labels[model]} − Buy & hold | {b['diff_vs_buy_and_hold']:+.3f} | "
                f"[{b['diff_ci'][0]:+.3f}, {b['diff_ci'][1]:+.3f}] | {b['p_boot']:.3g} |"
            )

    lines.extend(
        [
            "",
            "## Cost sensitivity (portfolio Sharpe, delay 0)",
            "",
            "| Model | 0 bp | 2 bp | 5 bp | 10 bp |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model, d_map in primary["portfolio"].items():
        vals = " | ".join(_fmt_num(d_map["delay_0"][f"{c}bp"]["sharpe"]) for c in COSTS_BPS)
        lines.append(f"| {labels[model]} | {vals} |")

    lines.extend(
        [
            "",
            "## Execution delay sensitivity (portfolio Sharpe, 0 bp and 5 bp)",
            "",
            "| Model | Delay 0 (0 bp) | Delay 1 (0 bp) | Delay 0 (5 bp) | Delay 1 (5 bp) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model, d_map in primary["portfolio"].items():
        lines.append(
            f"| {labels[model]} | {_fmt_num(d_map['delay_0']['0bp']['sharpe'])} | "
            f"{_fmt_num(d_map['delay_1']['0bp']['sharpe'])} | "
            f"{_fmt_num(d_map['delay_0']['5bp']['sharpe'])} | "
            f"{_fmt_num(d_map['delay_1']['5bp']['sharpe'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-asset sleeve Sharpe & CAGR (5 bp, delay 0)",
            "",
            "| Asset | Model | CAGR | Sharpe | Max DD | Turnover/yr | Exposure |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for ticker, m_map in primary["sleeves"].items():
        for model, d_map in m_map.items():
            m = d_map["delay_0"]["5bp"]
            lines.append(
                f"| {ticker} | {labels[model]} | {_fmt_pct(m['cagr'])} | {_fmt_num(m['sharpe'])} | "
                f"{_fmt_pct(m['max_drawdown'])} | {_fmt_num(m['annual_turnover'], 1)}x | {_fmt_pct(m['exposure'])} |"
            )

    if payload.get("validation_context") is not None:
        vc = payload["validation_context"]
        lines.extend(
            [
                "",
                "## Validation context (2022–2023, point estimates, delay 0)",
                "",
                "| Model | Cost | CAGR | Sharpe | Max DD | Turnover/yr | Exposure |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model, d_map in vc["portfolio"].items():
            for cost in (0, 5):
                m = d_map["delay_0"][f"{cost}bp"]
                lines.append(
                    f"| {vc['models'][model]} | {cost} bp | {_fmt_pct(m['cagr'])} | {_fmt_num(m['sharpe'])} | "
                    f"{_fmt_pct(m['max_drawdown'])} | {_fmt_num(m['annual_turnover'], 1)}x | {_fmt_pct(m['exposure'])} |"
                )

    for img, alt in ((equity_img, "Equity curves at 5 bp"), (cost_img, "Cost sensitivity")):
        if img:
            lines.extend(["", f"![{alt}]({img})"])
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "results/p1/predictions.csv")
    parser.add_argument("--predictions2", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["buy_and_hold", "trend_20d", "logreg_closes", "resnet_pooled", "resnet_spy"],
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "results/p1/stage2")
    parser.add_argument("--equity-plot", type=Path, default=None)
    parser.add_argument("--cost-plot", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/multi_asset_samples.csv")
    parser.add_argument("--no-manifest-check", action="store_true")
    parser.add_argument("--no-raw-hash-check", action="store_true")
    parser.add_argument("--title", default="P1 Stage 2 — Cost-aware long/flat backtest")
    parser.add_argument("--study-status", default="Post-hoc exploratory evaluation; never used for model selection or tuning.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    json_path = Path(f"{args.output_prefix}_backtest.json")
    md_path = Path(f"{args.output_prefix}_backtest.md")
    eq_path = args.equity_plot if args.equity_plot is not None else Path(f"{args.output_prefix}_equity_{args.split}.png")
    cost_path = args.cost_plot if args.cost_plot is not None else Path(f"{args.output_prefix}_cost_sensitivity.png")

    targets = [p for p in (json_path, md_path, eq_path, cost_path) if p is not None]
    if not args.overwrite and any(p.exists() for p in targets):
        raise FileExistsError("Backtest output exists; pass --overwrite to replace")

    protocol = json.loads((ROOT / "experiments/p1_protocol.json").read_text(encoding="utf-8"))
    settings = protocol["stages"]["stage_1_probabilistic_evaluation"]["uncertainty"]
    summary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
    expected_hashes = None if args.no_raw_hash_check else {t: summary["asset_audit"][t]["raw_sha256"] for t in TICKERS}
    manifest_path = None if args.no_manifest_check else args.manifest

    df = pd.read_csv(args.predictions)
    if args.predictions2 is not None:
        df2 = pd.read_csv(args.predictions2)
        join_cols = [c for c in ("split", "ticker", "label_date", "label") if c in df.columns and c in df2.columns]
        extra_cols = [c for c in df2.columns if c not in df.columns]
        df = df.merge(df2[join_cols + extra_cols], on=join_cols, how="inner", validate="one_to_one")

    returns_by_ticker, manifest_map, raw_hashes = load_adj_returns(
        TICKERS, args.raw_dir, expected_hashes=expected_hashes, manifest_path=manifest_path
    )

    if args.split == "all" or "split" not in df.columns:
        primary_df = df.copy()
    else:
        primary_df = df[df["split"] == args.split].copy()
    if primary_df.empty:
        raise ValueError(f"No rows for split={args.split}")

    primary_res, eq_curves = run_backtest(primary_df, args.models, returns_by_ticker, manifest_map, settings, compute_bootstrap=True)

    val_res = None
    if args.val_split.lower() != "none" and "split" in df.columns and (df["split"] == args.val_split).any():
        val_df = df[df["split"] == args.val_split].copy()
        val_res, _ = run_backtest(val_df, args.models, returns_by_ticker, manifest_map, settings, compute_bootstrap=False)

    payload = {
        "title": args.title,
        "study_status": args.study_status,
        "primary_split": args.split,
        "raw_sha256": raw_hashes,
        "bootstrap": {k: settings[k] for k in ("block_length_sessions", "replicates", "seed", "interval")},
        "primary": primary_res,
        "validation_context": val_res,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    plot_equity_curves(eq_curves, primary_res["models"], eq_path, f"{args.title} ({args.split}, 5 bp, delay 0)")
    plot_cost_sensitivity(primary_res, primary_res["models"], cost_path, f"Portfolio Sharpe vs. transaction cost ({args.split}, delay 0)")
    write_backtest_markdown(payload, md_path, args.title, equity_img=eq_path.name, cost_img=cost_path.name)
    print("Saved", json_path.as_posix(), md_path.as_posix(), eq_path.as_posix(), cost_path.as_posix())


if __name__ == "__main__":
    main()

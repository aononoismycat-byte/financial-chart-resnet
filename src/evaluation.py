"""Shared probabilistic evaluation and markdown reporting across P1 stages."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.metrics import (
    accuracy,
    balanced_accuracy,
    bootstrap_p_value,
    brier,
    calibration_bins,
    date_block_bootstrap,
    log_loss,
    percentile_interval,
    roc_auc,
    summarize,
)

DEFAULT_BOOTSTRAP = {"block_length_sessions": 20, "replicates": 5000, "seed": 20261127, "interval": "95% percentile"}

BOOT_METRICS = {
    "accuracy": lambda y, p, d: accuracy(y, d),
    "balanced_accuracy": lambda y, p, d: balanced_accuracy(y, d),
    "roc_auc": lambda y, p, d: roc_auc(y, p),
    "brier": lambda y, p, d: brier(y, p),
    "log_loss": lambda y, p, d: log_loss(y, p),
}


def _reference_for(sub, reference_probability):
    if isinstance(reference_probability, str):
        return sub[reference_probability].to_numpy(dtype=float)
    return float(reference_probability)


def _arrays(sub, spec, label_col="label"):
    y = sub[label_col].to_numpy(dtype=np.int8)
    p = sub[spec["prob"]].to_numpy(dtype=float)
    d = sub[spec["pred"]].to_numpy(dtype=int)
    return y, p, d


def _bootstrap_replicates(sub, model_specs, settings, label_col="label", date_col="label_date"):
    data = {name: _arrays(sub, spec, label_col=label_col) for name, spec in model_specs.items()}
    reps = {name: {k: [] for k in BOOT_METRICS} for name in model_specs}
    for idx in date_block_bootstrap(
        sub[date_col].to_numpy(),
        settings["block_length_sessions"],
        settings["replicates"],
        settings["seed"],
    ):
        for name in model_specs:
            y, p, d = (arr[idx] for arr in data[name])
            for key, fn in BOOT_METRICS.items():
                reps[name][key].append(fn(y, p, d))
    return {name: {k: np.asarray(vals, dtype=float) for k, vals in metrics.items()} for name, metrics in reps.items()}


def evaluate_predictions(
    frame,
    models,
    reference_probability,
    settings,
    groups,
    comparisons,
    bootstrap=True,
    label_col="label",
    date_col="label_date",
    ticker_col="ticker",
):
    """Evaluate classification models across groups with date-block bootstrap uncertainty.

    models maps name -> {"prob": col, "pred": col, "hard": bool, "label": str}.
    """
    settings = {**DEFAULT_BOOTSTRAP, **(settings or {})}
    result = {
        "reference_probability": reference_probability if isinstance(reference_probability, str) else float(reference_probability),
        "bootstrap": (
            {k: settings[k] for k in ("block_length_sessions", "replicates", "seed", "interval") if k in settings}
            if bootstrap
            else None
        ),
        "models": {name: dict(spec) for name, spec in models.items()},
        "groups": {},
        "comparisons": {},
        "calibration": {},
    }

    for group in groups:
        sub = frame if group == "overall" else frame[frame[ticker_col] == group]
        valid_masks = {}
        full_models = {}
        partial_models = {}
        for name, spec in models.items():
            mask = sub[spec["prob"]].notna() & sub[spec["pred"]].notna()
            if not mask.any():
                continue
            valid_masks[name] = mask.to_numpy()
            if mask.all():
                full_models[name] = spec
            else:
                partial_models[name] = spec

        group_out = {}
        shared_reps = {}
        if full_models:
            ref_full = _reference_for(sub, reference_probability)
            for name, spec in full_models.items():
                y, p, d = _arrays(sub, spec, label_col=label_col)
                group_out[name] = {"metrics": summarize(y, p, d, ref_full, hard_scores=spec.get("hard", False))}
            if bootstrap:
                shared_reps = _bootstrap_replicates(sub, full_models, settings, label_col=label_col, date_col=date_col)
                for name in full_models:
                    group_out[name]["ci"] = {k: percentile_interval(shared_reps[name][k]) for k in BOOT_METRICS}

        for name, spec in partial_models.items():
            sub_m = sub.loc[valid_masks[name]]
            ref_m = _reference_for(sub_m, reference_probability)
            y, p, d = _arrays(sub_m, spec, label_col=label_col)
            group_out[name] = {"metrics": summarize(y, p, d, ref_m, hard_scores=spec.get("hard", False))}
            if bootstrap:
                reps_m = _bootstrap_replicates(sub_m, {name: spec}, settings, label_col=label_col, date_col=date_col)
                group_out[name]["ci"] = {k: percentile_interval(reps_m[name][k]) for k in BOOT_METRICS}

        result["groups"][group] = group_out

        if group == groups[0]:
            for name, spec in models.items():
                if name in valid_masks and not spec.get("hard", False):
                    sub_m = sub.loc[valid_masks[name]]
                    y, p, _ = _arrays(sub_m, spec, label_col=label_col)
                    result["calibration"][name] = calibration_bins(y, p)["bins"]

        comp_rows = []
        for a, b in comparisons or ():
            if a not in group_out or b not in group_out:
                continue
            if a in full_models and b in full_models:
                sub_pair = sub
                pair_reps = shared_reps
            else:
                pair_mask = valid_masks[a] & valid_masks[b]
                if not pair_mask.any():
                    continue
                sub_pair = sub.loc[pair_mask]
                pair_reps = (
                    _bootstrap_replicates(
                        sub_pair,
                        {a: models[a], b: models[b]},
                        settings,
                        label_col=label_col,
                        date_col=date_col,
                    )
                    if bootstrap
                    else None
                )
            ya, pa, da = _arrays(sub_pair, models[a], label_col=label_col)
            yb, pb, db = _arrays(sub_pair, models[b], label_col=label_col)
            for k, fn in BOOT_METRICS.items():
                diff = float(fn(ya, pa, da) - fn(yb, pb, db))
                entry = {"a": a, "b": b, "metric": k, "difference": diff}
                if bootstrap:
                    d_rep = pair_reps[a][k] - pair_reps[b][k]
                    entry["ci"] = percentile_interval(d_rep)
                    entry["p_boot"] = bootstrap_p_value(d_rep)
                comp_rows.append(entry)
        result["comparisons"][group] = comp_rows

    return result


def auc_vs_half(frame, model, settings=None, label_col="label", date_col="label_date"):
    """Bootstrap ROC-AUC minus 0.5, returning CI and two-sided sign p-value."""
    prob_col = model["prob"] if isinstance(model, dict) else str(model)
    settings = {**DEFAULT_BOOTSTRAP, **(settings or {})}
    sub = frame[frame[prob_col].notna()]
    y = sub[label_col].to_numpy(dtype=np.int8)
    p = sub[prob_col].to_numpy(dtype=float)
    point_auc = roc_auc(y, p)
    auc_reps = np.array(
        [
            roc_auc(y[idx], p[idx])
            for idx in date_block_bootstrap(
                sub[date_col].to_numpy(),
                settings["block_length_sessions"],
                settings["replicates"],
                settings["seed"],
            )
        ],
        dtype=float,
    )
    diff_reps = auc_reps - 0.5
    return {
        "n": int(len(sub)),
        "roc_auc": float(point_auc),
        "difference": float(point_auc - 0.5),
        "auc_ci": percentile_interval(auc_reps),
        "ci": percentile_interval(diff_reps),
        "p_boot": bootstrap_p_value(diff_reps),
    }


def fmt_pct(value):
    return "—" if value is None or not math.isfinite(value) else f"{100 * value:.2f}%"


def fmt_num(value, digits=4):
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def fmt_ci(ci, pct=False, digits=4):
    formatter = fmt_pct if pct else (lambda v: fmt_num(v, digits))
    return f"[{formatter(ci[0])}, {formatter(ci[1])}]"


def write_metrics_markdown(result, path, title, intro_lines=None, validation_result=None, image_name=None):
    """Write a markdown summary matching the Stage 1 table layout."""
    models = result["models"]
    groups = list(result["groups"].keys())
    primary = groups[0]
    boot = result.get("bootstrap")
    ref = result["reference_probability"]

    lines = [f"# {title}", ""]
    if intro_lines:
        lines.extend(intro_lines)
        lines.append("")
    if isinstance(ref, str):
        lines.append(f"Climatology reference probability column: `{ref}`.")
    else:
        lines.append(f"Climatology reference probability (training up-ratio): **{ref:.4f}**.")
    if boot:
        lines.append(
            f"Uncertainty intervals are {boot['interval']} circular moving-block bootstrap over label dates "
            f"(block {boot['block_length_sessions']} sessions, {boot['replicates']} replicates)."
        )
    lines.append("")

    primary_n = next(iter(result["groups"][primary].values()))["metrics"]["n"]
    lines.append(f"## {primary.capitalize()} ({primary_n:,} samples)")
    lines.append("")
    if boot:
        lines.append(
            "| Model | Accuracy [95% CI] | Balanced accuracy [95% CI] | ROC-AUC [95% CI] | Brier | Brier skill | Log loss | ECE | Binomial p |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    else:
        lines.append("| Model | Accuracy | Balanced accuracy | ROC-AUC | Brier | Brier skill | Log loss | ECE | Binomial p |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for name, entry in result["groups"][primary].items():
        s = entry["metrics"]
        hard = s["hard_scores"]
        label = models[name]["label"]
        if boot:
            ci = entry["ci"]
            acc_str = f"{fmt_pct(s['accuracy'])} {fmt_ci(ci['accuracy'], True)}"
            bal_str = f"{fmt_pct(s['balanced_accuracy'])} {fmt_ci(ci['balanced_accuracy'], True)}"
            auc_str = f"{fmt_num(s['roc_auc'], 3)} {fmt_ci(ci['roc_auc'], digits=3)}"
        else:
            acc_str = fmt_pct(s["accuracy"])
            bal_str = fmt_pct(s["balanced_accuracy"])
            auc_str = fmt_num(s["roc_auc"], 3)
        lines.append(
            f"| {label} | {acc_str} | {bal_str} | {auc_str} | "
            f"{'—' if hard else fmt_num(s['brier'])} | {'—' if hard else fmt_num(s['brier_skill_score'])} | "
            f"{'—' if hard else fmt_num(s['log_loss'])} | {'—' if hard else fmt_num(s['ece'])} | "
            f"{s['binomial_vs_50']['p_value']:.3g} |"
        )

    asset_groups = [g for g in groups if g != primary]
    if asset_groups:
        lines.extend(["", "## Per-asset metrics", ""])
        if boot:
            lines.append("| Asset | Up rate | Model | Accuracy | Balanced accuracy [95% CI] | ROC-AUC [95% CI] | Brier skill |")
            lines.append("|---|---:|---|---:|---|---|---:|")
        else:
            lines.append("| Asset | Up rate | Model | Accuracy | Balanced accuracy | ROC-AUC | Brier skill |")
            lines.append("|---|---:|---|---:|---:|---:|---:|")
        for g in asset_groups:
            for name, entry in result["groups"][g].items():
                s = entry["metrics"]
                label = models[name]["label"]
                if boot:
                    ci = entry["ci"]
                    bal_str = f"{fmt_pct(s['balanced_accuracy'])} {fmt_ci(ci['balanced_accuracy'], True)}"
                    auc_str = f"{fmt_num(s['roc_auc'], 3)} {fmt_ci(ci['roc_auc'], digits=3)}"
                else:
                    bal_str = fmt_pct(s["balanced_accuracy"])
                    auc_str = fmt_num(s["roc_auc"], 3)
                lines.append(
                    f"| {g} | {fmt_pct(s['up_ratio'])} | {label} | {fmt_pct(s['accuracy'])} | "
                    f"{bal_str} | {auc_str} | {'—' if s['hard_scores'] else fmt_num(s['brier_skill_score'])} |"
                )

    comp_rows = result["comparisons"].get(primary, [])
    if comp_rows:
        lines.extend(
            [
                "",
                f"## Paired differences on {primary} (A − B)",
                "",
                "Positive differences favour A for accuracy, balanced accuracy and AUC; negative differences favour A for Brier and log loss.",
                "",
            ]
        )
        if boot:
            lines.append("| A − B | Metric | Difference | 95% CI | Bootstrap p |")
            lines.append("|---|---|---:|---|---:|")
        else:
            lines.append("| A − B | Metric | Difference |")
            lines.append("|---|---|---:|")
        for row in comp_rows:
            if models[row["a"]].get("hard", False) and row["metric"] in ("brier", "log_loss"):
                continue
            pct = row["metric"] in ("accuracy", "balanced_accuracy")
            diff = f"{100 * row['difference']:+.2f} pp" if pct else f"{row['difference']:+.4f}"
            pair_name = f"{models[row['a']]['label']} − {models[row['b']]['label']}"
            if boot:
                ci = (
                    f"[{100 * row['ci'][0]:+.2f}, {100 * row['ci'][1]:+.2f}] pp"
                    if pct
                    else f"[{row['ci'][0]:+.4f}, {row['ci'][1]:+.4f}]"
                )
                lines.append(f"| {pair_name} | {row['metric']} | {diff} | {ci} | {row['p_boot']:.3g} |")
            else:
                lines.append(f"| {pair_name} | {row['metric']} | {diff} |")

    if validation_result is not None and primary in validation_result.get("groups", {}):
        lines.extend(
            [
                "",
                "## Validation context (point estimates)",
                "",
                "| Model | Accuracy | Balanced accuracy | ROC-AUC | Brier skill | Log loss |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, entry in validation_result["groups"][primary].items():
            s = entry["metrics"]
            label = models[name]["label"]
            lines.append(
                f"| {label} | {fmt_pct(s['accuracy'])} | {fmt_pct(s['balanced_accuracy'])} | "
                f"{fmt_num(s['roc_auc'], 3)} | {'—' if s['hard_scores'] else fmt_num(s['brier_skill_score'])} | "
                f"{'—' if s['hard_scores'] else fmt_num(s['log_loss'])} |"
            )

    if image_name:
        lines.extend(["", f"![Figure]({image_name})"])
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")

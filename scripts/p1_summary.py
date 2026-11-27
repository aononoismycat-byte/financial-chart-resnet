"""P1 Stage 7: generate unified results/p1/summary_all.md directly from committed JSON outputs."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

P1 = ROOT / "results/p1"


def fmt_pct(v):
    return "—" if v is None or not math.isfinite(v) else f"{100 * v:.2f}%"


def fmt_num(v, digits=3):
    return "—" if v is None or not math.isfinite(v) else f"{v:.{digits}f}"


def fmt_ci(ci, digits=3):
    return f"[{fmt_num(ci[0], digits)}, {fmt_num(ci[1], digits)}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=P1 / "summary_all.md")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.overwrite and args.output.exists():
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace")

    s3m = json.loads((P1 / "stage3_metrics.json").read_text(encoding="utf-8"))["test"]
    s3b = json.loads((P1 / "stage3_backtest.json").read_text(encoding="utf-8"))["primary"]
    s4 = json.loads((P1 / "stage4_walk_forward.json").read_text(encoding="utf-8"))
    s5 = json.loads((P1 / "stage5_seeds.json").read_text(encoding="utf-8"))
    s8 = json.loads((P1 / "stage8_holdout.json").read_text(encoding="utf-8"))

    # Table A: 2024-2025 held-out test (2,510 samples, post-hoc exploratory)
    rows_a = [
        ("majority", "buy_and_hold", "Majority / Buy & hold"),
        ("trend_20d", "trend_20d", "20-day trend"),
        ("resnet_pooled", "resnet_pooled", "Pooled ResNet-18"),
        ("logreg_closes", "logreg_closes", "Logistic regression (closes)"),
        ("logreg_features", "logreg_features", "Logistic regression (features)"),
        ("lightgbm", "lightgbm", "LightGBM (features)"),
        ("lightgbm_ext", "lightgbm_ext", "LightGBM (extended, exploratory)"),
    ]
    lines = [
        "# P1 Unified Summary Across All Evaluation Stages",
        "",
        "> **Scope**: Stages 1–6 cover the `2024–2025` test period (`2,510` samples, **post-hoc exploratory**) and `2014–2023` unseen walk-forward OOS (`12,580` samples). Stage 8 (`2026-01-02` to `2026-11-25`, `1,135` samples) is a **pre-registered confirmatory holdout** (`5aa00e4785af31c2f5acfd9c9b5685676bd10482`).",
        "",
        "## Table A — 2024–2025 Held-Out Test (2,510 samples, 5 bp one-way cost, delay 0)",
        "",
        "| Model | Accuracy | Balanced Acc | ROC-AUC [95% CI] | Brier Skill | 5 bp Sharpe [95% CI] | Max DD | Turnover/yr |",
        "|---|---:|---:|---|---:|---|---:|---:|",
    ]
    for m_key, b_key, label in rows_a:
        gm = s3m["groups"]["overall"][m_key]
        m, ci = gm["metrics"], gm["ci"]
        bs = "—" if m["hard_scores"] else fmt_num(m["brier_skill_score"], 4)
        bt = s3b["portfolio"][b_key]["delay_0"]["5bp"]
        bci = s3b["bootstrap_5bp_delay0"][b_key]["sharpe_ci"]
        lines.append(
            f"| {label} | {fmt_pct(m['accuracy'])} | {fmt_pct(m['balanced_accuracy'])} | "
            f"{fmt_num(m['roc_auc'], 3)} {fmt_ci(ci['roc_auc'], 3)} | {bs} | "
            f"{fmt_num(bt['sharpe'], 3)} {fmt_ci(bci, 3)} | {fmt_pct(bt['max_drawdown'])} | "
            f"{fmt_num(bt['annual_turnover'], 1)}x |"
        )

    # Table B: Robustness (Walk-forward 2014-2023, Stage 5 multi-seed, Stage 8 2026 holdout)
    wf_groups = s4["unseen_2014_2023"]["groups"]["overall"]
    fam = s5["family_summary"]
    s8_groups = s8["evaluation"]["groups"]["overall"]
    s8_bt = s8["backtest_5bp_delay0"]
    s8_bci = s8["backtest_bootstrap_5bp_delay0"]

    rows_b = [
        ("majority", "majority", "buy_and_hold", "Majority / Buy & hold"),
        ("trend_20d", "trend_20d", "trend_20d", "20-day trend"),
        ("resnet_embedding_head", "resnet_pooled", "resnet_pooled", "ResNet-18 (pooled / emb head)"),
        ("logreg_closes", "logreg_closes", "logreg_closes", "Logistic regression (closes)"),
        ("logreg_features", "logreg_features", "logreg_features", "Logistic regression (features)"),
        ("lightgbm", "lightgbm", "lightgbm", "LightGBM (features)"),
        (None, "logreg_closes_wf2026", "logreg_closes_wf2026", "LR (closes, WF2026)"),
        (None, "logreg_features_wf2026", "logreg_features_wf2026", "LR (features, WF2026)"),
        (None, "lightgbm_wf2026", "lightgbm_wf2026", "LightGBM (features, WF2026)"),
        (None, "resnet_embedding_head_wf2026", "resnet_embedding_head_wf2026", "ResNet-18 emb + LR (WF2026)"),
    ]
    lines.extend(
        [
            "",
            "## Table B — Walk-Forward, Multi-Seed & 2026 Confirmatory Holdout Robustness",
            "",
            "| Model | WF 2014–2023 AUC [95% CI] | Stage 5 Seed AUC (mean ± sd) | 2026 Holdout AUC [95% CI] | 2026 5 bp Sharpe |",
            "|---|---|---|---|---:|",
        ]
    )
    for wf_k, h_k, bt_k, label in rows_b:
        if wf_k is None or wf_k not in wf_groups:
            wf_str = "—"
        elif wf_k == "majority":
            wf_str = "0.500 (constant per fold; compare vs 0.500)"
        else:
            wm = wf_groups[wf_k]
            wf_str = f"{fmt_num(wm['metrics']['roc_auc'], 3)} {fmt_ci(wm['ci']['roc_auc'], 3)}"

        if h_k == "resnet_pooled":
            seed_str = f"{fam['resnet']['roc_auc']['mean']:.4f} ± {fam['resnet']['roc_auc']['sd']:.4f}"
        elif h_k == "lightgbm":
            seed_str = f"{fam['lightgbm']['roc_auc']['mean']:.4f} ± {fam['lightgbm']['roc_auc']['sd']:.4f}"
        else:
            seed_str = "—"

        hg = s8_groups[h_k]
        h_str = f"{fmt_num(hg['metrics']['roc_auc'], 3)} {fmt_ci(hg['ci']['roc_auc'], 3)}"
        bci = s8_bci[bt_k]["sharpe_ci"]
        sh_str = f"{fmt_num(s8_bt[bt_k]['sharpe'], 3)} {fmt_ci(bci, 3)}"
        lines.append(f"| {label} | {wf_str} | {seed_str} | {h_str} | {sh_str} |")

    lines.extend(
        [
            "",
            "> **Walk-forward climatology note**: Across pooled walk-forward folds, fold-varying `climatology_prob` induces a spurious `0.480` pooled AUC artifact; walk-forward model AUCs must be evaluated against `0.500` rather than against the pooled majority row.",
            "",
            "## Table C — Stage 8 Pre-Registered Confirmatory Family (2026 Holdout, Holm α = 0.05)",
            "",
            "| ID | Hypothesis | Estimate / Diff | 95% CI (diff vs null) | Raw p | Holm p | Significant? |",
            "|---|---|---:|---|---:|---:|---|",
        ]
    )
    for h in s8["confirmatory_family"]:
        lines.append(
            f"| {h['id']} | `{h['hypothesis']}` | {h['estimate']:.4f} | "
            f"[{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}] | {h['p_boot']:.4g} | {h['p_holm']:.4g} | "
            f"{'Yes' if h['significant_alpha_0_05'] else 'No'} |"
        )
    lines.append("")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

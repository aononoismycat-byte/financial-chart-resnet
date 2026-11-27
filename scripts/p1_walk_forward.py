"""P1 Stage 4: expanding-window annual walk-forward validation (2014-2025)."""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import evaluate_predictions, fmt_num, write_metrics_markdown
from src.features import ASSET_ONEHOT, PRIMARY_FEATURES
from src.metrics import accuracy, balanced_accuracy, roc_auc
from scripts.p1_backtest import load_adj_returns, plot_cost_sensitivity, plot_equity_curves, run_backtest, write_backtest_markdown
from scripts.p1_build_features import NORM_COLS, TICKERS, sha256
from scripts.p1_train_tabular import fit_lightgbm, fit_logreg

SEED = 42
GROUPS = ("overall", *TICKERS)
P1 = ROOT / "results/p1"


def build_walk_forward_folds(years, start_test_year=2014, end_test_year=2025, min_train_year=2010):
    """Build expanding-window (train<=Y-2, val=Y-1, test=Y) boolean masks."""
    years_arr = np.asarray(years, dtype=int)
    folds = []
    for test_year in range(start_test_year, end_test_year + 1):
        val_year = test_year - 1
        train_mask = (years_arr >= min_train_year) & (years_arr <= test_year - 2)
        val_mask = years_arr == val_year
        test_mask = years_arr == test_year
        if not (train_mask.any() and val_mask.any() and test_mask.any()):
            raise ValueError(f"Incomplete split for walk-forward test_year={test_year}")
        if (train_mask & val_mask).any() or (train_mask & test_mask).any() or (val_mask & test_mask).any():
            raise ValueError(f"Overlapping masks for test_year={test_year}")
        folds.append({"test_year": test_year, "val_year": val_year, "train": train_mask, "val": val_mask, "test": test_mask})
    return folds


def fit_embedding_head(x_train: np.ndarray, y_train: np.ndarray):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=SEED))
    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe.fit(x_train, y_train)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            converged = False
    return pipe, converged


def plot_auc_by_year(yearly_rows, output):
    df = pd.DataFrame(yearly_rows)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    series = [
        ("trend_20d", "20-day trend", "#7f7f7f", "--"),
        ("logreg_closes", "Logistic regression (closes)", "#1f77b4", "-"),
        ("logreg_features", "Logistic regression (features)", "#ff7f0e", "-"),
        ("lightgbm", "LightGBM (features)", "#2ca02c", "-"),
        ("resnet_embedding_head", "ResNet-18 embedding + LR", "#d62728", "-"),
    ]
    for key, label, color, ls in series:
        ax.plot(df["test_year"], df[f"{key}_auc"], marker="o", linestyle=ls, color=color, label=label, linewidth=1.7)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.0, label="Random (AUC = 0.50)")
    ax.axvspan(2023.5, 2025.5, color="#f0e442", alpha=0.22, label="2024–2025 (previously observed)")
    ax.set(
        xticks=df["test_year"].tolist(),
        xlabel="Held-out test year",
        ylabel="Out-of-sample ROC-AUC",
        title="P1 Stage 4 — Walk-forward out-of-sample ROC-AUC by year (2014–2025)",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_preds = P1 / "stage4_walk_forward_predictions.csv"
    out_meta = P1 / "stage4_walk_forward_meta.json"
    out_json = P1 / "stage4_walk_forward.json"
    out_md = P1 / "stage4_walk_forward.md"
    out_auc_plot = P1 / "stage4_auc_by_year.png"
    out_bt_prefix = P1 / "stage4"
    targets = [
        out_preds,
        out_meta,
        out_json,
        out_md,
        out_auc_plot,
        Path(f"{out_bt_prefix}_backtest.json"),
        Path(f"{out_bt_prefix}_backtest.md"),
        Path(f"{out_bt_prefix}_equity_oos.png"),
        Path(f"{out_bt_prefix}_cost_sensitivity.png"),
    ]
    if not args.overwrite and any(p.exists() for p in targets):
        raise FileExistsError("Stage 4 outputs exist; pass --overwrite to replace")

    feat_meta = json.loads((P1 / "stage3_features_meta.json").read_text(encoding="utf-8"))
    feat_csv = ROOT / feat_meta["features_csv"]
    if sha256(feat_csv) != feat_meta["features_sha256"]:
        raise ValueError("Feature CSV hash mismatch")
    df = pd.read_csv(feat_csv)

    emb_meta = json.loads((P1 / "stage4_embeddings_meta.json").read_text(encoding="utf-8"))
    emb_npy = ROOT / emb_meta["embeddings_npy"]
    keys_csv = ROOT / emb_meta["keys_csv"]
    if sha256(emb_npy) != emb_meta["embeddings_sha256"] or sha256(keys_csv) != emb_meta["keys_sha256"]:
        raise ValueError("Embeddings or keys hash mismatch")
    emb = np.load(emb_npy)
    emb_keys = pd.read_csv(keys_csv)
    pd.testing.assert_frame_equal(df[["ticker", "label_date", "split", "label"]], emb_keys, check_dtype=False)

    # Fixed-split sanity check for resnet_embedding_head vs Stage 1 resnet_pooled_prob
    s1 = pd.read_csv(P1 / "predictions.csv")
    train_orig = (df["split"] == "train").to_numpy()
    test_orig = (df["split"] == "test").to_numpy()
    sanity_pipe, sanity_conv = fit_embedding_head(emb[train_orig], df.loc[train_orig, "label"].to_numpy(dtype=int))
    sanity_prob = sanity_pipe.predict_proba(emb[test_orig])[:, 1]
    sanity_pred = sanity_pipe.predict(emb[test_orig])
    sanity_df = df.loc[test_orig, ["ticker", "label_date", "label"]].copy()
    sanity_df["emb_lr_prob"] = sanity_prob
    sanity_df["emb_lr_pred"] = sanity_pred
    sanity_merged = s1[s1["split"] == "test"].merge(sanity_df, on=["ticker", "label_date", "label"], how="inner", validate="one_to_one")
    pearson_r = float(np.corrcoef(sanity_merged["emb_lr_prob"], sanity_merged["resnet_pooled_prob"])[0, 1])
    fixed_sanity = {
        "converged": sanity_conv,
        "test_accuracy": accuracy(sanity_merged["label"], sanity_merged["emb_lr_pred"]),
        "test_roc_auc": roc_auc(sanity_merged["label"], sanity_merged["emb_lr_prob"]),
        "pearson_r_vs_resnet_pooled_prob": pearson_r,
    }

    years = pd.to_datetime(df["label_date"]).dt.year.to_numpy()
    folds = build_walk_forward_folds(years, 2014, 2025, min_train_year=2010)
    primary_cols = PRIMARY_FEATURES + ASSET_ONEHOT

    oos_frames = []
    fold_logs = []
    yearly_rows = []
    for fold in folds:
        ty = fold["test_year"]
        tr, va, te = fold["train"], fold["val"], fold["test"]
        y_tr = df.loc[tr, "label"].to_numpy(dtype=int)
        y_va = df.loc[va, "label"].to_numpy(dtype=int)
        y_te = df.loc[te, "label"].to_numpy(dtype=int)

        maj = int(np.count_nonzero(y_tr == 1) > np.count_nonzero(y_tr == 0))
        clim = float(y_tr.mean())
        trend_te = (df.loc[te, "ret_19"].to_numpy() > 0).astype(int)

        lr_c = fit_logreg(df.loc[tr, NORM_COLS], y_tr)
        lr_f = fit_logreg(df.loc[tr, primary_cols], y_tr)
        lgbm = fit_lightgbm(df.loc[tr, primary_cols], y_tr, df.loc[va, primary_cols], y_va, seed=SEED)
        emb_head, emb_conv = fit_embedding_head(emb[tr], y_tr)

        part = df.loc[te, ["ticker", "label_date", "label"]].copy().reset_index(drop=True)
        part.insert(0, "test_year", ty)
        part["climatology_prob"] = clim
        part["majority_pred"] = maj
        part["trend_pred"] = trend_te
        part["logreg_closes_prob"] = lr_c.predict_proba(df.loc[te, NORM_COLS])[:, 1]
        part["logreg_closes_pred"] = lr_c.predict(df.loc[te, NORM_COLS])
        part["logreg_features_prob"] = lr_f.predict_proba(df.loc[te, primary_cols])[:, 1]
        part["logreg_features_pred"] = lr_f.predict(df.loc[te, primary_cols])
        part["lightgbm_prob"] = lgbm.predict_proba(df.loc[te, primary_cols])[:, 1]
        part["lightgbm_pred"] = (part["lightgbm_prob"] > 0.5).astype(int)
        part["resnet_embedding_head_prob"] = emb_head.predict_proba(emb[te])[:, 1]
        part["resnet_embedding_head_pred"] = emb_head.predict(emb[te])
        oos_frames.append(part)

        fold_logs.append(
            {
                "test_year": ty,
                "val_year": fold["val_year"],
                "n_train": int(tr.sum()),
                "n_val": int(va.sum()),
                "n_test": int(te.sum()),
                "train_up_ratio": clim,
                "majority_class": maj,
                "lightgbm_best_iteration": int(lgbm.best_iteration_),
                "resnet_embedding_head_converged": emb_conv,
            }
        )
        y_row = {"test_year": ty, "n": int(te.sum()), "up_ratio": float(y_te.mean())}
        for name, p_col, d_col in (
            ("majority", "climatology_prob", "majority_pred"),
            ("trend_20d", "trend_pred", "trend_pred"),
            ("logreg_closes", "logreg_closes_prob", "logreg_closes_pred"),
            ("logreg_features", "logreg_features_prob", "logreg_features_pred"),
            ("lightgbm", "lightgbm_prob", "lightgbm_pred"),
            ("resnet_embedding_head", "resnet_embedding_head_prob", "resnet_embedding_head_pred"),
        ):
            y_row[f"{name}_acc"] = accuracy(y_te, part[d_col])
            y_row[f"{name}_bal_acc"] = balanced_accuracy(y_te, part[d_col])
            y_row[f"{name}_auc"] = roc_auc(y_te, part[p_col])
        yearly_rows.append(y_row)
        print(
            f"WF {ty}: LGBM_AUC={y_row['lightgbm_auc']:.3f} LR_feat_AUC={y_row['logreg_features_auc']:.3f} "
            f"ResNet_emb_AUC={y_row['resnet_embedding_head_auc']:.3f}",
            flush=True,
        )

    oos_df = pd.concat(oos_frames, ignore_index=True)
    oos_df.to_csv(out_preds, index=False, float_format="%.12g", lineterminator="\n")

    meta = {
        "stage": "P1 Stage 4 expanding walk-forward validation (2014-2025)",
        "features_sha256": feat_meta["features_sha256"],
        "embeddings_sha256": emb_meta["embeddings_sha256"],
        "fixed_split_sanity_check": fixed_sanity,
        "folds": fold_logs,
        "walk_forward_predictions_sha256": sha256(out_preds),
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    protocol = json.loads((ROOT / "experiments/p1_protocol.json").read_text(encoding="utf-8"))
    settings = protocol["stages"]["stage_1_probabilistic_evaluation"]["uncertainty"]
    model_specs = {
        "majority": {"prob": "climatology_prob", "pred": "majority_pred", "hard": False, "label": "Majority / climatology"},
        "trend_20d": {"prob": "trend_pred", "pred": "trend_pred", "hard": True, "label": "20-day trend"},
        "logreg_closes": {"prob": "logreg_closes_prob", "pred": "logreg_closes_pred", "hard": False, "label": "Logistic regression (closes)"},
        "logreg_features": {"prob": "logreg_features_prob", "pred": "logreg_features_pred", "hard": False, "label": "Logistic regression (features)"},
        "lightgbm": {"prob": "lightgbm_prob", "pred": "lightgbm_pred", "hard": False, "label": "LightGBM (features)"},
        "resnet_embedding_head": {
            "prob": "resnet_embedding_head_prob",
            "pred": "resnet_embedding_head_pred",
            "hard": False,
            "label": "ResNet-18 embedding + LR",
        },
    }
    comparisons = [
        ("lightgbm", "majority"),
        ("lightgbm", "logreg_closes"),
        ("lightgbm", "resnet_embedding_head"),
        ("logreg_features", "logreg_closes"),
        ("resnet_embedding_head", "logreg_closes"),
    ]
    eval_all = evaluate_predictions(oos_df, model_specs, "climatology_prob", settings, GROUPS, comparisons, bootstrap=True)
    eval_unseen = evaluate_predictions(
        oos_df[oos_df["test_year"] <= 2023], model_specs, "climatology_prob", settings, GROUPS, comparisons, bootstrap=True
    )

    payload = {
        "stage": "P1 Stage 4 walk-forward validation",
        "fixed_split_sanity_check": fixed_sanity,
        "yearly": yearly_rows,
        "pooled_2014_2025": eval_all,
        "unseen_2014_2023": eval_unseen,
    }
    out_json.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    plot_auc_by_year(yearly_rows, out_auc_plot)

    # Write combined markdown report (2014-2023 unseen + 2014-2025 pooled + per-year)
    write_metrics_markdown(
        eval_unseen,
        out_md,
        "P1 Stage 4 — Walk-forward validation (2014–2025)",
        intro_lines=[
            "> Expanding-window annual refit (`train: 2010..Y-2`, `val: Y-1`, `test: Y`). "
            "Below is the **2014–2023 unseen out-of-sample** summary (years never touched during Phase 1/2 model design), followed by the full 2014–2025 pooled OOS results and per-year breakdown.",
            f"> Fixed-split sanity check for `ResNet-18 embedding + LR`: test accuracy `{100 * fixed_sanity['test_accuracy']:.2f}%`, "
            f"ROC-AUC `{fixed_sanity['test_roc_auc']:.3f}`, Pearson r vs. Stage 1 `resnet_pooled_prob` = `{fixed_sanity['pearson_r_vs_resnet_pooled_prob']:.3f}`.",
        ],
        image_name=out_auc_plot.name,
    )
    extra = [
        "",
        "## Pooled 2014–2025 OOS overall (includes previously observed 2024–2025)",
        "",
        "| Model | Accuracy [95% CI] | Balanced accuracy [95% CI] | ROC-AUC [95% CI] | Brier skill | Log loss |",
        "|---|---|---|---|---:|---:|",
    ]
    for name, entry in eval_all["groups"]["overall"].items():
        s, ci = entry["metrics"], entry["ci"]
        bs_str = "—" if s["hard_scores"] else fmt_num(s["brier_skill_score"])
        ll_str = "—" if s["hard_scores"] else fmt_num(s["log_loss"])
        extra.append(
            f"| {model_specs[name]['label']} | {100*s['accuracy']:.2f}% [{100*ci['accuracy'][0]:.2f}%, {100*ci['accuracy'][1]:.2f}%] | "
            f"{100*s['balanced_accuracy']:.2f}% [{100*ci['balanced_accuracy'][0]:.2f}%, {100*ci['balanced_accuracy'][1]:.2f}%] | "
            f"{s['roc_auc']:.3f} [{ci['roc_auc'][0]:.3f}, {ci['roc_auc'][1]:.3f}] | "
            f"{bs_str} | {ll_str} |"
        )
    extra.extend(
        [
            "",
            "## Annual test ROC-AUC by year",
            "",
            "| Year | Up rate | 20-day trend | LR (closes) | LR (features) | LightGBM | ResNet-18 emb + LR | Note |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for r in yearly_rows:
        note = "previously observed" if r["test_year"] >= 2024 else "unseen OOS"
        extra.append(
            f"| {r['test_year']} | {100*r['up_ratio']:.2f}% | {r['trend_20d_auc']:.3f} | {r['logreg_closes_auc']:.3f} | "
            f"{r['logreg_features_auc']:.3f} | {r['lightgbm_auc']:.3f} | {r['resnet_embedding_head_auc']:.3f} | {note} |"
        )
    with out_md.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(extra) + "\n")

    # Backtest concatenated OOS signals 2014-2025
    summary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
    expected_hashes = {t: summary["asset_audit"][t]["raw_sha256"] for t in TICKERS}
    returns_by_ticker, manifest_map, raw_hashes = load_adj_returns(
        TICKERS, ROOT / "data/raw", expected_hashes=expected_hashes, manifest_path=ROOT / "data/processed/multi_asset_samples.csv"
    )
    bt_models = ["buy_and_hold", "trend_20d", "logreg_closes", "logreg_features", "lightgbm", "resnet_embedding_head"]
    bt_res, eq_oos = run_backtest(oos_df, bt_models, returns_by_ticker, manifest_map, settings, compute_bootstrap=True)
    bt_payload = {
        "title": "P1 Stage 4 — Walk-forward OOS backtest (2014–2025)",
        "study_status": "Walk-forward expanding OOS backtest (2014-2023 unseen; 2024-2025 previously observed).",
        "primary_split": "oos_2014_2025",
        "raw_sha256": raw_hashes,
        "bootstrap": {k: settings[k] for k in ("block_length_sessions", "replicates", "seed", "interval")},
        "primary": bt_res,
        "validation_context": None,
    }
    Path(f"{out_bt_prefix}_backtest.json").write_text(json.dumps(bt_payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    plot_equity_curves(eq_oos, bt_res["models"], P1 / "stage4_equity_oos.png", "P1 Stage 4 — Walk-forward OOS equity (2014–2025, 5 bp, delay 0)")
    plot_cost_sensitivity(bt_res, bt_res["models"], Path(f"{out_bt_prefix}_cost_sensitivity.png"), "P1 Stage 4 — Walk-forward OOS Sharpe vs. cost")
    write_backtest_markdown(
        bt_payload,
        Path(f"{out_bt_prefix}_backtest.md"),
        "P1 Stage 4 — Walk-forward OOS backtest (2014–2025)",
        equity_img="stage4_equity_oos.png",
        cost_img="stage4_cost_sensitivity.png",
    )
    print("Stage 4 complete.")


if __name__ == "__main__":
    main()

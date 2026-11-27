"""P1 Stage 3: train tabular baselines (LogReg features & LightGBM), evaluate & backtest.

Post-hoc exploratory extension on the frozen chronological split (2010-2021 train,
2022-2023 validation for LightGBM early stopping only, 2024-2025 single test pass).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import evaluate_predictions, write_metrics_markdown
from src.features import ASSET_ONEHOT, EXTENDED_FEATURES, PRIMARY_FEATURES, predict_logreg_json, save_logreg_json
from scripts.p1_backtest import load_adj_returns, plot_cost_sensitivity, plot_equity_curves, run_backtest, write_backtest_markdown
from scripts.p1_build_features import NORM_COLS, TICKERS, sha256

SEED = 42
GROUPS = ("overall", *TICKERS)
P1 = ROOT / "results/p1"


def fit_logreg(x_train: pd.DataFrame, y_train: np.ndarray):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipe.fit(x_train, y_train)
    return pipe


def fit_lightgbm(x_train: pd.DataFrame, y_train: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray, seed: int = SEED):
    model = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=15,
        learning_rate=0.02,
        n_estimators=2000,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=4,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_preds = P1 / "stage3_predictions.csv"
    out_meta = P1 / "stage3_predictions_meta.json"
    out_metrics_json = P1 / "stage3_metrics.json"
    out_metrics_md = P1 / "stage3_metrics.md"
    out_bt_prefix = P1 / "stage3"
    models_dir = P1 / "models"
    model_paths = [
        models_dir / "lightgbm_seed42.txt",
        models_dir / "lightgbm_ext_seed42.txt",
        models_dir / "logreg_features.json",
        models_dir / "logreg_closes.json",
    ]
    all_targets = [
        out_preds,
        out_meta,
        out_metrics_json,
        out_metrics_md,
        Path(f"{out_bt_prefix}_backtest.json"),
        Path(f"{out_bt_prefix}_backtest.md"),
        Path(f"{out_bt_prefix}_equity_test.png"),
        Path(f"{out_bt_prefix}_cost_sensitivity.png"),
        *model_paths,
    ]
    if not args.overwrite and any(p.exists() for p in all_targets):
        raise FileExistsError("Stage 3 outputs exist; pass --overwrite to replace")

    feat_meta = json.loads((P1 / "stage3_features_meta.json").read_text(encoding="utf-8"))
    feat_csv = ROOT / feat_meta["features_csv"]
    if sha256(feat_csv) != feat_meta["features_sha256"]:
        raise ValueError("Feature CSV hash mismatch against stage3_features_meta.json")
    df = pd.read_csv(feat_csv)

    s1_meta = json.loads((P1 / "predictions_meta.json").read_text(encoding="utf-8"))
    s1_csv = P1 / "predictions.csv"
    if sha256(s1_csv) != s1_meta["predictions_sha256"]:
        raise ValueError("Stage 1 predictions.csv hash mismatch")
    s1 = pd.read_csv(s1_csv)

    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    vt_mask = df["split"].isin(["val", "test"]).to_numpy()
    y_train = df.loc[train_mask, "label"].to_numpy(dtype=int)
    y_val = df.loc[val_mask, "label"].to_numpy(dtype=int)

    # Gate 1: LogReg on norm_close_* must exactly match Stage 1 logreg_pred.
    lr_closes = fit_logreg(df.loc[train_mask, NORM_COLS], y_train)
    vt_df = df.loc[vt_mask, ["split", "ticker", "label_date", "label"]].copy().reset_index(drop=True)
    vt_df["logreg_closes_check_prob"] = lr_closes.predict_proba(df.loc[vt_mask, NORM_COLS])[:, 1]
    vt_df["logreg_closes_check_pred"] = lr_closes.predict(df.loc[vt_mask, NORM_COLS])
    merged_check = s1.merge(vt_df, on=["split", "ticker", "label_date", "label"], how="inner", validate="one_to_one")
    if len(merged_check) != len(s1):
        raise ValueError("Stage 1 / feature merge row count mismatch")
    if not (merged_check["logreg_closes_check_pred"].to_numpy() == merged_check["logreg_pred"].to_numpy()).all():
        raise ValueError("Reproduction gate failed: logreg_closes_check predictions != Stage 1 logreg_pred")
    max_logreg_prob_diff = float(np.max(np.abs(merged_check["logreg_closes_check_prob"] - merged_check["logreg_prob"])))

    primary_cols = PRIMARY_FEATURES + ASSET_ONEHOT
    ext_cols = PRIMARY_FEATURES + EXTENDED_FEATURES + ASSET_ONEHOT

    lr_feat = fit_logreg(df.loc[train_mask, primary_cols], y_train)
    lgbm_primary = fit_lightgbm(df.loc[train_mask, primary_cols], y_train, df.loc[val_mask, primary_cols], y_val, seed=SEED)
    lgbm_ext = fit_lightgbm(df.loc[train_mask, ext_cols], y_train, df.loc[val_mask, ext_cols], y_val, seed=SEED)

    lgb_iter_p = int(lgbm_primary.best_iteration_)
    lgb_iter_e = int(lgbm_ext.best_iteration_)
    models_dir.mkdir(parents=True, exist_ok=True)
    lgbm_primary.booster_.save_model(str(models_dir / "lightgbm_seed42.txt"), num_iteration=lgb_iter_p)
    lgbm_ext.booster_.save_model(str(models_dir / "lightgbm_ext_seed42.txt"), num_iteration=lgb_iter_e)
    save_logreg_json(lr_feat, primary_cols, models_dir / "logreg_features.json")
    save_logreg_json(lr_closes, NORM_COLS, models_dir / "logreg_closes.json")

    # Verify serialized models reproduce predictions identically.
    p_lr_feat, d_lr_feat = predict_logreg_json(models_dir / "logreg_features.json", df.loc[vt_mask])
    if not (d_lr_feat == lr_feat.predict(df.loc[vt_mask, primary_cols])).all():
        raise ValueError("Serialized logreg_features.json prediction mismatch")
    b_p = lgb.Booster(model_file=str(models_dir / "lightgbm_seed42.txt"))
    b_e = lgb.Booster(model_file=str(models_dir / "lightgbm_ext_seed42.txt"))
    p_lgb_p = b_p.predict(df.loc[vt_mask, primary_cols])
    p_lgb_e = b_e.predict(df.loc[vt_mask, ext_cols])
    if float(np.max(np.abs(p_lgb_p - lgbm_primary.predict_proba(df.loc[vt_mask, primary_cols])[:, 1]))) > 1e-12:
        raise ValueError("Serialized lightgbm_seed42.txt prediction mismatch")
    if float(np.max(np.abs(p_lgb_e - lgbm_ext.predict_proba(df.loc[vt_mask, ext_cols])[:, 1]))) > 1e-12:
        raise ValueError("Serialized lightgbm_ext_seed42.txt prediction mismatch")

    out_df = df.loc[vt_mask, ["split", "ticker", "label_date", "label"]].copy().reset_index(drop=True)
    out_df["logreg_features_prob"] = p_lr_feat
    out_df["logreg_features_pred"] = d_lr_feat
    out_df["lightgbm_prob"] = p_lgb_p
    out_df["lightgbm_pred"] = (p_lgb_p > 0.5).astype(int)
    out_df["lightgbm_ext_prob"] = p_lgb_e
    out_df["lightgbm_ext_pred"] = (p_lgb_e > 0.5).astype(int)

    P1.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_preds, index=False, float_format="%.12g", lineterminator="\n")

    meta = {
        "stage": "P1 Stage 3 tabular baselines (post-hoc exploratory)",
        "features_sha256": feat_meta["features_sha256"],
        "stage1_predictions_sha256": s1_meta["predictions_sha256"],
        "versions": {"lightgbm": lgb.__version__, "scikit_learn": sklearn.__version__},
        "gates": {
            "logreg_closes_exact_pred_match": True,
            "logreg_closes_max_abs_prob_diff": max_logreg_prob_diff,
            "serialized_models_verified": True,
        },
        "lightgbm": {
            "seed": SEED,
            "features": primary_cols,
            "best_iteration": lgb_iter_p,
            "best_val_logloss": float(lgbm_primary.best_score_["valid_0"]["binary_logloss"]),
        },
        "lightgbm_ext": {
            "seed": SEED,
            "features": ext_cols,
            "best_iteration": lgb_iter_e,
            "best_val_logloss": float(lgbm_ext.best_score_["valid_0"]["binary_logloss"]),
        },
        "model_files_sha256": {p.name: sha256(p) for p in model_paths},
        "stage3_predictions_sha256": sha256(out_preds),
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Evaluation via src.evaluation
    protocol = json.loads((ROOT / "experiments/p1_protocol.json").read_text(encoding="utf-8"))
    settings = protocol["stages"]["stage_1_probabilistic_evaluation"]["uncertainty"]
    merged = s1.merge(out_df, on=["split", "ticker", "label_date", "label"], how="inner", validate="one_to_one")
    model_specs = {
        "majority": {"prob": "climatology_prob", "pred": "majority_pred", "hard": False, "label": "Majority / climatology"},
        "trend_20d": {"prob": "trend_pred", "pred": "trend_pred", "hard": True, "label": "20-day trend"},
        "logreg_closes": {"prob": "logreg_prob", "pred": "logreg_pred", "hard": False, "label": "Logistic regression (closes)"},
        "resnet_pooled": {"prob": "resnet_pooled_prob", "pred": "resnet_pooled_pred", "hard": False, "label": "Pooled ResNet-18"},
        "logreg_features": {"prob": "logreg_features_prob", "pred": "logreg_features_pred", "hard": False, "label": "Logistic regression (features)"},
        "lightgbm": {"prob": "lightgbm_prob", "pred": "lightgbm_pred", "hard": False, "label": "LightGBM (features)"},
        "lightgbm_ext": {"prob": "lightgbm_ext_prob", "pred": "lightgbm_ext_pred", "hard": False, "label": "LightGBM (extended, exploratory)"},
    }
    comparisons = [
        ("lightgbm", "majority"),
        ("lightgbm", "logreg_closes"),
        ("lightgbm", "resnet_pooled"),
        ("logreg_features", "logreg_closes"),
        ("lightgbm_ext", "lightgbm"),
    ]
    ref_prob = float(s1_meta["climatology_probability"])
    val_eval = evaluate_predictions(
        merged[merged["split"] == "val"], model_specs, ref_prob, settings, GROUPS, comparisons, bootstrap=False
    )
    test_eval = evaluate_predictions(
        merged[merged["split"] == "test"], model_specs, ref_prob, settings, GROUPS, comparisons, bootstrap=True
    )
    metrics_payload = {
        "stage": "P1 Stage 3 metrics (post-hoc exploratory)",
        "lightgbm_ext_note": "lightgbm_ext uses OHLCV + 60-session exploratory extended features; reported separately from the primary 20-close information set.",
        "validation": val_eval,
        "test": test_eval,
    }
    out_metrics_json.write_text(json.dumps(metrics_payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    write_metrics_markdown(
        test_eval,
        out_metrics_md,
        "P1 Stage 3 — Tabular feature baselines vs. ResNet-18",
        intro_lines=[
            "> Post-hoc exploratory analysis on the frozen 2024–2025 test split. Primary features use only the exact 20 unadjusted closes seen by ResNet-18 plus asset identity; `LightGBM (extended, exploratory)` uses OHLCV and 60-session history and is reported separately."
        ],
        validation_result=val_eval,
    )

    # Cost-aware backtest for Stage 3 models alongside baselines
    summary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
    expected_hashes = {t: summary["asset_audit"][t]["raw_sha256"] for t in TICKERS}
    returns_by_ticker, manifest_map, raw_hashes = load_adj_returns(
        TICKERS, ROOT / "data/raw", expected_hashes=expected_hashes, manifest_path=ROOT / "data/processed/multi_asset_samples.csv"
    )
    bt_models = ["buy_and_hold", "trend_20d", "logreg_closes", "resnet_pooled", "logreg_features", "lightgbm", "lightgbm_ext"]
    bt_test, eq_test = run_backtest(merged[merged["split"] == "test"], bt_models, returns_by_ticker, manifest_map, settings, compute_bootstrap=True)
    bt_val, _ = run_backtest(merged[merged["split"] == "val"], bt_models, returns_by_ticker, manifest_map, settings, compute_bootstrap=False)
    bt_payload = {
        "title": "P1 Stage 3 — Tabular vs. ResNet cost-aware backtest",
        "study_status": "Post-hoc exploratory evaluation; LightGBM (extended) uses OHLCV + 60-session exploratory features.",
        "primary_split": "test",
        "raw_sha256": raw_hashes,
        "bootstrap": {k: settings[k] for k in ("block_length_sessions", "replicates", "seed", "interval")},
        "primary": bt_test,
        "validation_context": bt_val,
    }
    Path(f"{out_bt_prefix}_backtest.json").write_text(json.dumps(bt_payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    plot_equity_curves(eq_test, bt_test["models"], Path(f"{out_bt_prefix}_equity_test.png"), "P1 Stage 3 — Portfolio equity (test, 5 bp, delay 0)")
    plot_cost_sensitivity(bt_test, bt_test["models"], Path(f"{out_bt_prefix}_cost_sensitivity.png"), "P1 Stage 3 — Sharpe vs. cost (test, delay 0)")
    write_backtest_markdown(
        bt_payload,
        Path(f"{out_bt_prefix}_backtest.md"),
        "P1 Stage 3 — Tabular vs. ResNet cost-aware backtest",
        equity_img="stage3_equity_test.png",
        cost_img="stage3_cost_sensitivity.png",
    )
    print(f"Stage 3 complete. LightGBM best_iter={lgb_iter_p}, LightGBM_ext best_iter={lgb_iter_e}")


if __name__ == "__main__":
    main()

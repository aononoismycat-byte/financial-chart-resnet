"""P1 Stage 8d (step 2): confirmatory evaluation & backtest on fresh 2026 holdout."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.transforms import Normalize

from src.charts import normalized_prices, render_chart
from src.evaluation import auc_vs_half, evaluate_predictions, write_metrics_markdown
from src.features import ASSET_ONEHOT, PRIMARY_FEATURES, predict_logreg_json, window_features
from src.metrics import holm
from src.model import ResNet18Classifier
from src.training import load_checkpoint
from scripts.p1_backtest import load_adj_returns, plot_cost_sensitivity, plot_equity_curves, run_backtest, write_backtest_markdown
from scripts.p1_build_features import NORM_COLS, TICKERS, sha256
from scripts.p1_download_holdout import verify_holdout_file
from scripts.train_spy_resnet import MEAN, STD

P1 = ROOT / "results/p1"
MODELS = P1 / "models"


def build_2026_samples():
    rows = []
    norm_rows = []
    feat_rows = []
    chart_tensors = []
    norm_transform = Normalize(MEAN, STD)

    for ticker in TICKERS:
        csv_path = ROOT / "data/raw_holdout_2026" / f"{ticker}.csv"
        df = pd.read_csv(csv_path, dtype={"Date": "string"})
        verify_holdout_file(ticker, df)
        dates = df["Date"].tolist()
        closes = df["Close"].to_numpy(dtype=float)
        onehot = {f"asset_{t}": int(t == ticker) for t in TICKERS}

        for idx in range(20, len(df)):
            label_date = dates[idx]
            if label_date < "2026-01-02" or label_date > "2026-11-25":
                continue
            window_closes = closes[idx - 20 : idx]
            close_t = float(closes[idx - 1])
            close_t1 = float(closes[idx])
            rows.append(
                {
                    "ticker": ticker,
                    "label_date": label_date,
                    "prev_session": dates[idx - 1],
                    "label": int(close_t1 > close_t),
                }
            )
            norm = normalized_prices(window_closes)
            norm_rows.append({f"norm_close_{i:02d}": float(v) for i, v in enumerate(norm)})
            f_dict = window_features(window_closes)
            f_dict.update(onehot)
            feat_rows.append(f_dict)

            pixels = np.array(render_chart(window_closes), dtype=np.uint8, copy=True)
            tensor = torch.from_numpy(pixels).permute(2, 0, 1).to(torch.float32).div_(255.0)
            chart_tensors.append(norm_transform(tensor))

    meta_df = pd.DataFrame(rows)
    norm_df = pd.DataFrame(norm_rows)
    feat_df = pd.DataFrame(feat_rows)
    images = torch.stack(chart_tensors, dim=0)
    return meta_df, norm_df, feat_df, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-reg-commit", required=True, help="Git commit SHA recorded at Stage 8c before downloading 2026 data")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_preds = P1 / "stage8_holdout_predictions.csv"
    out_meta = P1 / "stage8_holdout_meta.json"
    out_json = P1 / "stage8_holdout.json"
    out_md = P1 / "stage8_holdout.md"
    out_bt_prefix = P1 / "stage8"
    out_eq = P1 / "stage8_equity_2026.png"
    targets = [
        out_preds,
        out_meta,
        out_json,
        out_md,
        out_eq,
        Path(f"{out_bt_prefix}_backtest.json"),
        Path(f"{out_bt_prefix}_backtest.md"),
        Path(f"{out_bt_prefix}_cost_sensitivity.png"),
    ]
    if not args.overwrite and any(p.exists() for p in targets):
        raise FileExistsError("Stage 8 outputs exist; pass --overwrite to replace")

    protocol = json.loads((ROOT / "experiments/p1_protocol.json").read_text(encoding="utf-8"))
    amendments = [a for a in protocol.get("amendments", []) if a.get("stage") == "stage_8_optional_fresh_holdout"]
    if not amendments:
        raise ValueError("Stage 8 amendment must be recorded before running holdout evaluation")

    # Verify frozen artifacts match pre-registered SHA-256 values
    summary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
    freeze_8a = json.loads((P1 / "stage8_freeze_meta.json").read_text(encoding="utf-8"))
    s3_meta = json.loads((P1 / "stage3_predictions_meta.json").read_text(encoding="utf-8"))
    pooled_ckpt = ROOT / summary["freeze"]["pooled_checkpoint"]
    if sha256(pooled_ckpt) != summary["freeze"]["pooled_checkpoint_sha256"]:
        raise ValueError("Pooled ResNet checkpoint hash mismatch")
    for fname, expected in s3_meta["model_files_sha256"].items():
        if fname != "lightgbm_ext_seed42.txt" and sha256(MODELS / fname) != expected:
            raise ValueError(f"Frozen model hash mismatch for {fname}")
    for fname, expected in freeze_8a["frozen_artifacts_sha256"].items():
        if sha256(MODELS / fname) != expected:
            raise ValueError(f"Frozen WF2026 model hash mismatch for {fname}")

    meta_df, norm_df, feat_df, images = build_2026_samples()
    full_df = pd.concat([meta_df[["ticker", "label_date", "label"]], norm_df, feat_df], axis=1)

    # ResNet pooled + embedding extraction
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18Classifier(pretrained=False, training_mode="classifier_only").to(device)
    load_checkpoint(pooled_ckpt, model, device)
    model.eval()
    fc_layer = model.backbone.fc
    model.backbone.fc = nn.Identity()

    emb_chunks, prob_chunks, pred_chunks = [], [], []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            batch = images[start : start + 32].to(device)
            emb = model(batch)
            logits = fc_layer(emb)
            emb_chunks.append(emb.cpu().numpy().astype(np.float32))
            prob_chunks.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            pred_chunks.append(logits.argmax(dim=1).cpu().numpy())
    embeddings = np.concatenate(emb_chunks, axis=0)

    out_df = meta_df[["ticker", "label_date", "label"]].copy()
    out_df["climatology_prob"] = float(freeze_8a["climatology_prob_2010_2024"])
    out_df["majority_pred"] = int(freeze_8a["majority_class_2010_2024"])
    out_df["trend_pred"] = (feat_df["ret_19"].to_numpy() > 0).astype(int)
    out_df["resnet_pooled_prob"] = np.concatenate(prob_chunks, axis=0)
    out_df["resnet_pooled_pred"] = np.concatenate(pred_chunks, axis=0)

    # Fixed Phase 2/3 models
    p_lc, d_lc = predict_logreg_json(MODELS / "logreg_closes.json", full_df)
    out_df["logreg_prob"], out_df["logreg_pred"] = p_lc, d_lc
    p_lf, d_lf = predict_logreg_json(MODELS / "logreg_features.json", full_df)
    out_df["logreg_features_prob"], out_df["logreg_features_pred"] = p_lf, d_lf
    b_p = lgb.Booster(model_file=str(MODELS / "lightgbm_seed42.txt"))
    p_lgb = b_p.predict(full_df[PRIMARY_FEATURES + ASSET_ONEHOT])
    out_df["lightgbm_prob"], out_df["lightgbm_pred"] = p_lgb, (p_lgb > 0.5).astype(int)

    # WF2026 frozen models
    p_lc26, d_lc26 = predict_logreg_json(MODELS / "logreg_closes_wf2026.json", full_df)
    out_df["logreg_closes_wf2026_prob"], out_df["logreg_closes_wf2026_pred"] = p_lc26, d_lc26
    p_lf26, d_lf26 = predict_logreg_json(MODELS / "logreg_features_wf2026.json", full_df)
    out_df["logreg_features_wf2026_prob"], out_df["logreg_features_wf2026_pred"] = p_lf26, d_lf26
    b_26 = lgb.Booster(model_file=str(MODELS / "lightgbm_wf2026.txt"))
    p_lgb26 = b_26.predict(full_df[PRIMARY_FEATURES + ASSET_ONEHOT])
    out_df["lightgbm_wf2026_prob"], out_df["lightgbm_wf2026_pred"] = p_lgb26, (p_lgb26 > 0.5).astype(int)
    p_eh26, d_eh26 = predict_logreg_json(MODELS / "resnet_embedding_head_wf2026.json", embeddings)
    out_df["resnet_embedding_head_wf2026_prob"], out_df["resnet_embedding_head_wf2026_pred"] = p_eh26, d_eh26

    out_df.to_csv(out_preds, index=False, float_format="%.12g", lineterminator="\n")

    settings = protocol["stages"]["stage_1_probabilistic_evaluation"]["uncertainty"]
    model_specs = {
        "majority": {"prob": "climatology_prob", "pred": "majority_pred", "hard": False, "label": "Majority / climatology"},
        "trend_20d": {"prob": "trend_pred", "pred": "trend_pred", "hard": True, "label": "20-day trend"},
        "resnet_pooled": {"prob": "resnet_pooled_prob", "pred": "resnet_pooled_pred", "hard": False, "label": "Pooled ResNet-18"},
        "logreg_closes": {"prob": "logreg_prob", "pred": "logreg_pred", "hard": False, "label": "LR (closes, 2010–21)"},
        "logreg_features": {"prob": "logreg_features_prob", "pred": "logreg_features_pred", "hard": False, "label": "LR (features, 2010–21)"},
        "lightgbm": {"prob": "lightgbm_prob", "pred": "lightgbm_pred", "hard": False, "label": "LightGBM (2010–21)"},
        "logreg_closes_wf2026": {"prob": "logreg_closes_wf2026_prob", "pred": "logreg_closes_wf2026_pred", "hard": False, "label": "LR (closes, WF2026)"},
        "logreg_features_wf2026": {"prob": "logreg_features_wf2026_prob", "pred": "logreg_features_wf2026_pred", "hard": False, "label": "LR (features, WF2026)"},
        "lightgbm_wf2026": {"prob": "lightgbm_wf2026_prob", "pred": "lightgbm_wf2026_pred", "hard": False, "label": "LightGBM (WF2026)"},
        "resnet_embedding_head_wf2026": {"prob": "resnet_embedding_head_wf2026_prob", "pred": "resnet_embedding_head_wf2026_pred", "hard": False, "label": "ResNet-18 emb + LR (WF2026)"},
    }
    comparisons = [
        ("resnet_pooled", "logreg_closes_wf2026"),
        ("lightgbm_wf2026", "majority"),
        ("logreg_features_wf2026", "logreg_closes_wf2026"),
    ]
    eval_res = evaluate_predictions(
        out_df, model_specs, float(freeze_8a["climatology_prob_2010_2024"]), settings, ("overall", *TICKERS), comparisons, bootstrap=True
    )

    # Pre-registered 4-test confirmatory family with Holm at alpha = 0.05
    h1 = auc_vs_half(out_df, model_specs["lightgbm_wf2026"], settings=settings)
    h2 = auc_vs_half(out_df, model_specs["resnet_pooled"], settings=settings)
    h3 = auc_vs_half(out_df, model_specs["logreg_closes_wf2026"], settings=settings)
    h4_row = next(
        r for r in eval_res["comparisons"]["overall"]
        if r["a"] == "resnet_pooled" and r["b"] == "logreg_closes_wf2026" and r["metric"] == "roc_auc"
    )
    confirmatory = [
        {"id": "H1", "hypothesis": "AUC(lightgbm_wf2026) != 0.5", "estimate": h1["roc_auc"], "diff": h1["difference"], "ci": h1["ci"], "p_boot": h1["p_boot"]},
        {"id": "H2", "hypothesis": "AUC(resnet_pooled) != 0.5", "estimate": h2["roc_auc"], "diff": h2["difference"], "ci": h2["ci"], "p_boot": h2["p_boot"]},
        {"id": "H3", "hypothesis": "AUC(logreg_closes_wf2026) != 0.5", "estimate": h3["roc_auc"], "diff": h3["difference"], "ci": h3["ci"], "p_boot": h3["p_boot"]},
        {"id": "H4", "hypothesis": "AUC(resnet_pooled) - AUC(logreg_closes_wf2026) != 0", "estimate": h4_row["difference"], "diff": h4_row["difference"], "ci": h4_row["ci"], "p_boot": h4_row["p_boot"]},
    ]
    adj = holm([h["p_boot"] for h in confirmatory])
    for h, p_adj in zip(confirmatory, adj):
        h["p_holm"] = p_adj
        h["significant_alpha_0_05"] = bool(p_adj < 0.05)

    pre_reg_sha = args.pre_reg_commit
    if len(pre_reg_sha.strip()) < 7:
        raise ValueError("--pre-reg-commit must be a valid commit SHA")

    meta = {
        "stage": "P1 Stage 8 confirmatory 2026 holdout",
        "pre_registration_commit": pre_reg_sha,
        "window": "2026-01-02 to 2026-11-25",
        "rows_total": int(len(out_df)),
        "sessions_per_asset": int(len(out_df) // len(TICKERS)),
        "predictions_sha256": sha256(out_preds),
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Backtest on 2026 Adj Close returns (reusing load_adj_returns without manifest check, checking prev_session)
    returns_by_ticker, _, raw_hashes = load_adj_returns(
        TICKERS, ROOT / "data/raw_holdout_2026", expected_hashes=None, manifest_path=None
    )
    for row in meta_df.itertuples(index=False):
        prev_raw, _ = returns_by_ticker[row.ticker][row.label_date]
        if prev_raw != row.prev_session:
            raise ValueError(f"2026 session alignment mismatch for {row.ticker} {row.label_date}")

    bt_models = [
        "buy_and_hold",
        "trend_20d",
        "resnet_pooled",
        "logreg_closes",
        "logreg_features",
        "lightgbm",
        "logreg_closes_wf2026",
        "logreg_features_wf2026",
        "lightgbm_wf2026",
        "resnet_embedding_head_wf2026",
    ]
    bt_res, eq_2026 = run_backtest(out_df, bt_models, returns_by_ticker, None, settings, compute_bootstrap=True)
    bt_payload = {
        "title": "P1 Stage 8 — 2026 fresh holdout cost-aware backtest",
        "study_status": "Confirmatory 2026 holdout (secondary economic evaluation on frozen models).",
        "primary_split": "holdout_2026",
        "raw_sha256": raw_hashes,
        "bootstrap": {k: settings[k] for k in ("block_length_sessions", "replicates", "seed", "interval")},
        "primary": bt_res,
        "validation_context": None,
    }
    Path(f"{out_bt_prefix}_backtest.json").write_text(json.dumps(bt_payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    plot_equity_curves(eq_2026, bt_res["models"], out_eq, "P1 Stage 8 — Fresh 2026 holdout equity (5 bp, delay 0)")
    plot_cost_sensitivity(bt_res, bt_res["models"], P1 / "stage8_cost_sensitivity.png", "P1 Stage 8 — 2026 holdout Sharpe vs. cost")
    write_backtest_markdown(
        bt_payload,
        Path(f"{out_bt_prefix}_backtest.md"),
        "P1 Stage 8 — 2026 fresh holdout cost-aware backtest",
        equity_img=out_eq.name,
        cost_img="stage8_cost_sensitivity.png",
    )

    payload = {
        "stage": "P1 Stage 8 confirmatory 2026 holdout",
        "pre_registration_commit": pre_reg_sha,
        "confirmatory_family": confirmatory,
        "evaluation": eval_res,
        "backtest_5bp_delay0": {m: bt_res["portfolio"][m]["delay_0"]["5bp"] for m in bt_res["portfolio"]},
        "backtest_bootstrap_5bp_delay0": bt_res["bootstrap_5bp_delay0"],
    }
    out_json.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    write_metrics_markdown(
        eval_res,
        out_md,
        "P1 Stage 8 — Confirmatory 2026 holdout evaluation",
        intro_lines=[
            f"> **Confirmatory out-of-time evaluation** (`2026-01-02` to `2026-11-25`, `{len(out_df):,}` samples across 5 ETFs). All models and hypotheses were frozen and pushed prior to download (pre-registration commit `{pre_reg_sha}`)."
        ],
        image_name=out_eq.name,
    )
    conf_lines = [
        "",
        "## Confirmatory hypothesis family (Holm-Bonferroni corrected, α = 0.05)",
        "",
        "| ID | Hypothesis | Estimate / Diff | 95% CI (diff) | Raw p | Holm p | Significant? |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for h in confirmatory:
        conf_lines.append(
            f"| {h['id']} | `{h['hypothesis']}` | {h['estimate']:.4f} | "
            f"[{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}] | {h['p_boot']:.4g} | {h['p_holm']:.4g} | "
            f"{'Yes' if h['significant_alpha_0_05'] else 'No'} |"
        )
    conf_lines.extend(
        [
            "",
            "## Secondary economic evaluation (5 bp one-way, delay 0; details in `stage8_backtest.md`)",
            "",
            "| Model | CAGR | Sharpe [95% CI] | Max DD | Turnover/yr | Exposure |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for m, perf in payload["backtest_5bp_delay0"].items():
        b = bt_res["bootstrap_5bp_delay0"].get(m, {})
        ci_s = f"[{b['sharpe_ci'][0]:.3f}, {b['sharpe_ci'][1]:.3f}]" if "sharpe_ci" in b else ""
        conf_lines.append(
            f"| {bt_res['models'][m]} | {100*perf['cagr']:.2f}% | {perf['sharpe']:.3f} {ci_s} | "
            f"{100*perf['max_drawdown']:.2f}% | {perf['annual_turnover']:.1f}x | {100*perf['exposure']:.2f}% |"
        )
    with out_md.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(conf_lines) + "\n")
    print("Stage 8 evaluation complete.")


if __name__ == "__main__":
    main()

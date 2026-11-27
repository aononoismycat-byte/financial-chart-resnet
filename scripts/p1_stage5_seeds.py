"""P1 Stage 5: multi-seed robustness for ResNet-18 head and LightGBM + Holm correction."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import ChartDataset
from src.evaluation import auc_vs_half, evaluate_predictions
from src.features import ASSET_ONEHOT, PRIMARY_FEATURES
from src.metrics import accuracy, balanced_accuracy, holm, log_loss, roc_auc
from src.model import ResNet18Classifier
from src.training import load_checkpoint, set_seed
from scripts.p1_build_features import TICKERS, sha256
from scripts.p1_train_tabular import fit_lightgbm
from scripts.train_spy_resnet import NormalizedDataset

SEEDS = [42, 0, 1, 2, 3, 4]
GROUPS = ("overall", *TICKERS)
P1 = ROOT / "results/p1"
MANIFEST = ROOT / "data/processed/multi_asset_samples.csv"


def verify_logits_equivalence(emb: np.ndarray, keys_df: pd.DataFrame) -> float:
    """Gate: verify full ResNet-18 logits match fc(cached_embedding) on 64 val images."""
    summary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
    ckpt_path = ROOT / summary["freeze"]["pooled_checkpoint"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18Classifier(pretrained=False, training_mode="classifier_only").to(device)
    load_checkpoint(ckpt_path, model, device)
    model.eval()

    manifest = pd.read_csv(MANIFEST)
    val_rows = manifest[manifest["split"] == "val"].head(64).reset_index(drop=True)
    val_emb_idx = keys_df.index[(keys_df["split"] == "val")].to_numpy()[:64]
    if not (keys_df.loc[val_emb_idx, ["ticker", "label_date"]].reset_index(drop=True).equals(val_rows[["ticker", "label_date"]])):
        raise ValueError("Val alignment mismatch during Stage 5 equivalence gate")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_manifest = Path(tmp) / "val64.csv"
        val_rows.to_csv(tmp_manifest, index=False)
        loader = DataLoader(NormalizedDataset(ChartDataset(tmp_manifest, "val", project_root=ROOT)), batch_size=32, shuffle=False)
        full_chunks = []
        with torch.inference_mode():
            for images, _ in loader:
                full_chunks.append(model(images.to(device)).cpu())
    full_logits = torch.cat(full_chunks, dim=0)
    with torch.inference_mode():
        fc_logits = model.backbone.fc(torch.from_numpy(emb[val_emb_idx]).to(device)).cpu()
    max_diff = float((full_logits - fc_logits).abs().max().item())
    if max_diff >= 1e-4:
        raise ValueError(f"Stage 5 equivalence gate failed: max |logit diff| = {max_diff} >= 1e-4")
    return max_diff


def train_resnet_head_seed(emb_tr, y_tr, emb_va, y_va, emb_te, seed: int):
    set_seed(seed)
    head = nn.Linear(512, 2)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(emb_tr), torch.from_numpy(y_tr).long()),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    va_x = torch.from_numpy(emb_va)
    va_y = torch.from_numpy(y_va).long()
    te_x = torch.from_numpy(emb_te)

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, 11):
        head.train()
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(bx), by)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            val_loss = float(criterion(head(va_x), va_y).item())
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        p_val = torch.softmax(head(va_x), dim=1)[:, 1].numpy()
        p_test = torch.softmax(head(te_x), dim=1)[:, 1].numpy()
    return p_val, p_test, best_epoch, best_loss


def plot_seed_spread(per_seed, ens_aucs, refs, output):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    rng = np.random.default_rng(7)
    families = [("resnet", "ResNet-18 head (6 seeds)", "#d62728"), ("lightgbm", "LightGBM (6 seeds)", "#2ca02c")]
    for x_pos, (fam, label, color) in enumerate(families):
        aucs = [r["roc_auc"] for r in per_seed[fam]]
        jitter = rng.uniform(-0.08, 0.08, size=len(aucs))
        ax.scatter(x_pos + jitter, aucs, color=color, s=55, alpha=0.85, zorder=3, label=label)
        for j, r in zip(jitter, per_seed[fam]):
            ax.annotate(str(r["seed"]), (x_pos + j, r["roc_auc"]), textcoords="offset points", xytext=(5, 2), fontsize=7)
        ax.scatter([x_pos], [ens_aucs[fam]], marker="D", s=90, color="black", zorder=4, label=f"{fam} ensemble" if x_pos == 0 else None)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.0, label="Random (0.500)")
    ax.axhline(refs["logreg_closes"], color="#1f77b4", linestyle="--", linewidth=1.2, label=f"LR closes ({refs['logreg_closes']:.3f})")
    ax.axhline(refs["logreg_features"], color="#ff7f0e", linestyle="-.", linewidth=1.2, label=f"LR features ({refs['logreg_features']:.3f})")
    ax.set(xticks=[0, 1], xticklabels=["ResNet-18 head", "LightGBM"], xlim=(-0.5, 1.5), ylabel="Held-out test ROC-AUC (2024–2025)",
           title="P1 Stage 5 — Multi-seed ROC-AUC spread (seeds 42, 0, 1, 2, 3, 4)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_preds = P1 / "stage5_seed_predictions.csv"
    out_meta = P1 / "stage5_seed_predictions_meta.json"
    out_json = P1 / "stage5_seeds.json"
    out_md = P1 / "stage5_seeds.md"
    out_plot = P1 / "stage5_seed_spread.png"
    if not args.overwrite and any(p.exists() for p in (out_preds, out_meta, out_json, out_md, out_plot)):
        raise FileExistsError("Stage 5 outputs exist; pass --overwrite to replace")

    protocol = json.loads((ROOT / "experiments/p1_protocol.json").read_text(encoding="utf-8"))
    if not any(a.get("stage") == "stage_5_robustness" for a in protocol.get("amendments", [])):
        raise ValueError("Stage 5 amendment must be recorded in experiments/p1_protocol.json before running")

    feat_meta = json.loads((P1 / "stage3_features_meta.json").read_text(encoding="utf-8"))
    feat_csv = ROOT / feat_meta["features_csv"]
    if sha256(feat_csv) != feat_meta["features_sha256"]:
        raise ValueError("Feature CSV hash mismatch")
    df = pd.read_csv(feat_csv)

    emb_meta = json.loads((P1 / "stage4_embeddings_meta.json").read_text(encoding="utf-8"))
    emb_npy = ROOT / emb_meta["embeddings_npy"]
    keys_csv = ROOT / emb_meta["keys_csv"]
    if sha256(emb_npy) != emb_meta["embeddings_sha256"] or sha256(keys_csv) != emb_meta["keys_sha256"]:
        raise ValueError("Embeddings hash mismatch")
    emb = np.load(emb_npy)
    emb_keys = pd.read_csv(keys_csv)
    pd.testing.assert_frame_equal(df[["ticker", "label_date", "split", "label"]], emb_keys, check_dtype=False)

    max_logit_diff = verify_logits_equivalence(emb, emb_keys)

    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "val").to_numpy()
    te = (df["split"] == "test").to_numpy()
    vt = (df["split"].isin(["val", "test"])).to_numpy()
    y_tr = df.loc[tr, "label"].to_numpy(dtype=int)
    y_va = df.loc[va, "label"].to_numpy(dtype=int)
    y_te = df.loc[te, "label"].to_numpy(dtype=int)

    primary_cols = PRIMARY_FEATURES + ASSET_ONEHOT
    x_tr_f, x_va_f, x_te_f = df.loc[tr, primary_cols], df.loc[va, primary_cols], df.loc[te, primary_cols]

    seed_df = df.loc[vt, ["split", "ticker", "label_date", "label"]].copy().reset_index(drop=True)
    va_sub = (seed_df["split"] == "val").to_numpy()
    te_sub = (seed_df["split"] == "test").to_numpy()

    seed_logs = {"resnet": [], "lightgbm": []}
    per_seed_test = {"resnet": [], "lightgbm": []}
    rn_val_probs, rn_test_probs = [], []
    lgb_val_probs, lgb_test_probs = [], []

    for seed in SEEDS:
        pv, pt, best_ep, best_vl = train_resnet_head_seed(emb[tr], y_tr, emb[va], y_va, emb[te], seed)
        col_r = f"resnet_seed{seed}_prob"
        seed_df.loc[va_sub, col_r] = pv
        seed_df.loc[te_sub, col_r] = pt
        rn_val_probs.append(pv)
        rn_test_probs.append(pt)
        seed_logs["resnet"].append({"seed": seed, "best_epoch": best_ep, "best_val_loss": best_vl})
        per_seed_test["resnet"].append(
            {
                "seed": seed,
                "accuracy": accuracy(y_te, (pt > 0.5).astype(int)),
                "balanced_accuracy": balanced_accuracy(y_te, (pt > 0.5).astype(int)),
                "roc_auc": roc_auc(y_te, pt),
                "log_loss": log_loss(y_te, pt),
            }
        )

        lgbm = fit_lightgbm(x_tr_f, y_tr, x_va_f, y_va, seed=seed)
        lv = lgbm.predict_proba(x_va_f)[:, 1]
        lt = lgbm.predict_proba(x_te_f)[:, 1]
        col_l = f"lightgbm_seed{seed}_prob"
        seed_df.loc[va_sub, col_l] = lv
        seed_df.loc[te_sub, col_l] = lt
        lgb_val_probs.append(lv)
        lgb_test_probs.append(lt)
        seed_logs["lightgbm"].append(
            {
                "seed": seed,
                "best_iteration": int(lgbm.best_iteration_),
                "best_val_logloss": float(lgbm.best_score_["valid_0"]["binary_logloss"]),
            }
        )
        per_seed_test["lightgbm"].append(
            {
                "seed": seed,
                "accuracy": accuracy(y_te, (lt > 0.5).astype(int)),
                "balanced_accuracy": balanced_accuracy(y_te, (lt > 0.5).astype(int)),
                "roc_auc": roc_auc(y_te, lt),
                "log_loss": log_loss(y_te, lt),
            }
        )

    seed_df.loc[va_sub, "resnet_ens_prob"] = np.mean(rn_val_probs, axis=0)
    seed_df.loc[te_sub, "resnet_ens_prob"] = np.mean(rn_test_probs, axis=0)
    seed_df["resnet_ens_pred"] = (seed_df["resnet_ens_prob"] > 0.5).astype(int)

    seed_df.loc[va_sub, "lightgbm_ens_prob"] = np.mean(lgb_val_probs, axis=0)
    seed_df.loc[te_sub, "lightgbm_ens_prob"] = np.mean(lgb_test_probs, axis=0)
    seed_df["lightgbm_ens_pred"] = (seed_df["lightgbm_ens_prob"] > 0.5).astype(int)

    seed_df.to_csv(out_preds, index=False, float_format="%.12g", lineterminator="\n")
    meta = {
        "stage": "P1 Stage 5 multi-seed robustness (post-hoc exploratory)",
        "seeds": SEEDS,
        "logits_equivalence_max_abs_diff": max_logit_diff,
        "seed_training": seed_logs,
        "stage5_seed_predictions_sha256": sha256(out_preds),
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Summary stats per family
    family_summary = {}
    for fam in ("resnet", "lightgbm"):
        family_summary[fam] = {}
        for metric in ("accuracy", "balanced_accuracy", "roc_auc", "log_loss"):
            vals = np.array([r[metric] for r in per_seed_test[fam]], dtype=float)
            family_summary[fam][metric] = {
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }

    # Pre-specified 10-test family on ensemble probabilities + Holm correction
    s1 = pd.read_csv(P1 / "predictions.csv")
    s3 = pd.read_csv(P1 / "stage3_predictions.csv")
    merged = (
        s1.merge(s3, on=["split", "ticker", "label_date", "label"], how="inner", validate="one_to_one")
        .merge(seed_df, on=["split", "ticker", "label_date", "label"], how="inner", validate="one_to_one")
    )
    test_df = merged[merged["split"] == "test"].reset_index(drop=True)
    s1_meta = json.loads((P1 / "predictions_meta.json").read_text(encoding="utf-8"))
    settings = protocol["stages"]["stage_1_probabilistic_evaluation"]["uncertainty"]
    ref_prob = float(s1_meta["climatology_probability"])

    model_specs = {
        "majority": {"prob": "climatology_prob", "pred": "majority_pred", "hard": False, "label": "Majority / climatology"},
        "logreg_closes": {"prob": "logreg_prob", "pred": "logreg_pred", "hard": False, "label": "Logistic regression (closes)"},
        "logreg_features": {"prob": "logreg_features_prob", "pred": "logreg_features_pred", "hard": False, "label": "Logistic regression (features)"},
        "resnet_ens": {"prob": "resnet_ens_prob", "pred": "resnet_ens_pred", "hard": False, "label": "ResNet-18 head (6-seed ens)"},
        "lightgbm_ens": {"prob": "lightgbm_ens_prob", "pred": "lightgbm_ens_pred", "hard": False, "label": "LightGBM (6-seed ens)"},
    }
    comparisons = [
        ("lightgbm_ens", "majority"),
        ("lightgbm_ens", "logreg_closes"),
        ("resnet_ens", "logreg_closes"),
        ("resnet_ens", "lightgbm_ens"),
        ("logreg_features", "logreg_closes"),
    ]
    eval_res = evaluate_predictions(test_df, model_specs, ref_prob, settings, ("overall",), comparisons, bootstrap=True)
    family_tests = [row for row in eval_res["comparisons"]["overall"] if row["metric"] in ("roc_auc", "log_loss")]
    adj = holm([row["p_boot"] for row in family_tests])
    for row, p_adj in zip(family_tests, adj):
        row["p_holm"] = p_adj

    ens_auc_vs_half = {
        "resnet_ens": auc_vs_half(test_df, model_specs["resnet_ens"], settings=settings),
        "lightgbm_ens": auc_vs_half(test_df, model_specs["lightgbm_ens"], settings=settings),
    }

    payload = {
        "stage": "P1 Stage 5 multi-seed robustness (post-hoc exploratory)",
        "logits_equivalence_max_abs_diff": max_logit_diff,
        "per_seed": per_seed_test,
        "family_summary": family_summary,
        "ensemble_eval": eval_res,
        "ensemble_auc_vs_half": ens_auc_vs_half,
        "holm_family_10_tests": family_tests,
    }
    out_json.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    refs = {
        "logreg_closes": eval_res["groups"]["overall"]["logreg_closes"]["metrics"]["roc_auc"],
        "logreg_features": eval_res["groups"]["overall"]["logreg_features"]["metrics"]["roc_auc"],
    }
    ens_aucs = {
        "resnet": eval_res["groups"]["overall"]["resnet_ens"]["metrics"]["roc_auc"],
        "lightgbm": eval_res["groups"]["overall"]["lightgbm_ens"]["metrics"]["roc_auc"],
    }
    plot_seed_spread(per_seed_test, ens_aucs, refs, out_plot)

    lines = [
        "# P1 Stage 5 — Multi-seed robustness and Holm-corrected comparisons",
        "",
        f"> Post-hoc exploratory analysis over 6 seeds (`{SEEDS}`). Per protocol amendment, ResNet-18 classifier heads are retrained on cached frozen-backbone embeddings (equivalence gate on 64 val images: `max |logit diff| = {max_logit_diff:.2e} < 1e-4`).",
        "",
        "## Across-seed distribution (2024–2025 held-out test, 2,510 samples)",
        "",
        "| Family | Metric | Mean ± SD | Min | Max | Ensemble |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for fam, label in (("resnet", "ResNet-18 head"), ("lightgbm", "LightGBM (features)")):
        ens_m = eval_res["groups"]["overall"][f"{fam}_ens"]["metrics"]
        for metric, fmt in (("accuracy", True), ("balanced_accuracy", True), ("roc_auc", False), ("log_loss", False)):
            st = family_summary[fam][metric]
            if fmt:
                lines.append(
                    f"| {label} | {metric} | {100*st['mean']:.2f}% ± {100*st['sd']:.2f}% | "
                    f"{100*st['min']:.2f}% | {100*st['max']:.2f}% | {100*ens_m[metric]:.2f}% |"
                )
            else:
                lines.append(
                    f"| {label} | {metric} | {st['mean']:.4f} ± {st['sd']:.4f} | "
                    f"{st['min']:.4f} | {st['max']:.4f} | {ens_m[metric]:.4f} |"
                )

    lines.extend(
        [
            "",
            "## Ensemble AUC vs. 0.50 (date-block bootstrap)",
            "",
            "| Ensemble | ROC-AUC [95% CI] | Bootstrap p (vs 0.5) |",
            "|---|---|---:|",
        ]
    )
    for key, label in (("resnet_ens", "ResNet-18 head (6-seed ens)"), ("lightgbm_ens", "LightGBM (6-seed ens)")):
        h = ens_auc_vs_half[key]
        lines.append(f"| {label} | {h['roc_auc']:.4f} [{h['auc_ci'][0]:.4f}, {h['auc_ci'][1]:.4f}] | {h['p_boot']:.3g} |")

    lines.extend(
        [
            "",
            "## Pre-specified 10-test family with Holm step-down correction",
            "",
            "| Comparison (A − B) | Metric | Difference | 95% CI | Raw p | Holm adjusted p |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for row in family_tests:
        name = f"{model_specs[row['a']]['label']} − {model_specs[row['b']]['label']}"
        lines.append(
            f"| {name} | {row['metric']} | {row['difference']:+.4f} | "
            f"[{row['ci'][0]:+.4f}, {row['ci'][1]:+.4f}] | {row['p_boot']:.4g} | {row['p_holm']:.4g} |"
        )
    lines.extend(["", "![Seed spread](stage5_seed_spread.png)", ""])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("Stage 5 complete.")


if __name__ == "__main__":
    main()

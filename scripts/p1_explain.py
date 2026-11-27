"""P1 Stage 6: Grad-CAM on pooled ResNet-18 and native TreeSHAP on LightGBM."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from src.dataset import ChartDataset
from src.explain import gradcam, line_lift, right_quarter_share
from src.features import ASSET_ONEHOT, PRIMARY_FEATURES
from src.model import ResNet18Classifier
from src.training import load_checkpoint
from scripts.p1_build_features import sha256
from scripts.train_spy_resnet import NormalizedDataset

P1 = ROOT / "results/p1"
MANIFEST = ROOT / "data/processed/multi_asset_samples.csv"


def plot_gradcam_examples(records, cams, raw_images, output):
    """3x3 grid: top-3 highest P(Up), bottom-3 lowest P(Up), 3 closest to 0.5."""
    probs = records["resnet_pooled_prob"].to_numpy()
    order_desc = np.argsort(-probs, kind="stable")
    order_asc = np.argsort(probs, kind="stable")
    order_mid = np.argsort(np.abs(probs - 0.5), kind="stable")

    selected = list(order_desc[:3]) + list(order_asc[:3]) + list(order_mid[:3])
    row_labels = ["Highest P(Up)"] * 3 + ["Lowest P(Up)"] * 3 + ["Closest to 0.5"] * 3

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.2))
    for ax, idx, r_lbl in zip(axes.flat, selected, row_labels):
        row = records.iloc[idx]
        ax.imshow(raw_images[idx])
        ax.imshow(cams[idx], cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
        ax.axvline(168, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(
            f"{r_lbl}\n{row['ticker']} {row['label_date']} | P={row['resnet_pooled_prob']:.3f} y={int(row['label'])}",
            fontsize=8.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("P1 Stage 6 — Grad-CAM (layer4, target class = Up) on 2024–2025 test charts", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_gradcam_mean(cams, preds, output):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, cls, title in ((axes[0], 0, "Mean CAM: Predicted Non-up"), (axes[1], 1, "Mean CAM: Predicted Up")):
        mask = preds == cls
        mean_cam = cams[mask].mean(axis=0)
        im = ax.imshow(mean_cam, cmap="jet", vmin=0.0, vmax=max(0.4, float(mean_cam.max())))
        ax.axvline(168, color="white", linestyle="--", linewidth=1.2, label="x = 168 (right quarter)")
        ax.set(title=f"{title} (n={int(mask.sum())})", xlabel="Chart x pixel (day 0 → day 19)", ylabel="Chart y pixel")
        ax.legend(fontsize=7.5, loc="upper left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    x_pixels = np.arange(224)
    for cls, label, color in ((0, "Predicted Non-up", "#1f77b4"), (1, "Predicted Up", "#d62728")):
        profile = cams[preds == cls].mean(axis=(0, 1))
        axes[2].plot(x_pixels, profile, color=color, linewidth=1.7, label=label)
    axes[2].axvline(168, color="black", linestyle="--", linewidth=1.0, label="Right quarter (x ≥ 168)")
    axes[2].set(xlabel="Chart x pixel (left = t-19, right = t)", ylabel="Mean Grad-CAM intensity", title="Column-attention profile")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_shap_bar(feature_names, mean_abs, output, top_k=15):
    order = np.argsort(mean_abs)[::-1][:top_k]
    names = [feature_names[i] for i in order][::-1]
    vals = mean_abs[order][::-1]
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.barh(range(len(names)), vals, color="#2ca02c", alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set(xlabel="Mean |TreeSHAP value| (log-odds)", title="P1 Stage 6 — LightGBM global feature importance (test 2024–2025)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_shap_beeswarm(feature_names, x_mat, shap_vals, order, output, top_k=10):
    top_idx = order[:top_k][::-1]
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    sm = None
    for y_pos, idx in enumerate(top_idx):
        sv = shap_vals[:, idx]
        fv = x_mat[:, idx]
        pct = rankdata(fv, method="average") / len(fv)
        jitter = rng.uniform(-0.22, 0.22, size=len(sv))
        sm = ax.scatter(sv, np.full_like(sv, y_pos, dtype=float) + jitter, c=pct, cmap="coolwarm", s=8, alpha=0.65, vmin=0, vmax=1)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
    ax.set(xlabel="TreeSHAP contribution to P(Up) log-odds", title="P1 Stage 6 — LightGBM TreeSHAP beeswarm (top 10 features)")
    ax.grid(axis="x", alpha=0.25)
    if sm is not None:
        cb = fig.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label("Feature value percentile (low → high)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_shap_dependence(feature_names, x_mat, shap_vals, order, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, idx in zip(axes, order[:2]):
        ax.scatter(x_mat[:, idx], shap_vals[:, idx], s=10, alpha=0.5, color="#1f77b4")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.set(xlabel=feature_names[idx], ylabel="TreeSHAP value (log-odds)", title=f"Dependence: {feature_names[idx]}")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = [
        P1 / "stage6_gradcam_examples.png",
        P1 / "stage6_gradcam_mean.png",
        P1 / "stage6_shap_bar.png",
        P1 / "stage6_shap_beeswarm.png",
        P1 / "stage6_shap_dependence.png",
        P1 / "stage6_explainability.json",
        P1 / "stage6_explainability.md",
    ]
    if not args.overwrite and any(p.exists() for p in outputs):
        raise FileExistsError("Stage 6 outputs exist; pass --overwrite to replace")

    protocol = json.loads((ROOT / "experiments/p1_protocol.json").read_text(encoding="utf-8"))
    if not any(a.get("stage") == "stage_6_explainability_governance" for a in protocol.get("amendments", [])):
        raise ValueError("Stage 6 native TreeSHAP amendment must be recorded before running")

    summary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
    freeze = summary["freeze"]
    if sha256(MANIFEST) != freeze["samples_sha256"]:
        raise ValueError("Manifest hash mismatch")
    ckpt_path = ROOT / freeze["pooled_checkpoint"]
    if sha256(ckpt_path) != freeze["pooled_checkpoint_sha256"]:
        raise ValueError("Pooled checkpoint hash mismatch")

    s1 = pd.read_csv(P1 / "predictions.csv")
    test_s1 = s1[s1["split"] == "test"].reset_index(drop=True)
    manifest = pd.read_csv(MANIFEST)
    test_manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)
    test_records = test_manifest.merge(
        test_s1, on=["split", "ticker", "label_date", "label"], how="inner", validate="one_to_one"
    )
    if len(test_records) != len(test_s1):
        raise ValueError("Test manifest / Stage 1 predictions alignment mismatch")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18Classifier(pretrained=False, training_mode="classifier_only").to(device)
    load_checkpoint(ckpt_path, model, device)
    model.eval()

    with tempfile.TemporaryDirectory() as tmp:
        m_path = Path(tmp) / "test_manifest.csv"
        test_records.to_csv(m_path, index=False)
        loader = DataLoader(NormalizedDataset(ChartDataset(m_path, "test", project_root=ROOT)), batch_size=32, shuffle=False)
        cam_chunks, prob_chunks, pred_chunks = [], [], []
        for images, _ in loader:
            imgs = images.to(device)
            with torch.inference_mode():
                logits = model(imgs)
                prob_chunks.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                pred_chunks.append(logits.argmax(dim=1).cpu().numpy())
            cam_chunks.append(gradcam(model, imgs, target_class=1))

    cams = np.concatenate(cam_chunks, axis=0)
    check_probs = np.concatenate(prob_chunks, axis=0)
    check_preds = np.concatenate(pred_chunks, axis=0)
    if not (check_preds == test_records["resnet_pooled_pred"].to_numpy(dtype=int)).all():
        raise ValueError("Grad-CAM pass predictions mismatch Stage 1 predictions")
    max_p_diff = float(np.max(np.abs(check_probs - test_records["resnet_pooled_prob"].to_numpy())))

    raw_images = []
    lifts = []
    for i, img_rel in enumerate(test_records["image_path"]):
        with Image.open(ROOT / img_rel) as im:
            arr = np.array(im, dtype=np.uint8, copy=True)
        raw_images.append(arr)
        lifts.append(line_lift(cams[i], arr))
    lifts_arr = np.asarray(lifts, dtype=float)
    rq_arr = right_quarter_share(cams)

    gradcam_stats = {
        "n_test": int(len(test_records)),
        "max_prob_diff_vs_stage1": max_p_diff,
        "uniform_right_quarter_baseline": 0.25,
        "overall": {
            "mean_right_quarter_share": float(rq_arr.mean()),
            "sd_right_quarter_share": float(rq_arr.std(ddof=1)),
            "mean_line_lift": float(lifts_arr.mean()),
            "sd_line_lift": float(lifts_arr.std(ddof=1)),
        },
        "by_predicted_class": {},
    }
    for cls, name in ((0, "predicted_non_up"), (1, "predicted_up")):
        mask = check_preds == cls
        gradcam_stats["by_predicted_class"][name] = {
            "n": int(mask.sum()),
            "mean_right_quarter_share": float(rq_arr[mask].mean()),
            "mean_line_lift": float(lifts_arr[mask].mean()),
        }

    plot_gradcam_examples(test_records, cams, raw_images, outputs[0])
    plot_gradcam_mean(cams, check_preds, outputs[1])

    # Native TreeSHAP on LightGBM Stage 3 model
    feat_meta = json.loads((P1 / "stage3_features_meta.json").read_text(encoding="utf-8"))
    feat_csv = ROOT / feat_meta["features_csv"]
    if sha256(feat_csv) != feat_meta["features_sha256"]:
        raise ValueError("Feature CSV hash mismatch")
    feat_df = pd.read_csv(feat_csv)
    test_feat = feat_df[feat_df["split"] == "test"].reset_index(drop=True)
    feature_names = PRIMARY_FEATURES + ASSET_ONEHOT
    x_test = test_feat[feature_names].to_numpy(dtype=float)

    booster = lgb.Booster(model_file=str(P1 / "models/lightgbm_seed42.txt"))
    contrib = booster.predict(x_test, pred_contrib=True)
    raw_scores = booster.predict(x_test, raw_score=True)
    additivity_err = float(np.max(np.abs(contrib.sum(axis=1) - raw_scores)))
    if additivity_err >= 1e-6:
        raise ValueError(f"TreeSHAP additivity gate failed: max error {additivity_err} >= 1e-6")

    shap_vals = contrib[:, :-1]
    base_value = float(contrib[0, -1])
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(-mean_abs, kind="stable")

    plot_shap_bar(feature_names, mean_abs, outputs[2])
    plot_shap_beeswarm(feature_names, x_test, shap_vals, order, outputs[3])
    plot_shap_dependence(feature_names, x_test, shap_vals, order, outputs[4])

    ranking = [
        {"rank": int(r + 1), "feature": feature_names[idx], "mean_abs_shap": float(mean_abs[idx])}
        for r, idx in enumerate(order)
    ]
    payload = {
        "stage": "P1 Stage 6 explainability (post-hoc exploratory)",
        "gradcam": gradcam_stats,
        "treeshap": {
            "model": "results/p1/models/lightgbm_seed42.txt",
            "n_test": int(len(test_feat)),
            "expected_value_log_odds": base_value,
            "additivity_max_abs_error": additivity_err,
            "ranking": ranking,
        },
    }
    outputs[5].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P1 Stage 6 — Explainability: Grad-CAM (ResNet-18) & TreeSHAP (LightGBM)",
        "",
        "## Grad-CAM on pooled ResNet-18 (`layer4`, 2,510 test charts)",
        "",
        "| Subset | n | Mean right-quarter share (vs 0.250 uniform) | Mean line lift |",
        "|---|---:|---:|---:|",
        f"| All test charts | {gradcam_stats['n_test']} | {gradcam_stats['overall']['mean_right_quarter_share']:.4f} | {gradcam_stats['overall']['mean_line_lift']:.3f}x |",
        f"| Predicted Non-up | {gradcam_stats['by_predicted_class']['predicted_non_up']['n']} | "
        f"{gradcam_stats['by_predicted_class']['predicted_non_up']['mean_right_quarter_share']:.4f} | "
        f"{gradcam_stats['by_predicted_class']['predicted_non_up']['mean_line_lift']:.3f}x |",
        f"| Predicted Up | {gradcam_stats['by_predicted_class']['predicted_up']['n']} | "
        f"{gradcam_stats['by_predicted_class']['predicted_up']['mean_right_quarter_share']:.4f} | "
        f"{gradcam_stats['by_predicted_class']['predicted_up']['mean_line_lift']:.3f}x |",
        "",
        "## Native TreeSHAP on LightGBM (`lightgbm_seed42.txt`, 2,510 test samples)",
        "",
        f"- **Additivity verification gate**: `max |sum(SHAP) + bias - raw_score| = {additivity_err:.2e} < 1e-6`.",
        f"- **Base value (log-odds)**: `{base_value:.4f}`.",
        "",
        "| Rank | Feature | Mean |SHAP| (log-odds) |",
        "|---:|---|---:|",
    ]
    for item in ranking[:10]:
        lines.append(f"| {item['rank']} | `{item['feature']}` | {item['mean_abs_shap']:.5f} |")
    lines.extend(
        [
            "",
            "![Grad-CAM examples](stage6_gradcam_examples.png)",
            "",
            "![Grad-CAM mean](stage6_gradcam_mean.png)",
            "",
            "![TreeSHAP importance](stage6_shap_bar.png)",
            "",
            "![TreeSHAP beeswarm](stage6_shap_beeswarm.png)",
            "",
            "![TreeSHAP dependence](stage6_shap_dependence.png)",
            "",
        ]
    )
    outputs[6].write_text("\n".join(lines), encoding="utf-8")
    print("Stage 6 complete.")


if __name__ == "__main__":
    main()

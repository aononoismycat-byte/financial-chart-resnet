"""P1 Stage 8a: train and freeze walk-forward 2026 models BEFORE downloading 2026 data.

Trains on label years 2010-2024 and uses 2025 for LightGBM early stopping only.
Saves portable model artifacts to results/p1/models/*_wf2026.(txt|json) and records
SHA-256 hashes in results/p1/stage8_freeze_meta.json.
"""
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

from src.features import ASSET_ONEHOT, PRIMARY_FEATURES, predict_logreg_json, save_logreg_json
from scripts.p1_build_features import NORM_COLS, sha256
from scripts.p1_train_tabular import fit_lightgbm, fit_logreg
from scripts.p1_walk_forward import fit_embedding_head

SEED = 42
P1 = ROOT / "results/p1"
MODELS = P1 / "models"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_files = [
        MODELS / "lightgbm_wf2026.txt",
        MODELS / "logreg_features_wf2026.json",
        MODELS / "logreg_closes_wf2026.json",
        MODELS / "resnet_embedding_head_wf2026.json",
        P1 / "stage8_freeze_meta.json",
    ]
    if not args.overwrite and any(p.exists() for p in out_files):
        raise FileExistsError("Stage 8a frozen models exist; pass --overwrite to replace")

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

    years = pd.to_datetime(df["label_date"]).dt.year.to_numpy()
    if (years > 2025).any():
        raise ValueError("Unexpected post-2025 rows in pre-2026 feature dataset")
    tr = (years >= 2010) & (years <= 2024)
    va = years == 2025
    y_tr = df.loc[tr, "label"].to_numpy(dtype=int)
    y_va = df.loc[va, "label"].to_numpy(dtype=int)

    primary_cols = PRIMARY_FEATURES + ASSET_ONEHOT
    emb_cols = [f"emb_{i:03d}" for i in range(emb.shape[1])]

    lr_closes = fit_logreg(df.loc[tr, NORM_COLS], y_tr)
    lr_features = fit_logreg(df.loc[tr, primary_cols], y_tr)
    lgbm = fit_lightgbm(df.loc[tr, primary_cols], y_tr, df.loc[va, primary_cols], y_va, seed=SEED)
    emb_head, emb_conv = fit_embedding_head(emb[tr], y_tr)

    MODELS.mkdir(parents=True, exist_ok=True)
    lgb_iter = int(lgbm.best_iteration_)
    lgbm.booster_.save_model(str(MODELS / "lightgbm_wf2026.txt"), num_iteration=lgb_iter)
    save_logreg_json(lr_features, primary_cols, MODELS / "logreg_features_wf2026.json")
    save_logreg_json(lr_closes, NORM_COLS, MODELS / "logreg_closes_wf2026.json")
    save_logreg_json(emb_head, emb_cols, MODELS / "resnet_embedding_head_wf2026.json")

    # Verify serialized models reproduce 2025 predictions identically.
    b = lgb.Booster(model_file=str(MODELS / "lightgbm_wf2026.txt"))
    if float(np.max(np.abs(b.predict(df.loc[va, primary_cols]) - lgbm.predict_proba(df.loc[va, primary_cols])[:, 1]))) > 1e-12:
        raise ValueError("Serialized lightgbm_wf2026 mismatch")
    p_lf, _ = predict_logreg_json(MODELS / "logreg_features_wf2026.json", df.loc[va])
    if float(np.max(np.abs(p_lf - lr_features.predict_proba(df.loc[va, primary_cols])[:, 1]))) > 1e-12:
        raise ValueError("Serialized logreg_features_wf2026 mismatch")
    p_lc, _ = predict_logreg_json(MODELS / "logreg_closes_wf2026.json", df.loc[va])
    if float(np.max(np.abs(p_lc - lr_closes.predict_proba(df.loc[va, NORM_COLS])[:, 1]))) > 1e-12:
        raise ValueError("Serialized logreg_closes_wf2026 mismatch")
    p_eh, _ = predict_logreg_json(MODELS / "resnet_embedding_head_wf2026.json", emb[va])
    if float(np.max(np.abs(p_eh - emb_head.predict_proba(emb[va].astype(np.float64))[:, 1]))) > 1e-12:
        raise ValueError("Serialized resnet_embedding_head_wf2026 mismatch")

    majority = int(np.count_nonzero(y_tr == 1) > np.count_nonzero(y_tr == 0))
    climatology = float(y_tr.mean())
    meta = {
        "stage": "P1 Stage 8a pre-registration model freeze (2010-2024 train, 2025 early-stopping val)",
        "n_train_2010_2024": int(tr.sum()),
        "n_val_2025": int(va.sum()),
        "majority_class_2010_2024": majority,
        "climatology_prob_2010_2024": climatology,
        "lightgbm_wf2026_best_iteration": lgb_iter,
        "lightgbm_wf2026_val_logloss": float(lgbm.best_score_["valid_0"]["binary_logloss"]),
        "resnet_embedding_head_wf2026_converged": emb_conv,
        "frozen_artifacts_sha256": {
            "lightgbm_wf2026.txt": sha256(MODELS / "lightgbm_wf2026.txt"),
            "logreg_features_wf2026.json": sha256(MODELS / "logreg_features_wf2026.json"),
            "logreg_closes_wf2026.json": sha256(MODELS / "logreg_closes_wf2026.json"),
            "resnet_embedding_head_wf2026.json": sha256(MODELS / "resnet_embedding_head_wf2026.json"),
        },
    }
    (P1 / "stage8_freeze_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("Stage 8a freeze complete:", meta["frozen_artifacts_sha256"])


if __name__ == "__main__":
    main()

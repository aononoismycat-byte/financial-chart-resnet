"""P1 Stage 4: extract frozen ImageNet ResNet-18 512-d embeddings for all samples."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import ChartDataset
from src.model import ResNet18Classifier
from src.training import load_checkpoint
from scripts.train_spy_resnet import NormalizedDataset

IMAGENET_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
IMAGENET_PATH = ROOT / "data/model_weights/torch_hub/checkpoints/resnet18-f37072fd.pth"
MANIFEST = ROOT / "data/processed/multi_asset_samples.csv"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-npy", type=Path, default=ROOT / "data/processed/p1_embeddings.npy")
    parser.add_argument("--output-keys", type=Path, default=ROOT / "data/processed/p1_embeddings_keys.csv")
    parser.add_argument("--output-meta", type=Path, default=ROOT / "results/p1/stage4_embeddings_meta.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.overwrite and any(p.exists() for p in (args.output_npy, args.output_keys, args.output_meta)):
        raise FileExistsError("Stage 4 embedding outputs exist; pass --overwrite to replace")

    summary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
    freeze = summary["freeze"]
    if sha256(MANIFEST) != freeze["samples_sha256"]:
        raise ValueError("Manifest hash mismatch")
    if sha256(IMAGENET_PATH) != IMAGENET_SHA256:
        raise ValueError("ImageNet pretrained weights hash mismatch")
    ckpt_path = ROOT / freeze["pooled_checkpoint"]
    if sha256(ckpt_path) != freeze["pooled_checkpoint_sha256"]:
        raise ValueError("Pooled checkpoint hash mismatch")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18Classifier(pretrained=False, training_mode="classifier_only").to(device)
    load_checkpoint(ckpt_path, model, device)

    # Gate 1: every backbone parameter and buffer except fc must equal ImageNet weights.
    imagenet_state = torch.load(IMAGENET_PATH, map_location="cpu", weights_only=True)
    model_state = model.state_dict()
    checked_tensors = 0
    for key, ref_tensor in imagenet_state.items():
        if key.startswith("fc."):
            continue
        full_key = f"backbone.{key}"
        if full_key not in model_state:
            raise KeyError(f"Missing backbone key {full_key}")
        if not torch.equal(model_state[full_key].cpu(), ref_tensor.cpu()):
            raise ValueError(f"Gate failed: backbone tensor {full_key} differs from ImageNet pretrained weights")
        checked_tensors += 1

    fc_weight = model.backbone.fc.weight.detach().cpu().clone()
    fc_bias = model.backbone.fc.bias.detach().cpu().clone()
    model.backbone.fc = nn.Identity()
    model.eval()

    manifest = pd.read_csv(MANIFEST)
    embeddings = np.empty((len(manifest), 512), dtype=np.float32)
    for split in ("train", "val", "test"):
        split_idx = manifest.index[manifest["split"] == split].to_numpy()
        loader = DataLoader(
            NormalizedDataset(ChartDataset(MANIFEST, split, project_root=ROOT)),
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )
        chunks = []
        with torch.inference_mode():
            for images, _ in loader:
                emb = model(images.to(device)).cpu().numpy().astype(np.float32)
                chunks.append(emb)
        embeddings[split_idx] = np.concatenate(chunks, axis=0)

    # Gate 2: original fc(embeddings) must reproduce Stage 1 resnet_pooled_pred and prob (< 1e-4).
    with torch.inference_mode():
        logits = torch.from_numpy(embeddings) @ fc_weight.T + fc_bias
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = logits.argmax(dim=1).numpy()

    keys = manifest[["ticker", "label_date", "split", "label"]].copy()
    keys["emb_prob"] = probs
    keys["emb_pred"] = preds
    s1 = pd.read_csv(ROOT / "results/p1/predictions.csv")
    merged = s1.merge(keys, on=["split", "ticker", "label_date", "label"], how="inner", validate="one_to_one")
    if len(merged) != len(s1):
        raise ValueError("Val/test row count mismatch against Stage 1 predictions.csv")
    if not (merged["emb_pred"].to_numpy(dtype=int) == merged["resnet_pooled_pred"].to_numpy(dtype=int)).all():
        raise ValueError("Gate failed: fc(embeddings) predictions do not exactly match resnet_pooled_pred")
    max_prob_diff = float(np.max(np.abs(merged["emb_prob"].to_numpy() - merged["resnet_pooled_prob"].to_numpy())))
    if max_prob_diff >= 1e-4:
        raise ValueError(f"Gate failed: max |prob diff| = {max_prob_diff} >= 1e-4")

    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, embeddings)
    manifest[["ticker", "label_date", "split", "label"]].to_csv(args.output_keys, index=False, lineterminator="\n")

    meta = {
        "stage": "P1 Stage 4 frozen-backbone embedding extraction",
        "manifest_sha256": freeze["samples_sha256"],
        "pooled_checkpoint_sha256": freeze["pooled_checkpoint_sha256"],
        "imagenet_weights_sha256": IMAGENET_SHA256,
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "device": str(device),
        "embeddings_npy": args.output_npy.relative_to(ROOT).as_posix(),
        "embeddings_sha256": sha256(args.output_npy),
        "keys_csv": args.output_keys.relative_to(ROOT).as_posix(),
        "keys_sha256": sha256(args.output_keys),
        "gates": {
            "imagenet_backbone_tensors_checked": checked_tensors,
            "imagenet_backbone_exact_match": True,
            "fc_reproduction_exact_pred_match": True,
            "fc_reproduction_max_abs_prob_diff": max_prob_diff,
        },
    }
    args.output_meta.parent.mkdir(parents=True, exist_ok=True)
    args.output_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"Gates passed ({checked_tensors} backbone tensors exact; max |prob diff|={max_prob_diff:.2e}). "
        f"Saved {args.output_npy}"
    )


if __name__ == "__main__":
    main()

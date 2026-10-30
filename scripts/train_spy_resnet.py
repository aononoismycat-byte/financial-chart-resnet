"""One fixed D2 SPY experiment; no test-set model evaluation or tuning."""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize
from src.dataset import ChartDataset
from src.model import ResNet18Classifier
from src.training import load_checkpoint, save_checkpoint, set_seed, train_one_epoch, validate
from scripts.evaluate_spy_baselines import calculate_metrics

SEED = 42
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
WEIGHTS = ROOT / "data/model_weights/torch_hub/checkpoints/resnet18-f37072fd.pth"
SAMPLES = ROOT / "data/processed/spy_samples.csv"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NormalizedDataset(Dataset):
    def __init__(self, base):
        self.base = base
        self.normalize = Normalize(MEAN, STD)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return self.normalize(image), label


def write_train_val_manifest(destination):
    """Copy only train/val rows; count test rows without using their labels."""
    counts = {"train": 0, "val": 0, "test": 0}
    dates = {"train": ("2010-01-01", "2021-12-31"), "val": ("2022-01-01", "2023-12-31")}
    with SAMPLES.open(encoding="utf-8", newline="") as source, destination.open("x", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            split = row["split"]
            if split not in counts:
                raise ValueError(f"unexpected split: {split}")
            counts[split] += 1
            if split == "test":
                continue
            first, last = dates[split]
            if (row["ticker"] != "SPY" or not first <= row["label_date"] <= last
                    or row["input_end_date"] >= row["label_date"]):
                raise ValueError(f"invalid {split} row: {row['label_date']}")
            path = (ROOT / row["image_path"]).resolve()
            if not path.is_relative_to(ROOT / "data/images/SPY" / split) or not path.is_file():
                raise ValueError(f"missing or misplaced image: {path}")
            writer.writerow(row)
    if counts != {"train": 3021, "val": 501, "test": 502}:
        raise ValueError(f"unexpected sample counts: {counts}")
    for split, count in counts.items():
        if len(list((ROOT / "data/images/SPY" / split).glob("*.png"))) != count:
            raise ValueError(f"image count mismatch for {split}")
    return counts


def plot_history(history, output, metric, title="SPY ResNet-18"):
    os.environ["MPLCONFIGDIR"] = str(output.parent / "matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [row[f"train_{metric}"] for row in history], marker="o", label="Train")
    ax.plot(epochs, [row[f"val_{metric}"] for row in history], marker="o", label="Validation")
    ax.set(xlabel="Epoch", ylabel=metric.capitalize(), title=f"{title}: {metric}")
    ax.set_xticks(epochs)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "training.log"
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        def report(message):
            print(message, flush=True)
            print(message, file=log, flush=True)
        try:
            report(f"Experiment directory: {output}")
            report(f"Started UTC: {datetime.now(timezone.utc).isoformat()}")
            if not WEIGHTS.is_file():
                raise FileNotFoundError(f"Pretrained weights not cached: {WEIGHTS}; no download attempted")
            weight_hash = sha256(WEIGHTS)
            if not weight_hash.startswith("f37072fd"):
                raise ValueError(f"Pretrained weight hash mismatch: {weight_hash}")
            summary = json.loads((ROOT / "data/processed/spy_summary.json").read_text(encoding="utf-8"))
            raw_hash = sha256(ROOT / "data/raw/SPY.csv")
            if raw_hash != summary["raw_sha256"]:
                raise ValueError("Raw SPY CSV differs from stage C checksum")
            manifest = output / "train_val_samples.csv"
            counts = write_train_val_manifest(manifest)
            set_seed(SEED)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else None
            torch.hub.set_dir(str(ROOT / "data/model_weights/torch_hub"))
            train_data = NormalizedDataset(ChartDataset(manifest, "train"))
            val_data = NormalizedDataset(ChartDataset(manifest, "val"))
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0,
                                      generator=torch.Generator().manual_seed(SEED))
            val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0,
                                    generator=torch.Generator().manual_seed(SEED))
            probe_images, probe_labels = next(iter(val_loader))
            if (tuple(probe_images.shape) != (32, 3, 224, 224) or probe_images.dtype != torch.float32
                    or tuple(probe_labels.shape) != (32,) or probe_labels.dtype != torch.long):
                raise ValueError("DataLoader contract check failed")
            # Move to the target device before building the optimizer.
            model = ResNet18Classifier(pretrained=True, training_mode="classifier_only").to(device)
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            config = {"ticker": "SPY", "seed": SEED, "architecture": "resnet18", "pretrained": True,
                      "pretrained_weight": str(WEIGHTS), "pretrained_sha256": weight_hash,
                      "training_mode": "classifier_only", "optimizer": "Adam", "learning_rate": 0.001,
                      "batch_size": 32, "epochs": 10, "loss": "CrossEntropyLoss", "device": str(device),
                      "gpu": gpu, "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                      "train_shuffle": True, "val_shuffle": False, "num_workers": 0,
                      "preprocessing": {"input": "RGB float32 [0,1]", "mean": MEAN, "std": STD,
                                        "size": [224, 224], "resize": False, "crop": False, "augmentation": False},
                      "dataset": {"samples_csv": str(SAMPLES), "samples_sha256": sha256(SAMPLES),
                                  "raw_csv": str(ROOT / "data/raw/SPY.csv"), "raw_sha256": raw_hash,
                                  "price_column": "Close", "label_dates": {"train": "2010-2021",
                                                                              "val": "2022-2023",
                                                                              "test_unused": "2024-2025"},
                                  "counts": counts, "train_val_manifest_sha256": sha256(manifest)}}
            (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            report(f"Device: {device}; GPU: {gpu}; counts: {counts}")
            report(f"Input batch: {tuple(probe_images.shape)} {probe_images.dtype}; labels: {tuple(probe_labels.shape)} {probe_labels.dtype}")
            report(f"Configuration: pretrained=True, classifier_only, Adam lr=0.001, batch=32, epochs=10, seed={SEED}")
            history = []
            best_loss, best_epoch, best_result = float("inf"), None, None
            checkpoint = output / "best_model.pt"
            for epoch in range(1, 11):
                started = time.perf_counter()
                train = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val = validate(model, val_loader, criterion, device)
                if not all(math.isfinite(v) for v in (train["loss"], train["accuracy"], val["loss"], val["accuracy"])):
                    raise RuntimeError(f"Non-finite training/validation metric at epoch {epoch}")
                row = {"epoch": epoch, "train_loss": train["loss"], "train_accuracy": train["accuracy"],
                       "val_loss": val["loss"], "val_accuracy": val["accuracy"]}
                history.append(row)
                improved = val["loss"] < best_loss
                if improved:
                    best_loss, best_epoch, best_result = val["loss"], epoch, val
                    save_checkpoint(checkpoint, model, seed=SEED,
                        training_parameters={**config, "best_epoch": best_epoch, "best_val_loss": best_loss,
                                             "best_val_accuracy": val["accuracy"]}, history=history)
                (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                report(f"Epoch {epoch:02d}/10 train_loss={train['loss']:.6f} train_acc={train['accuracy']:.6f} "
                       f"val_loss={val['loss']:.6f} val_acc={val['accuracy']:.6f} "
                       f"best={'yes' if improved else 'no'} seconds={time.perf_counter()-started:.2f}")
            # Instantiate without downloading weights: the complete trained state comes from the checkpoint.
            restored = ResNet18Classifier(pretrained=False, training_mode="classifier_only").to(device)
            metadata = load_checkpoint(checkpoint, restored, device=device)
            reloaded = validate(restored, val_loader, criterion, device)
            if (metadata["training_parameters"]["best_epoch"] != best_epoch
                    or not math.isclose(reloaded["loss"], best_loss, rel_tol=1e-6, abs_tol=1e-7)
                    or reloaded["accuracy"] != best_result["accuracy"]
                    or reloaded["labels"] != best_result["labels"]
                    or reloaded["predictions"] != best_result["predictions"]):
                raise RuntimeError("Reloaded checkpoint does not reproduce the saved best validation result")
            metrics = calculate_metrics(reloaded["labels"], reloaded["predictions"])
            result = {"best_epoch": best_epoch, "best_val_loss": best_loss,
                      "reloaded_val_loss": reloaded["loss"], "reload_verified": True,
                      "validation_metrics": metrics, "checkpoint_bytes": checkpoint.stat().st_size}
            (output / "validation_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            plot_history(history, output / "loss_curve.png", "loss")
            plot_history(history, output / "accuracy_curve.png", "accuracy")
            report(f"Best epoch: {best_epoch}; val_loss={best_loss:.9f}; reload_val_loss={reloaded['loss']:.9f}")
            report(f"Validation metrics: {json.dumps(metrics)}")
            report(f"Checkpoint: {checkpoint} ({checkpoint.stat().st_size} bytes)")
            report("Reload verified: loss, accuracy, labels and predictions match; test set not evaluated.")
            report(f"Finished UTC: {datetime.now(timezone.utc).isoformat()}")
        except Exception:
            report("FAILED: " + traceback.format_exc())
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new experiment directory; must not exist")
    args = parser.parse_args()
    run(args.output.resolve())

"""Check/download torchvision pretrained weights in a project-local cache."""
import hashlib
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torchvision.models import ResNet18_Weights
from src.model import ResNet18Classifier


def main():
    # Process-local override only; never change global network or proxy settings.
    cache = ROOT / "data" / "model_weights" / "torch_hub"
    torch.hub.set_dir(str(cache))
    path = cache / "checkpoints" / "resnet18-f37072fd.pth"
    print(f"Local weight cache: {path}", flush=True)
    print(f"Before loading, cached: {path.is_file()}", flush=True)
    print(f"Weight source: {ResNet18_Weights.DEFAULT.url}", flush=True)
    socket.setdefaulttimeout(30)
    model = ResNet18Classifier(pretrained=True, training_mode="classifier_only")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith("f37072fd"):
        raise ValueError(f"pretrained weight SHA-256 mismatch: {digest}")
    print(f"Loaded: {model.configuration()}")
    print(f"SHA-256: {digest}")
    print("Classifier shape:", tuple(model.backbone.fc.weight.shape))
    print("Recommended transforms:", ResNet18_Weights.DEFAULT.transforms())


if __name__ == "__main__":
    main()

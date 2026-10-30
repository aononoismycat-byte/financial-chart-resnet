"""Small training helpers for image/label batches, without dataset splitting.

Dataset items must be (float32 image [3, 224, 224], integer label 0 or 1).
DataLoader batches must contain images [B, 3, 224, 224] and long labels [B].
Split financial samples elsewhere by label date before creating loaders.
"""

import random
from pathlib import Path

import torch
from torch import nn


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch; this does not guarantee cross-device identity."""
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device=None):
    return torch.device(device if device is not None else
                        ("cuda" if torch.cuda.is_available() else "cpu"))


def _check_criterion(criterion):
    # This contract makes batch-size weighting an exact per-sample mean.
    if (not isinstance(criterion, nn.CrossEntropyLoss)
            or criterion.reduction != "mean" or criterion.weight is not None
            or criterion.ignore_index != -100):
        raise ValueError("Use unweighted CrossEntropyLoss with reduction='mean' and default ignore_index")


def _run_epoch(model, loader, criterion, device, optimizer=None):
    _check_criterion(criterion)
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    labels, predictions = [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if targets.dtype != torch.long or targets.ndim != 1:
                raise ValueError("batch labels must have dtype torch.long and shape [B]")
            if not torch.all((targets == 0) | (targets == 1)):
                raise ValueError("batch labels must be 0 or 1")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = targets.size(0)
            predicted = logits.argmax(dim=1)
            total_loss += loss.item() * batch_size
            total_correct += (predicted == targets).sum().item()
            total_count += batch_size
            if not training:
                labels.extend(targets.cpu().tolist())
                predictions.extend(predicted.cpu().tolist())
    if total_count == 0:
        raise ValueError("loader contains no samples")
    result = {"loss": total_loss / total_count, "accuracy": total_correct / total_count}
    if not training:
        result.update(labels=labels, predictions=predictions)
    return result


def train_one_epoch(model, loader, optimizer, criterion, device=None):
    device = resolve_device(device)
    model.to(device)
    # Moving a model between device types can replace Parameter objects. An
    # optimizer made before that move may silently update the old parameters.
    model_parameters = {id(parameter) for parameter in model.parameters()}
    if any(id(parameter) not in model_parameters
           for group in optimizer.param_groups for parameter in group["params"]):
        raise ValueError("Move the model to the target device before creating the optimizer")
    return _run_epoch(model, loader, criterion, device, optimizer)


def validate(model, loader, criterion, device=None):
    device = resolve_device(device)
    model.to(device)
    return _run_epoch(model, loader, criterion, device)


def fit(model, train_loader, validation_loader, optimizer, criterion, epochs,
        device=None, seed=42):
    """Train and return per-epoch metrics.

    Call set_seed(seed) before creating the model and loaders as well: setting
    it here cannot reproduce initialization that has already happened. Loaders
    using their own generators must have those generators seeded separately.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    set_seed(seed)
    device = resolve_device(device)
    history = []
    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, validation_loader, criterion, device)
        history.append({"epoch": epoch + 1, "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                        "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"]})
    return history


def _safe_metadata(value):
    """Convert paths to strings and reject metadata unsafe for weights_only."""
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("checkpoint metadata dictionary keys must be strings")
        return {key: _safe_metadata(item) for key, item in value.items()}
    raise TypeError(f"unsupported checkpoint metadata type: {type(value).__name__}")


def save_checkpoint(path, model, *, seed, training_parameters, history=None):
    """Save model state and safe metadata; caller supplies the initialization seed.

    Metadata supports primitive values, lists, tuples, string-keyed dictionaries
    and paths (saved as strings). This is not a resumable optimizer checkpoint.
    """
    torch.save({"model_state_dict": model.state_dict(),
                "model_config": _safe_metadata(model.configuration()),
                "seed": _safe_metadata(seed),
                "training_parameters": _safe_metadata(dict(training_parameters)),
                "history": _safe_metadata(history if history is not None else [])}, Path(path))


def load_checkpoint(path, model, device=None):
    """Load into an explicitly constructed compatible model and return metadata."""
    device = resolve_device(device)
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    config = checkpoint["model_config"]
    actual = model.configuration()
    # Pretrained initialization is irrelevant once all saved weights are loaded.
    for key in ("architecture", "num_classes", "training_mode"):
        if config[key] != actual[key]:
            raise ValueError(f"checkpoint {key} does not match the supplied model")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return {key: checkpoint[key] for key in
            ("model_config", "seed", "training_parameters", "history")}

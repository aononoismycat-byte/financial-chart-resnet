"""Synthetic smoke tests only; no financial data or prediction claims."""

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import ResNet18Classifier
from src.training import (fit, load_checkpoint, save_checkpoint, set_seed,
                          train_one_epoch, validate)


class TrainingSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep this small smoke test practical on machines with many CPU cores.
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def exercise_device(self, device):
        set_seed(42)
        model = ResNet18Classifier(pretrained=False, training_mode="classifier_only").to(device)
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.05)
        criterion = nn.CrossEntropyLoss()
        images = torch.randn(2, 3, 224, 224, device=device)
        labels = torch.tensor([0, 1], dtype=torch.long, device=device)
        frozen_before = {name: p.detach().clone() for name, p in model.named_parameters()
                         if not p.requires_grad}
        bn_before = {name: b.detach().clone() for name, b in model.named_buffers()
                     if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}
        head_before = model.backbone.fc.weight.detach().clone()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (2, 2))
        loss = criterion(logits, labels)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(any(torch.isfinite(g).all().item() and torch.count_nonzero(g).item() > 0
                            for g in grads))
        optimizer.step()
        self.assertFalse(torch.equal(head_before, model.backbone.fc.weight))
        for name, parameter in model.named_parameters():
            if name in frozen_before:
                self.assertTrue(torch.equal(frozen_before[name], parameter), name)
        for name, buffer in model.named_buffers():
            if name in bn_before:
                self.assertTrue(torch.equal(bn_before[name], buffer), name)
        self.assertTrue(bn_before)
        print(f"{device}: logits={tuple(logits.shape)}, finite loss/gradient, backward, "
              "parameter update, frozen backbone and BatchNorm checks passed", flush=True)
        return model, optimizer, criterion

    def test_cpu_training_validation_and_checkpoint(self):
        device = torch.device("cpu")
        model, optimizer, criterion = self.exercise_device(device)
        generator = torch.Generator().manual_seed(7)
        images = torch.randn(3, 3, 224, 224, generator=generator)
        targets = torch.tensor([0, 1, 0], dtype=torch.long)
        loader = DataLoader(TensorDataset(images, targets), batch_size=2, shuffle=False)
        bn_before = {name: b.detach().clone() for name, b in model.named_buffers()
                     if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}
        metrics = train_one_epoch(model, loader, optimizer, criterion, device)
        self.assertTrue(math.isfinite(metrics["loss"]))
        result = validate(model, loader, criterion, device)
        self.assertEqual(result["labels"], targets.tolist())
        self.assertEqual(len(result["predictions"]), len(targets))
        self.assertTrue(set(result["predictions"]).issubset({0, 1}))
        self.assertTrue(math.isfinite(result["loss"]))
        self.assertEqual(result["accuracy"], sum(a == b for a, b in
                         zip(result["labels"], result["predictions"])) / len(targets))
        # Compare against a single batch to catch incorrect unweighted batch means.
        single_batch = DataLoader(TensorDataset(images, targets), batch_size=3)
        single_loss = validate(model, single_batch, criterion, device)["loss"]
        self.assertTrue(math.isclose(result["loss"], single_loss, rel_tol=1e-5, abs_tol=1e-6))
        history = fit(model, loader, loader, optimizer, criterion, epochs=1, device=device, seed=42)
        self.assertEqual(len(history), 1)
        self.assertTrue(all(math.isfinite(history[0][key]) for key in
                            ("train_loss", "train_accuracy", "val_loss", "val_accuracy")))
        for name, buffer in model.named_buffers():
            if name in bn_before:
                self.assertTrue(torch.equal(bn_before[name], buffer), name)
        saved_state = copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            parameters = {"epochs": 1, "optimizer": "SGD", "learning_rate": 0.05,
                          "input_path": Path("data")}
            save_checkpoint(path, model, seed=42, training_parameters=parameters, history=history)
            restored = ResNet18Classifier(pretrained=False, training_mode="classifier_only")
            metadata = load_checkpoint(path, restored, device=device)
            self.assertEqual(metadata["seed"], 42)
            self.assertEqual(metadata["training_parameters"], {**parameters, "input_path": str(Path("data"))})
            self.assertEqual(metadata["model_config"], model.configuration())
            self.assertEqual(metadata["history"], history)
            for name, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, saved_state[name]), name)
            with self.assertRaisesRegex(TypeError, "unsupported checkpoint metadata type"):
                save_checkpoint(path, model, seed=42, training_parameters={"invalid": object()})
            # A rejected save must leave the prior checkpoint intact.
            self.assertEqual(load_checkpoint(path, restored, device=device)["seed"], 42)
        print("CPU: train/validation metrics, unequal batches, checkpoint save/load passed", flush=True)

    def test_full_finetune_and_pretrained_failure(self):
        set_seed(42)
        model = ResNet18Classifier(pretrained=False, training_mode="full_finetune")
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        loader = DataLoader(TensorDataset(torch.randn(2, 3, 224, 224),
                                          torch.tensor([0, 1])), batch_size=2)
        before = model.backbone.conv1.weight.detach().clone()
        bn_before = model.backbone.bn1.num_batches_tracked.detach().clone()
        metrics = train_one_epoch(model, loader, optimizer, nn.CrossEntropyLoss(), "cpu")
        self.assertTrue(math.isfinite(metrics["loss"]))
        self.assertFalse(torch.equal(before, model.backbone.conv1.weight))
        self.assertGreater(model.backbone.bn1.num_batches_tracked.item(), bn_before.item())
        other = ResNet18Classifier(pretrained=False)
        wrong_optimizer = torch.optim.SGD(other.parameters(), lr=0.01)
        with self.assertRaisesRegex(ValueError, "before creating the optimizer"):
            train_one_epoch(model, loader, wrong_optimizer, nn.CrossEntropyLoss(), "cpu")
        with patch("src.model.resnet18", side_effect=OSError("simulated download error")):
            with self.assertRaisesRegex(RuntimeError, "Could not load pretrained") as caught:
                ResNet18Classifier(pretrained=True)
            self.assertIsInstance(caught.exception.__cause__, OSError)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable: GPU test not executed")
    def test_cuda_forward_backward_and_update(self):
        device = torch.device("cuda:0")
        print(f"GPU: {torch.cuda.get_device_name(device)}; device={device}", flush=True)
        model, optimizer, criterion = self.exercise_device(device)
        before = model.backbone.fc.weight.detach().clone()
        bn_before = model.backbone.bn1.running_mean.detach().clone()
        loader = DataLoader(TensorDataset(torch.randn(2, 3, 224, 224),
                                          torch.tensor([0, 1])), batch_size=2)
        metrics = train_one_epoch(model, loader, optimizer, criterion, device)
        self.assertTrue(math.isfinite(metrics["loss"]))
        self.assertFalse(torch.equal(before, model.backbone.fc.weight))
        self.assertTrue(torch.equal(bn_before, model.backbone.bn1.running_mean))
        self.assertEqual(len(validate(model, loader, criterion, device)["labels"]), 2)
        print("GPU: train_one_epoch and validate passed", flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

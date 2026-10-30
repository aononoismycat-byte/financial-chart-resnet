"""ResNet-18 image classifier; logits are suitable for CrossEntropyLoss."""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Classifier(nn.Module):
    """Two-class classifier with a frozen or trainable ResNet-18 backbone."""

    def __init__(self, pretrained: bool = False, training_mode: str = "classifier_only"):
        super().__init__()
        if training_mode not in {"classifier_only", "full_finetune"}:
            raise ValueError("training_mode must be 'classifier_only' or 'full_finetune'")
        self.pretrained = bool(pretrained)
        self.training_mode = training_mode
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        try:
            self.backbone = resnet18(weights=weights)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    "Could not load pretrained ResNet-18 weights; check the cache and network."
                ) from exc
            raise
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
        if training_mode == "classifier_only":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.backbone.fc.parameters():
                parameter.requires_grad = True
        self.train()

    def train(self, mode: bool = True):
        """Keep frozen backbone BatchNorm statistics fixed even during training."""
        super().train(mode)
        if self.training_mode == "classifier_only":
            self.backbone.eval()
            self.backbone.fc.train(mode)
        return self

    def forward(self, images):
        return self.backbone(images)

    def configuration(self):
        return {"architecture": "resnet18", "num_classes": 2,
                "pretrained": self.pretrained, "training_mode": self.training_mode}

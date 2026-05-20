"""
CNN classifier wrappers for brain tumor MRI classification.

Provides a unified interface for ResNet50, DenseNet121, and InceptionV3
backbones with a custom classification head suited for medical imaging.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


SUPPORTED_BACKBONES = ["resnet50", "densenet121", "inception_v3"]


class CNNClassifier(nn.Module):
    """
    Unified CNN classifier wrapper for medical image classification.

    Supports multiple backbones with a consistent classification head:
        - ResNet50
        - DenseNet121
        - InceptionV3

    Args:
        backbone: One of 'resnet50', 'densenet121', 'inception_v3'
        num_classes: Number of output classes
        pretrained: Load ImageNet pretrained weights
        dropout: Dropout rate in classification head
        freeze_backbone: Whether to freeze backbone initially
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"backbone must be one of {SUPPORTED_BACKBONES}, got {backbone}"
            )

        self.backbone_name = backbone
        self.num_classes = num_classes

        self.model, self._head_attr = self._build_backbone(
            backbone, pretrained, num_classes, dropout
        )

        if freeze_backbone:
            self.freeze_backbone()

    def _build_backbone(
        self,
        name: str,
        pretrained: bool,
        num_classes: int,
        dropout: float,
    ):
        """Build the chosen backbone and replace its classification head."""
        if name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = self._build_head(in_features, num_classes, dropout)
            return model, "fc"

        elif name == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.densenet121(weights=weights)
            in_features = model.classifier.in_features
            model.classifier = self._build_head(in_features, num_classes, dropout)
            return model, "classifier"

        elif name == "inception_v3":
            weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.inception_v3(weights=weights, aux_logits=True)
            in_features = model.fc.in_features
            model.fc = self._build_head(in_features, num_classes, dropout)
            # Auxiliary classifier head also needs updating
            aux_in = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(aux_in, num_classes)
            return model, "fc"

        raise ValueError(f"Unknown backbone: {name}")

    def _build_head(
        self, in_features: int, num_classes: int, dropout: float
    ) -> nn.Sequential:
        """Standard classification head: Linear -> ReLU -> Dropout -> Linear."""
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def freeze_backbone(self) -> None:
        """Freeze all parameters except the classification head."""
        for name, param in self.model.named_parameters():
            if not name.startswith(self._head_attr):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (B, 3, H, W)

        Returns:
            Logits of shape (B, num_classes).
            For InceptionV3 during training, returns main logits only
            (auxiliary logits accessible via model.AuxLogits during .train()).
        """
        output = self.model(x)
        if isinstance(output, tuple):
            return output[0]
        return output

    def count_parameters(self) -> dict:
        """Return total, trainable, and frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }


def build_model(
    backbone: str = "resnet50",
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
    device: Optional[str] = None,
) -> CNNClassifier:
    """
    Factory function to build a CNN classifier.

    Args:
        backbone: Backbone name
        num_classes: Number of output classes
        pretrained: Load ImageNet weights
        dropout: Head dropout rate
        freeze_backbone: Whether to freeze initially
        device: 'cuda' or 'cpu' (auto-detected if None)

    Returns:
        Configured CNNClassifier on the chosen device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNNClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
    return model.to(device)


if __name__ == "__main__":
    print(f"Supported backbones: {SUPPORTED_BACKBONES}")

    for bb in ["resnet50", "densenet121"]:
        print(f"\n{'=' * 50}")
        print(f"Backbone: {bb}")
        print('=' * 50)

        model = CNNClassifier(backbone=bb, num_classes=4, freeze_backbone=True)
        counts = model.count_parameters()
        print(f"  Total params:     {counts['total']:>12,}")
        print(f"  Trainable params: {counts['trainable']:>12,}")
        print(f"  Frozen params:    {counts['frozen']:>12,}")

        size = 224
        dummy = torch.randn(2, 3, size, size)
        out = model(dummy)
        print(f"  Output shape:     {out.shape}")

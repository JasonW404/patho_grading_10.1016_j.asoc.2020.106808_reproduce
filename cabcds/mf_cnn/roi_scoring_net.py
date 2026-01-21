"""AlexNet-based ROI scoring network."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import AlexNet_Weights, alexnet


class RoiScoringNet(nn.Module):
    """AlexNet-based multi-class ROI scorer."""

    def __init__(self, num_classes: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        base = alexnet(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool

        classifier_layers = list(base.classifier)
        classifier_layers[-1] = nn.Linear(classifier_layers[-1].in_features, num_classes)
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor.

        Returns:
            Logits tensor.
        """

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

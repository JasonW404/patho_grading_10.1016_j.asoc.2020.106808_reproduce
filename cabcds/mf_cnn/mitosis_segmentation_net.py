"""VGG16-based FCN for mitosis segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16


@dataclass(frozen=True)
class SegmentationOutput:
    """Segmentation network output.

    Attributes:
        logits: Raw segmentation logits.
        features: Intermediate encoder features for downstream use.
    """

    logits: torch.Tensor
    features: tuple[torch.Tensor, ...]


class MitosisSegmentationNet(nn.Module):
    """VGG16-based FCN with skip connections for mitosis blob segmentation."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        base = vgg16(weights=weights)
        self.encoder = base.features

        self.pool_indices = [4, 9, 16, 23, 30]

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        self.skip_projection = nn.ModuleList(
            [
                nn.Conv2d(64, 32, kernel_size=1),
                nn.Conv2d(128, 32, kernel_size=1),
                nn.Conv2d(256, 32, kernel_size=1),
                nn.Conv2d(512, 32, kernel_size=1),
            ]
        )

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        """Forward pass.

        Args:
            x: Input image tensor.

        Returns:
            SegmentationOutput containing logits and encoder features.
        """

        features: list[torch.Tensor] = []
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx in self.pool_indices:
                features.append(x)

        deepest = features[-1]
        decoded = self.decoder(deepest)

        skip_sources = features[:-1]
        for skip_tensor, projection in zip(reversed(skip_sources), self.skip_projection, strict=False):
            resized = nn.functional.interpolate(
                skip_tensor,
                size=decoded.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded = decoded + projection(resized)

        logits = self.classifier(decoded)
        return SegmentationOutput(logits=logits, features=tuple(features))

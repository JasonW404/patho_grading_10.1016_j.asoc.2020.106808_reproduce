"""Torchvision-based CNNs used in MF-CNN.

Notes
-----
The original paper uses:
- `CNN_seg`: VGG16-VD converted to FCN, with transposed-conv upsampling and skip connections.
- `CNN_det`: AlexNet binary classifier on resized 227x227 crops.
- `CNN_global`: AlexNet 3-class classifier on resized 227x227 ROI patches.

This module keeps the architecture lightweight and focused on model definitions.
"""

from __future__ import annotations

from typing import cast
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


@dataclass(frozen=True)
class SegmentationOutput:
    """Segmentation network output.

    Attributes:
        logits: Raw segmentation logits.
        features: Intermediate encoder features for downstream use.
    """

    logits: torch.Tensor
    features: tuple[torch.Tensor, ...]


class CNNSeg(nn.Module):
    """
    CNN_Seg: VGG16-based FCN with skip connections.
    
    1. generated cell count based useful handcrafted features for Hybrid-Descriptor. 
    2. produced 512 x 512 pixel patches with prospective mitoses for CNNGlobal. 
    3. produced a relatively balanced dataset of prospective mitotic patches (of size 80 × 80 pixels) for CNNDet by reducing the class imbalance.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()

        self.num_classes = int(num_classes)

        # Load pretrained VGG16 model
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)\
                      .features
        vgg16_feature = cast(nn.Sequential, vgg16)  # Type casting for IDEs

        # Encoder: "Blue" rectangles. From pretrained VGG16 (Fig. 8. CNN_seg)
        self.stage1 = vgg16_feature[:5]    # P1 Conv + ReLU (x2) + Pool => 256x256
        self.stage2 = vgg16_feature[5:10]  # P2 Conv + ReLU (x2) + Pool => 128x128
        self.stage3 = vgg16_feature[10:17] # P3 Conv + ReLU (x3) + Pool => 64x64
        self.stage4 = vgg16_feature[17:24] # P4 Conv + ReLU (x3) + Pool => 32x32
        self.stage5 = vgg16_feature[24:31] # P5 Conv + ReLU (x3) + Pool => 16x16

        # Bottleneck: FCN-style replacement of VGG's FC layers.
        # Classic FCN (and many VGG16-to-FCN conversions) replace:
        # - fc6 with a 7x7 conv
        # - fc7 with a 1x1 conv
        # followed by a per-pixel class score conv.
        # We keep spatial resolution by using padding=3 on the 7x7 conv.
        self.fc_block = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(4096, self.num_classes, kernel_size=1),
        )

        # Decoder: Up Sampling + Skip Connections
        
        self.conv_trans_1 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=4, stride=2, padding=1)  # 16 -> 32
        self.conv_trans_2 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=4, stride=2, padding=1)  # 32 -> 64
        self.conv_trans_3 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=4, stride=2, padding=1)  # 64 -> 128
        self.conv_trans_4 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=8, stride=4, padding=2)  # 128 -> 512
        
        self.upsample_1 = nn.Conv2d(512, self.num_classes, kernel_size=1)  # [P4 (F:512) -> + CONV-TRAN1]
        self.upsample_2 = nn.Conv2d(256, self.num_classes, kernel_size=1)  # [P3 (F:256) -> + CONV-TRAN2]
        self.upsample_3 = nn.Conv2d(128, self.num_classes, kernel_size=1)  # [P2 (F:128) -> + CONV-TRAN3]

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        def _resize_like(tensor: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            if tensor.shape[-2:] == ref.shape[-2:]:
                return tensor
            return F.interpolate(tensor, size=ref.shape[-2:], mode="bilinear", align_corners=False)

        # Encoder
        p1 = self.stage1(x)
        p2 = self.stage2(p1)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)

        # Bottleneck
        fc3 = self.fc_block(p5)  # 16x16

        # Decoder (fusing according to red arrows in the figure)
        skip1 = self.upsample_1(p4)
        out = _resize_like(self.conv_trans_1(fc3), skip1) + skip1  # SKIP 1

        skip2 = self.upsample_2(p3)
        out = _resize_like(self.conv_trans_2(out), skip2) + skip2  # SKIP 2

        skip3 = self.upsample_3(p2)
        out = _resize_like(self.conv_trans_3(out), skip3) + skip3  # SKIP 3
        
        logits = self.conv_trans_4(out)
        # Ensure logits match input size for loss computation and downstream mask extraction.
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # Return logits (for CrossEntropyLoss) and encoder features for downstream use.
        return SegmentationOutput(logits=logits, features=(p1, p2, p3, p4, p5))


class CNNSegLegacy(nn.Module):
    """Legacy CNN_seg variant.

    This matches an earlier implementation that used a lightweight 1x1-conv
    approximation of VGG's FC layers (i.e., kernel_size=1 throughout).

    Kept to load older checkpoints that are not compatible with the current
    FCN-like 7x7/1x1 head.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()

        self.num_classes = int(num_classes)

        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None).features
        vgg16_feature = cast(nn.Sequential, vgg16)

        self.stage1 = vgg16_feature[:5]
        self.stage2 = vgg16_feature[5:10]
        self.stage3 = vgg16_feature[10:17]
        self.stage4 = vgg16_feature[17:24]
        self.stage5 = vgg16_feature[24:31]

        self.fc_block = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, self.num_classes, kernel_size=1),
        )

        self.conv_trans_1 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=4, stride=2, padding=1)
        self.conv_trans_2 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=4, stride=2, padding=1)
        self.conv_trans_3 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=4, stride=2, padding=1)
        self.conv_trans_4 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=8, stride=4, padding=2)

        self.upsample_1 = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.upsample_2 = nn.Conv2d(256, self.num_classes, kernel_size=1)
        self.upsample_3 = nn.Conv2d(128, self.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        def _resize_like(tensor: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            if tensor.shape[-2:] == ref.shape[-2:]:
                return tensor
            return F.interpolate(tensor, size=ref.shape[-2:], mode="bilinear", align_corners=False)

        p1 = self.stage1(x)
        p2 = self.stage2(p1)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)

        fc3 = self.fc_block(p5)
        skip1 = self.upsample_1(p4)
        out = _resize_like(self.conv_trans_1(fc3), skip1) + skip1
        skip2 = self.upsample_2(p3)
        out = _resize_like(self.conv_trans_2(out), skip2) + skip2
        skip3 = self.upsample_3(p2)
        out = _resize_like(self.conv_trans_3(out), skip3) + skip3
        logits = self.conv_trans_4(out)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return SegmentationOutput(logits=logits, features=(p1, p2, p3, p4, p5))


class CNNDet(nn.Module):
    """
    CNN_Det: AlexNet-based CNN for mitosis detection.
    
    The input should be the 227x227 RGB patches, resized from 80x80 patches at the blobs detected by CNN_seg.
    """
   
    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        
        # Load pretrained AlexNet model
        alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = alexnet.features
        
        # Modify the classifier to match the number of classes
        self.classifier = alexnet.classifier
        self.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        
        # Flatten for fully connected layers (AlexNet features output is typically 6x6x256)
        x = torch.flatten(x, 1)
        
        logits = self.classifier(x)
        return logits


class CNNGlobal(CNNDet):
    """CNN_Global: AlexNet-based CNN for ROI scoring."""
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__(num_classes=3, pretrained=pretrained)
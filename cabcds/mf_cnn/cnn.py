"""The CNNs used in MF-CNN"""

from __future__ import annotations

from typing import cast
from dataclasses import dataclass

import torch
from torch import nn
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

        # Bottleneck: "Red" rectangles. (Fig. 8. CNN_seg)
        self.fc_block = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 2, kernel_size=1) # Corresponds to FC3 => 2 classes
        )

        # Decoder: Up Sampling + Skip Connections
        
        self.conv_trans_1 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)  # 16 -> 32
        self.conv_trans_2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)  # 32 -> 64
        self.conv_trans_3 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)  # 64 -> 128
        self.conv_trans_4 = nn.ConvTranspose2d(2, 2, kernel_size=8, stride=4, padding=2)  # 128 -> 512
        
        self.upsample_1 = nn.Conv2d(512, 2, kernel_size=1)  # [P4 (F:512) -> + CONV-TRAN1]
        self.upsample_2 = nn.Conv2d(256, 2, kernel_size=1)  # [P3 (F:256) -> + CONV-TRAN2]
        self.upsample_3 = nn.Conv2d(128, 2, kernel_size=1)  # [P2 (F:128) -> + CONV-TRAN3]

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder
        p1 = self.stage1(x)
        p2 = self.stage2(p1)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)

        # Bottleneck
        fc3 = self.fc_block(p5) # 16x16

        # Decoder (fusing according to red arrows in the figure)
        out = self.conv_trans_1(fc3) + self.upsample_1(p4)  # SKIP 1
        out = self.conv_trans_2(out) + self.upsample_2(p3)  # SKIP 2
        out = self.conv_trans_3(out) + self.upsample_3(p2)  # SKIP 3
        
        out = self.conv_trans_4(out)                        
        
        return self.softmax(out)


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
    def __init__(self, pretrained: bool = True):
        super().__init__(num_classes=3, pretrained=pretrained)
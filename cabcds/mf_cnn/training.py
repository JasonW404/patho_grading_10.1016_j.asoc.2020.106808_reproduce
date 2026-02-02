from typing import Literal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from cabcds.mf_cnn.cnn import CNNSeg, CNNDet, CNNGlobal

def train_mf_cnn_modules(
    seg_model: CNNSeg, 
    det_model: CNNDet, 
    global_model: CNNGlobal,
    train_loader_seg: DataLoader,
    train_loader_det: DataLoader,
    train_loader_global: DataLoader,
    device: Literal["cuda", "mps", "cpu"] = "cpu",
    *,
    lr: float = 1e-4,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    seg_epochs: int = 10,
    det_epochs: int = 1,
    global_epochs: int = 1,
):
    """
    Implements the training logic for MF-CNN modules.
    
    The paper utilizes Stochastic Gradient Descent (SGD) to minimize segmentation 
    and classification errors.
    """

    # --- 1. Training CNN_seg ---
    # CNN_seg uses a Softmax output for pixel-wise segmentation[cite: 200].
    # It was fine-tuned for approximately 7.2 hours[cite: 211].
    seg_model.to(device)
    seg_optimizer = optim.SGD(seg_model.parameters(), lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay))
    seg_criterion = nn.CrossEntropyLoss()  # Standard for pixel-wise classification
    
    seg_model.train()
    for _epoch in range(int(seg_epochs)):
        for images, masks in train_loader_seg:
            images, masks = images.to(device), masks.to(device)
            seg_optimizer.zero_grad()
            outputs = seg_model(images)
            loss = seg_criterion(outputs.logits, masks)
            loss.backward()
            seg_optimizer.step()

    # --- 2. Training CNN_det ---
    # CNN_det is an AlexNet fine-tuned on an 80x80 patch dataset produced by CNN_seg[cite: 212].
    # These patches are resized to 227x227 for the network[cite: 214].
    # Fine-tuning took approximately 1.3 hours[cite: 216].
    det_model.to(device)
    det_optimizer = optim.SGD(det_model.parameters(), lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay))
    det_criterion = nn.CrossEntropyLoss()
    
    det_model.train()
    for _epoch in range(int(det_epochs)):
        for images, labels in train_loader_det:
            images, labels = images.to(device), labels.to(device)
            det_optimizer.zero_grad()
            outputs = det_model(images)
            loss = det_criterion(outputs, labels)
            loss.backward()
            det_optimizer.step()

    # --- 3. Training CNN_global ---
    # CNN_global is also an AlexNet, but modified for 3-class ROI scoring[cite: 222].
    # It takes 512x512 patches (resized to 227x227) containing potential mitoses[cite: 220, 223].
    # Fine-tuning took approximately 2.6 hours[cite: 228].
    global_model.to(device)
    global_optimizer = optim.SGD(global_model.parameters(), lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay))
    global_criterion = nn.CrossEntropyLoss()
    
    global_model.train()
    for _epoch in range(int(global_epochs)):
        for images, labels in train_loader_global:
            images, labels = images.to(device), labels.to(device)
            global_optimizer.zero_grad()
            outputs = global_model(images)
            loss = global_criterion(outputs, labels)
            loss.backward()
            global_optimizer.step()
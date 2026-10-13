"""
Tongue model — Hybrid CNN (EfficientNet-B0) + ViT (ViT-Base/16) architecture
and preprocessing, matching the training notebook exactly.

Model weights: models/best_model (1).pt (state_dict only, PyTorch).
Classes (alphabetical, as assigned by torchvision.datasets.ImageFolder):
    0: cracked
    1: healthy
    2: pale
    3: red
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=4, cnn_name="efficientnet_b0",
                 vit_name="vit_base_patch16_224", pretrained=False, dropout=0.3):
        super().__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=pretrained, num_classes=0)
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)

        cnn_dim = self.cnn.num_features
        vit_dim = self.vit.num_features

        self.fusion = nn.Sequential(
            nn.Linear(cnn_dim + vit_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        f_cnn = self.cnn(x)
        f_vit = self.vit(x)
        fused = torch.cat([f_cnn, f_vit], dim=1)
        z = self.fusion(fused)
        return self.classifier(z)


def extract_tongue_roi(img_rgb):
    """Color + contour based tongue ROI extraction.
    Falls back to the full image if no confident region is found."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower1 = np.array([0, 20, 40])
    upper1 = np.array([25, 255, 255])
    lower2 = np.array([160, 20, 40])
    upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img_rgb.shape[:2]
    if not contours:
        return img_rgb

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 0.05 * h * w:
        return img_rgb

    x, y, cw, ch = cv2.boundingRect(largest)
    pad = int(0.05 * max(cw, ch))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + cw + pad), min(h, y + ch + pad)
    return img_rgb[y0:y1, x0:x1]


def clahe_normalize(img_rgb):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)


def resize_with_padding(img_rgb, size=IMG_SIZE):
    h, w = img_rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def preprocess_tongue_bytes(file_bytes: bytes):
    """Raw uploaded image bytes -> preprocessed RGB uint8 array (224,224,3)."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = extract_tongue_roi(img_rgb)
    img_rgb = clahe_normalize(img_rgb)
    img_rgb = resize_with_padding(img_rgb, IMG_SIZE)
    return img_rgb


def tongue_bytes_to_tensor(file_bytes: bytes):
    """Raw uploaded image bytes -> torch tensor shaped (1, 3, 224, 224), normalized."""
    processed = preprocess_tongue_bytes(file_bytes)
    if processed is None:
        return None
    return to_tensor(processed).unsqueeze(0)
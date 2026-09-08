"""Build segmentation model for surface defect detection.

Uses torchvision DeepLabV3+ with ResNet-50 backbone, adapted for
6-class segmentation on 256x256 inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


def build_model(num_classes: int = 6, pretrained_backbone: bool = True) -> nn.Module:
    """Build DeepLabV3+ with ResNet-50 backbone for segmentation.

    Args:
        num_classes: Number of output segmentation classes (including background).
        pretrained_backbone: If True, load ImageNet-pretrained ResNet-50 weights.

    Returns:
        nn.Module ready for training or inference.
    """
    if pretrained_backbone:
        # Build with pretrained weights (default 21 classes from COCO),
        # then replace classifier heads for our num_classes.
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights)
    else:
        model = deeplabv3_resnet50(weights=None, num_classes=num_classes)

    # Replace final Conv2d in classifier to match num_classes
    old_conv = model.classifier[-1]
    model.classifier[-1] = nn.Conv2d(
        old_conv.in_channels, num_classes, kernel_size=1
    )

    # Replace final Conv2d in aux_classifier to match num_classes
    if model.aux_classifier is not None:
        old_aux_conv = model.aux_classifier[-1]
        model.aux_classifier[-1] = nn.Conv2d(
            old_aux_conv.in_channels, num_classes, kernel_size=1
        )

    return model


def build_model_for_export(num_classes: int = 6) -> nn.Module:
    """Build model in eval mode with fixed 256x256 input for ONNX export.

    Args:
        num_classes: Number of output segmentation classes.

    Returns:
        nn.Module in eval mode with no pretrained weights.
    """
    model = build_model(num_classes=num_classes, pretrained_backbone=False)
    model.eval()
    return model


if __name__ == "__main__":
    model = build_model(num_classes=6)
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    # DeepLabV3+ returns a dict with 'out' and optionally 'aux'
    print(f"Output shape: {out['out'].shape}")  # [1, 6, 256, 256]

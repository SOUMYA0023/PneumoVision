"""
Grad-CAM visualization using pytorch-grad-cam library.
Returns: original image, heatmap overlay (both as PIL Images).
"""

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn


def get_target_layer(model: nn.Module, model_name: str):
    """Return the last convolutional layer for each model architecture."""
    if model_name == "resnet50":
        return [model.layer4[-1]]
    elif model_name == "efficientnetb0":
        return [model.features[-1]]
    elif model_name == "vgg16":
        return [model.features[-1]]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def compute_gradcam(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    target_class: int,
    original_pil_image: Image.Image
) -> tuple:
    """
    Compute Grad-CAM heatmap for the target class.

    Returns:
        overlay_pil: PIL Image of X-ray with heatmap overlay
        raw_cam: numpy array of raw CAM (H x W), values 0.0 to 1.0
    """
    target_layers = get_target_layer(model, model_name)
    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0), targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # (H, W)

    # Prepare original image as float numpy array normalized to [0, 1]
    img_resized = original_pil_image.convert("RGB").resize((224, 224))
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    # Create colored overlay
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    overlay_pil = Image.fromarray(cam_image)

    return overlay_pil, grayscale_cam

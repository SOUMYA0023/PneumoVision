"""
Monte Carlo Dropout uncertainty estimation.
Runs N forward passes with dropout enabled, computes std deviation.
"""

import torch
import torch.nn as nn
import numpy as np


def enable_dropout(model: nn.Module):
    """Enable dropout layers during inference for MC Dropout."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()  # Keep dropout active


def mc_dropout_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_passes: int = 20
) -> tuple:
    """
    Run Monte Carlo Dropout inference.

    Returns:
        mean_probs: Average softmax probabilities across N passes (num_classes,)
        std_probs:  Std deviation of probabilities across N passes (num_classes,)
        uncertainty_level: "Low" | "Medium" | "High"
    """
    model.eval()
    enable_dropout(model)

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            output = model(input_tensor.unsqueeze(0))
            probs = torch.softmax(output, dim=1).squeeze(0).numpy()
            all_probs.append(probs)

    all_probs = np.array(all_probs)  # (N, num_classes)
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)

    # Uncertainty = std of predicted class probability
    predicted_class = np.argmax(mean_probs)
    uncertainty_val = std_probs[predicted_class]

    if uncertainty_val < 0.05:
        uncertainty_level = "Low"
    elif uncertainty_val < 0.15:
        uncertainty_level = "Medium"
    else:
        uncertainty_level = "High"

    # Restore model to eval (disabling dropout)
    model.eval()

    return mean_probs, std_probs, uncertainty_level

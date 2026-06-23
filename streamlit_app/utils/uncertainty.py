"""
Monte Carlo Dropout uncertainty estimation.
Runs N forward passes with dropout enabled, computes std deviation.
"""

import torch
import torch.nn as nn
import numpy as np
import streamlit as st


def _enable_dropout(model: nn.Module):
    """Enable dropout layers during inference for MC Dropout."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()  # Keep dropout active


def _restore_eval_all_dropout(model: nn.Module):
    """
    Explicitly restore every Dropout submodule to eval mode.
    This is stronger than model.eval() alone, which only sets the
    top-level module flag and does NOT recursively guarantee that
    nested Dropout layers inside Sequential blocks are restored.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


def mc_dropout_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_passes: int = 20
) -> tuple:
    """
    Run Monte Carlo Dropout inference.

    Returns:
        mean_probs: Average softmax probabilities across N passes (num_classes,)
                    SOFTMAX APPLIED HERE — DO NOT APPLY AGAIN DOWNSTREAM
        std_probs:  Std deviation of probabilities across N passes (num_classes,)
        uncertainty_level: "Low" | "Medium" | "High"
    """
    # RULE A: Verify model starts in eval mode
    if model.training:
        st.warning("Model state was unexpectedly in training mode — reset automatically.")
        model.eval()

    # Enable dropout for MC inference (this mutates the cached model)
    _enable_dropout(model)

    try:
        all_probs = []
        # RULE D: Gradients not needed for MC Dropout inference
        with torch.no_grad():
            for _ in range(n_passes):
                output = model(input_tensor.unsqueeze(0))
                # SOFTMAX APPLIED HERE — DO NOT APPLY AGAIN DOWNSTREAM
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

        return mean_probs, std_probs, uncertainty_level

    finally:
        # RULE B: Always restore model to clean eval state, even on exception.
        # This uses a dedicated restore function that iterates every submodule
        # instead of relying on model.eval() alone.
        _restore_eval_all_dropout(model)

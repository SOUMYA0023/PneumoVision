"""
Severity scoring based on Grad-CAM activation area.
Only applicable when prediction is BACTERIAL or VIRAL.
"""

import numpy as np


def compute_severity(raw_cam: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute severity score from raw Grad-CAM heatmap.

    Args:
        raw_cam: numpy array of shape (H, W), values 0.0 to 1.0
        threshold: activation threshold above which a pixel is "active"

    Returns:
        dict with keys: level (str), percentage (float), color (str), description (str)
    """
    active_pixels = (raw_cam > threshold).sum()
    total_pixels = raw_cam.size
    active_pct = (active_pixels / total_pixels) * 100.0

    if active_pct <= 15.0:
        level = "Mild"
        color = "#00C49A"  # Green
        description = "Limited consolidation detected."
    elif active_pct <= 40.0:
        level = "Moderate"
        color = "#FFB347"  # Amber
        description = "Moderate consolidation detected across lung region."
    else:
        level = "Severe"
        color = "#FF4B4B"  # Red
        description = "Extensive consolidation detected. Urgent review recommended."

    return {
        "level":       level,
        "percentage":  round(active_pct, 2),
        "color":       color,
        "description": description
    }

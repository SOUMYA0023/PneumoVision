"""
Loads PyTorch model weights from the models/ directory.
Supports: resnet50, efficientnetb0, vgg16
Returns model in eval mode on CPU (for Streamlit deployment compatibility).
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from pathlib import Path
import streamlit as st

MODEL_DIR = Path(__file__).parent.parent / "models"
DEVICE = torch.device("cpu")  # CPU for Streamlit Cloud compatibility

CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
NUM_CLASSES = 3


def set_deterministic_mode():
    """
    Set all random seeds and enable deterministic algorithms for reproducible inference.
    Call this once at app startup, before any model loading or inference.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some CPU ops may not have deterministic implementations;
        # fall back gracefully but seeds are still set.
        pass


def _force_eval_all_dropout(model: nn.Module):
    """
    Recursively iterate through every submodule and explicitly set
    every Dropout instance to eval mode. This is stronger than
    model.eval() alone, which only sets the top-level module flag.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


def build_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )
    return model


def build_efficientnetb0(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_vgg16(num_classes: int) -> nn.Module:
    model = models.vgg16(weights=None)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    return model


MODEL_BUILDERS = {
    "resnet50":       build_resnet50,
    "efficientnetb0": build_efficientnetb0,
    "vgg16":          build_vgg16,
}


@st.cache_resource(show_spinner="Loading model weights...")
def load_model(model_name: str) -> nn.Module:
    """Load and cache model. Cached per model_name key."""
    weight_path = MODEL_DIR / f"{model_name}_pneumo.pth"

    if not weight_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weight_path}. "
            "Please run the Colab training notebook first and copy .pth files to streamlit_app/models/"
        )

    builder = MODEL_BUILDERS[model_name]
    model = builder(NUM_CLASSES)
    checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_clean_model(model_name: str) -> nn.Module:
    """
    Get the cached model and defensively reset it to a clean deterministic eval state.

    Always call this instead of load_model() directly from pages and utilities.
    Even if the cached model was mutated by MC Dropout or other code, this function
    guarantees the returned model has all layers (including nested Dropout) in eval mode.
    """
    model = load_model(model_name)
    model.eval()
    _force_eval_all_dropout(model)
    return model


def get_transform():
    """Return inference transform."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

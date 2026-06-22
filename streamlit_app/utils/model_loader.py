"""
Loads PyTorch model weights from the models/ directory.
Supports: resnet50, efficientnetb0, vgg16
Returns model in eval mode on CPU (for Streamlit deployment compatibility).
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import streamlit as st

MODEL_DIR = Path(__file__).parent.parent / "models"
DEVICE = torch.device("cpu")  # CPU for Streamlit Cloud compatibility

CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
NUM_CLASSES = 3


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


def get_transform():
    """Return inference transform."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

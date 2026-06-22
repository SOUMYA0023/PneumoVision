"""
Benchmarks page (Feature F9).
Display model performance metrics, confusion matrices, ROC curves,
and training history from the exported metrics JSON/PNG files.
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

METRICS_DIR = Path(__file__).parent.parent / "metrics"
MODEL_NAMES = ["resnet50", "efficientnetb0", "vgg16"]
DISPLAY_NAMES = {
    "resnet50": "ResNet-50",
    "efficientnetb0": "EfficientNet-B0",
    "vgg16": "VGG-16"
}


def load_metrics() -> dict:
    """Load all_metrics.json or individual metric files."""
    combined_path = METRICS_DIR / "all_metrics.json"
    if combined_path.exists():
        with open(combined_path) as f:
            return json.load(f)

    # Try individual files
    metrics = {}
    for name in MODEL_NAMES:
        path = METRICS_DIR / f"{name}_metrics.json"
        if path.exists():
            with open(path) as f:
                metrics[name] = json.load(f)
    return metrics


def render():
    st.markdown("## 📊 Model Benchmarks & Performance")
    st.markdown("Compare all three CNN models trained on the chest X-ray dataset.")

    if not METRICS_DIR.exists():
        st.warning(
            "⚠️ **Metrics directory not found.**\n\n"
            "Please run the Colab training notebook first and copy the metrics files "
            "(JSON + PNG) to `streamlit_app/metrics/`."
        )
        _render_placeholder_benchmarks()
        return

    metrics = load_metrics()
    if not metrics:
        st.warning(
            "⚠️ **No metrics files found.**\n\n"
            "Please run the Colab training notebook and copy metrics JSON/PNG files "
            "to `streamlit_app/metrics/`."
        )
        _render_placeholder_benchmarks()
        return

    # ---- Overview Comparison Table ----
    st.markdown("### 🏆 Model Comparison")
    table_data = []
    for model_name in MODEL_NAMES:
        if model_name in metrics:
            m = metrics[model_name]
            table_data.append({
                "Model":            DISPLAY_NAMES.get(model_name, model_name),
                "Accuracy":         f"{m.get('accuracy', 0) * 100:.2f}%",
                "Precision (Macro)": f"{m.get('precision_macro', 0) * 100:.2f}%",
                "Recall (Macro)":   f"{m.get('recall_macro', 0) * 100:.2f}%",
                "F1 (Macro)":       f"{m.get('f1_macro', 0) * 100:.2f}%",
                "F1 (Weighted)":    f"{m.get('f1_weighted', 0) * 100:.2f}%",
            })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ---- AUC Scores ----
    st.markdown("### 📈 AUC Scores (One-vs-Rest)")
    auc_data = []
    for model_name in MODEL_NAMES:
        if model_name in metrics and "auc_scores" in metrics[model_name]:
            auc = metrics[model_name]["auc_scores"]
            row = {"Model": DISPLAY_NAMES.get(model_name, model_name)}
            for cls, score in auc.items():
                row[f"AUC ({cls})"] = f"{score:.4f}"
            auc_data.append(row)

    if auc_data:
        auc_df = pd.DataFrame(auc_data)
        st.dataframe(auc_df, use_container_width=True, hide_index=True)

    # ---- Visualizations Per Model ----
    st.markdown("### 📉 Detailed Model Analysis")

    tabs = st.tabs([DISPLAY_NAMES.get(n, n) for n in MODEL_NAMES if n in metrics])
    available_models = [n for n in MODEL_NAMES if n in metrics]

    for tab, model_name in zip(tabs, available_models):
        with tab:
            st.markdown(f"#### {DISPLAY_NAMES.get(model_name, model_name)} Performance")

            # Key metrics as metric cards
            m = metrics[model_name]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy", f"{m.get('accuracy', 0) * 100:.1f}%")
            mc2.metric("Precision", f"{m.get('precision_macro', 0) * 100:.1f}%")
            mc3.metric("Recall", f"{m.get('recall_macro', 0) * 100:.1f}%")
            mc4.metric("F1 Score", f"{m.get('f1_macro', 0) * 100:.1f}%")

            # Confusion Matrix
            cm_path = METRICS_DIR / f"{model_name}_confusion_matrix.png"
            if cm_path.exists():
                st.markdown("##### Confusion Matrix")
                cm_img = Image.open(cm_path)
                st.image(cm_img, caption=f"{DISPLAY_NAMES[model_name]} — Confusion Matrix",
                         use_container_width=True)

            # ROC Curves
            roc_path = METRICS_DIR / f"{model_name}_roc_curves.png"
            if roc_path.exists():
                st.markdown("##### ROC Curves (One-vs-Rest)")
                roc_img = Image.open(roc_path)
                st.image(roc_img, caption=f"{DISPLAY_NAMES[model_name]} — ROC Curves",
                         use_container_width=True)

            # Training Loss Curves
            loss_path = METRICS_DIR / f"{model_name}_loss_curves.png"
            if loss_path.exists():
                st.markdown("##### Training / Validation Curves")
                loss_img = Image.open(loss_path)
                st.image(loss_img, caption=f"{DISPLAY_NAMES[model_name]} — Loss & Accuracy",
                         use_container_width=True)

            # AUC details
            if "auc_scores" in m:
                st.markdown("##### Per-Class AUC")
                auc_cols = st.columns(len(m["auc_scores"]))
                for i, (cls, score) in enumerate(m["auc_scores"].items()):
                    auc_cols[i].metric(cls, f"{score:.4f}")


def _render_placeholder_benchmarks():
    """Show placeholder benchmark information when no trained metrics are available."""
    st.markdown("---")
    st.markdown("### 📋 Expected Model Architecture Summary")

    arch_data = [
        {
            "Model": "ResNet-50",
            "Parameters": "~25.6M",
            "Architecture": "Deep residual network with skip connections",
            "Classifier Head": "FC → 256 → ReLU → FC → 3 classes",
            "Dropout": "0.5, 0.3"
        },
        {
            "Model": "EfficientNet-B0",
            "Parameters": "~5.3M",
            "Architecture": "Compound-scaled mobile-optimized CNN",
            "Classifier Head": "FC → 3 classes",
            "Dropout": "0.4"
        },
        {
            "Model": "VGG-16",
            "Parameters": "~138M",
            "Architecture": "Sequential deep convolutional network",
            "Classifier Head": "FC → 3 classes (replaces original FC layer)",
            "Dropout": "0.5"
        }
    ]
    df = pd.DataFrame(arch_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### 🔧 Training Configuration")
    config_data = {
        "Transfer Learning": "ImageNet pretrained, head-only → full fine-tuning",
        "Optimizer": "Adam (lr=1e-4 head, lr=1e-5 full)",
        "Scheduler": "CosineAnnealingLR",
        "Loss": "CrossEntropyLoss with class weights",
        "Augmentation": "HorizontalFlip, Rotation(±10°), ColorJitter",
        "Early Stopping": "Patience = 5 on validation loss",
        "Unfreeze Epoch": "Epoch 10 (backbone unfrozen)",
    }
    for key, val in config_data.items():
        st.markdown(f"- **{key}:** {val}")

    st.info(
        "💡 **To see actual metrics:** Run the `PneumoVision_Training.ipynb` notebook in Google Colab, "
        "then copy the exported `.pth` files to `streamlit_app/models/` and "
        "metrics files to `streamlit_app/metrics/`."
    )

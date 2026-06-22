"""
Side-by-side comparison page (Feature F4).
Upload two X-rays and compare predictions, confidence, and Grad-CAM side by side.
"""

import time
import streamlit as st
import numpy as np
from PIL import Image

from utils.model_loader import load_model, get_transform, CLASS_NAMES
from utils.gradcam import compute_gradcam
from utils.uncertainty import mc_dropout_inference
from utils.severity import compute_severity


def analyze_single(model, model_name, pil_image):
    """Run full analysis pipeline on one image, return result dict."""
    transform = get_transform()
    tensor = transform(pil_image)

    mean_probs, std_probs, uncertainty_level = mc_dropout_inference(model, tensor, n_passes=20)
    predicted_idx = int(np.argmax(mean_probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(mean_probs[predicted_idx])

    heatmap_image, raw_cam = compute_gradcam(model, model_name, tensor, predicted_idx, pil_image)

    severity = None
    if predicted_class != "NORMAL":
        severity = compute_severity(raw_cam)

    return {
        "prediction":       predicted_class,
        "confidence":       confidence,
        "uncertainty":      uncertainty_level,
        "heatmap":          heatmap_image,
        "severity":         severity,
        "mean_probs":       mean_probs,
    }


def render_result_panel(result: dict, label: str):
    """Render prediction results for one image in a column."""
    pred_color = "#00C49A" if result["prediction"] == "NORMAL" else "#FF4B4B"
    label_map = {"NORMAL": "✅ Normal", "BACTERIAL": "⚠️ Bacterial", "VIRAL": "⚠️ Viral"}

    st.markdown(f"""
    <div style="background:{pred_color}22; border:1px solid {pred_color};
                border-radius:8px; padding:12px; text-align:center;">
        <b style="color:{pred_color}; font-size:16px;">{label_map[result['prediction']]}</b><br>
        <span style="color:#ccc;">Confidence: {result['confidence']*100:.2f}%</span><br>
        <span style="color:#888;">Uncertainty: {result['uncertainty']}</span>
    </div>
    """, unsafe_allow_html=True)

    st.image(result["heatmap"], caption=f"{label} — Grad-CAM", use_container_width=True)

    if result["severity"]:
        sev = result["severity"]
        st.markdown(f"**Severity:** <span style='color:{sev['color']}'>{sev['level']}</span> "
                    f"({sev['percentage']:.1f}%)", unsafe_allow_html=True)


def render():
    st.markdown("## 🔄 Compare — Side by Side")
    st.markdown("Upload two chest X-rays to compare AI predictions and attention heatmaps.")

    model_name = st.session_state.get("selected_model", "resnet50")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### X-Ray A")
        upload_a = st.file_uploader("Upload X-Ray A", type=["jpg", "jpeg", "png"], key="compare_a")
    with col2:
        st.markdown("#### X-Ray B")
        upload_b = st.file_uploader("Upload X-Ray B", type=["jpg", "jpeg", "png"], key="compare_b")

    if upload_a is None or upload_b is None:
        st.info("Please upload both X-rays to run comparison.")
        return

    image_a = Image.open(upload_a).convert("RGB")
    image_b = Image.open(upload_b).convert("RGB")

    # Show originals
    orig_col1, orig_col2 = st.columns(2)
    with orig_col1:
        st.image(image_a, caption="X-Ray A (Original)", use_container_width=True)
    with orig_col2:
        st.image(image_b, caption="X-Ray B (Original)", use_container_width=True)

    # Run analysis
    try:
        model = load_model(model_name)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    with st.spinner("Analyzing both X-rays..."):
        result_a = analyze_single(model, model_name, image_a)
        result_b = analyze_single(model, model_name, image_b)

    # Show results
    st.markdown("### 📊 Analysis Results")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        render_result_panel(result_a, "X-Ray A")
    with res_col2:
        render_result_panel(result_b, "X-Ray B")

    # Probability comparison table
    st.markdown("### 📈 Probability Comparison")
    comp_cols = st.columns(3)
    for i, cls in enumerate(CLASS_NAMES):
        with comp_cols[i]:
            st.markdown(f"**{cls}**")
            st.caption(f"A: {result_a['mean_probs'][i]*100:.1f}%")
            st.caption(f"B: {result_b['mean_probs'][i]*100:.1f}%")

"""
Single X-Ray inference page.
Features: F1 (detection), F2 (Grad-CAM), F3 (confidence + uncertainty),
          F5 (PDF report), F10 (severity scoring)
"""

import time
import streamlit as st
import torch
import numpy as np
from PIL import Image

from utils.model_loader import load_model, get_transform, CLASS_NAMES
from utils.gradcam import compute_gradcam
from utils.uncertainty import mc_dropout_inference
from utils.severity import compute_severity
from utils.report_generator import generate_pdf_report
from utils.history import add_to_history


def render():
    st.markdown("## 🫁 Diagnose — Single X-Ray")
    st.markdown("Upload a chest X-ray to get an AI-powered analysis with explainability heatmap.")

    # Model selector (also in sidebar)
    model_name = st.session_state.get("selected_model", "resnet50")

    uploaded = st.file_uploader(
        "Upload Chest X-Ray (JPG or PNG)", type=["jpg", "jpeg", "png"],
        key="single_upload"
    )

    if uploaded is None:
        st.info("Please upload a chest X-ray image to begin analysis.")
        return

    original_image = Image.open(uploaded).convert("RGB")

    # ---- Load model ----
    try:
        model = load_model(model_name)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    transform = get_transform()
    input_tensor = transform(original_image)

    # ---- Inference ----
    with st.spinner("Running analysis..."):
        t_start = time.time()

        # F3: MC Dropout for uncertainty
        mean_probs, std_probs, uncertainty_level = mc_dropout_inference(model, input_tensor, n_passes=20)

        predicted_idx = int(np.argmax(mean_probs))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(mean_probs[predicted_idx])

        # F2: Grad-CAM
        heatmap_image, raw_cam = compute_gradcam(
            model, model_name, input_tensor, predicted_idx, original_image
        )

        # F10: Severity (only for pneumonia)
        severity = None
        if predicted_class != "NORMAL":
            severity = compute_severity(raw_cam)

        t_end = time.time()
        inference_time_ms = (t_end - t_start) * 1000

    # ---- F1: Display Prediction ----
    pred_color = "#00C49A" if predicted_class == "NORMAL" else "#FF4B4B"
    label_map = {"NORMAL": "✅ Normal", "BACTERIAL": "⚠️ Bacterial Pneumonia", "VIRAL": "⚠️ Viral Pneumonia"}

    st.markdown(f"""
    <div style="background:{pred_color}22; border:2px solid {pred_color}; border-radius:12px;
                padding:20px; text-align:center; margin:16px 0;">
        <h2 style="color:{pred_color}; margin:0;">{label_map[predicted_class]}</h2>
        <p style="color:#ccc; margin:4px 0;">Model: {model_name.upper()} | Inference: {inference_time_ms:.0f}ms</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- F3: Confidence + Uncertainty ----
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.progress(confidence)
    with col2:
        unc_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        st.metric("Uncertainty", f"{unc_colors.get(uncertainty_level, '')} {uncertainty_level}")
    with col3:
        if severity:
            st.metric("Severity", severity["level"])
        else:
            st.metric("Severity", "N/A")

    # ---- F2: Images (Original + Grad-CAM) ----
    st.markdown("### 🔍 Visual Analysis")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(original_image, caption="Original X-Ray", use_container_width=True)
    with img_col2:
        st.image(heatmap_image, caption="Grad-CAM Heatmap (Red = High Attention)", use_container_width=True)

    # ---- F10: Severity Details ----
    if severity:
        st.markdown("### 📊 Severity Assessment")
        sev_color = severity["color"]
        st.markdown(f"""
        <div style="background:{sev_color}22; border-left:4px solid {sev_color};
                    padding:12px 16px; border-radius:4px;">
            <b style="color:{sev_color};">{severity['level']} Pneumonia</b><br>
            <span style="color:#ccc;">{severity['description']}</span><br>
            <small style="color:#888;">Activation coverage: {severity['percentage']:.1f}% of image area</small>
        </div>
        """, unsafe_allow_html=True)

    # ---- Class Probabilities ----
    st.markdown("### 📈 Class Probabilities")
    prob_cols = st.columns(3)
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, prob_cols)):
        with col:
            pct = mean_probs[i] * 100
            std = std_probs[i] * 100
            st.markdown(f"**{cls}**")
            st.progress(float(mean_probs[i]))
            st.caption(f"{pct:.2f}% ± {std:.2f}%")

    # ---- F5: PDF Report ----
    st.markdown("### 📄 Export Report")
    pdf_bytes = generate_pdf_report(
        original_image=original_image,
        heatmap_image=heatmap_image,
        prediction=predicted_class,
        confidence=confidence,
        uncertainty_level=uncertainty_level,
        severity=severity,
        model_name=model_name,
        inference_time_ms=inference_time_ms,
        filename=uploaded.name
    )
    st.download_button(
        label="⬇ Download Diagnostic Report (PDF)",
        data=pdf_bytes,
        file_name=f"pneumovision_report_{uploaded.name.split('.')[0]}.pdf",
        mime="application/pdf",
        type="primary"
    )

    # ---- Add to session history ----
    add_to_history(
        filename=uploaded.name,
        prediction=predicted_class,
        confidence=confidence,
        uncertainty=uncertainty_level,
        model_name=model_name,
        severity=severity["level"] if severity else None
    )

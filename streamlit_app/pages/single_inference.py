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

from utils.model_loader import get_clean_model, get_transform, CLASS_NAMES
from utils.gradcam import compute_gradcam
from utils.uncertainty import mc_dropout_inference
from utils.severity import compute_severity
from utils.report_generator import generate_pdf_report
from utils.history import add_to_history


def render():
    # ---- Workflow Header ----
    uploaded = st.session_state.get("single_upload", None)
    is_analyzed = (uploaded is not None)

    step_1_style = "color: var(--primary); font-weight: 600;" if not is_analyzed else "color: var(--muted);"
    step_2_style = "color: var(--primary); font-weight: 600;" if is_analyzed else "color: var(--muted);"
    
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; 
                background: #141822; border: 1px solid var(--border); border-radius: 8px; 
                padding: 10px 16px; margin-bottom: 24px;">
        <span style="font-size: 0.8rem; {step_1_style}">1. Upload X-Ray</span>
        <span style="font-size: 0.8rem; color: var(--border);">➔</span>
        <span style="font-size: 0.8rem; {step_2_style}">2. AI Analysis & Predictions</span>
        <span style="font-size: 0.8rem; color: var(--border);">➔</span>
        <span style="font-size: 0.8rem; {step_2_style}">3. Grad-CAM & Severity</span>
        <span style="font-size: 0.8rem; color: var(--border);">➔</span>
        <span style="font-size: 0.8rem; {step_2_style}">4. Export Report</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🫁 Diagnose — Single X-Ray")
    st.markdown("Upload a chest X-ray to get an AI-powered analysis with explainability heatmap.")

    model_name = st.session_state.get("selected_model", "resnet50")

    uploaded = st.file_uploader(
        "Upload Chest X-Ray (JPG or PNG)", type=["jpg", "jpeg", "png"],
        key="single_upload"
    )

    if uploaded is None:
        st.info("Please upload a chest X-ray image to begin analysis.")
        return

    original_image = Image.open(uploaded).convert("RGB")

    # ---- Load model (always get a clean eval-state model) ----
    try:
        model = get_clean_model(model_name)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    transform = get_transform()
    input_tensor = transform(original_image)

    # ---- Inference ----
    with st.spinner("Running analysis..."):
        t_start = time.time()

        # F3: MC Dropout for uncertainty
        # mean_probs are already softmax probabilities — DO NOT apply softmax again
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
    pred_color = "var(--primary)" if predicted_class == "NORMAL" else "var(--danger)"
    bg_color = "rgba(0, 196, 154, 0.08)" if predicted_class == "NORMAL" else "rgba(255, 75, 75, 0.08)"
    label_map = {"NORMAL": "✅ Normal", "BACTERIAL": "⚠️ Bacterial Pneumonia", "VIRAL": "⚠️ Viral Pneumonia"}

    st.markdown(f"""
    <div style="background: {bg_color}; border: 1px solid {pred_color}; border-radius: 12px;
                padding: 24px; text-align: center; margin: 20px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
        <small style="color: var(--muted); text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">AI Diagnostic Results</small>
        <h2 style="color: {pred_color}; margin: 8px 0; font-size: 1.8rem; border: none; padding: 0;">{label_map[predicted_class]}</h2>
        <div style="display: inline-flex; gap: 16px; margin-top: 8px; color: var(--muted); font-size: 0.85rem;">
            <span>Model: <b>{model_name.upper()}</b></span>
            <span>•</span>
            <span>Inference: <b>{inference_time_ms:.0f}ms</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- F3: Confidence + Uncertainty ----
    st.markdown("### 📊 Metrics Summary")
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
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.image(original_image, caption="Original X-Ray", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with img_col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.image(heatmap_image, caption="Grad-CAM Heatmap (Red = High Attention)", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- F10: Severity Details ----
    if severity:
        st.markdown("### 📈 Severity Assessment")
        sev_color = severity["color"]
        # Convert hex to rgb for opacity background
        sev_color_clean = sev_color.lstrip('#')
        sev_rgb = tuple(int(sev_color_clean[i:i+2], 16) for i in (0, 2, 4))
        
        st.markdown(f"""
        <div style="background: rgba({sev_rgb[0]},{sev_rgb[1]},{sev_rgb[2]}, 0.06); border: 1px solid {sev_color};
                    padding: 20px; border-radius: 12px; margin-bottom: 24px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background-color: {sev_color};"></span>
                <b style="color: {sev_color}; font-size: 1.1rem;">{severity['level']} Severity Classified</b>
            </div>
            <p style="color: var(--text) !important; margin: 8px 0 !important; font-size: 0.95rem;">{severity['description']}</p>
            <div style="color: var(--muted); font-size: 0.8rem; margin-top: 12px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 8px;">
                Grad-CAM Activation Coverage: <b>{severity['percentage']:.1f}%</b> of lung region
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Class Probabilities ----
    st.markdown("### 📈 Class Probabilities")
    st.markdown('<div class="premium-card" style="padding: 20px 24px;">', unsafe_allow_html=True)
    prob_cols = st.columns(3)
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, prob_cols)):
        with col:
            pct = mean_probs[i] * 100
            std = std_probs[i] * 100
            st.markdown(f"**{cls}**")
            st.progress(float(mean_probs[i]))
            st.caption(f"{pct:.2f}% ± {std:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- F5: PDF Report ----
    st.markdown("### 📄 Export Report")
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<p style='margin-bottom:16px;'>Generate and download a comprehensive diagnostic report containing original X-Ray, AI prediction, uncertainty scoring, severity classification, and Grad-CAM attention maps.</p>", unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Add to session history ----
    add_to_history(
        filename=uploaded.name,
        prediction=predicted_class,
        confidence=confidence,
        uncertainty=uncertainty_level,
        model_name=model_name,
        severity=severity["level"] if severity else None
    )

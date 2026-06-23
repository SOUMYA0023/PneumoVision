"""
Side-by-side comparison page (Feature F4).
Upload two X-rays and compare predictions, confidence, and Grad-CAM side by side.
"""

import time
import streamlit as st
import numpy as np
from PIL import Image

from utils.model_loader import get_clean_model, get_transform, CLASS_NAMES
from utils.gradcam import compute_gradcam
from utils.uncertainty import mc_dropout_inference
from utils.severity import compute_severity


def analyze_single(model, model_name, pil_image):
    """Run full analysis pipeline on one image, return result dict."""
    transform = get_transform()
    tensor = transform(pil_image)

    # mean_probs are already softmax probabilities — DO NOT apply softmax again
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
    pred_color = "var(--primary)" if result["prediction"] == "NORMAL" else "var(--danger)"
    bg_color = "rgba(0, 196, 154, 0.06)" if result["prediction"] == "NORMAL" else "rgba(255, 75, 75, 0.06)"
    label_map = {"NORMAL": "✅ Normal", "BACTERIAL": "⚠️ Bacterial", "VIRAL": "⚠️ Viral"}

    st.markdown(f"""
    <div style="background:{bg_color}; border:1px solid {pred_color};
                border-radius:12px; padding:16px; text-align:center; margin-bottom: 16px;">
        <small style="color: var(--muted); text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">{label} Result</small>
        <div style="color:{pred_color}; font-size: 1.25rem; font-weight: 700; margin: 6px 0 8px 0;">{label_map[result['prediction']]}</div>
        <div style="color:var(--text); font-size:0.9rem; margin-bottom: 2px;">Confidence: <b>{result['confidence']*100:.2f}%</b></div>
        <div style="color:var(--muted); font-size:0.85rem;">Uncertainty: <b>{result['uncertainty']}</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.image(result["heatmap"], caption=f"{label} — Grad-CAM Heatmap", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if result["severity"]:
        sev = result["severity"]
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border); 
                    padding: 12px 16px; border-radius: 8px; font-size: 0.9rem;">
            <b>Severity:</b> <span style='color:{sev['color']}; font-weight: 600;'>{sev['level']}</span> ({sev['percentage']:.1f}%)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border); 
                    padding: 12px 16px; border-radius: 8px; font-size: 0.9rem; color: var(--muted);">
            <b>Severity:</b> N/A
        </div>
        """, unsafe_allow_html=True)


def render():
    st.markdown("## 🔄 Compare — Side by Side")
    st.markdown("Upload two chest X-rays to compare AI predictions and attention heatmaps.")

    model_name = st.session_state.get("selected_model", "resnet50")

    # Layout selection columns
    st.markdown('<div class="premium-card" style="padding: 20px 24px; margin-bottom: 24px;">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Chest X-Ray A")
        upload_a = st.file_uploader("Upload X-Ray A", type=["jpg", "jpeg", "png"], key="compare_a", label_visibility="collapsed")
    with col2:
        st.markdown("#### Chest X-Ray B")
        upload_b = st.file_uploader("Upload X-Ray B", type=["jpg", "jpeg", "png"], key="compare_b", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if upload_a is None or upload_b is None:
        st.info("Please upload both chest X-ray images to run comparisons.")
        return

    image_a = Image.open(upload_a).convert("RGB")
    image_b = Image.open(upload_b).convert("RGB")

    # Show originals
    st.markdown("### 🔍 Original X-Rays")
    orig_col1, orig_col2 = st.columns(2)
    with orig_col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.image(image_a, caption="X-Ray A (Original)", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with orig_col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.image(image_b, caption="X-Ray B (Original)", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Run analysis — always use get_clean_model() for deterministic state
    try:
        model = get_clean_model(model_name)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    with st.spinner("Analyzing both chest X-rays..."):
        result_a = analyze_single(model, model_name, image_a)
        result_b = analyze_single(model, model_name, image_b)

    # Show results
    st.markdown("### 📊 Diagnostic Results")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        render_result_panel(result_a, "X-Ray A")
    with res_col2:
        render_result_panel(result_b, "X-Ray B")

    # Probability comparison table
    st.markdown("### 📈 Probability Comparison")
    st.markdown('<div class="premium-card" style="padding: 24px;">', unsafe_allow_html=True)
    comp_cols = st.columns(3)
    for i, cls in enumerate(CLASS_NAMES):
        with comp_cols[i]:
            st.markdown(f"**{cls}**")
            
            # Progress bar for model A
            st.markdown(f"<small style='color: var(--muted);'>X-Ray A</small>", unsafe_allow_html=True)
            st.progress(float(result_a['mean_probs'][i]))
            st.markdown(f"<span style='color: var(--primary); font-weight:600;'>{result_a['mean_probs'][i]*100:.2f}%</span>", unsafe_allow_html=True)
            
            # Progress bar for model B
            st.markdown(f"<small style='color: var(--muted); margin-top: 8px; display:block;'>X-Ray B</small>", unsafe_allow_html=True)
            st.progress(float(result_b['mean_probs'][i]))
            st.markdown(f"<span style='color: var(--primary); font-weight:600;'>{result_b['mean_probs'][i]*100:.2f}%</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

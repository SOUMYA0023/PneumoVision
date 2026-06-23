"""
Batch upload page (Feature F8).
Process multiple X-rays at once and display results in a summary table.
"""

import time
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from utils.model_loader import get_clean_model, get_transform, CLASS_NAMES
from utils.gradcam import compute_gradcam
from utils.uncertainty import mc_dropout_inference
from utils.severity import compute_severity
from utils.history import add_to_history


def render():
    st.markdown("## 📦 Batch Upload — Analyze Multiple X-Rays")
    st.markdown("Upload multiple chest X-rays for batch processing. Results are displayed in a summary table.")

    model_name = st.session_state.get("selected_model", "resnet50")

    st.markdown('<div class="premium-card" style="padding: 24px; margin-bottom: 24px;">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload X-Ray Images (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_upload",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if not uploaded_files:
        st.info("Please upload one or more chest X-ray images for batch analysis.")
        return

    st.markdown(f"**{len(uploaded_files)} file(s)** selected for analysis.")

    # Load model — always use get_clean_model() for deterministic state
    try:
        model = get_clean_model(model_name)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    transform = get_transform()

    if st.button("🚀 Run Batch Analysis", type="primary", key="run_batch"):
        results = []
        progress_bar = st.progress(0, text="Analyzing X-rays...")

        for idx, uploaded in enumerate(uploaded_files):
            pil_image = Image.open(uploaded).convert("RGB")
            input_tensor = transform(pil_image)

            t_start = time.time()

            # MC Dropout for uncertainty
            # mean_probs are already softmax probabilities — DO NOT apply softmax again
            mean_probs, std_probs, uncertainty_level = mc_dropout_inference(
                model, input_tensor, n_passes=10  # Fewer passes for batch speed
            )

            predicted_idx = int(np.argmax(mean_probs))
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = float(mean_probs[predicted_idx])

            # Grad-CAM
            heatmap_image, raw_cam = compute_gradcam(
                model, model_name, input_tensor, predicted_idx, pil_image
            )

            # Severity
            severity = None
            if predicted_class != "NORMAL":
                severity = compute_severity(raw_cam)

            t_end = time.time()
            inference_time_ms = (t_end - t_start) * 1000

            results.append({
                "filename":        uploaded.name,
                "prediction":      predicted_class,
                "confidence":      confidence,
                "uncertainty":     uncertainty_level,
                "severity":        severity["level"] if severity else "N/A",
                "severity_pct":    severity["percentage"] if severity else 0.0,
                "inference_ms":    round(inference_time_ms, 1),
                "heatmap":         heatmap_image,
                "original":        pil_image,
                "probs":           mean_probs,
            })

            # Add to session history
            add_to_history(
                filename=uploaded.name,
                prediction=predicted_class,
                confidence=confidence,
                uncertainty=uncertainty_level,
                model_name=model_name,
                severity=severity["level"] if severity else None
            )

            progress_bar.progress(
                (idx + 1) / len(uploaded_files),
                text=f"Analyzed {idx + 1}/{len(uploaded_files)}: {uploaded.name}"
            )

        progress_bar.empty()
        st.success(f"✅ Batch analysis complete! {len(results)} images processed.")

        # ---- Summary Statistics ----
        st.markdown("### 📊 Batch Summary")
        total = len(results)
        normal_count = sum(1 for r in results if r["prediction"] == "NORMAL")
        bacterial_count = sum(1 for r in results if r["prediction"] == "BACTERIAL")
        viral_count = sum(1 for r in results if r["prediction"] == "VIRAL")

        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        sum_col1.metric("Total Scanned", total)
        sum_col2.metric("Normal", normal_count)
        sum_col3.metric("Bacterial", bacterial_count)
        sum_col4.metric("Viral", viral_count)

        # Avg confidence
        avg_conf = np.mean([r["confidence"] for r in results]) * 100
        st.markdown(f"<div style='margin: 16px 0; font-size: 0.95rem; color: var(--muted);'>Average Confidence Across Batch: <b style='color: var(--primary);'>{avg_conf:.1f}%</b></div>", unsafe_allow_html=True)

        # ---- Results Table ----
        st.markdown("### 📋 Detailed Results")
        df_data = []
        for r in results:
            df_data.append({
                "File":          r["filename"],
                "Prediction":    r["prediction"],
                "Confidence":    f"{r['confidence'] * 100:.2f}%",
                "Uncertainty":   r["uncertainty"],
                "Severity":      r["severity"],
                "Time (ms)":     r["inference_ms"],
            })

        df = pd.DataFrame(df_data)

        # Color-coded display
        def color_prediction(val):
            if val == "NORMAL":
                return "color: var(--primary); font-weight: bold"
            else:
                return "color: var(--danger); font-weight: bold"

        styled_df = df.style.map(color_prediction, subset=["Prediction"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # ---- CSV Export ----
        st.markdown('<div class="premium-card" style="margin-top: 24px;">', unsafe_allow_html=True)
        st.markdown("##### Export Results")
        st.markdown("<p style='margin-bottom: 16px;'>Export the summary metrics and detailed prediction output as a standard CSV format.</p>", unsafe_allow_html=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Results as CSV",
            data=csv_bytes,
            file_name="pneumovision_batch_results.csv",
            mime="text/csv"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Individual Heatmaps (Expandable) ----
        st.markdown("### 🔍 Individual Heatmaps")
        for r in results:
            pred_color = "var(--primary)" if r["prediction"] == "NORMAL" else "var(--danger)"
            with st.expander(
                f"{'✅' if r['prediction'] == 'NORMAL' else '⚠️'} "
                f"{r['filename']} — {r['prediction']} ({r['confidence']*100:.1f}%)"
            ):
                hm_col1, hm_col2 = st.columns(2)
                with hm_col1:
                    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                    st.image(r["original"], caption="Original", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with hm_col2:
                    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                    st.image(r["heatmap"], caption="Grad-CAM", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Probability bars
                st.markdown('<div class="premium-card" style="padding: 16px 20px;">', unsafe_allow_html=True)
                prob_cols = st.columns(3)
                for i, cls in enumerate(CLASS_NAMES):
                    with prob_cols[i]:
                        st.markdown(f"**{cls}**")
                        st.progress(float(r["probs"][i]))
                        st.caption(f"{r['probs'][i]*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)

                if r["severity"] != "N/A":
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border); 
                                padding: 12px; border-radius: 8px; font-size: 0.9rem; margin-top: 8px;">
                        <b>Severity:</b> {r['severity']} (Activation: <b>{r['severity_pct']:.1f}%</b>)
                    </div>
                    """, unsafe_allow_html=True)

"""
Batch upload page (Feature F8).
Process multiple X-rays at once and display results in a summary table.
"""

import time
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from utils.model_loader import load_model, get_transform, CLASS_NAMES
from utils.gradcam import compute_gradcam
from utils.uncertainty import mc_dropout_inference
from utils.severity import compute_severity
from utils.history import add_to_history


def render():
    st.markdown("## 📦 Batch Upload — Analyze Multiple X-Rays")
    st.markdown("Upload multiple chest X-rays for batch processing. Results are displayed in a summary table.")

    model_name = st.session_state.get("selected_model", "resnet50")

    uploaded_files = st.file_uploader(
        "Upload X-Ray Images (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if not uploaded_files:
        st.info("Please upload one or more chest X-ray images for batch analysis.")
        return

    st.markdown(f"**{len(uploaded_files)} file(s)** selected for analysis.")

    # Load model
    try:
        model = load_model(model_name)
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
        st.markdown(f"**Average Confidence:** {avg_conf:.1f}%")

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
                return "color: #00C49A; font-weight: bold"
            else:
                return "color: #FF4B4B; font-weight: bold"

        styled_df = df.style.map(color_prediction, subset=["Prediction"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # ---- CSV Export ----
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Results as CSV",
            data=csv_bytes,
            file_name="pneumovision_batch_results.csv",
            mime="text/csv"
        )

        # ---- Individual Heatmaps (Expandable) ----
        st.markdown("### 🔍 Individual Heatmaps")
        for r in results:
            pred_color = "#00C49A" if r["prediction"] == "NORMAL" else "#FF4B4B"
            with st.expander(
                f"{'✅' if r['prediction'] == 'NORMAL' else '⚠️'} "
                f"{r['filename']} — {r['prediction']} ({r['confidence']*100:.1f}%)"
            ):
                hm_col1, hm_col2 = st.columns(2)
                with hm_col1:
                    st.image(r["original"], caption="Original", use_container_width=True)
                with hm_col2:
                    st.image(r["heatmap"], caption="Grad-CAM", use_container_width=True)

                # Probability bars
                prob_cols = st.columns(3)
                for i, cls in enumerate(CLASS_NAMES):
                    with prob_cols[i]:
                        st.markdown(f"**{cls}**")
                        st.progress(float(r["probs"][i]))
                        st.caption(f"{r['probs'][i]*100:.1f}%")

                if r["severity"] != "N/A":
                    st.markdown(f"**Severity:** {r['severity']} "
                                f"(Activation: {r['severity_pct']:.1f}%)")

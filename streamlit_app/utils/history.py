"""
Session history management using Streamlit session_state.
"""

import streamlit as st
import pandas as pd
from datetime import datetime


def init_history():
    """Initialize history list in session state if not present."""
    if "inference_history" not in st.session_state:
        st.session_state.inference_history = []


def add_to_history(filename: str, prediction: str, confidence: float,
                   uncertainty: str, model_name: str, severity: str | None = None):
    """Append one inference record to session history."""
    init_history()
    record = {
        "Timestamp":   datetime.now().strftime("%H:%M:%S"),
        "File":        filename,
        "Prediction":  prediction,
        "Confidence":  f"{confidence * 100:.2f}%",
        "Uncertainty": uncertainty,
        "Severity":    severity if severity else "N/A",
        "Model":       model_name.upper(),
    }
    st.session_state.inference_history.insert(0, record)


def get_history_df() -> pd.DataFrame:
    """Return history as a DataFrame."""
    init_history()
    if not st.session_state.inference_history:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.inference_history)


def clear_history():
    """Clear all history."""
    st.session_state.inference_history = []


def render_history_sidebar():
    """Render the session history panel in the Streamlit sidebar."""
    init_history()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🕓 Session History")

    if not st.session_state.inference_history:
        st.sidebar.caption("No inferences yet this session.")
        return

    for record in st.session_state.inference_history[:10]:
        color = "#00C49A" if record["Prediction"] == "NORMAL" else "#FF4B4B"
        st.sidebar.markdown(
            f"<small style='color:{color}'><b>{record['Prediction']}</b></small> "
            f"<small style='color:#888'>{record['File'][:20]} · {record['Timestamp']}</small>",
            unsafe_allow_html=True
        )

    df = get_history_df()
    csv = df.to_csv(index=False).encode("utf-8")
    col1, col2 = st.sidebar.columns(2)
    col1.download_button("⬇ CSV", csv, "history.csv", "text/csv", key="dl_history")
    if col2.button("🗑 Clear", key="clear_history"):
        clear_history()
        st.rerun()

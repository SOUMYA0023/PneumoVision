"""
PneumoVision — AI-Powered Pneumonia Detection from Chest X-Rays
Main Streamlit Application Entry Point

Features:
    F1:  3-Class Pneumonia Detection (Normal, Bacterial, Viral)
    F2:  Grad-CAM Explainability Heatmaps
    F3:  Confidence Scores + MC Dropout Uncertainty
    F4:  Side-by-Side X-Ray Comparison
    F5:  PDF Diagnostic Report Export
    F6:  Session History with CSV Export
    F7:  Multi-Model Selector (ResNet50, EfficientNetB0, VGG16)
    F8:  Batch Upload & Processing
    F9:  Model Benchmarks & Metrics Dashboard
    F10: Severity Scoring (Mild / Moderate / Severe)
"""

import streamlit as st
from assets.logo_placeholder import get_logo_html
from utils.history import render_history_sidebar
from utils.model_loader import set_deterministic_mode

# ---- Page Configuration ----
st.set_page_config(
    page_title="PneumoVision — AI Chest X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "PneumoVision v1.0 — AI-powered pneumonia detection from chest X-rays. "
                 "Built with PyTorch & Streamlit. For educational/research purposes only."
    }
)

# ---- Initialize deterministic mode (Bug 5 fix) ----
set_deterministic_mode()

# ---- Initialize all session state keys upfront (Step 3) ----
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Diagnose"

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "resnet50"

if "inference_history" not in st.session_state:
    st.session_state.inference_history = []

# ---- Custom CSS ----
st.markdown("""
<style>
    /* Global styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #00C49A;
        --background: #0E1117;
        --card: #1A1F2E;
        --border: #2A3242;
        --danger: #FF4B4B;
        --warning: #FFB347;
        --muted: #8892A6;
        --text: #FAFAFA;
    }

    html, body, div.stApp {
        font-family: 'Inter', sans-serif;
        background-color: var(--background);
        color: var(--text);
    }

    /* Unified content width & centered layout */
    .block-container {
        max-width: 1200px !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin: 0 auto !important;
    }

    /* Typography */
    h1, .stApp h1 {
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        color: var(--text) !important;
        margin-bottom: 0.75rem !important;
        letter-spacing: -0.025em;
    }

    h2, .stApp h2 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: var(--text) !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.5rem;
    }

    h3, .stApp h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: var(--text) !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.5rem !important;
    }

    p, .stApp p {
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        color: var(--muted) !important;
    }

    .stApp caption, .muted-text {
        font-size: 0.8rem !important;
        color: var(--muted) !important;
    }

    /* Premium card styling */
    .premium-card {
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    
    .premium-card:hover {
        border-color: rgba(0, 196, 154, 0.4) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
    }

    /* Hero header */
    .main-header {
        background: linear-gradient(135deg, #141824 0%, #1A1F2E 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(0,196,154,0.06) 0%, transparent 50%),
                    radial-gradient(circle at 70% 50%, rgba(255,75,75,0.03) 0%, transparent 50%);
    }
    .main-header h1 {
        color: var(--text);
        font-size: 2rem !important;
        margin: 0 !important;
    }
    .main-header p {
        color: var(--muted);
        font-size: 0.95rem !important;
        margin: 6px 0 0 0 !important;
    }
    .accent { color: var(--primary); }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0E1117 !important;
        border-right: 1px solid var(--border) !important;
    }

    /* Metric cards styling */
    [data-testid="stMetric"] {
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        margin-top: 4px !important;
    }

    /* Primary and Secondary Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary), #00A080) !important;
        color: #0E1117 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #00D4A4, #00B090) !important;
        box-shadow: 0 4px 12px rgba(0,196,154,0.2) !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: var(--primary) !important;
        color: var(--primary) !important;
        background-color: rgba(0,196,154,0.03) !important;
    }

    /* File uploader refactoring */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border) !important;
        border-radius: 12px !important;
        background-color: var(--card) !important;
        padding: 24px !important;
        transition: all 0.2s ease !important;
        text-align: center !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary) !important;
        background-color: #1e2536 !important;
    }
    [data-testid="stFileUploader"] section {
        padding: 0 !important;
    }
    /* Fix overlap and styling of uploader button */
    [data-testid="stFileUploader"] button {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        margin-top: 12px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploader"] button:hover {
        border-color: var(--primary) !important;
        color: var(--primary) !important;
        background-color: rgba(0, 196, 154, 0.05) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        background-color: #141822 !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        padding: 8px 18px !important;
        font-weight: 500 !important;
        color: var(--muted) !important;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--card) !important;
        color: var(--primary) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }

    /* Progress bars styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary), #00E5B5) !important;
        border-radius: 6px !important;
        height: 8px !important;
    }
    .stProgress div {
        font-size: 0.85rem !important;
        color: var(--muted) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        color: var(--text) !important;
    }
    .streamlit-expanderContent {
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        background-color: #151a26 !important;
        padding: 16px !important;
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* Download buttons */
    .stDownloadButton > button {
        border: 1px solid var(--primary) !important;
        color: var(--primary) !important;
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        background-color: rgba(0, 196, 154, 0.08) !important;
        box-shadow: 0 4px 12px rgba(0,196,154,0.1) !important;
    }

    /* Selectbox styling */
    div[data-baseweb="select"] > div {
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }

    /* Disclaimer banner */
    .disclaimer {
        background-color: rgba(255, 75, 75, 0.05) !important;
        border: 1px solid rgba(255, 75, 75, 0.15) !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        font-size: 0.78rem !important;
        color: var(--muted) !important;
        text-align: center !important;
        margin-top: 40px !important;
    }
</style>
""", unsafe_allow_html=True)


# ---- Sidebar ----
with st.sidebar:
    # Logo
    st.markdown(get_logo_html(), unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#888; font-size:0.8rem; margin-top:-8px;'>"
        "AI-Powered Chest X-Ray Analysis</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # F7: Model Selector
    st.markdown("### 🧠 Model Selection")
    model_options = {
        "resnet50":       "ResNet-50 (Recommended)",
        "efficientnetb0": "EfficientNet-B0 (Lightweight)",
        "vgg16":          "VGG-16 (Classic)",
    }
    selected_model = st.selectbox(
        "Choose CNN Architecture",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        key="selected_model",
        help="Select the pre-trained model for inference. ResNet-50 offers the best balance of accuracy and speed."
    )

    st.markdown("---")

    # Navigation — using native st.sidebar.button() for instant rendering
    st.markdown("### 🧭 Navigation")

    nav_options = {
        "🔬 Diagnose":     "Diagnose",
        "🔄 Compare":      "Compare",
        "📂 Batch Upload": "Batch",
        "📊 Benchmarks":   "Benchmarks",
    }

    for label, key in nav_options.items():
        is_active = st.session_state.active_tab == key
        if st.sidebar.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.active_tab = key
            st.rerun()

    # F6: Session History
    render_history_sidebar()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#555; font-size:0.7rem;'>"
        "PneumoVision v1.0<br>"
        "Research & Educational Use Only<br>"
        "© 2024 PneumoVision Team"
        "</div>",
        unsafe_allow_html=True
    )


# ---- Main Content ----
# Header
st.markdown("""
<div class="main-header">
    <h1>🫁 Pneumo<span class="accent">Vision</span></h1>
    <p>AI-powered pneumonia detection from chest X-rays using deep learning</p>
</div>
""", unsafe_allow_html=True)

# ---- Session-state driven routing (no st.tabs) ----
from pages import single_inference, comparison, batch_upload, benchmarks

active = st.session_state.get("active_tab", "Diagnose")

# Render a subtle tab indicator bar at the top of main content
tab_labels = {
    "Diagnose":   "🔬 Diagnose",
    "Compare":    "🔄 Compare",
    "Batch":      "📂 Batch Upload",
    "Benchmarks": "📊 Benchmarks",
}

tab_html = "<div style='display:flex; gap:8px; margin-bottom:24px; border-bottom:1px solid #333; padding-bottom:8px;'>"
for key, label in tab_labels.items():
    if key == active:
        tab_html += (
            f"<span style='color:#00C49A; font-weight:700; font-size:15px; "
            f"border-bottom:2px solid #00C49A; padding-bottom:6px;'>{label}</span>"
        )
    else:
        tab_html += (
            f"<span style='color:#555; font-size:15px; padding-bottom:6px;'>{label}</span>"
        )
tab_html += "</div>"
st.markdown(tab_html, unsafe_allow_html=True)

# Route to correct page
if active == "Diagnose":
    single_inference.render()
elif active == "Compare":
    comparison.render()
elif active == "Batch":
    batch_upload.render()
elif active == "Benchmarks":
    benchmarks.render()

# ---- Disclaimer Footer ----
st.markdown("""
<div class="disclaimer">
    ⚕️ <b>Medical Disclaimer:</b> PneumoVision is an AI research tool for educational purposes only.
    It is <b>NOT</b> a certified medical device and should <b>NOT</b> be used for clinical diagnosis.
    All results must be reviewed by a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)

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

# ---- Custom CSS ----
st.markdown("""
<style>
    /* Global styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 50%, #0E1117 100%);
        border: 1px solid #2A2F3E;
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(0,196,154,0.08) 0%, transparent 50%),
                    radial-gradient(circle at 70% 50%, rgba(255,75,75,0.05) 0%, transparent 50%);
        animation: pulse-glow 8s ease-in-out infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    .main-header h1 {
        color: #FAFAFA;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        position: relative;
    }
    .main-header p {
        color: #888;
        font-size: 1rem;
        margin: 8px 0 0 0;
        position: relative;
    }
    .accent { color: #00C49A; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #141822 100%);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1A1F2E;
        border: 1px solid #2A2F3E;
        border-radius: 12px;
        padding: 16px;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00C49A, #00A080);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #00D4A4, #00B090);
        box-shadow: 0 4px 20px rgba(0,196,154,0.3);
        transform: translateY(-1px);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2A2F3E;
        border-radius: 12px;
        padding: 8px;
        transition: border-color 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #00C49A;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #1A1F2E;
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #00C49A22;
    }

    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00C49A, #00D4A4);
        border-radius: 4px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #1A1F2E;
        border-radius: 8px;
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #2A2F3E;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Download buttons */
    .stDownloadButton > button {
        border: 1px solid #00C49A;
        color: #00C49A;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stDownloadButton > button:hover {
        background: #00C49A22;
    }

    /* Disclaimer banner */
    .disclaimer {
        background: #FF4B4B11;
        border: 1px solid #FF4B4B33;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.75rem;
        color: #888;
        text-align: center;
        margin-top: 32px;
    }

    /* Hide default header decorations */
    header[data-testid="stHeader"] {
        background: transparent;
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

    # Navigation
    st.markdown("### 🧭 Navigation")
    page_options = {
        "diagnose":    "🫁 Diagnose (Single)",
        "compare":     "🔄 Compare (Side-by-Side)",
        "batch":       "📦 Batch Upload",
        "benchmarks":  "📊 Model Benchmarks",
    }
    selected_page = st.radio(
        "Select Page",
        options=list(page_options.keys()),
        format_func=lambda x: page_options[x],
        key="selected_page",
        label_visibility="collapsed"
    )

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

# Route to selected page
if selected_page == "diagnose":
    from pages.single_inference import render
    render()
elif selected_page == "compare":
    from pages.comparison import render
    render()
elif selected_page == "batch":
    from pages.batch_upload import render
    render()
elif selected_page == "benchmarks":
    from pages.benchmarks import render
    render()

# ---- Disclaimer Footer ----
st.markdown("""
<div class="disclaimer">
    ⚕️ <b>Medical Disclaimer:</b> PneumoVision is an AI research tool for educational purposes only.
    It is <b>NOT</b> a certified medical device and should <b>NOT</b> be used for clinical diagnosis.
    All results must be reviewed by a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)

# 🫁 PneumoVision — AI-Powered Pneumonia Detection

A production-grade pneumonia detection system from chest X-rays using deep learning.

## Features

| # | Feature | Description |
|---|---------|-------------|
| F1 | 3-Class Detection | Normal, Bacterial Pneumonia, Viral Pneumonia |
| F2 | Grad-CAM Heatmaps | Explainability overlays showing model attention areas |
| F3 | Confidence + Uncertainty | MC Dropout-based uncertainty estimation |
| F4 | Side-by-Side Comparison | Compare two X-rays with parallel analysis |
| F5 | PDF Diagnostic Report | Professional one-page report with images and findings |
| F6 | Session History | Track all inferences with CSV export |
| F7 | Multi-Model Selector | Switch between ResNet-50, EfficientNet-B0, VGG-16 |
| F8 | Batch Upload | Process multiple X-rays simultaneously |
| F9 | Model Benchmarks | Performance metrics, confusion matrices, ROC curves |
| F10 | Severity Scoring | Mild / Moderate / Severe based on Grad-CAM activation |

## Project Structure

```
PneumoVision/
├── PneumoVision_Training.ipynb    # Google Colab training notebook
└── streamlit_app/
    ├── app.py                     # Main Streamlit entry point
    ├── requirements.txt
    ├── .streamlit/config.toml     # Theme configuration
    ├── utils/
    │   ├── model_loader.py        # Model loading + caching
    │   ├── gradcam.py             # Grad-CAM visualization
    │   ├── uncertainty.py         # MC Dropout uncertainty
    │   ├── severity.py            # Severity scoring
    │   ├── report_generator.py    # PDF report generation
    │   └── history.py             # Session history management
    ├── pages/
    │   ├── single_inference.py    # Single X-Ray diagnosis
    │   ├── comparison.py          # Side-by-side comparison
    │   ├── batch_upload.py        # Batch processing
    │   └── benchmarks.py          # Model metrics dashboard
    ├── assets/
    │   └── logo_placeholder.py    # SVG logo generator
    ├── models/                    # Place .pth files here
    └── metrics/                   # Place JSON + PNG metrics here
```

## Quick Start

### Step 1: Train Models (Google Colab)

1. Open `PneumoVision_Training.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Upload your `kaggle.json` API key when prompted
4. Run all cells sequentially
5. Download the exported ZIP files

### Step 2: Set Up Streamlit App

```bash
# Extract model weights
unzip pneumovision_export.zip -d streamlit_app/models/

# Extract metrics
unzip pneumovision_metrics.zip -d streamlit_app/metrics/

# Install dependencies
cd streamlit_app
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Models

| Model | Parameters | Architecture |
|-------|-----------|-------------|
| ResNet-50 | ~25.6M | Deep residual network with skip connections |
| EfficientNet-B0 | ~5.3M | Compound-scaled mobile-optimized CNN |
| VGG-16 | ~138M | Sequential deep convolutional network |

All models use:
- ImageNet pretrained weights with transfer learning
- Two-phase training: head-only → full fine-tuning
- CosineAnnealing LR scheduler
- Early stopping (patience=5)
- Weighted CrossEntropy loss for class imbalance

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney.

Original: 2 classes (Normal, Pneumonia)
Remapped: 3 classes (Normal, Bacterial, Viral) using filename prefixes.

## ⚕️ Disclaimer

PneumoVision is an AI research tool for **educational purposes only**. It is **NOT** a certified medical device and should **NOT** be used for clinical diagnosis. All results must be reviewed by a qualified healthcare professional.

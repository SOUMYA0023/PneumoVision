"""
PDF diagnostic report generator using ReportLab.
Produces a professional, one-page report with X-ray, heatmap, and findings.
"""

import io
import numpy as np
from datetime import datetime
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


def pil_to_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


def generate_pdf_report(
    original_image: Image.Image,
    heatmap_image: Image.Image,
    prediction: str,
    confidence: float,
    uncertainty_level: str,
    severity: dict | None,
    model_name: str,
    inference_time_ms: float,
    filename: str = "X-Ray"
) -> bytes:
    """
    Generate a PDF diagnostic report and return as bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm
    )

    styles = getSampleStyleSheet()
    story = []

    # ---- Header ----
    title_style = ParagraphStyle(
        "ReportTitle", fontSize=20, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#00C49A"), alignment=TA_CENTER, spaceAfter=4
    )
    sub_style = ParagraphStyle(
        "ReportSub", fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#888888"), alignment=TA_CENTER, spaceAfter=12
    )

    story.append(Paragraph("PneumoVision Diagnostic Report", title_style))
    story.append(Paragraph("AI-Assisted Chest X-Ray Analysis", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#333333")))
    story.append(Spacer(1, 0.4 * cm))

    # ---- Meta info table ----
    now = datetime.now().strftime("%B %d, %Y  %H:%M:%S")
    meta_data = [
        ["Date / Time", now],
        ["File Analyzed", filename],
        ["Model Used", model_name.upper()],
        ["Inference Time", f"{inference_time_ms:.1f} ms"],
    ]
    meta_table = Table(meta_data, colWidths=[4 * cm, 12 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#222222")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F0F0")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#F0F0F0"), colors.HexColor("#E8E8E8")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5 * cm))

    # ---- Images ----
    img_buf_orig = io.BytesIO(pil_to_bytes(original_image.resize((300, 300))))
    img_buf_heat = io.BytesIO(pil_to_bytes(heatmap_image.resize((300, 300))))

    img_orig = RLImage(img_buf_orig, width=7 * cm, height=7 * cm)
    img_heat = RLImage(img_buf_heat, width=7 * cm, height=7 * cm)

    label_style = ParagraphStyle(
        "ImgLabel", fontSize=9, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#555555"), alignment=TA_CENTER
    )
    img_table = Table(
        [[img_orig, img_heat],
         [Paragraph("Original X-Ray", label_style), Paragraph("Grad-CAM Heatmap", label_style)]],
        colWidths=[8 * cm, 8 * cm]
    )
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CCCCCC")),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 0.5 * cm))

    # ---- Findings ----
    finding_color = {
        "NORMAL":    "#00C49A",
        "BACTERIAL": "#FF4B4B",
        "VIRAL":     "#FF4B4B",
    }.get(prediction, "#222222")

    finding_style = ParagraphStyle(
        "Finding", fontSize=16, fontName="Helvetica-Bold",
        textColor=colors.HexColor(finding_color), alignment=TA_CENTER, spaceAfter=4
    )
    story.append(Paragraph(f"Prediction: {prediction}", finding_style))

    findings_data = [
        ["Confidence Score", f"{confidence * 100:.2f}%"],
        ["Uncertainty Level", uncertainty_level],
    ]
    if severity:
        findings_data.append(["Severity Assessment", severity["level"]])
        findings_data.append(["Activation Area", f"{severity['percentage']:.1f}% of lung region"])

    findings_table = Table(findings_data, colWidths=[5 * cm, 11 * cm])
    findings_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#222222")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#F0F0F0"), colors.HexColor("#E8E8E8")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(findings_table)
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))

    # ---- Disclaimer ----
    disclaimer_style = ParagraphStyle(
        "Disclaimer", fontSize=8, fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#999999"), alignment=TA_JUSTIFY, spaceAfter=0
    )
    disclaimer = (
        "DISCLAIMER: This report is generated by an AI model for research and educational purposes only. "
        "It is NOT a substitute for professional medical diagnosis, advice, or treatment. "
        "All findings must be reviewed and confirmed by a qualified radiologist or physician. "
        "PneumoVision does not assume any clinical liability for decisions made based on this output."
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(disclaimer, disclaimer_style))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

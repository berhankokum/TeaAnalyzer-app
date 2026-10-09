"""
PDF Rapor Üretimi
=================
Tek ve toplu analiz için çiftçi dostu PDF raporları.
"""

import io
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def _register_turkish_fonts():
    """Windows Arial fontunu Türkçe karakterler için kaydet."""
    font_candidates = [
        ("TR", r"C:\Windows\Fonts\arial.ttf"),
        ("TR-Bold", r"C:\Windows\Fonts\arialbd.ttf"),
        ("TR-Italic", r"C:\Windows\Fonts\ariali.ttf"),
    ]
    registered = {}
    for name, path in font_candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                registered[name] = True
            except Exception:
                registered[name] = False
        else:
            registered[name] = False

    # Bold/italic ailesini bağla (yoksa normale düşer)
    if registered.get("TR"):
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily(
            "TR",
            normal="TR",
            bold="TR-Bold" if registered.get("TR-Bold") else "TR",
            italic="TR-Italic" if registered.get("TR-Italic") else "TR",
            boldItalic="TR-Bold" if registered.get("TR-Bold") else "TR",
        )
    return registered.get("TR", False)


# Modül yüklenince fontu kaydet
_TR_FONT_OK = _register_turkish_fonts()
FONT_NORMAL = "TR" if _TR_FONT_OK else "Helvetica"
FONT_BOLD = "TR-Bold" if _TR_FONT_OK else "Helvetica-Bold"

# Tema renkleri
GREEN = colors.HexColor("#4a7c59")
DARK_GREEN = colors.HexColor("#2e5d3a")
LIGHT_GREEN = colors.HexColor("#eef3ee")
GRAY = colors.HexColor("#666666")


def _health_color(score):
    """Sağlık skoruna göre renk."""
    if score >= 70:
        return colors.HexColor("#2e7d32")
    elif score >= 50:
        return colors.HexColor("#f9a825")
    return colors.HexColor("#d84315")


def _health_text(score):
    if score >= 70:
        return "İyi durumda"
    elif score >= 50:
        return "Orta düzeyde"
    return "Dikkat gerekli"


def _numpy_to_reportlab_image(img_rgb, width_cm=8):
    """NumPy RGB array'i ReportLab Image nesnesine çevir."""
    # BGR'ye çevir (cv2 encode için), PNG olarak belleğe yaz
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        return None
    bio = io.BytesIO(buffer.tobytes())
    h, w = img_rgb.shape[:2]
    aspect = h / w
    width = width_cm * cm
    height = width * aspect
    return RLImage(bio, width=width, height=height)


def _styles():
    """Ortak paragraf stilleri."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "TitleTR", fontName=FONT_BOLD, fontSize=20,
        textColor=DARK_GREEN, alignment=TA_CENTER, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "SubtitleTR", fontName=FONT_NORMAL, fontSize=11,
        textColor=GRAY, alignment=TA_CENTER, spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        "SectionTR", fontName=FONT_BOLD, fontSize=13,
        textColor=GREEN, spaceBefore=14, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        "BodyTR", fontName=FONT_NORMAL, fontSize=10,
        textColor=colors.black, alignment=TA_LEFT, spaceAfter=4,
    ))
    return styles


# ============================================================
# TEK GÖRÜNTÜ RAPORU
# ============================================================
def generate_single_report(result, output_path, garden_name="Çay Bahçesi"):
    """Tek görüntü analizi için PDF rapor."""
    styles = _styles()
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
        leftMargin=2*cm, rightMargin=2*cm,
    )
    elements = []

    # Başlık
    elements.append(Paragraph("ÇAY BAHÇESİ SAĞLIK RAPORU", styles["TitleTR"]))
    elements.append(Paragraph(
        f"{garden_name} &nbsp;•&nbsp; {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        styles["SubtitleTR"]
    ))
    
    # Renk tonu özet kutusu
    hue_data = [[
        Paragraph("YAPRAK RENK TONU DAĞILIMI",
                  ParagraphStyle("ht", fontName=FONT_NORMAL, fontSize=10,
                                 textColor=GRAY, alignment=TA_CENTER)),
    ], [
        Table([[
            Paragraph(f"<font color='#b8a000'>Sarımsı<br/>%{result.yellow_green_pct:.0f}</font>",
                       ParagraphStyle("c1", fontName=FONT_BOLD, fontSize=11, alignment=TA_CENTER)),
            Paragraph(f"<font color='#6aab3a'>Açık Y.<br/>%{result.light_green_pct:.0f}</font>",
                       ParagraphStyle("c2", fontName=FONT_BOLD, fontSize=11, alignment=TA_CENTER)),
            Paragraph(f"<font color='#4a7c59'>Orta Y.<br/>%{result.mid_green_pct:.0f}</font>",
                       ParagraphStyle("c3", fontName=FONT_BOLD, fontSize=11, alignment=TA_CENTER)),
            Paragraph(f"<font color='#1e4d2b'>Koyu Y.<br/>%{result.dark_green_pct:.0f}</font>",
                       ParagraphStyle("c4", fontName=FONT_BOLD, fontSize=11, alignment=TA_CENTER)),
        ]], colWidths=[3.5*cm]*4),
    ], [
        Paragraph(f"Baskın ton: {result.dominant_color}",
                  ParagraphStyle("dom", fontName=FONT_BOLD, fontSize=11,
                                 textColor=DARK_GREEN, alignment=TA_CENTER)),
    ]]
    hue_table = Table(hue_data, colWidths=[16*cm])
    hue_table.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_GREEN),
        ("BOX",           (0, 0), (-1, -1), 1.5, GREEN),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(hue_table)
    elements.append(Spacer(1, 0.5*cm))

    # Görüntüler (orijinal + segmentasyon yan yana)
    if result.original_rgb is not None and result.segmentation_rgb is not None:
        elements.append(Paragraph("Görüntü Analizi", styles["SectionTR"]))
        img_orig = _numpy_to_reportlab_image(result.original_rgb, width_cm=7.5)
        img_seg = _numpy_to_reportlab_image(result.segmentation_rgb, width_cm=7.5)
        if img_orig and img_seg:
            img_table = Table([
                [img_orig, img_seg],
                [Paragraph("<font size=9 color='#666666'>Orijinal Görüntü</font>", styles["BodyTR"]),
                 Paragraph("<font size=9 color='#666666'>Bitki Örtüsü Haritası</font>", styles["BodyTR"])],
            ], colWidths=[8*cm, 8*cm])
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            elements.append(img_table)
            elements.append(Spacer(1, 0.3*cm))

    # Detaylı sonuçlar tablosu
    elements.append(Paragraph("Detaylı Sonuçlar", styles["SectionTR"]))
    detail_data = [
        ["Ölçüm", "Değer", "Açıklama"],
        ["Yeşil Örtü Oranı", f"%{result.vegetation_pct:.1f}",
         "Görüntüdeki bitki örtüsü kapsama alanı"],
        ["Gölge Alan Oranı", f"%{result.shadow_pct:.1f}",
         "Gölge altında kalan, ölçülemeyen alan"],
        ["Diğer", f"%{result.other_pct:.1f}",
         "Toprak, yol vb. bitki olmayan alanlar"],
        ["Örtü Homojenliği", "İyi" if result.exg_std < 35 else "Değişken",
         "Bahçenin tek düze mi parçalı mı olduğu"],
    ]
    detail_table = Table(detail_data, colWidths=[5*cm, 3*cm, 8*cm])
    detail_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT_NORMAL),
        ("BACKGROUND", (0, 0), (-1, 0), GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREEN]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
    ]))
    elements.append(detail_table)
    elements.append(Spacer(1, 0.5*cm))

    # Vejetasyon indeksleri (yorumlu tablo)
    from core.index_display import index_rows

    elements.append(Paragraph("Vejetasyon İndeksleri", styles["SectionTR"]))
    idx_data = [["Gösterge", "Değer", "Normalize (%)", "Durum"]]
    idx_rows = index_rows(result)
    for row in idx_rows:
        idx_data.append([
            f"{row['name']} — {row['label']}",
            row["value_str"],
            f"%{row['norm']:.0f}",
            row["status"],
        ])

    idx_table = Table(idx_data, colWidths=[7*cm, 2.5*cm, 3*cm, 3.5*cm])
    idx_style = [
        ("FONTNAME",      (0, 0), (-1, -1), FONT_NORMAL),
        ("BACKGROUND",    (0, 0), (-1, 0), GREEN),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT_GREEN]),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("ALIGN",         (0, 0), (0, -1), "LEFT"),
        ("FONTNAME",      (3, 1), (3, -1), FONT_BOLD),
    ]
    # "Durum" hücrelerini status() rengiyle renklendir
    for i, row in enumerate(idx_rows, start=1):
        idx_style.append(("TEXTCOLOR", (3, i), (3, i), colors.HexColor(row["color"])))
    idx_table.setStyle(TableStyle(idx_style))
    elements.append(idx_table)
    elements.append(Spacer(1, 0.2*cm))
    elements.append(Paragraph(
        "<font size=8 color='#999999'><i>Not: İndeks eşikleri (Düşük/Orta/Yüksek) "
        "başlangıç değerleridir. RGB vejetasyon indeksleri aydınlatmaya duyarlı "
        "olduğundan, eşikler kendi veri setinizden kalibre edilmelidir.</i></font>",
        styles["BodyTR"]
    ))
    elements.append(Spacer(1, 0.3*cm))

    # Dosya bilgisi
    elements.append(Paragraph(
        f"<font size=8 color='#999999'>Analiz edilen dosya: {result.filename} "
        f"({result.image_width}x{result.image_height} piksel)</font>",
        styles["BodyTR"]
    ))

    # Açıklama notu
    elements.append(Spacer(1, 0.3*cm))
    elements.append(Paragraph(
        "<font size=8 color='#999999'><i>Not: Bu rapor RGB drone görüntüsünden otomatik "
        "üretilmiştir. 'Yeşil örtü' çay ve diğer yeşil bitkileri (portakal vb.) birlikte "
        "kapsar; tür ayrımı yapmaz. Sonuçlar bahçe izleme amaçlıdır.</i></font>",
        styles["BodyTR"]
    ))

    doc.build(elements)
    return output_path


# ============================================================
# TOPLU RAPOR
# ============================================================
def generate_batch_report(results, output_path, garden_name="Çay Bahçesi"):
    """Toplu analiz için özet PDF rapor."""
    styles = _styles()
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
    )
    elements = []

    # Başlık
    elements.append(Paragraph("ÇAY BAHÇESİ TOPLU ANALİZ RAPORU", styles["TitleTR"]))
    elements.append(Paragraph(
        f"{garden_name} • {len(results)} görüntü • "
        f"{__import__('datetime').datetime.now().strftime('%d.%m.%Y %H:%M')}",
        styles["SubtitleTR"]
    ))

    # Özet istatistikler
    avg_veg    = sum(r.vegetation_pct    for r in results) / len(results)
    avg_shadow = sum(r.shadow_pct        for r in results) / len(results)
    avg_exg    = sum(r.exg_mean          for r in results) / len(results)
    avg_yg     = sum(r.yellow_green_pct  for r in results) / len(results)
    avg_dg     = sum(r.dark_green_pct    for r in results) / len(results)

    elements.append(Paragraph("Genel Özet", styles["SectionTR"]))
    summary_data = [
        ["Ort. Yeşil Örtü", "Ort. Gölge", "Ort. ExG",
         "Ort. Sarımsı", "Ort. Koyu Yeşil"],
        [f"%{avg_veg:.1f}", f"%{avg_shadow:.1f}", f"{avg_exg:.0f}",
         f"%{avg_yg:.1f}", f"%{avg_dg:.1f}"],
    ]
    summary_table = Table(summary_data, colWidths=[3.5*cm]*5)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), GREEN),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME",     (0, 1), (-1, 1), FONT_BOLD),
        ("FONTSIZE",     (0, 0), (-1, 0), 9),
        ("FONTSIZE",     (0, 1), (-1, 1), 13),
        ("TEXTCOLOR",    (0, 1), (-1, 1), DARK_GREEN),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",   (0, 1), (-1, 1), LIGHT_GREEN),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.white),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.5*cm))

    # Detaylı tablo
    elements.append(Paragraph("Görüntü Bazında Sonuçlar", styles["SectionTR"]))
    table_data = [[
        "Dosya", "Çay %", "Yeşil %",
        "Sarımsı %", "Açık Y. %", "Orta Y. %", "Koyu Y. %",
        "Baskın Ton"
    ]]
    for r in results:
        table_data.append([
            r.filename,
            f"{r.tea_coverage_pct:.1f}" if r.analysis_mode == "tea_only" else "—",
            f"{r.vegetation_pct:.1f}",
            f"{r.yellow_green_pct:.1f}",
            f"{r.light_green_pct:.1f}",
            f"{r.mid_green_pct:.1f}",
            f"{r.dark_green_pct:.1f}",
            r.dominant_color,
        ])

    col_widths = [4.5*cm, 1.6*cm, 1.6*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.8*cm, 2.5*cm]
    detail_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    detail_table.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (-1, -1), FONT_NORMAL),
        ("BACKGROUND",    (0, 0), (-1, 0), GREEN),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT_GREEN]),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(detail_table)

    elements.append(Spacer(1, 0.3*cm))
    elements.append(Paragraph(
        "<font size=8 color='#999999'><i>Not: Yüzdeler YOLO ile tespit edilen "
        "çay alanı içinde hesaplanmıştır. Renk tonu dağılımı HSV uzayındaki "
        "Hue kanalından türetilmiştir.</i></font>",
        styles["BodyTR"]
    ))

    doc.build(elements)
    return output_path
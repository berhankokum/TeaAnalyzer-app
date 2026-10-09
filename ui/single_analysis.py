"""Tek görüntü analiz sekmesi."""

import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QFrame, QButtonGroup, QRadioButton, QSizePolicy,
    QMessageBox, QProgressDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect
from PyQt6.QtGui import QFont, QPixmap, QImage, QPainter, QColor

from core.analyzer import TeaAnalyzer, AnalysisResult
from core.pipeline import AnalysisPipeline
from core.index_display import INDEX_CONFIG, index_rows

def numpy_to_qpixmap(img_rgb, max_size=560):
    """NumPy RGB array'i QPixmap'e çevir, oranı koruyarak küçült."""
    h, w = img_rgb.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized = np.ascontiguousarray(resized)
    qimg = QImage(resized.data, new_w, new_h, 3 * new_w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class _IndexBar(QWidget):
    """
    Renk bölgeli yatay dolu çubuk (vejetasyon indeksi gösterimi).
    Dolu kısmın uzunluğu normalize yüzdeye, rengi status() durumuna göre belirlenir.
    """

    def __init__(self):
        super().__init__()
        self._pct = 0.0
        self._color = QColor("#cccccc")
        self.setMinimumHeight(14)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, pct, color_hex):
        self._pct = max(0.0, min(100.0, float(pct)))
        self._color = QColor(color_hex)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        radius = rect.height() / 2.0

        # Boş kısım (açık gri zemin)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#e0e6e0"))
        painter.drawRoundedRect(rect, radius, radius)

        # Dolu kısım (durum renginde)
        fill_w = int(rect.width() * self._pct / 100.0)
        if fill_w > 0:
            fill_rect = QRect(rect.left(), rect.top(), fill_w, rect.height())
            painter.setBrush(self._color)
            painter.drawRoundedRect(fill_rect, radius, radius)
        painter.end()


class AnalysisWorker(QThread):
    """Analizi arka planda yapar (arayüz donmasın)."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, pipeline, image_path):
        super().__init__()
        self.pipeline = pipeline
        self.image_path = image_path

    def run(self):
        try:
            result = self.pipeline.analyze(self.image_path, generate_visuals=True)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class SingleAnalysisTab(QWidget):
    def __init__(self, rf_model_path, yolo_model_path=None):
        super().__init__()
        self.pipeline = AnalysisPipeline(rf_model_path, yolo_model_path)
        self.current_result = None
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        # ---- ÜST BAR: Görüntü seç butonu + dosya adı ----
        top_bar = QHBoxLayout()
        self.btn_select = QPushButton("📷  Görüntü Seç")
        self.btn_select.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.btn_select.setMinimumHeight(48)
        self.btn_select.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_select.clicked.connect(self.select_image)
        self.btn_select.setStyleSheet("""
            QPushButton {
                background-color: #4a7c59; color: white;
                border: none; border-radius: 8px; padding: 0 24px;
            }
            QPushButton:hover { background-color: #5a9169; }
            QPushButton:pressed { background-color: #3a6249; }
        """)
        top_bar.addWidget(self.btn_select)

        self.lbl_filename = QLabel("Henüz görüntü seçilmedi")
        self.lbl_filename.setFont(QFont("Segoe UI", 11))
        self.lbl_filename.setStyleSheet("color: #666;")
        top_bar.addWidget(self.lbl_filename)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # ---- ANA İÇERİK: Sol görüntü, sağ sonuçlar ----
        content = QHBoxLayout()
        content.setSpacing(15)

        # SOL: Görüntü gösterimi
        left_panel = QVBoxLayout()
        self.image_label = QLabel("Görüntü buraya gelecek")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(560, 480)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #fafafa;
                border: 2px dashed #ccc;
                border-radius: 8px;
                color: #999;
                font-size: 14px;
            }
        """)
        left_panel.addWidget(self.image_label)

        # Görüntü modu seçimi (orijinal / segmentasyon / overlay)
        mode_bar = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.rb_original   = QRadioButton("Orijinal")
        self.rb_tea        = QRadioButton("Çay Alanı")
        self.rb_yellow     = QRadioButton("Sarımsı")
        self.rb_lightgreen = QRadioButton("Açık Yeşil")
        self.rb_midgreen   = QRadioButton("Orta Yeşil")
        self.rb_darkgreen  = QRadioButton("Koyu Yeşil")

        self.rb_original.setChecked(True)

        btn_colors = {
            self.rb_original:   "#555555",
            self.rb_tea:        "#4a7c59",
            self.rb_yellow:     "#b8a000",
            self.rb_lightgreen: "#6aab3a",
            self.rb_midgreen:   "#4a7c59",
            self.rb_darkgreen:  "#1e4d2b",
        }

        for rb, color in btn_colors.items():
            rb.setFont(QFont("Segoe UI", 9))
            rb.setStyleSheet(f"""
                QRadioButton::indicator:checked {{
                    background-color: {color};
                    border: 2px solid {color};
                    border-radius: 6px;
                }}
                QRadioButton::indicator {{
                    width: 12px; height: 12px;
                    border-radius: 6px;
                    border: 2px solid #aaa;
                }}
            """)
            rb.toggled.connect(self._update_image_display)
            self.mode_group.addButton(rb)
            mode_bar.addWidget(rb)
        mode_bar.addStretch()
        left_panel.addLayout(mode_bar)
        content.addLayout(left_panel, stretch=3)

        # SAĞ: Sonuç panosu
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # Renk tonu kartı (sağlık skoru yerine)
        self.hue_card = self._make_hue_card()
        right_panel.addWidget(self.hue_card["frame"])

        # Sınıf yüzdesi kartları
        self.veg_card = self._make_result_card("🟢", "Yeşil Örtü", "#4a7c59")
        self.shadow_card = self._make_result_card("⬛", "Gölge Alan", "#555")
        self.other_card = self._make_result_card("⬜", "Diğer", "#aaa")
        right_panel.addWidget(self.veg_card["frame"])
        right_panel.addWidget(self.shadow_card["frame"])
        right_panel.addWidget(self.other_card["frame"])

        # Vejetasyon indeksleri kartı (ExG / GLI / VARI renk bölgeli çubuklar)
        self.index_card = self._make_index_card()
        right_panel.addWidget(self.index_card["frame"])

        # Çay alanı tespit bilgisi
        self.lbl_intensity = QLabel("")
        self.lbl_intensity.setFont(QFont("Segoe UI", 10))
        self.lbl_intensity.setStyleSheet("color: #666; padding: 4px;")
        self.lbl_intensity.setWordWrap(True)
        right_panel.addWidget(self.lbl_intensity)

        right_panel.addStretch()

        # Rapor indir butonu
        self.btn_report = QPushButton("📄  Rapor İndir")
        self.btn_report.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.btn_report.setMinimumHeight(44)
        self.btn_report.setEnabled(False)
        self.btn_report.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_report.clicked.connect(self.export_report)
        self.btn_report.setStyleSheet("""
            QPushButton {
                background-color: #2c5f8a; color: white;
                border: none; border-radius: 8px;
            }
            QPushButton:hover { background-color: #366fa0; }
            QPushButton:disabled { background-color: #ccc; color: #888; }
        """)
        right_panel.addWidget(self.btn_report)

        content.addLayout(right_panel, stretch=2)
        main_layout.addLayout(content)
        self.setLayout(main_layout)

    def _make_hue_card(self):
        """Yeşil-sarı renk tonu dağılım kartı."""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #f0f4f0;
                border: 2px solid #4a7c59;
                border-radius: 12px;
            }
        """)
        layout = QVBoxLayout()

        title = QLabel("YAPRAK RENK TONU DAĞILIMI")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        title.setStyleSheet("color: #4a7c59; border: none;")
        layout.addWidget(title)

        # 4 renk çubuğu
        bars = {}
        colors_map = {
            "yellow_green": ("#c8b400", "Sarımsı yeşil"),
            "light_green":  ("#7bc142", "Açık yeşil"),
            "mid_green":    ("#4a7c59", "Orta yeşil"),
            "dark_green":   ("#1e4d2b", "Koyu yeşil"),
        }

        for key, (color, label) in colors_map.items():
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFont(QFont("Segoe UI", 9))
            lbl.setStyleSheet("border: none; color: #333;")
            lbl.setFixedWidth(100)

            bar = QLabel("█")
            bar.setStyleSheet(f"border: none; color: {color};")
            bar.setFont(QFont("Segoe UI", 9))

            pct_lbl = QLabel("—")
            pct_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            pct_lbl.setStyleSheet(f"border: none; color: {color};")
            pct_lbl.setFixedWidth(45)
            pct_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            row.addWidget(lbl)
            row.addWidget(bar, stretch=1)
            row.addWidget(pct_lbl)

            container = QWidget()
            container.setLayout(row)
            layout.addWidget(container)
            bars[key] = {"bar": bar, "pct": pct_lbl, "color": color}

        self.lbl_dominant = QLabel("")
        self.lbl_dominant.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_dominant.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_dominant.setStyleSheet("border: none; color: #333; padding: 4px;")
        layout.addWidget(self.lbl_dominant)

        frame.setLayout(layout)
        return {"frame": frame, "bars": bars}

    def _make_index_card(self):
        """Vejetasyon indeksleri kartı (ExG / GLI / VARI)."""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #f0f4f0;
                border: 2px solid #4a7c59;
                border-radius: 12px;
            }
        """)
        layout = QVBoxLayout()

        title = QLabel("VEJETASYON İNDEKSLERİ")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        title.setStyleSheet("color: #4a7c59; border: none;")
        layout.addWidget(title)

        rows = {}
        for name in ("ExG", "GLI", "VARI"):
            cfg = INDEX_CONFIG[name]

            container = QWidget()
            container.setStyleSheet("border: none;")
            vbox = QVBoxLayout()
            vbox.setContentsMargins(2, 2, 2, 4)
            vbox.setSpacing(3)

            # Üst satır: indeks adı + açıklama (sol), durum etiketi (sağ)
            top = QHBoxLayout()
            name_lbl = QLabel(f"{name} — {cfg['label']}")
            name_lbl.setFont(QFont("Segoe UI", 9))
            name_lbl.setStyleSheet("border: none; color: #333;")
            status_lbl = QLabel("—")
            status_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            status_lbl.setStyleSheet("border: none; color: #666;")
            status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            top.addWidget(name_lbl)
            top.addStretch()
            top.addWidget(status_lbl)
            vbox.addLayout(top)

            # Alt satır: dolu çubuk (sol), ham değer (sağ)
            bottom = QHBoxLayout()
            bar = _IndexBar()
            value_lbl = QLabel("—")
            value_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            value_lbl.setStyleSheet("border: none; color: #333;")
            value_lbl.setFixedWidth(55)
            value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            bottom.addWidget(bar, stretch=1)
            bottom.addWidget(value_lbl)
            vbox.addLayout(bottom)

            container.setLayout(vbox)
            layout.addWidget(container)
            rows[name] = {"bar": bar, "value": value_lbl, "status": status_lbl}

        frame.setLayout(layout)
        return {"frame": frame, "rows": rows}

    def _make_result_card(self, icon, title, color):
        """Sınıf yüzdesi kartı."""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border: 1px solid #ddd;
                border-left: 5px solid {color};
                border-radius: 6px;
            }}
        """)
        layout = QHBoxLayout()
        lbl_icon = QLabel(icon)
        lbl_icon.setFont(QFont("Segoe UI", 18))
        lbl_icon.setStyleSheet("border: none;")

        lbl_title = QLabel(title)
        lbl_title.setFont(QFont("Segoe UI", 11))
        lbl_title.setStyleSheet("border: none; color: #444;")

        lbl_value = QLabel("—")
        lbl_value.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        lbl_value.setStyleSheet(f"border: none; color: {color};")
        lbl_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(lbl_icon)
        layout.addWidget(lbl_title)
        layout.addStretch()
        layout.addWidget(lbl_value)
        frame.setLayout(layout)
        return {"frame": frame, "value": lbl_value}

    # --------------------------------------------------------
    # GÖRÜNTÜ SEÇME VE ANALİZ
    # --------------------------------------------------------
    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Çay Bahçesi Görüntüsü Seç", "",
            "Görüntü Dosyaları (*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG)"
        )
        if not path:
            return

        self.lbl_filename.setText(f"Dosya: {Path(path).name}")

        # Progress dialog
        self.progress = QProgressDialog("Analiz ediliyor...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setCancelButton(None)
        self.progress.show()

        # Arka planda analiz
        self.worker = AnalysisWorker(self.pipeline, path)
        self.worker.finished.connect(self._on_analysis_done)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.start()

    def _on_analysis_done(self, result):
        self.progress.close()
        self.current_result = result
        self._update_image_display()
        self._update_results(result)
        self.btn_report.setEnabled(True)

    def _on_analysis_error(self, msg):
        self.progress.close()
        QMessageBox.critical(self, "Hata", f"Analiz sırasında hata oluştu:\n{msg}")

    def _update_image_display(self):
        if self.current_result is None:
            return
        r = self.current_result

        if self.rb_original.isChecked():
            img = r.original_rgb
        elif self.rb_tea.isChecked():
            img = self._make_tea_overlay(r)
        elif self.rb_yellow.isChecked():
            img = self._make_tone_overlay(r, "yellow_green")
        elif self.rb_lightgreen.isChecked():
            img = self._make_tone_overlay(r, "light_green")
        elif self.rb_midgreen.isChecked():
            img = self._make_tone_overlay(r, "mid_green")
        elif self.rb_darkgreen.isChecked():
            img = self._make_tone_overlay(r, "dark_green")
        else:
            img = r.original_rgb

        if img is not None:
            self.image_label.setPixmap(numpy_to_qpixmap(img))

    def _update_results(self, r):
        # Yüzde kartları
        self.veg_card["value"].setText(f"%{r.vegetation_pct:.1f}")
        self.shadow_card["value"].setText(f"%{r.shadow_pct:.1f}")
        self.other_card["value"].setText(f"%{r.other_pct:.1f}")

        # Renk tonu çubukları
        bars = self.hue_card["bars"]
        values = {
            "yellow_green": r.yellow_green_pct,
            "light_green":  r.light_green_pct,
            "mid_green":    r.mid_green_pct,
            "dark_green":   r.dark_green_pct,
        }
        for key, pct in values.items():
            b = bars[key]
            # Çubuk uzunluğu: en fazla 20 █ karakteri
            bar_len = max(1, int(pct / 5))
            b["bar"].setText("█" * bar_len)
            b["pct"].setText(f"%{pct:.0f}")

        self.lbl_dominant.setText(f"Baskın: {r.dominant_color}")

        # Vejetasyon indeksleri çubukları (ExG / GLI / VARI)
        for row in index_rows(r):
            w = self.index_card["rows"][row["name"]]
            w["bar"].set_value(row["norm"], row["color"])
            w["value"].setText(row["value_str"])
            w["status"].setText(row["status"])
            w["status"].setStyleSheet(f"border: none; color: {row['color']};")

        # Çay alanı bilgisi
        if r.analysis_mode == "tea_only":
            mode_text = (f"✓ Çay alanı tespit edildi "
                         f"(görüntünün %{r.tea_coverage_pct:.0f}'i)\n"
                         f"Yüzdeler çay alanına göre hesaplandı.")
        else:
            mode_text = "⚠ Çay tespit edilemedi, tüm görüntü analiz edildi."

        self.lbl_intensity.setText(mode_text)

    def export_report(self):
        if self.current_result is None:
            return
        from core.report import generate_single_report

        default_name = f"{Path(self.current_result.filename).stem}_rapor.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Raporu Kaydet", default_name, "PDF Dosyası (*.pdf)"
        )
        if not path:
            return
        try:
            generate_single_report(self.current_result, path)
            QMessageBox.information(self, "Başarılı", f"Rapor kaydedildi:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Rapor oluşturulamadı:\n{e}")

    def _make_tea_overlay(self, r):
        """Çay maskesini orijinal görüntü üzerine yeşil vurgu olarak bindir."""
        if r.original_rgb is None:
            return None
        if r.tea_mask is None:
            return r.original_rgb  # YOLO çalışmadıysa orijinali göster
        img = r.original_rgb.copy()
        green = np.zeros_like(img)
        green[..., 1] = 255
        # Çay alanı: yeşil vurgu, dışı: koyulaştırılmış
        overlay = np.where(
            r.tea_mask[..., None],
            cv2.addWeighted(img, 0.6, green, 0.4, 0),
            img // 3
        ).astype(np.uint8)
        return overlay
    
    def _make_tone_overlay(self, r, tone_key):
        """
        Belirli bir renk tonuna ait pikselleri vurgula.
        tone_key: "yellow_green" | "light_green" | "mid_green" | "dark_green"
        """
        if r.original_rgb is None:
            return None

        img = r.original_rgb
        h, w = img.shape[:2]

        # HSV hesapla
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        H = hsv[..., 0].astype(np.float32)
        V = hsv[..., 2].astype(np.float32)

        # Ton aralıkları ve vurgu renkleri
        tone_config = {
            "yellow_green": {
                "h_range": (22, 38),
                "highlight": np.array([220, 185, 0], dtype=np.uint8),   # altın sarı
                "label": "Sarımsı yeşil"
            },
            "light_green": {
                "h_range": (38, 52),
                "highlight": np.array([120, 200, 60], dtype=np.uint8),  # açık yeşil
                "label": "Açık yeşil"
            },
            "mid_green": {
                "h_range": (52, 68),
                "highlight": np.array([60, 140, 80], dtype=np.uint8),   # orta yeşil
                "label": "Orta yeşil"
            },
            "dark_green": {
                "h_range": (68, 90),
                "highlight": np.array([20, 80, 40], dtype=np.uint8),    # koyu yeşil
                "label": "Koyu yeşil"
            },
        }

        cfg = tone_config[tone_key]
        h_lo, h_hi = cfg["h_range"]
        highlight_color = cfg["highlight"]

        # Ton maskesi: hem H aralığında hem yeterince parlak
        bright = V > 40
        tone_mask = ((H >= h_lo) & (H < h_hi) & bright)

        # Eğer çay maskesi varsa sadece çay alanında uygula
        if r.tea_mask is not None:
            tone_mask = tone_mask & r.tea_mask

        # Görsel: vurgulanan = renkli, diğerleri = koyulaştırılmış
        dim = (img // 4).astype(np.uint8)  # karanlık arka plan
        highlight_layer = np.full_like(img, highlight_color)
        result_img = np.where(tone_mask[..., None], highlight_layer, dim).astype(np.uint8)

        # Orijinali hafifçe karıştır (tamamen düz renk değil, doku görünsün)
        blended = cv2.addWeighted(result_img, 0.7, img // 2, 0.3, 0)
        return blended
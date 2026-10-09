"""Yaprak tabanlı analiz penceresi."""

import cv2
import numpy as np
from pathlib import Path
from collections import Counter
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QFileDialog, QMessageBox,
    QProgressDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QButtonGroup, QRadioButton, QSizePolicy,
    QComboBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage, QColor

from core.leaf_detector import LeafDetector, LeafAnalysisResult, _is_healthy


# ---------------------------------------------------------------------------
# Yardımcı
# ---------------------------------------------------------------------------

def _to_qpixmap(img_rgb: np.ndarray, max_size: int = 560) -> QPixmap:
    h, w = img_rgb.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    resized = np.ascontiguousarray(cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA))
    qimg = QImage(resized.data, nw, nh, 3 * nw, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Arka plan iş parçacığı
# ---------------------------------------------------------------------------

class _LeafWorker(QThread):
    """Birden fazla görüntüyü sırayla analiz eder."""
    progress = pyqtSignal(int, int)        # (tamamlanan, toplam)
    finished = pyqtSignal(object)          # List[LeafAnalysisResult]
    error = pyqtSignal(str)

    def __init__(self, detector: LeafDetector, image_paths):
        super().__init__()
        self.detector = detector
        self.image_paths = list(image_paths)

    def run(self):
        try:
            results = []
            total = len(self.image_paths)
            for i, path in enumerate(self.image_paths):
                results.append(self.detector.analyze(path))
                self.progress.emit(i + 1, total)
            self.finished.emit(results)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")


def _aggregate(results):
    """Tüm fotoğraflardaki yaprakları toplayıp dağılım çıkarır."""
    health_counter = Counter()
    age_counter = Counter()
    total = 0
    healthy = 0
    for res in results:
        for lf in res.leaves:
            total += 1
            health_counter[lf.health_label] += 1
            age_counter[lf.age_label] += 1
            if _is_healthy(lf.health_label):
                healthy += 1
    return total, healthy, health_counter, age_counter


# ---------------------------------------------------------------------------
# Ana pencere
# ---------------------------------------------------------------------------

class LeafAnalysisWindow(QMainWindow):
    back_requested = pyqtSignal()

    def __init__(self, models_dir: Path):
        super().__init__()
        self.models_dir = models_dir
        self.detector: LeafDetector = None
        self.results: list = []          # List[LeafAnalysisResult]
        self.paths: list = []            # seçilen dosya yolları

        self._init_detector()
        self.setWindowTitle("Çay Yaprak Analizi")
        self.setMinimumSize(1100, 750)
        self._build_ui()
        self._apply_style()

    # ------------------------------------------------------------------
    # Başlatma
    # ------------------------------------------------------------------

    def _init_detector(self):
        det = self.models_dir / "tea-leaf-detection-model.pt"
        health = self.models_dir / "best_tealeaf_model.pth"
        age = self.models_dir / "tea-leaf-age.pt"
        missing = [p.name for p in (det, health, age) if not p.exists()]
        if missing:
            print(f"[LeafAnalysis] Eksik model dosyaları: {missing}")
            return
        try:
            self.detector = LeafDetector(det, health, age)
        except Exception as e:
            print(f"[LeafAnalysis] Detector başlatılamadı: {e}")

    # ------------------------------------------------------------------
    # Arayüz
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(15, 15, 15, 15)
        root.setSpacing(12)

        # ---- Üst çubuk ----
        top_bar = QHBoxLayout()

        btn_back = QPushButton("← Ana Menü")
        btn_back.setFont(QFont("Segoe UI", 11))
        btn_back.setMinimumHeight(42)
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(self.back_requested)
        btn_back.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #4a7c59;
                border: 2px solid #4a7c59;
                border-radius: 8px;
                padding: 0 16px;
            }
            QPushButton:hover { background-color: #eef5ee; }
        """)
        top_bar.addWidget(btn_back)

        self.btn_select = QPushButton("📷  Görüntüleri Seç (en az 3)")
        self.btn_select.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.btn_select.setMinimumHeight(48)
        self.btn_select.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_select.clicked.connect(self._select_images)
        self.btn_select.setStyleSheet("""
            QPushButton {
                background-color: #4a7c59; color: white;
                border: none; border-radius: 8px; padding: 0 24px;
            }
            QPushButton:hover  { background-color: #5a9169; }
            QPushButton:pressed { background-color: #3a6249; }
        """)
        top_bar.addWidget(self.btn_select)

        self.lbl_filename = QLabel("Henüz görüntü seçilmedi (en az 3 fotoğraf yükleyin)")
        self.lbl_filename.setFont(QFont("Segoe UI", 11))
        self.lbl_filename.setStyleSheet("color: #666;")
        top_bar.addWidget(self.lbl_filename)
        top_bar.addStretch()
        root.addLayout(top_bar)

        # ---- Ana içerik ----
        content = QHBoxLayout()
        content.setSpacing(15)

        # SOL: görüntü paneli
        left_panel = QVBoxLayout()

        # Fotoğraflar arası geçiş
        self.photo_combo = QComboBox()
        self.photo_combo.setFont(QFont("Segoe UI", 10))
        self.photo_combo.setMinimumHeight(32)
        self.photo_combo.currentIndexChanged.connect(self._update_image_display)
        self.photo_combo.setEnabled(False)
        self.photo_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                color: #1b3a2a;
                border: 1px solid #ccc;
                border-radius: 6px;
                padding: 2px 8px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: #1b3a2a;
                selection-background-color: #c8e6c9;
                selection-color: #1b3a2a;
                outline: none;
            }
        """)
        left_panel.addWidget(self.photo_combo)

        self.image_label = QLabel("Görüntü buraya gelecek")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(560, 480)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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

        # Görüntü modu radyo butonları
        mode_bar = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.rb_original = QRadioButton("Orijinal")
        self.rb_annotated = QRadioButton("Tespit Sonucu")
        self.rb_original.setChecked(True)

        _rb_style = """
            QRadioButton::indicator:checked {
                background-color: #4a7c59;
                border: 2px solid #4a7c59;
                border-radius: 6px;
            }
            QRadioButton::indicator {
                width: 12px; height: 12px;
                border-radius: 6px;
                border: 2px solid #aaa;
            }
        """
        for rb in (self.rb_original, self.rb_annotated):
            rb.setFont(QFont("Segoe UI", 9))
            rb.setStyleSheet(_rb_style)
            rb.toggled.connect(self._update_image_display)
            self.mode_group.addButton(rb)
            mode_bar.addWidget(rb)
        mode_bar.addStretch()
        left_panel.addLayout(mode_bar)

        content.addLayout(left_panel, stretch=3)

        # SAĞ: sonuç paneli
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        right_panel.addWidget(self._build_summary_card())
        right_panel.addWidget(self._build_legend())
        right_panel.addWidget(self._build_disease_card())

        lbl_list = QLabel("Sınıf Dağılımı (tüm yapraklar)")
        lbl_list.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        lbl_list.setStyleSheet("color: #2e5d3a;")
        right_panel.addWidget(lbl_list)

        self.table = self._build_table()
        right_panel.addWidget(self.table)

        content.addLayout(right_panel, stretch=2)
        root.addLayout(content)

    def _build_summary_card(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #f0f4f0;
                border: 2px solid #4a7c59;
                border-radius: 12px;
            }
        """)
        layout = QVBoxLayout(frame)
        layout.setSpacing(6)
        layout.setContentsMargins(14, 12, 14, 12)

        title = QLabel("BAHÇE RAPORU")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        title.setStyleSheet("color: #4a7c59; border: none;")
        layout.addWidget(title)

        self._sum_photos = self._summary_row(layout, "Analiz edilen fotoğraf:")
        self._sum_total = self._summary_row(layout, "Toplam yaprak:")
        self._sum_healthy = self._summary_row(layout, "Sağlıklı oran:")
        self._sum_diseased = self._summary_row(layout, "Hastalıklı oran:")

        # Sözel bahçe değerlendirmesi (yaş dağılımı vb.)
        self._report_text = QLabel("")
        self._report_text.setWordWrap(True)
        self._report_text.setFont(QFont("Segoe UI", 9))
        self._report_text.setStyleSheet("border: none; color: #444; margin-top: 4px;")
        layout.addWidget(self._report_text)

        return frame

    def _summary_row(self, parent_layout, label_text: str) -> QLabel:
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFont(QFont("Segoe UI", 10))
        lbl.setStyleSheet("border: none; color: #555;")
        val = QLabel("—")
        val.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        val.setStyleSheet("border: none; color: #2e5d3a;")
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(val)
        parent_layout.addLayout(row)
        return val

    def _build_legend(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
            }
        """)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(16)

        for color, text in (("#22c55e", "Sağlıklı yaprak"), ("#dc3535", "Hastalıklı yaprak")):
            dot = QLabel("■")
            dot.setFont(QFont("Segoe UI", 12))
            dot.setStyleSheet(f"color: {color}; border: none;")
            lbl = QLabel(text)
            lbl.setFont(QFont("Segoe UI", 9))
            lbl.setStyleSheet("color: #555; border: none;")
            layout.addWidget(dot)
            layout.addWidget(lbl)

        layout.addStretch()
        return frame

    def _build_disease_card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("diseaseCard")
        frame.setStyleSheet("""
            QFrame#diseaseCard {
                background-color: white;
                border: 1px solid #e6cccc;
                border-radius: 12px;
            }
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title = QLabel("🦠  Tespit Edilen Hastalıklar")
        title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        title.setStyleSheet("color: #b00000; border: none;")
        layout.addWidget(title)

        self._disease_container = QVBoxLayout()
        self._disease_container.setSpacing(6)
        layout.addLayout(self._disease_container)

        # Başlangıç bilgisi
        self._disease_placeholder = QLabel("Analiz bekleniyor…")
        self._disease_placeholder.setFont(QFont("Segoe UI", 9))
        self._disease_placeholder.setStyleSheet("color: #888; border: none;")
        self._disease_container.addWidget(self._disease_placeholder)

        return frame

    def _build_table(self) -> QTableWidget:
        tbl = QTableWidget()
        tbl.setColumnCount(3)
        tbl.setHorizontalHeaderLabels(["Durum / Sınıf", "Yaprak Sayısı", "Yüzde"])
        hh = tbl.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        tbl.setColumnWidth(1, 110)
        tbl.setColumnWidth(2, 80)
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        tbl.setAlternatingRowColors(True)
        tbl.setFont(QFont("Segoe UI", 10))
        tbl.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 6px;
                gridline-color: #eee;
                color: #1b3a2a;
                background-color: white;
            }
            QHeaderView::section {
                background-color: #4a7c59;
                color: white;
                font-weight: bold;
                padding: 6px;
                border: none;
            }
            QTableWidget::item:alternate {
                background-color: #f5faf5;
            }
            QTableWidget::item:selected {
                background-color: #c8e6c9;
                color: #1b3a2a;
            }
        """)
        return tbl

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f0; }
            QWidget     { background-color: #f5f5f0; }
        """)

    # ------------------------------------------------------------------
    # Eylemler
    # ------------------------------------------------------------------

    def _select_images(self):
        if self.detector is None:
            QMessageBox.critical(
                self, "Model Hatası",
                "Yaprak analiz modelleri yüklenemedi.\n"
                "Lütfen models/ klasöründeki dosyaları kontrol edin:\n"
                "  • tea-leaf-detection-model.pt\n"
                "  • best_tealeaf_model.pth\n"
                "  • tea-leaf-age.pt"
            )
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self, "Çay Yaprağı Görüntülerini Seç (en az 3)", "",
            "Görüntü Dosyaları (*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG)"
        )
        if not paths:
            return

        if len(paths) < 3:
            QMessageBox.warning(
                self, "Yetersiz Fotoğraf",
                f"Bahçe analizi için en az 3 fotoğraf yüklemelisiniz.\n"
                f"Seçtiğiniz fotoğraf sayısı: {len(paths)}\n\n"
                "Lütfen en az 3 fotoğraf seçin."
            )
            return

        self.paths = paths
        self.lbl_filename.setText(f"{len(paths)} fotoğraf seçildi")
        self.btn_select.setEnabled(False)

        self.progress = QProgressDialog(
            "Yapraklar analiz ediliyor…", None, 0, len(paths), self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setCancelButton(None)
        self.progress.show()

        self._worker = _LeafWorker(self.detector, paths)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, done: int, total: int):
        self.progress.setLabelText(f"Analiz ediliyor… ({done}/{total})")
        self.progress.setValue(done)

    def _on_done(self, results: list):
        self.progress.close()
        self.btn_select.setEnabled(True)
        self.results = results

        # Fotoğraf geçiş listesini doldur
        self.photo_combo.blockSignals(True)
        self.photo_combo.clear()
        for i, path in enumerate(self.paths):
            leaf_n = results[i].total_leaves if i < len(results) else 0
            self.photo_combo.addItem(f"Fotoğraf {i + 1}/{len(self.paths)} — "
                                     f"{Path(path).name}  ({leaf_n} yaprak)")
        self.photo_combo.setCurrentIndex(0)
        self.photo_combo.setEnabled(True)
        self.photo_combo.blockSignals(False)

        self._update_image_display()
        self._update_summary()
        self._populate_disease_card()
        self._populate_table()

    def _on_error(self, msg: str):
        self.progress.close()
        self.btn_select.setEnabled(True)
        QMessageBox.critical(self, "Analiz Hatası", f"Hata oluştu:\n{msg}")

    def _update_image_display(self):
        if not self.results:
            return
        idx = max(0, self.photo_combo.currentIndex())
        if idx >= len(self.results):
            return
        result = self.results[idx]
        img = (result.original_rgb
               if self.rb_original.isChecked()
               else result.annotated_rgb)
        self.image_label.setPixmap(_to_qpixmap(img))

    def _update_summary(self):
        total, healthy, health_counter, age_counter = _aggregate(self.results)
        self._sum_photos.setText(str(len(self.results)))
        self._sum_total.setText(str(total))

        if total == 0:
            self._sum_healthy.setText("—")
            self._sum_diseased.setText("—")
            self._report_text.setText("Hiçbir fotoğrafta yaprak tespit edilemedi.")
            return

        healthy_pct = healthy / total * 100
        self._sum_healthy.setText(f"%{healthy_pct:.0f}")
        self._sum_diseased.setText(f"%{100 - healthy_pct:.0f}")

        # Yaş dağılımı cümlesi
        if age_counter:
            parts = [f"%{cnt / total * 100:.0f} → {label}"
                     for label, cnt in age_counter.most_common()]
            self._report_text.setText(
                "Olgunluk dağılımı: " + ",  ".join(parts))
        else:
            self._report_text.setText("")

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
            else:
                child = item.layout()
                if child is not None:
                    LeafAnalysisWindow._clear_layout(child)

    def _populate_disease_card(self):
        self._clear_layout(self._disease_container)
        total, _, health_counter, _ = _aggregate(self.results)

        diseased = [(lbl, cnt) for lbl, cnt in health_counter.most_common()
                    if not _is_healthy(lbl)]

        if total == 0 or not diseased:
            msg = ("Hastalıklı yaprak tespit edilmedi  🌿"
                   if total else "Henüz yaprak tespit edilmedi")
            lbl = QLabel(msg)
            lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            lbl.setStyleSheet("color: #2e7d32; border: none;")
            self._disease_container.addWidget(lbl)
            return

        diseased_total = sum(cnt for _, cnt in diseased)

        for label, count in diseased:
            row = QFrame()
            row.setStyleSheet("""
                QFrame {
                    background-color: #fdf3f3;
                    border: 1px solid #f0dada;
                    border-radius: 8px;
                }
            """)
            rl = QHBoxLayout(row)
            rl.setContentsMargins(10, 6, 10, 6)
            rl.setSpacing(8)

            # Kırmızı kutularda görünen hastalık ID'si
            id_badge = QLabel(f"#{self.detector.disease_id(label)}")
            id_badge.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            id_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            id_badge.setFixedWidth(34)
            id_badge.setStyleSheet("""
                color: white;
                background-color: #dc3535;
                border: none;
                border-radius: 9px;
                padding: 2px 0;
            """)
            rl.addWidget(id_badge)

            name = QLabel(label)
            name.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            name.setStyleSheet("color: #5a1a1a; border: none;")
            rl.addWidget(name)
            rl.addStretch()

            pct = count / total * 100
            badge = QLabel(f"{count} yaprak  •  %{pct:.0f}")
            badge.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            badge.setStyleSheet("""
                color: white;
                background-color: #c0392b;
                border: none;
                border-radius: 9px;
                padding: 2px 10px;
            """)
            rl.addWidget(badge)

            self._disease_container.addWidget(row)

        # Toplam hastalıklı özeti
        summary = QLabel(f"Toplam hastalıklı yaprak: {diseased_total} / {total}")
        summary.setFont(QFont("Segoe UI", 9))
        summary.setStyleSheet("color: #7a3030; border: none; margin-top: 2px;")
        self._disease_container.addWidget(summary)

    def _populate_table(self):
        self.table.setRowCount(0)
        total, _, health_counter, _ = _aggregate(self.results)

        if total == 0:
            self.table.setRowCount(1)
            item = QTableWidgetItem("Yaprak tespit edilemedi")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setForeground(QColor("#888"))
            self.table.setItem(0, 0, item)
            self.table.setSpan(0, 0, 1, 3)
            return

        # En çok görülen sınıftan aza doğru
        for label, count in health_counter.most_common():
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Sütun 0 — sınıf adı (renkli)
            name_item = QTableWidgetItem(label)
            healthy = _is_healthy(label)
            name_item.setForeground(QColor("#1a6b1a") if healthy else QColor("#b00000"))
            name_item.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            self.table.setItem(row, 0, name_item)

            # Sütun 1 — yaprak sayısı
            cnt_item = QTableWidgetItem(str(count))
            cnt_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            cnt_item.setForeground(QColor("#1b3a2a"))
            cnt_item.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            self.table.setItem(row, 1, cnt_item)

            # Sütun 2 — yüzde
            pct_item = QTableWidgetItem(f"%{count / total * 100:.0f}")
            pct_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            pct_item.setForeground(QColor("#1b3a2a"))
            pct_item.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            self.table.setItem(row, 2, pct_item)

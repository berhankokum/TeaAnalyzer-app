"""Toplu analiz sekmesi."""

from pathlib import Path
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTableWidget, QTableWidgetItem, QProgressBar,
    QFrame, QMessageBox, QHeaderView, QAbstractItemView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from core.analyzer import TeaAnalyzer
from core.pipeline import AnalysisPipeline

IMG_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]


class BatchWorker(QThread):
    progress = pyqtSignal(int, int, str)
    one_done = pyqtSignal(object)
    finished_all = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, pipeline, image_paths):
        super().__init__()
        self.pipeline = pipeline
        self.image_paths = image_paths
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        results = []
        total = len(self.image_paths)
        for i, path in enumerate(self.image_paths, 1):
            if self._stop:
                break
            try:
                r = self.pipeline.analyze(path, generate_visuals=False)
                results.append(r)
                self.one_done.emit(r)
                self.progress.emit(i, total, Path(path).name)
            except Exception as e:
                self.progress.emit(i, total, f"{Path(path).name} (HATA)")
        self.finished_all.emit(results)


class BatchAnalysisTab(QWidget):
    def __init__(self, rf_model_path, yolo_model_path=None):
        super().__init__()
        self.pipeline = AnalysisPipeline(rf_model_path, yolo_model_path)
        self.results = []
        self.folder = None
        self.image_paths = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        # ---- ÜST BAR: klasör seç + başlat + excel ----
        top = QHBoxLayout()

        self.btn_folder = QPushButton("📁  Klasör Seç")
        self.btn_folder.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.btn_folder.setMinimumHeight(44)
        self.btn_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_folder.clicked.connect(self.select_folder)
        self.btn_folder.setStyleSheet(self._btn_style("#4a7c59", "#5a9169"))
        top.addWidget(self.btn_folder)

        self.lbl_folder = QLabel("Klasör seçilmedi")
        self.lbl_folder.setFont(QFont("Segoe UI", 11))
        self.lbl_folder.setStyleSheet("color: #666;")
        top.addWidget(self.lbl_folder)
        top.addStretch()

        self.btn_start = QPushButton("▶  Analizi Başlat")
        self.btn_start.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.btn_start.setMinimumHeight(44)
        self.btn_start.setEnabled(False)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_start.setStyleSheet(self._btn_style("#2c5f8a", "#366fa0"))
        top.addWidget(self.btn_start)

        self.btn_excel = QPushButton("📄  Excel İndir")
        self.btn_excel.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.btn_excel.setMinimumHeight(44)
        self.btn_excel.setEnabled(False)
        self.btn_excel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_excel.clicked.connect(self.export_excel)
        self.btn_excel.setStyleSheet(self._btn_style("#6a4a8a", "#7d5aa0"))
        top.addWidget(self.btn_excel)

        self.btn_pdf = QPushButton("📑  PDF Rapor")
        self.btn_pdf.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.btn_pdf.setMinimumHeight(44)
        self.btn_pdf.setEnabled(False)
        self.btn_pdf.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_pdf.clicked.connect(self.export_pdf)
        self.btn_pdf.setStyleSheet(self._btn_style("#a0522d", "#b5613a"))
        top.addWidget(self.btn_pdf)

        layout.addLayout(top)

        # ---- İLERLEME ÇUBUĞU ----
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(28)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc; border-radius: 6px;
                text-align: center; background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4a7c59; border-radius: 5px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # ---- ÖZET BAR ----
        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f4f0; border: 1px solid #4a7c59;
                border-radius: 8px;
            }
        """)
        self.summary_frame.setVisible(False)
        summary_layout = QHBoxLayout()
        self.lbl_summary = QLabel("")
        self.lbl_summary.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.lbl_summary.setStyleSheet("border: none; color: #2e5d3a; padding: 8px;")
        self.lbl_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        summary_layout.addWidget(self.lbl_summary)
        self.summary_frame.setLayout(summary_layout)
        layout.addWidget(self.summary_frame)

        # ---- SONUÇ TABLOSU ----
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Dosya", "Çay %", "Yeşil Örtü %",
            "Sarımsı Y. %", "Açık Y. %", "Orta Y. %", "Koyu Y. %",
            "Baskın Ton", "ExG"
        ])
        self.table.setFont(QFont("Segoe UI", 10))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                background-color: white;
                alternate-background-color: #eef3ee;
                color: #222222;
            }
            QTableWidget::item {
                color: #222222;
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #4a7c59;
                color: white;
            }
            QHeaderView::section {
                background-color: #4a7c59; color: white;
                padding: 8px; border: none; font-weight: bold;
            }
            QTableWidget QTableCornerButton::section {
                background-color: #4a7c59;
            }
        """)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, 8):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)

        self.setLayout(layout)

    def _btn_style(self, color, hover):
        return f"""
            QPushButton {{
                background-color: {color}; color: white;
                border: none; border-radius: 8px; padding: 0 20px;
            }}
            QPushButton:hover {{ background-color: {hover}; }}
            QPushButton:disabled {{ background-color: #ccc; color: #888; }}
        """

    # --------------------------------------------------------
    # KLASÖR SEÇME
    # --------------------------------------------------------
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Görüntü Klasörü Seç", ""
        )
        if not folder:
            return

        self.folder = Path(folder)
        # Klasördeki görüntüleri bul
        images = []
        for ext in IMG_EXTENSIONS:
            images.extend(self.folder.glob(ext))
        self.image_paths = sorted(set(str(p) for p in images))

        if not self.image_paths:
            QMessageBox.warning(self, "Uyarı",
                                "Seçilen klasörde görüntü dosyası bulunamadı.")
            self.lbl_folder.setText("Klasör seçilmedi")
            self.btn_start.setEnabled(False)
            return

        self.lbl_folder.setText(
            f"Seçili: {self.folder.name} ({len(self.image_paths)} görüntü)"
        )
        self.btn_start.setEnabled(True)

    # --------------------------------------------------------
    # ANALİZ BAŞLAT
    # --------------------------------------------------------
    def start_analysis(self):
        if not self.image_paths:
            return

        # Tabloyu temizle
        self.table.setRowCount(0)
        self.results = []

        # Arayüzü kilitle
        self.btn_start.setEnabled(False)
        self.btn_folder.setEnabled(False)
        self.btn_excel.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.image_paths))
        self.progress_bar.setValue(0)
        self.summary_frame.setVisible(False)

        # Worker başlat
        self.worker = BatchWorker(self.pipeline, self.image_paths)
        self.worker.progress.connect(self._on_progress)
        self.worker.one_done.connect(self._on_one_done)
        self.worker.finished_all.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, current, total, filename):
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{current}/{total} — {filename}")

    def _on_one_done(self, result):
        # Tabloya bir satır ekle
        row = self.table.rowCount()
        self.table.insertRow(row)

        

        values = [
            result.filename,
            f"{result.tea_coverage_pct:.1f}" if result.analysis_mode == "tea_only" else "—",
            f"{result.vegetation_pct:.1f}",
            f"{result.yellow_green_pct:.1f}",
            f"{result.light_green_pct:.1f}",
            f"{result.mid_green_pct:.1f}",
            f"{result.dark_green_pct:.1f}",
            result.dominant_color,
            f"{result.exg_mean:.0f}",
        ]

        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            # Renk kodlaması: sarımsı yeşil = sarı, koyu yeşil = koyu yeşil
            if col == 3:
                item.setForeground(QColor("#b8a000"))
            elif col == 4:
                item.setForeground(QColor("#6aab3a"))
            elif col == 5:
                item.setForeground(QColor("#4a7c59"))
            elif col == 6:
                item.setForeground(QColor("#1e4d2b"))
            else:
                item.setForeground(QColor("#222222"))
            if col >= 1:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, col, item)

    def _on_finished(self, results):
        self.results = results
        self.progress_bar.setVisible(False)
        self.btn_start.setEnabled(True)
        self.btn_folder.setEnabled(True)
        self.btn_excel.setEnabled(len(results) > 0)
        self.btn_pdf.setEnabled(len(results) > 0)

        if not results:
            QMessageBox.warning(self, "Uyarı", "Hiç görüntü analiz edilemedi.")
            return

        # Özet hesapla
        avg_veg = sum(r.vegetation_pct for r in results) / len(results)
        avg_shadow = sum(r.shadow_pct for r in results) / len(results)
        avg_yg = sum(r.yellow_green_pct for r in results) / len(results)
        avg_dg = sum(r.dark_green_pct for r in results) / len(results)

        self.lbl_summary.setText(
            f"📊  {len(results)} görüntü analiz edildi     |     "
            f"Ort. Yeşil Örtü: %{avg_veg:.1f}     |     "
            f"Ort. Gölge: %{avg_shadow:.1f}     |     "
            f"Sarımsı: %{avg_yg:.1f}  →  Koyu: %{avg_dg:.1f}"
        )
        self.summary_frame.setVisible(True)

    # --------------------------------------------------------
    # EXCEL İNDİR
    # --------------------------------------------------------
    def export_excel(self):
        if not self.results:
            return

        default_name = f"{self.folder.name}_analiz.xlsx" if self.folder else "analiz.xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self, "Excel Olarak Kaydet", default_name,
            "Excel Dosyası (*.xlsx)"
        )
        if not path:
            return

        try:
            df = pd.DataFrame([r.to_dict() for r in self.results])

            # Özet satırı ekle
            summary_row = {
                "Dosya": "ORTALAMA",
                "Çay Kapsamı (%)": round(df["Çay Kapsamı (%)"].mean(), 1),
                "Yeşil Örtü (%)": round(df["Yeşil Örtü (%)"].mean(), 1),
                "Gölge (%)": round(df["Gölge (%)"].mean(), 1),
                "Sarımsı Yeşil (%)": round(df["Sarımsı Yeşil (%)"].mean(), 1),
                "Açık Yeşil (%)": round(df["Açık Yeşil (%)"].mean(), 1),
                "Orta Yeşil (%)": round(df["Orta Yeşil (%)"].mean(), 1),
                "Koyu Yeşil (%)": round(df["Koyu Yeşil (%)"].mean(), 1),
                "ExG": round(df["ExG"].mean(), 3),
                "GLI": round(df["GLI"].mean(), 3),
                "VARI": round(df["VARI"].mean(), 3),
            }
            df_out = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

            df_out.to_excel(path, index=False)
            QMessageBox.information(self, "Başarılı",
                                    f"Rapor kaydedildi:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Excel kaydedilemedi:\n{e}")

    def export_pdf(self):
        if not self.results:
            return
        from core.report import generate_batch_report

        default_name = f"{self.folder.name}_rapor.pdf" if self.folder else "toplu_rapor.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "PDF Rapor Kaydet", default_name, "PDF Dosyası (*.pdf)"
        )
        if not path:
            return
        try:
            garden = self.folder.name if self.folder else "Çay Bahçesi"
            generate_batch_report(self.results, path, garden_name=garden)
            QMessageBox.information(self, "Başarılı", f"PDF rapor kaydedildi:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"PDF oluşturulamadı:\n{e}")
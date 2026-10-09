"""
Çay Bahçesi Analiz Uygulaması
==============================
Drone görüntülerinden çay bahçesi sağlık analizi.
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QLabel, QStatusBar
)
from PyQt6.QtCore import Qt, QObject, QEvent
from PyQt6.QtGui import QFont

# Proje kök dizinini path'e ekle
sys.path.insert(0, str(Path(__file__).parent))

from ui.single_analysis import SingleAnalysisTab
from ui.batch_analysis import BatchAnalysisTab

# Model yolları
MODELS_DIR = Path(__file__).parent / "models"
RF_MODEL_PATH = MODELS_DIR / "rf_model_binary.pkl"
YOLO_MODEL_PATH = MODELS_DIR / "tea_yolo.pt"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Çay Bahçesi Sağlık Analizi")
        self.setMinimumSize(1100, 750)

        # Merkezi widget ve sekmeler
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Segoe UI", 11))

        # Sekmeleri oluştur
        self.single_tab = SingleAnalysisTab(RF_MODEL_PATH, YOLO_MODEL_PATH)
        self.batch_tab = BatchAnalysisTab(RF_MODEL_PATH, YOLO_MODEL_PATH)

        self.tabs.addTab(self.single_tab, "  📷  Tek Görüntü Analizi  ")
        self.tabs.addTab(self.batch_tab, "  📁  Toplu Analiz  ")

        self.setCentralWidget(self.tabs)

        # Durum çubuğu
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Hazır — Bir görüntü yükleyerek başlayın")

        # Genel stil
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f0;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #e8e8e0;
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #4a7c59;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #6b9b7a;
                color: white;
            }
        """)


class _AreaCloseFilter(QObject):
    """Alan analiz penceresinin kapatma olayını yakalayarak menüye geri döner."""

    def __init__(self, callback):
        super().__init__()
        self._cb = callback

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Close:
            event.ignore()
            self._cb()
            return True
        return False


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # tutarlı görünüm
    app.setQuitOnLastWindowClosed(False)  # çıkışı manuel yönetiyoruz

    # Açılır pencerelerdeki (hata kutuları, yükleme/ilerleme penceresi) yazıları
    # sistem koyu temasından bağımsız olarak siyah göster.
    app.setStyleSheet("""
        QMessageBox, QProgressDialog, QDialog, QInputDialog {
            background-color: #f5f5f0;
        }
        QMessageBox QLabel, QProgressDialog QLabel,
        QDialog QLabel, QInputDialog QLabel {
            color: #1b1b1b;
            background-color: transparent;
        }
        QMessageBox QPushButton, QProgressDialog QPushButton,
        QDialog QPushButton, QInputDialog QPushButton {
            color: #1b1b1b;
            background-color: #e8e8e0;
            border: 1px solid #b8b8b0;
            border-radius: 6px;
            padding: 5px 16px;
            min-width: 70px;
        }
        QMessageBox QPushButton:hover, QProgressDialog QPushButton:hover,
        QDialog QPushButton:hover, QInputDialog QPushButton:hover {
            background-color: #d8d8d0;
        }
        QProgressBar {
            color: #1b1b1b;
            background-color: #e8e8e0;
            border: 1px solid #b8b8b0;
            border-radius: 6px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4a7c59;
            border-radius: 6px;
        }
    """)

    # Model var mı kontrol et
    if not RF_MODEL_PATH.exists():
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(
            None, "Model Bulunamadı",
            f"RF model dosyası bulunamadı:\n{RF_MODEL_PATH}"
        )
        sys.exit(1)

    from ui.main_menu import MainMenuWindow
    from ui.leaf_analysis import LeafAnalysisWindow

    menu = MainMenuWindow()
    area_window = None
    leaf_window = None

    def open_area():
        nonlocal area_window
        menu.hide()
        if area_window is None:
            area_window = MainWindow()
            # Pencere kapatılınca ana menüye dön
            _filter = _AreaCloseFilter(lambda: (area_window.hide(), menu.show()))
            area_window.installEventFilter(_filter)
            area_window._close_filter = _filter   # referansı koru
        area_window.show()
        area_window.raise_()
        area_window.activateWindow()

    def open_leaf():
        nonlocal leaf_window
        menu.hide()
        if leaf_window is None:
            leaf_window = LeafAnalysisWindow(MODELS_DIR)
            leaf_window.back_requested.connect(lambda: (leaf_window.hide(), menu.show()))
            # Pencere X ile kapatılınca da menüye dön (süreç askıda kalmasın).
            _leaf_filter = _AreaCloseFilter(lambda: (leaf_window.hide(), menu.show()))
            leaf_window.installEventFilter(_leaf_filter)
            leaf_window._close_filter = _leaf_filter   # referansı koru
        leaf_window.show()
        leaf_window.raise_()
        leaf_window.activateWindow()

    menu.area_requested.connect(open_area)
    menu.leaf_requested.connect(open_leaf)
    menu.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
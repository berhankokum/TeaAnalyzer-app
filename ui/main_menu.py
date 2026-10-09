"""Ana menü penceresi — analiz türü seçimi."""

from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QApplication
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class _MenuCard(QFrame):
    """Tıklanabilir analiz kartı."""

    clicked = pyqtSignal()

    def __init__(self, icon: str, title: str, description: str, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumSize(290, 220)
        self._build(icon, title, description)
        self._normal_style = """
            _MenuCard {
                background-color: #f5faf5;
                border: 2px solid #c0d8c0;
                border-radius: 16px;
            }
        """
        self._hover_style = """
            _MenuCard {
                background-color: #e4f4e4;
                border: 2px solid #4a7c59;
                border-radius: 16px;
            }
        """
        self.setStyleSheet(self._normal_style)

    def _build(self, icon: str, title: str, description: str):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(14)
        layout.setContentsMargins(30, 34, 30, 34)

        lbl_icon = QLabel(icon)
        lbl_icon.setFont(QFont("Segoe UI Emoji", 38))
        lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_icon.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(lbl_icon)

        lbl_title = QLabel(title)
        lbl_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("border: none; background: transparent; color: #2e5d3a;")
        layout.addWidget(lbl_title)

        lbl_desc = QLabel(description)
        lbl_desc.setFont(QFont("Segoe UI", 10))
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("border: none; background: transparent; color: #666;")
        layout.addWidget(lbl_desc)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self.setStyleSheet(self._hover_style)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(self._normal_style)
        super().leaveEvent(event)


class MainMenuWindow(QMainWindow):
    area_requested = pyqtSignal()
    leaf_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Çay Bahçesi Analiz Sistemi")
        self.setMinimumSize(820, 540)
        self.resize(920, 600)
        self._build_ui()
        self.setStyleSheet("QMainWindow { background-color: #f5f5f0; }"
                           "QWidget   { background-color: #f5f5f0; }")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.setSpacing(36)
        root.setContentsMargins(60, 55, 60, 55)

        # Başlık
        lbl_title = QLabel("Çay Bahçesi Analiz Sistemi")
        lbl_title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("color: #2e5d3a;")
        root.addWidget(lbl_title)

        lbl_sub = QLabel("Gerçekleştirmek istediğiniz analiz türünü seçin")
        lbl_sub.setFont(QFont("Segoe UI", 12))
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sub.setStyleSheet("color: #888;")
        root.addWidget(lbl_sub)

        # Kart satırı
        cards_row = QHBoxLayout()
        cards_row.setSpacing(32)

        self.card_area = _MenuCard(
            "\U0001f5fa️",
            "Alan Tabanlı Analiz",
            "Drone görüntülerinde çay bahçesi\nalanı sağlık analizi yapın."
        )
        self.card_area.clicked.connect(self.area_requested)

        self.card_leaf = _MenuCard(
            "\U0001f343",
            "Yaprak Tabanlı Analiz",
            "Çay yapraklarında hastalık ve\nyaş tespiti gerçekleştirin."
        )
        self.card_leaf.clicked.connect(self.leaf_requested)

        cards_row.addWidget(self.card_area)
        cards_row.addWidget(self.card_leaf)
        root.addLayout(cards_row)

    def closeEvent(self, event):
        # Ana menü kapatıldığında uygulamadan tamamen çık (süreç terminalde kalmasın).
        QApplication.quit()
        super().closeEvent(event)

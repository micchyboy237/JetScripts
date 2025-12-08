# ─────────────────────────────────────────────────────────────────────────────
# Add these imports at the top (replace the old ones)
# ─────────────────────────────────────────────────────────────────────────────

import sys

import logging
from rich.logging import RichHandler

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QFont

# ─────────────────────────────────────────────────────────────────────────────
# New: Logging setup helper (add near the top of the file)
# ─────────────────────────────────────────────────────────────────────────────

def setup_overlay_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure rich logging for the overlay (beautiful console output)."""
    logger = logging.getLogger("SubtitleOverlay")
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_path=False,
        )
        handler.setLevel(level)
        formatter = logging.Formatter("[bold magenta]%(name)s[/] | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# ─────────────────────────────────────────────────────────────────────────────
# Updated class with rich logging + signals
# ─────────────────────────────────────────────────────────────────────────────

class SubtitleOverlay(QWidget):
    # Signal for external code to react to events (optional but handy)
    textChanged = pyqtSignal(str)
    minimized = pyqtSignal(bool)
    closed = pyqtSignal()

    def __init__(
        self,
        initial_text: str = "Subtitle Overlay Ready",
        font_size: int = 32,
        width: int = 800,
        height: int = 200,
        parent=None
    ):
        super().__init__(parent)

        self.logger = setup_overlay_logging()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 180); border-radius: 15px;")

        self._drag_position = QPoint()
        self._minimized = False
        self._original_geometry = QRect(100, 100, width, height)

        self._setup_ui(initial_text, font_size, width, height)
        self.setGeometry(self._original_geometry)

        self.logger.info("[green]SubtitleOverlay created[/] – drag, minimize, close ready")

    def _setup_ui(self, initial_text: str, font_size: int, width: int, height: int):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Control bar (minimize + close)
        control_bar = QHBoxLayout()
        control_bar.addStretch()

        self.minimize_btn = QPushButton("−")
        self.minimize_btn.setFixedSize(30, 30)
        self.minimize_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 50);
                border: none;
                border-radius: 15px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover { background-color: rgba(255, 255, 255, 100); }
        """)
        self.minimize_btn.clicked.connect(self.toggle_minimize)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 100, 100, 150);
                border: none;
                border-radius: 15px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover { background-color: rgba(255, 50, 50, 200); }
        """)
        close_btn.clicked.connect(self.close)

        control_bar.addWidget(self.minimize_btn)
        control_bar.addWidget(close_btn)

        # Subtitle label
        self.text_label = QLabel(initial_text)
        self.text_label.setStyleSheet("color: white;")
        self.text_label.setFont(QFont("Arial", font_size, QFont.Weight.Bold))
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setWordWrap(True)

        layout.addLayout(control_bar)
        layout.addWidget(self.text_label, stretch=1)

    # ──────────────────────────────
    # Updated methods with logging
    # ──────────────────────────────
    def set_text(self, text: str):
        """Update the displayed subtitle text."""
        old_text = self.text_label.text()
        self.text_label.setText(text or "")
        self.textChanged.emit(text)
        self.logger.info(f"[cyan]Text updated[/]: '{old_text}' → '[bold] {text or '(empty)'}[/]'")

    def toggle_minimize(self):
        """Minimize to only show control bar or restore full size."""
        self._minimized = not self._minimized
        if self._minimized:
            minimized_geom = QRect(
                self.x(), self.y(), self.width(), 60
            )
            self.setGeometry(minimized_geom)
            self.text_label.hide()
            self.minimize_btn.setText("□")
            self.logger.info("[yellow]Overlay minimized[/]")
        else:
            self.setGeometry(self._original_geometry)
            self.text_label.show()
            self.minimize_btn.setText("−")
            self.logger.info("[green]Overlay restored[/]")
        self.minimized.emit(self._minimized)

    def closeEvent(self, event):
        """Override close to emit signal and log."""
        self.closed.emit()
        self.logger.info("[red]Overlay closed[/]")
        super().closeEvent(event)

    # Dragging support (unchanged, but still needed)
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and int(self._drag_position):
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_position = QPoint()

# ─────────────────────────────────────────────────────────────────────────────
# Updated example usage (replace the old if __name__ block)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)

    overlay = SubtitleOverlay(
        initial_text="Live Subtitle Overlay • Ready",
        font_size=36,
        width=900,
        height=220
    )
    overlay.show()

    # Simulate live subtitle updates
    import time
    from threading import Thread

    def demo_updates():
        subtitles = [
            "Hello! This is a live subtitle overlay.",
            "You can drag me anywhere on the screen.",
            "Click the minus button to minimize.",
            "Rich logging is now active below!",
            "Supports very long subtitles that automatically wrap because word-wrap is enabled.",
            "Enjoy!",
            "",
        ]
        for i, line in enumerate(subtitles * 3):  # repeat a few times
            time.sleep(3.5)
            overlay.set_text(line)
            if i == 5:
                overlay.toggle_minimize()
                time.sleep(2)
                overlay.toggle_minimize()

    Thread(target=demo_updates, daemon=True).start()

    sys.exit(app.exec())
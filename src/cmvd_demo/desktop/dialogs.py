"""Consistent task dismissal and scrollable windows on small/high-DPI screens."""
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QScrollArea, QWidget


def fit_window(window, width, height):
    screen = window.screen()
    if screen is not None:
        rect = screen.availableGeometry()
        window.resize(min(width, max(320, rect.width()-24)),
                      min(height, max(240, rect.height()-60)))


class TaskDialog(QDialog):
    dismissing = Signal()

    def done(self, result):
        self.dismissing.emit()
        super().done(result)

    def closeEvent(self, event):
        # QDialog.closeEvent routes through reject/done (also used by Escape).
        super().closeEvent(event)


def scroll_layout(dialog):
    outer = QVBoxLayout(dialog)
    outer.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea(dialog)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    body = QWidget()
    layout = QVBoxLayout(body)
    scroll.setWidget(body)
    outer.addWidget(scroll)
    return layout

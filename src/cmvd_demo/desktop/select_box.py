"""Styled vertical dropdowns; keep native QComboBox selection semantics."""
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QPainterPath, QPen, QColor
from PySide6.QtWidgets import QComboBox, QListView, QStyledItemDelegate, QStyle


class SelectBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        view = QListView(self)
        view.setSpacing(3)
        view.setTextElideMode(Qt.ElideNone)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        view.setItemDelegate(QStyledItemDelegate(view))
        self.setView(view)
        self.setMaxVisibleItems(10)
        self.setStyleSheet("""
            QComboBox {
                background: #172a3c; color: #dae4ef;
                border: 1px solid #385268; border-radius: 8px;
                padding: 7px 34px 7px 11px; min-height: 18px;
                combobox-popup: 0;
            }
            QComboBox:hover { background: #1d3247; border-color: #5b7b91; }
            QComboBox:focus, QComboBox:on { border-color: #63ccbd; background: #1a3345; }
            QComboBox:disabled { background: #152434; color: #60768b; border-color: #2a3c4f; }
            QComboBox::drop-down {
                subcontrol-origin: padding; subcontrol-position: top right;
                width: 30px; border: none;
            }
            QComboBox::down-arrow { image: none; width: 0px; height: 0px; }
            QComboBox QAbstractItemView {
                background: #132536; color: #dae4ef; border: 1px solid #426277;
                border-radius: 9px; padding: 5px; outline: 0;
                selection-background-color: #245765; selection-color: #c6fff0;
            }
            QComboBox QAbstractItemView::item {
                padding: 7px 12px; min-height: 20px; border-radius: 6px;
                border: 1px solid transparent;
            }
            QComboBox QAbstractItemView::item:hover { background: #203e52; }
            QComboBox QAbstractItemView::item:selected {
                background: #245765; color: #c6fff0; border-color: #367782;
            }
            QComboBox QAbstractItemView::item:disabled { color: #60768b; }
        """)

    def paintEvent(self, event):
        super().paintEvent(event)
        # Draw a small vector chevron without external bitmap/SVG resources.
        logical = self.rect().adjusted(self.width()-27, 0, -11, 0)
        rect = QStyle.visualRect(self.layoutDirection(), self.rect(), logical)
        center = rect.center()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = '#9ce5d7' if self.view().isVisible() else '#94aebf'
        if not self.isEnabled():
            color = '#52697e'
        pen = QPen(QColor(color), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        direction = -1 if self.view().isVisible() else 1
        path = QPainterPath(QPointF(center.x()-4, center.y()-2*direction))
        path.lineTo(center.x(), center.y()+2*direction)
        path.lineTo(center.x()+4, center.y()-2*direction)
        painter.drawPath(path)

    def showPopup(self):
        super().showPopup()
        self.update()

    def hidePopup(self):
        super().hidePopup()
        self.update()

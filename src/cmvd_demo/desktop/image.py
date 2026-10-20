"""A paint acknowledgement, distinct from ImageItem.sigImageChanged.

This acknowledges Qt drawing, not monitor scan-out or compositor presentation.
"""

from PySide6.QtCore import Signal
import pyqtgraph as pg


class PresentedImageItem(pg.ImageItem):
    frame_painted = Signal(int)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.serial = 0
        self.painted_serial = -1

    def new_frame(self):
        self.serial += 1

    def clear(self):
        self.serial += 1
        super().clear()

    def paint(self, painter, *args):
        super().paint(painter, *args)
        if self.image is not None and self.qimage is not None and self.painted_serial != self.serial:
            self.painted_serial = self.serial
            self.frame_painted.emit(self.serial)

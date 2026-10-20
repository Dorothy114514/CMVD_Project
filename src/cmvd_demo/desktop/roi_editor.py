"""Separate image navigation from ROI editing; publish topology edits atomically."""

from PySide6.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ROI import Handle,_PolyLineSegment


class _EditingHandle(Handle):
    def removeClicked(self):
        if self.isEnabled() and all(r.checkRemoveHandle(self) for r in self.rois):
            super().removeClicked()

    def hoverEvent(self, event):
        if self.isEnabled():
            super().hoverEvent(event)

    def mouseDragEvent(self, event):
        if self.isEnabled():
            super().mouseDragEvent(event)
        else:
            event.ignore()

    def mouseClickEvent(self, event):
        if self.isEnabled():
            super().mouseClickEvent(event)
        else:
            event.ignore()


class _EditingSegment(_PolyLineSegment):
    def hoverEvent(self, event):
        if not self.isEnabled():
            return
        parent = self.parentItem()
        if parent.translatable:
            if not event.isExit():
                event.acceptDrags(Qt.LeftButton)
        else:
            super().hoverEvent(event)

    def mouseDragEvent(self, event):
        parent = self.parentItem()
        if self.isEnabled() and parent.translatable:
            # pyqtgraph's default segment claims these drags but discards them.
            # Rebind coordinates before forwarding the gesture to the contour.
            event.currentItem = parent
            parent.mouseDragEvent(event)
        else:
            event.ignore()


class RoiViewBox(pg.ViewBox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.roi_tool = "pan"
        self.setMouseMode(self.PanMode)

    def set_roi_tool(self, tool):
        self.roi_tool = tool
        self.setCursor(Qt.OpenHandCursor if tool == "pan" else Qt.CrossCursor)

    def mouseDragEvent(self, event, axis=None):
        if self.roi_tool != "pan" and event.button() in (Qt.LeftButton, Qt.RightButton):
            # A missed point must not fall through to background pan or zoom.
            event.accept()
            return
        super().mouseDragEvent(event, axis=axis)


class EditablePolyLineROI(pg.PolyLineROI):
    def __init__(self, *args, **kwargs):
        self.topology_enabled = True
        super().__init__(*args, **kwargs)

    def addHandle(self, info, index=None):
        if info.get("item") is None:
            info = dict(info)
            info["item"] = _EditingHandle(8, typ=info["type"], pen=self.handlePen,
                                           hoverPen=self.handleHoverPen,parent=self,
                                           antialias=self._antialias)
        handle = super().addHandle(info, index=index)
        handle.setCursor(Qt.PointingHandCursor)
        return handle

    def addSegment(self, h1, h2, index=None):
        # Mirror the pinned pyqtgraph 0.14 wiring; keep shared endpoint handles.
        segment = _EditingSegment(handles=(h1,h2),pen=self.pen,hoverPen=self.hoverPen,
                                  parent=self,movable=False,antialias=self._antialias)
        if index is None:
            self.segments.append(segment)
        else:
            self.segments.insert(index,segment)
        segment.sigClicked.connect(self.segmentClicked)
        segment.setAcceptedMouseButtons(Qt.LeftButton)
        segment.setZValue(self.zValue()+1)
        for handle in segment.handles:
            handle["item"].setDeletable(True)
            handle["item"].setAcceptedMouseButtons(handle["item"].acceptedMouseButtons()|Qt.LeftButton)

    def hoverEvent(self,event):
        if self.isEnabled():
            super().hoverEvent(event)

    def mouseDragEvent(self,event):
        if self.isEnabled():
            super().mouseDragEvent(event)
        else:
            event.ignore()

    def _atomic(self, function):
        blocked = self.blockSignals(True)
        try:
            function()
        finally:
            self.blockSignals(blocked)
        if not blocked:
            self.sigRegionChanged.emit(self)
            self.sigRegionChangeFinished.emit(self)

    def segmentClicked(self, segment, ev=None, pos=None):
        if self.topology_enabled and self.isEnabled():
            self._atomic(lambda: super(EditablePolyLineROI, self).segmentClicked(segment, ev, pos))

    def removeHandle(self, handle, updateSegments=True):
        # setPoints/initial construction also use this; callers already block
        # signals while repainting. Keep the editor and its Qt parent alive.
        self._atomic(lambda: super(EditablePolyLineROI, self).removeHandle(handle, updateSegments))

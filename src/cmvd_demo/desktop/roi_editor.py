"""Separate image navigation from ROI editing; publish topology edits atomically."""

from PySide6.QtCore import Qt, Signal
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ROI import Handle,_PolyLineSegment


def _own_menu(menu, item):
    """pyqtgraph creates parentless QWidget menus, unsafe for worker-thread GC."""
    scene = item.scene()
    if scene is not None and scene.views():
        owner = scene.views()[0].window()
        if menu.parent() is not owner:
            menu.setParent(owner, menu.windowFlags())
    return menu


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
    def getMenu(self):
        return _own_menu(super().getMenu(), self)

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
    drawing_click = Signal(object, object, bool)
    drawing_drag = Signal(object, object, bool)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.roi_tool = "pan"
        self.drawing_mode = None
        self.setMouseMode(self.PanMode)

    def set_roi_tool(self, tool):
        self.roi_tool = tool
        self.setCursor(Qt.OpenHandCursor if tool == "pan" else Qt.CrossCursor)

    def mouseDragEvent(self, event, axis=None):
        if self.drawing_mode and event.button() == Qt.LeftButton:
            event.accept()
            if self.drawing_mode == 'ellipse':
                self.drawing_drag.emit(event.buttonDownScenePos(), event.scenePos(), event.isFinish())
            return
        if self.roi_tool != "pan" and event.button() in (Qt.LeftButton, Qt.RightButton):
            # A missed point must not fall through to background pan or zoom.
            event.accept()
            return
        super().mouseDragEvent(event, axis=axis)

    def mouseClickEvent(self, event):
        if self.drawing_mode and event.button() in (Qt.LeftButton, Qt.RightButton):
            event.accept()
            self.drawing_click.emit(event.scenePos(), event.button(), event.double())
            return
        super().mouseClickEvent(event)


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
        _own_menu(handle.menu, self)
        handle.menu.setTitle("轮廓控制点")
        handle.removeAction.setText("删除控制点")
        handle.setCursor(Qt.PointingHandCursor)
        return handle

    def getMenu(self):
        return _own_menu(super().getMenu(), self)

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
        item = self.handles[self.indexOfHandle(handle)]['item']
        self._atomic(lambda: super(EditablePolyLineROI, self).removeHandle(item, updateSegments))
        if not item.rois:
            item.menu.deleteLater()
            item.deleteLater()

    def removeSegment(self, segment):
        super().removeSegment(segment)
        if segment.menu is not None:
            segment.menu.deleteLater()
        segment.deleteLater()

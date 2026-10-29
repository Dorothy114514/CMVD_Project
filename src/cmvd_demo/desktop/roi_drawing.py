"""Transient native drawing gestures; only a validated completion edits the ROI."""
import numpy as np
from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QKeySequence, QShortcut
import pyqtgraph as pg

from cmvd_demo.roi import RoiError, document_hash, set_track
from cmvd_demo.roi.placement import ellipse_vertices, placed_contour
from .presentation import error_message


class RoiDrawing(QObject):
    def __init__(self, panel):
        super().__init__(panel)
        self.panel, self.window = panel, panel.window
        self.active = False
        self.points = []
        self.mode = None
        self.error = ''
        self.preview = pg.PlotDataItem(parent=self.window.image, pen=pg.mkPen('#75e4be', width=2, style=Qt.DashLine),
                                       symbol='o', symbolSize=6, symbolBrush='#75e4be')
        self.preview.setZValue(40)
        self.preview.setAcceptedMouseButtons(Qt.NoButton)
        self.preview.hide()
        self.shortcuts = []
        for key, callback in (('Escape', self.cancel), ('Return', self.finish), ('Enter', self.finish), ('Backspace', self.back)):
            shortcut = QShortcut(QKeySequence(key), self.window)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(callback)
            shortcut.setEnabled(False)
            self.shortcuts.append(shortcut)
        self.window.view.drawing_click.connect(self.click)
        self.window.view.drawing_drag.connect(self.drag)

    def _tell(self, message):
        self.panel.status.setText(message)
        self.window.status.setText(message)

    def context(self):
        p, w = self.panel, self.window
        return (w.case_revision, p.key, w.displayed_indices, p.role.currentData(), p.scope.currentIndex(),
                p.boundary.currentData(), document_hash(p.document) if p.document else None)

    def start(self):
        p, w = self.panel, self.window
        if not p._ready() or p.invalid_edit or any(r.busy for r in (w.runner, p.runner, w.feature_panel.runner)):
            return
        if p.scope.currentIndex()==1 and not any(r['role']==p.role.currentData() and r['slice_index']==w.displayed_indices[1]
                                                for r in p.document['tracks']):
            self._tell('请先在参考轮廓模式勾画，再修正单帧。')
            return
        if p.scope.currentIndex()==0 and any(r['role']==p.role.currentData() and r['slice_index']==w.displayed_indices[1]
                    and any(o['frame_index']==w.displayed_indices[0] for o in r['overrides']) for r in p.document['tracks']):
            self._tell('本帧有独立轮廓，请切换“仅修正当前帧”后勾画。')
            return
        if p.role.currentData()=='myocardium' and p.boundary.currentData()=='inner' and not any(
                r['role']=='myocardium' and r['slice_index']==w.displayed_indices[1] for r in p.document['tracks']):
            self._tell('请先勾画心外膜，再单独勾画心内膜。')
            return
        w._stop_play()
        w.set_roi_tool('points')
        self.mode = p.draw_shape.currentData()
        self._context = self.context()
        self.include_inner = p.hole.isChecked()
        self.active, self.error, self.points = True, '', []
        w.view.drawing_mode = self.mode
        w.view.setCursor(Qt.CrossCursor)
        for pair in p.overlays.values():
            for item in pair:
                if item is not None:
                    item.setEnabled(False)
                    for handle in item.getHandles():
                        handle.hide()
        for shortcut in self.shortcuts:
            shortcut.setEnabled(True)
        self._refresh()
        w.graphics.setFocus()
        self._tell('在目标区域按住左键拖出椭圆，松开完成；Esc或右键取消。' if self.mode=='ellipse'
                         else '沿边界依次左键点选；Enter、双击或点回起点完成，Backspace撤回一点，Esc或右键取消。')

    def _refresh(self):
        self.panel.refresh_buttons()
        if hasattr(self.window, 'feature_panel'):
            self.window.feature_panel.refresh_buttons()
        if hasattr(self.window, 'work'):
            self.window.work.refresh()
        if hasattr(self.window, 'curve_link'):
            self.window.curve_link.refresh()

    def _valid(self):
        if self.active and self.context() != self._context:
            self.cancel()
        return self.active

    def rc(self, scene_pos):
        p = self.window.image.mapFromScene(scene_pos)
        return [p.y()-.5, p.x()-.5]

    def _show(self, points, bad=False):
        if not points:
            self.preview.hide()
            return
        xy = np.asarray(points, dtype=float)[:, ::-1]+.5
        if len(xy)>2:
            xy = np.vstack([xy, xy[0]])
        self.preview.setPen(pg.mkPen('#ff7e80' if bad else '#75e4be', width=2, style=Qt.DashLine))
        self.preview.setData(x=xy[:,0], y=xy[:,1])
        self.preview.show()

    def click(self, scene_pos, button, double):
        if not self._valid():
            return
        if button == Qt.RightButton:
            self.cancel()
            return
        if self.mode != 'polygon' or button != Qt.LeftButton:
            return
        point = self.rc(scene_pos)
        rows, cols = self.panel.grid.shape[-2:]
        if not (-.5 <= point[0] <= rows-.5 and -.5 <= point[1] <= cols-.5):
            self._tell('请在影像范围内点选；已选点保留，Esc可取消。')
            return
        if len(self.points)>=3:
            first = self.window.image.mapToScene(pg.Point(self.points[0][1]+.5, self.points[0][0]+.5))
            if (scene_pos-first).manhattanLength()<=10:
                self.finish()
                return
        if not self.points or not np.allclose(self.points[-1], point, atol=1e-6, rtol=0):
            if len(self.points)>=128:
                self._tell('最多128个顶点，请完成或撤回后调整。')
                return
            self.points.append(point)
        self.error = ''
        self._show(self.points)
        self.panel.finish_draw_button.setEnabled(len(self.points)>=3)
        self._tell(f'已选{len(self.points)}点；Enter完成，Backspace撤回一点，Esc取消。')
        if double:
            self.finish()

    def drag(self, start, end, finished):
        if not self._valid() or self.mode != 'ellipse':
            return
        try:
            self.points = ellipse_vertices(self.rc(start), self.rc(end))
            self._show(self.points)
            if finished:
                self.finish()
        except RoiError as error:
            self.points = []
            self.preview.hide()
            if finished:
                self._tell(error_message(error.code)+' 请重新拖出轮廓，或按Esc取消。')

    def back(self):
        if not self._valid():
            return
        if self.mode != 'polygon':
            self.cancel()
            return
        if self.points:
            self.points.pop()
        self.error = ''
        self._show(self.points)
        self.panel.finish_draw_button.setEnabled(len(self.points)>=3)
        self._tell(f'已选{len(self.points)}点；继续点选，Enter完成，Esc取消。')

    def finish(self):
        if not self._valid():
            return
        p, w = self.panel, self.window
        try:
            t, z = w.displayed_indices
            contour = placed_contour(p.document, p.role.currentData(), z, t, self.points,
                                     p.boundary.currentData(), self.include_inner)
            updated = set_track(p.document, p.role.currentData(), z, t, contour, frame_only=p.scope.currentIndex()==1)
            self._stop()
            p._replace_document(updated)
            p._overlay_key = None
            p.on_frame()
            self._tell('轮廓已完成，可拖点细调或撤销；内膜初始比例仅为手工起点，请单独核对内外膜。'
                             if p.role.currentData()=='myocardium' and p.boundary.currentData()=='both' and self.include_inner
                             else '轮廓已完成，可拖点细调或撤销；请逐帧核对后重新计算曲线。')
            self._refresh()
        except RoiError as error:
            self.error = error.code
            self._show(self.points, bad=True)
            self._tell(error_message(error.code)+' 草稿未应用；Backspace撤回一点，或Esc取消重画。')

    def _stop(self):
        self.active = False
        self.window.view.drawing_mode = None
        self.preview.hide()
        for shortcut in self.shortcuts:
            shortcut.setEnabled(False)
        self.points = []

    def cancel(self):
        if not self.active:
            return
        self._stop()
        self.panel._overlay_key = None
        self.panel.on_frame()
        self._refresh()
        self._tell('已取消勾画；原有轮廓和计算结果保留。')

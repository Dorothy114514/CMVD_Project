"""Manual polygon editor and asynchronous original-curve workspace."""

import copy
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt,Signal
from PySide6.QtGui import QKeySequence,QShortcut
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QCheckBox,QFileDialog,QApplication
import pyqtgraph as pg

from cmvd_demo.roi import new_document,binding_for,validate_document,document_hash,set_track,extract_curves,save_snapshot,load_document,RoiError
from cmvd_demo.roi.model import contour_at
from cmvd_demo.roi.history import RoiHistory
from cmvd_demo.paths import workspace_root
from .tasks import TaskRunner
from .roi_editor import EditablePolyLineROI
from .roi_drawing import RoiDrawing
from .presentation import error_message
from .select_box import SelectBox

ROLES = {"myocardium":"心肌","blood_pool":"血池"}
COLORS = {"myocardium":"#f6bd60","blood_pool":"#63d8eb"}


class RoiPanel(QWidget):
    curve_ready = Signal()
    snapshot_saved = Signal()
    state_changed = Signal()

    def __init__(self,window):
        super().__init__(window)
        self.window = window
        self.runner = TaskRunner(self)
        self.runner.result.connect(self._result)
        self.runner.failed.connect(self._failed)
        self.runner.progress.connect(self._progress)
        self.runner.idle.connect(window._idle)
        self.runner.idle.connect(self.refresh_buttons)
        self.output_root = workspace_root()/"outputs"/"roi"
        self.documents = {}
        self.document,self.grid,self.key,self.result = None,None,None,None
        self.overlays = {}
        self._overlay_pool = {}
        self._overlay_key = None
        self._drawing = False
        self.invalid_edit = False
        self.last_error = ""
        self.saved_output = None
        self._job = None
        self.history = RoiHistory()
        self._edit_before = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8,8,8,8)
        title = QLabel("心肌与血池轮廓（ROI）")
        title.setObjectName("section")
        layout.addWidget(title)
        row = QHBoxLayout()
        self.role = SelectBox()
        self.role.setAccessibleName("勾画对象")
        for key,name in ROLES.items():
            self.role.addItem(name,key)
        self.role.currentIndexChanged.connect(self._role_changed)
        self.scope = SelectBox()
        self.scope.setAccessibleName("轮廓作用范围")
        self.scope.addItems(["参考轮廓","仅修正当前帧"])
        self.scope.currentIndexChanged.connect(self._scope_changed)
        row.addWidget(self.role)
        row.addWidget(self.scope)
        layout.addLayout(row)
        self.boundary = SelectBox()
        self.boundary.setAccessibleName("可编辑边界")
        for label, key in (("内外膜一起编辑", "both"), ("只编辑心外膜", "outer"), ("只编辑心内膜", "inner")):
            self.boundary.addItem(label, key)
        self.boundary.setToolTip("单独编辑时另一条边界锁定；移动整个轮廓仍同时移动内外膜。")
        self.boundary.currentIndexChanged.connect(self._boundary_changed)
        layout.addWidget(self.boundary)
        row = QHBoxLayout()
        self.draw_shape = SelectBox()
        self.draw_shape.setAccessibleName("勾画方式")
        self.draw_shape.addItem("拖出椭圆", "ellipse")
        self.draw_shape.addItem("逐点勾画", "polygon")
        self.draw_button = QPushButton("在图上勾画")
        self.draw_button.setObjectName("primary")
        self.draw_button.clicked.connect(lambda: self.drawing.start())
        row.addWidget(self.draw_shape)
        row.addWidget(self.draw_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.finish_draw_button = QPushButton("完成勾画（Enter）")
        self.cancel_draw_button = QPushButton("取消勾画（Esc）")
        self.finish_draw_button.clicked.connect(lambda: self.drawing.finish())
        self.cancel_draw_button.clicked.connect(lambda: self.drawing.cancel())
        row.addWidget(self.finish_draw_button)
        row.addWidget(self.cancel_draw_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.undo_button = QPushButton("撤销（Ctrl+Z）")
        self.redo_button = QPushButton("重做（Ctrl+Y）")
        for button in (self.undo_button,self.redo_button):
            button.setToolTip("只恢复轮廓；复核和曲线需重做。当前病例最多保留50步/16 MiB，按序列隔离，关闭不保存历史。")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        row.addWidget(self.undo_button)
        row.addWidget(self.redo_button)
        layout.addLayout(row)
        self._shortcuts = []
        for keys,callback in (("Ctrl+Z",self.undo),("Ctrl+Y",self.redo),("Ctrl+Shift+Z",self.redo)):
            shortcut = QShortcut(QKeySequence(keys),window)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
        row = QHBoxLayout()
        self.add_button = QPushButton("居中参考轮廓")
        self.add_button.setToolTip("放置居中的几何起点；可改用“在图上勾画”直接指定位置和大小。")
        self.add_button.clicked.connect(self.add_reference)
        self.remove_button = QPushButton("删除所选轮廓")
        self.remove_button.clicked.connect(self.remove_track)
        row.addWidget(self.add_button)
        row.addWidget(self.remove_button)
        layout.addLayout(row)
        self.hole = QCheckBox("排除心内膜以内区域")
        self.hole.setToolTip("以外轮廓以内、内轮廓以外的区域计算心肌信号；请手动核对内外膜位置。")
        self.hole.setChecked(True)
        self.hole.toggled.connect(self._hole_changed)
        layout.addWidget(self.hole)
        self.reset_frame_button = QPushButton("取消本帧修正，恢复参考轮廓")
        self.reset_frame_button.clicked.connect(self.reset_frame)
        layout.addWidget(self.reset_frame_button)
        hint = QLabel("先选心肌或血池，再在图上拖椭圆或逐点勾画。\n逐点时Enter完成、Backspace撤一点、Esc取消；完成后可拖点细调。\n可单独锁定内外膜；滚轮缩放、中键平移。切换病例前请保存。")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        layout.addWidget(hint)
        self.orientation_review = QCheckBox("已核对采集方向与覆盖")
        self.motion_review = QCheckBox("已逐帧检查全部标注层的轮廓与运动")
        for check in (self.orientation_review,self.motion_review):
            check.toggled.connect(self._reviews_changed)
            layout.addWidget(check)
        row = QHBoxLayout()
        self.calculate_button = QPushButton("计算原始曲线")
        self.calculate_button.clicked.connect(self.calculate)
        self.cancel_button = QPushButton("取消曲线任务")
        self.cancel_button.clicked.connect(self.cancel_analysis)
        row.addWidget(self.calculate_button)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)
        self.curve_controls = QHBoxLayout()
        layout.addLayout(self.curve_controls)
        self.curve_hint = QLabel("计算曲线后可点击定位影像，MR曲线可选择基线/首过区间。")
        self.curve_hint.setWordWrap(True)
        layout.addWidget(self.curve_hint)
        self.plot = pg.PlotWidget(background="#080f1b")
        # Parentless pyqtgraph menus otherwise survive window DeferredDelete
        # and may be finalized during a subsequent reader's Python GC.
        for menu in (self.plot.getPlotItem().ctrlMenu, self.plot.getViewBox().menu):
            menu.setParent(self.plot, menu.windowFlags())
        self.plot.setMinimumHeight(180)
        self.plot.setLabel("bottom","时间",units="s")
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.plot.showGrid(x=True,y=True,alpha=.15)
        self.plot.addLegend()
        layout.addWidget(self.plot,1)
        self.measurement = QLabel("当前帧ROI：—")
        self.measurement.setWordWrap(True)
        layout.addWidget(self.measurement)
        row = QHBoxLayout()
        self.save_button = QPushButton("保存ROI / 曲线")
        self.save_button.clicked.connect(self.save)
        self.load_button = QPushButton("载入已存轮廓")
        self.load_button.clicked.connect(self._choose_load)
        row.addWidget(self.save_button)
        row.addWidget(self.load_button)
        layout.addLayout(row)
        self.status = QLabel("加载影像后可放置ROI。所有真实结果默认未审核。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.qc = QLabel("原始曲线预览 · 未临床验证\n尚不计算MBF、MFR或诊断。")
        self.qc.setObjectName("pill")
        self.qc.setWordWrap(True)
        layout.addWidget(self.qc)
        self._cursor = None
        self.drawing = RoiDrawing(self)
        self.refresh_buttons()

    def _ready(self):
        w = self.window
        return (self.grid is not None and self.document is not None and w.grid is self.grid
                and w.current_pixels is not None and w.displayed_indices==(w.time_slider.value(),w.slice_slider.value())
                and not (hasattr(self,"drawing") and self.drawing.active))

    def refresh_buttons(self):
        ready = self._ready()
        drawing = hasattr(self,"drawing") and self.drawing.active
        polygon = drawing and self.drawing.mode=="polygon"
        self.undo_button.setText("撤回一点（Ctrl+Z）" if polygon else "取消草稿（Ctrl+Z）" if drawing else "撤销（Ctrl+Z）")
        self.window.roi_undo_button.setText("撤点" if polygon else "取消草稿" if drawing else "撤销")
        self.draw_button.setEnabled(ready and not self.invalid_edit and not self.runner.busy)
        self.draw_shape.setEnabled(not drawing)
        self.boundary.setEnabled(ready and self.role.currentData()=="myocardium")
        self.finish_draw_button.setVisible(drawing)
        self.cancel_draw_button.setVisible(drawing)
        self.finish_draw_button.setEnabled(drawing and len(self.drawing.points)>=3)
        locked_reference = (ready and self.scope.currentIndex()==0 and any(
            r['role']==self.role.currentData() and r['slice_index']==self.window.displayed_indices[1]
            and any(o['frame_index']==self.window.displayed_indices[0] for o in r['overrides'])
            for r in self.document['tracks']))
        self.hole.setEnabled(ready and self.role.currentData()=="myocardium" and not locked_reference
                             and self.window.roi_tool.currentData()!="pan")
        tracks = bool(self.document and self.document["tracks"])
        for button in (self.add_button,self.remove_button,self.reset_frame_button,self.load_button):
            button.setEnabled(ready)
        self.save_button.setEnabled(ready and tracks and not self.invalid_edit and not self.runner.busy)
        self.calculate_button.setEnabled(ready and tracks and not self.invalid_edit and not self.runner.busy
                                         and self.grid.times_s is not None and self.grid.shape[0]>1)
        self.cancel_button.setEnabled(self.runner.busy)
        for check in (self.orientation_review,self.motion_review):
            check.setEnabled(ready and tracks and not self.invalid_edit)
        self.undo_button.setEnabled(drawing or (ready and (self.invalid_edit or self.history.can_undo(self.key))))
        self.redo_button.setEnabled(ready and not self.invalid_edit and self.history.can_redo(self.key))
        self.window.roi_undo_button.setEnabled(self.undo_button.isEnabled())
        self.window.roi_redo_button.setEnabled(self.redo_button.isEnabled())

    def _remove_overlays(self):
        # Retain three editor objects under the image's Qt ownership. Repeated
        # remove/reparent/destruction of PolyLineROI segment graphs can invalidate
        # native objects on PySide6. Hidden objects never display another case.
        for item in self._overlay_pool.values():
            item.setVisible(False)
            item.setEnabled(False)
        self.overlays = {}
        self._overlay_key = None

    def _clear_curve(self):
        self.result = None
        self.plot.clear()
        self._cursor = None
        self.measurement.setText("当前帧ROI：—")
        self.qc.setText("原始曲线预览 · 未临床验证\n尚不计算MBF、MFR或诊断。")
        self.state_changed.emit()

    def suspend(self):
        if hasattr(self,"drawing"):
            self.drawing.cancel()
        self.runner.invalidate()
        self._job = None
        self._edit_before = None
        self._remove_overlays()
        self._clear_curve()
        self.refresh_buttons()

    def reset_case(self):
        self.suspend()
        self.documents.clear()
        self.history.clear()
        self.document,self.grid,self.key = None,None,None
        self.invalid_edit = False
        self.boundary.setCurrentIndex(0)
        self._sync_reviews()
        self.refresh_buttons()

    def detach_grid(self):
        self.suspend()
        self.document,self.grid,self.key = None,None,None
        self.refresh_buttons()

    def set_grid(self,grid,key):
        self.suspend()
        binding = binding_for(grid)
        old = self.documents.get(key)
        if old is None or old["binding"]!=binding:
            self.history.clear(key)
        self.document = old if old is not None and old["binding"]==binding else new_document(grid)
        if getattr(grid,"synthetic_id",None):
            self.document["purpose"] = "engineering_fixture"
        self.documents[key] = self.document
        self.grid,self.key = grid,key
        self.invalid_edit = False
        self.last_error = ""
        self._sync_reviews()

    def navigation_pending(self):
        self.drawing.cancel()
        for pair in self.overlays.values():
            for item in pair:
                if item is not None:
                    item.setEnabled(False)
        if self._job and self._job[0]=="curve" and self.window.slice_slider.value()!=self._job[-1]:
            self.cancel_analysis()
        self.refresh_buttons()

    def on_frame(self):
        if not self._ready():
            return
        t,z = self.window.displayed_indices
        if self.invalid_edit:
            self.invalid_edit = False
            self._edit_before = None
            self.last_error = ""
            self._overlay_key = None
        if self.result is not None and self.result["slice_index"]!=z:
            self._clear_curve()
        self._draw_overlays(t,z)
        if self._cursor is not None:
            self._cursor.setValue(self.grid.times_s[z,t])
            self._show_measurement(t)
        self.refresh_buttons()

    def _role_changed(self):
        self.drawing.cancel()
        self._overlay_key = None
        self.on_frame()

    def _boundary_changed(self):
        if hasattr(self,"drawing"):
            self.drawing.cancel()
        self._overlay_key = None
        self.on_frame()

    def _show_measurement(self,t):
        if self.result is not None:
            self.measurement.setText("；".join(
                f"{ROLES[c['role']]}：{c['pixel_count'][t]} px / {c['area_mm2'][t]:.2f} mm²，均值 {c['mean'][t]:.5g}"
                for c in self.result["curves"]))

    def _scope_changed(self):
        self.drawing.cancel()
        self._overlay_key = None
        self.on_frame()

    def _draw_overlays(self,t,z):
        tracks = [r for r in self.document["tracks"] if r["slice_index"]==z]
        role = self.role.currentData()
        contours = [(r["role"],contour_at(r,t)) for r in tracks]
        tool = self.window.roi_tool.currentData()
        boundary_choice = self.boundary.currentData()
        key = document_hash([contours,role,self.scope.currentIndex(),tool,boundary_choice])
        scope_editable = self.scope.currentIndex()==1 or not any(
            r["role"]==role and any(o["frame_index"]==t for o in r["overrides"]) for r in tracks)
        editable = scope_editable and tool!="pan"
        def can_edit(kind, boundary):
            return (kind==role and editable and (kind=="blood_pool" or tool=="contour"
                    or boundary_choice=="both" or boundary_choice==("outer","inner")[boundary]))
        if key==self._overlay_key:
            for kind,pair in self.overlays.items():
                for boundary,item in enumerate(pair):
                    if item is not None:
                        item.setEnabled(can_edit(kind,boundary))
            self.hole.setEnabled(role=="myocardium" and editable)
            if not scope_editable:
                self.status.setText("本帧有独立轮廓；请选择“仅修正当前帧”编辑。")
            return
        self._remove_overlays()
        self._drawing = True
        try:
            for track in tracks:
                kind = track["role"]
                contour = contour_at(track,t)
                pair = []
                for boundary,vertices in enumerate((contour["outer_rc"],contour["inner_rc"])):
                    if vertices is None:
                        pair.append(None)
                        continue
                    points = [(c+.5,r+.5) for r,c in vertices]
                    pool_key = kind,boundary
                    item = self._overlay_pool.get(pool_key)
                    if item is None:
                        item = EditablePolyLineROI(points,closed=True,parent=self.window.image,
                                              pen=pg.mkPen(COLORS[kind],width=2),removable=False,
                                              movable=False,rotatable=False,resizable=False)
                        self._overlay_pool[pool_key] = item
                        item.sigRegionChanged.connect(self._editing)
                        item.sigRegionChangeFinished.connect(self._commit)
                    else:
                        current = [item.mapToItem(self.window.image,p) for name,p in item.getLocalHandlePositions()]
                        if len(current)!=len(points) or not np.allclose([[p.x(),p.y()] for p in current],points,atol=1e-9,rtol=0):
                            item.blockSignals(True)
                            item.setPos(0,0)
                            item.setAngle(0)
                            item.setSize((1,1))
                            if len(item.getHandles())==len(points):
                                for handle,point in zip(item.getHandles(),points):
                                    item.movePoint(handle,pg.Point(point),finish=False,coords="parent")
                            else:
                                item.setPoints(points)
                            item.blockSignals(False)
                    item.setZValue(20)
                    item.translatable = tool=="contour" and kind==role and editable
                    item.topology_enabled = tool=="points"
                    item.setVisible(True)
                    item.setEnabled(can_edit(kind,boundary))
                    for handle in item.getHandles():
                        handle.setVisible(can_edit(kind,boundary) and tool=="points")
                    pair.append(item)
                self.overlays[kind] = tuple(pair)
        finally:
            self._drawing = False
        self._overlay_key = key
        self.hole.blockSignals(True)
        active = next((c for k,c in contours if k==role),None)
        if role=="myocardium" and active is not None:
            self.hole.setChecked(active["inner_rc"] is not None)
        self.hole.setEnabled(role=="myocardium" and editable)
        self.hole.blockSignals(False)
        if not scope_editable:
            self.status.setText("本帧有独立轮廓；请选择“仅修正当前帧”，或回到未单独修正帧编辑参考。")

    def _sync_reviews(self):
        for key,check in (("orientation_and_coverage",self.orientation_review),("motion_and_contours",self.motion_review)):
            check.blockSignals(True)
            check.setChecked(bool(self.document and self.document["reviews"][key]))
            check.blockSignals(False)

    def _edit_context(self):
        t,z = self.window.displayed_indices or (0,0)
        return t,z,self.role.currentData(),self.scope.currentIndex()

    def _replace_document(self,document,*,record=True):
        if record and self.document is not None and self.document["binding"]==document["binding"]:
            before,context = self._edit_before or (self.document["tracks"],self._edit_context())
            self.history.record(self.key,before,document["tracks"],context)
        self._edit_before = None
        self.document = document
        self.documents[self.key] = document
        self.runner.invalidate()
        self._job = None
        self._clear_curve()
        self.invalid_edit = False
        self.last_error = ""
        self._sync_reviews()
        self.refresh_buttons()

    def undo(self):
        if self.drawing.active:
            self.drawing.back()
            return
        self._restore_history(False)

    def redo(self):
        self._restore_history(True)

    def _restore_history(self,redo):
        if not self._ready() or QApplication.mouseButtons()!=Qt.NoButton:
            return
        self.window._stop_play()
        if self.invalid_edit:
            if redo:
                return
            # An invalid drag never entered the document or redo stack.
            tracks,context = self._edit_before or (self.document["tracks"],self._edit_context())
        else:
            restored = self.history.restore(self.key,self.document["tracks"],redo=redo)
            if restored is None:
                self.refresh_buttons()
                return
            tracks,context = restored
        updated = copy.deepcopy(self.document)
        updated["tracks"] = copy.deepcopy(tracks)
        updated["reviews"] = {k:False for k in updated["reviews"]}
        self._replace_document(updated,record=False)
        t,z,role,scope = context
        for control,value in ((self.role,self.role.findData(role)),(self.scope,scope)):
            control.blockSignals(True)
            control.setCurrentIndex(value)
            control.blockSignals(False)
        self.window.set_roi_tool("points")
        self._overlay_key = None
        self.on_frame()
        if self.window.displayed_indices!=(t,z):
            for slider,value in ((self.window.time_slider,t),(self.window.slice_slider,z)):
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)
            self.window._position_changed()
        self.status.setText(("已重做" if redo else "已撤销")+"轮廓更改；请重新复核并计算曲线。")
        self.refresh_buttons()

    def _editing(self,*args):
        if self._drawing or self.grid is None:
            return
        if self._edit_before is None:
            self._edit_before = copy.deepcopy(self.document["tracks"]),self._edit_context()
        pair = self.overlays.get(self.role.currentData(),())
        moved = args[0] if args else None
        if (self.window.roi_tool.currentData()=="contour" and len(pair)==2
                and all(item is not None for item in pair) and moved in pair):
            # A myocardium ROI comprises two boundaries. Translate both by the
            # same amount, whether the gesture starts on its inner or outer edge.
            t,z,role,_ = self._edit_before[1]
            track = next(r for r in self._edit_before[0] if r["role"]==role and r["slice_index"]==z)
            contour = contour_at(track,t)
            index = pair.index(moved)
            current = moved.mapToItem(self.window.image,moved.getLocalHandlePositions()[0][1])
            original = contour[("outer_rc","inner_rc")[index]][0]
            dr,dc = current.y()-.5-original[0],current.x()-.5-original[1]
            partner = pair[1-index]
            point = partner.mapToItem(self.window.image,partner.getLocalHandlePositions()[0][1])
            target = contour[("outer_rc","inner_rc")[1-index]][0]
            offset = pg.Point(target[1]+dc+.5-point.x(),target[0]+dr+.5-point.y())
            blocked = partner.blockSignals(True)
            try:
                partner.setPos(partner.pos()+offset)
            finally:
                partner.blockSignals(blocked)
        w = self.window
        w._stop_play()
        w.runner.invalidate()
        w._cancel_prefetch()
        w.debounce.stop()
        w._operation = None
        if w.displayed_indices is not None:
            t,z = w.displayed_indices
            for slider,value in ((w.time_slider,t),(w.slice_slider,z)):
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)
        self.runner.invalidate()
        self._job = None
        self.invalid_edit = True
        self.document = copy.deepcopy(self.document)
        self.document["reviews"] = {k:False for k in self.document["reviews"]}
        self.documents[self.key] = self.document
        self._clear_curve()
        self._sync_reviews()
        self.status.setText("正在修改轮廓；松开控制点后校验。")
        self.refresh_buttons()

    def _commit(self,*args):
        if self._drawing or not self._ready():
            return
        pair = self.overlays.get(self.role.currentData())
        if not pair:
            return
        contour = {}
        for key,item in zip(("outer_rc","inner_rc"),pair):
            contour[key] = None if item is None else [[item.mapToItem(self.window.image,p).y()-.5,
                                                     item.mapToItem(self.window.image,p).x()-.5]
                                                    for name,p in item.getLocalHandlePositions()]
        t,z = self.window.displayed_indices
        try:
            updated = set_track(self.document,self.role.currentData(),z,t,contour,frame_only=self.scope.currentIndex()==1)
            self._replace_document(updated)
            self.status.setText("轮廓已更新；请逐帧复核运动，再计算原始曲线。")
            self._overlay_key = None
            self.on_frame()
        except RoiError as error:
            self.invalid_edit = True
            self.last_error = error.code
            self.status.setText("轮廓无效，可按Ctrl+Z恢复，或修正越界、自交与内外轮廓关系。")
            self.status.setToolTip("技术代码："+error.code)
            self.refresh_buttons()

    def add_reference(self):
        if not self._ready():
            return
        self.window._stop_play()
        self.window.set_roi_tool("points")
        t,z = self.window.displayed_indices
        rows,cols = self.grid.shape[-2:]
        center = np.array([(rows-1)/2,(cols-1)/2])
        angles = np.arange(8)*2*np.pi/8
        circle = np.column_stack([np.sin(angles),np.cos(angles)])
        role = self.role.currentData()
        radius = np.array([rows,cols])*(.22 if role=="myocardium" else .07)
        contour = {"outer_rc":(center+circle*radius).tolist(),
                   "inner_rc":(center+circle*radius*.55).tolist() if role=="myocardium" and self.hole.isChecked() else None}
        self.scope.setCurrentIndex(0)
        self._replace_document(set_track(self.document,role,z,t,contour))
        self._overlay_key = None
        self.on_frame()
        self.status.setText("已放置示例轮廓；位置不是自动识别结果，请手动调整。")

    def remove_track(self):
        if not self._ready():
            return
        updated = copy.deepcopy(self.document)
        z = self.window.displayed_indices[1]
        updated["tracks"] = [r for r in updated["tracks"] if (r["role"],r["slice_index"])!=(self.role.currentData(),z)]
        updated["reviews"] = {k:False for k in updated["reviews"]}
        self._replace_document(updated)
        self._overlay_key = None
        self.on_frame()

    def reset_frame(self):
        if not self._ready():
            return
        t,z = self.window.displayed_indices
        updated = copy.deepcopy(self.document)
        for track in updated["tracks"]:
            if (track["role"],track["slice_index"])==(self.role.currentData(),z):
                track["overrides"] = [r for r in track["overrides"] if r["frame_index"]!=t]
        updated["reviews"] = {k:False for k in updated["reviews"]}
        self._replace_document(updated)
        self._overlay_key = None
        self.on_frame()

    def _hole_changed(self):
        if not self._ready() or self.role.currentData()!="myocardium":
            return
        t,z = self.window.displayed_indices
        track = next((r for r in self.document["tracks"] if r["role"]=="myocardium" and r["slice_index"]==z),None)
        if track is None:
            return
        contour = copy.deepcopy(contour_at(track,t))
        if self.hole.isChecked():
            outer = np.asarray(contour["outer_rc"])
            center = outer.mean(axis=0)
            contour["inner_rc"] = (center+(outer-center)*.55).tolist()
        else:
            contour["inner_rc"] = None
        try:
            self._replace_document(set_track(self.document,"myocardium",z,t,contour,frame_only=self.scope.currentIndex()==1))
            self._overlay_key = None
            self.on_frame()
        except RoiError as error:
            self._failed(0,error.code)

    def _reviews_changed(self):
        if self.document is not None:
            updated = copy.deepcopy(self.document)
            updated["reviews"] = {"orientation_and_coverage":self.orientation_review.isChecked(),
                                  "motion_and_contours":self.motion_review.isChecked()}
            self._replace_document(updated)

    def _context(self):
        return self.window.case_revision,self.key,document_hash(self.document)

    def calculate(self):
        if not self._ready() or self.invalid_edit:
            return
        self.window._stop_play()
        self.window._cancel_prefetch()
        grid,doc,z = self.grid,copy.deepcopy(self.document),self.window.displayed_indices[1]
        self._clear_curve()
        self.last_error = ""
        self._job = ("curve",self._context(),z)
        def calculate(progress,cancel):
            return extract_curves(grid,doc,z,read_plane=lambda t,z:self.window.backend.plane(grid,t,z,None,progress,cancel),
                                  progress=progress,cancel=cancel)
        self.runner.submit(calculate)
        self.refresh_buttons()
        self.status.setText("正在读取当前层的原始时序…")

    def save(self):
        if not self._ready() or self.invalid_edit or not self.document["tracks"]:
            return
        doc,result = copy.deepcopy(self.document),copy.deepcopy(self.result)
        self._job = ("save",self._context())
        self.saved_output = None
        self.runner.submit(lambda progress,cancel:save_snapshot(self.output_root,doc,result))
        self.refresh_buttons()

    def _choose_load(self):
        path,_ = QFileDialog.getOpenFileName(self,"重载当前影像的ROI",str(self.output_root),"ROI JSON (*.json)")
        if path:
            self.load_roi(path)

    def load_roi(self,path):
        if self.grid is None:
            return
        expected = copy.deepcopy(self.document["binding"])
        self._job = ("load",self._context())
        self.last_error = ""
        self.runner.submit(lambda progress,cancel:load_document(path,expected))
        self.refresh_buttons()

    def cancel_analysis(self):
        self.runner.invalidate()
        self._job = None
        self._clear_curve()
        self.status.setText("ROI任务已取消；未发布部分曲线。")
        self.refresh_buttons()

    def present_curves(self, payload):
        if (self.document is None or payload["document_sha256"] != document_hash(self.document)
                or self.window.displayed_indices is None or payload["slice_index"] != self.window.displayed_indices[1]):
            raise RoiError("roi_stale_curve_result")
        self.result = payload
        self.plot.clear()
        for curve in payload["curves"]:
            self.plot.plot(curve["time_s"],curve["mean"],pen=pg.mkPen(COLORS[curve["role"]],width=2),
                           symbol="o",symbolSize=4,name=ROLES[curve["role"]])
        self.plot.setLabel("left","原始均值",units=payload["curves"][0]["units"])
        self._cursor = pg.InfiniteLine(angle=90,movable=False,pen="#9aaec1")
        self.plot.addItem(self._cursor)
        t,z = self.window.displayed_indices
        self._cursor.setValue(self.grid.times_s[z,t])
        self._show_measurement(t)
        pending = bool(payload["qc"]["warnings"])
        self.qc.setText("原始曲线 · 未临床验证\n"+("方向/轮廓运动或分支条件待核对，禁止作为已审核特征。" if pending else "本地人工复核已记录；仍非临床定量验证。"))
        self.status.setText("曲线完成；未平滑、未插值、未重采样。")
        self.curve_ready.emit()
        self.refresh_buttons()

    def _result(self,revision,payload):
        job = self._job
        self._job = None
        if job is None or self.document is None or job[1]!=self._context():
            return
        if job[0]=="curve":
            if self.window.displayed_indices is None or self.window.displayed_indices[1]!=job[2]:
                return
            self.present_curves(payload)
        elif job[0]=="load":
            self._replace_document(payload)
            self.window.set_roi_tool("points")
            self._overlay_key = None
            self.on_frame()
            self.status.setText("ROI已重载，来源/几何/时间匹配；请核对显示轮廓。")
        elif job[0]=="save":
            self.saved_output = payload
            self.status.setText("轮廓与曲线已保存，可从“文件→打开工作目录”查看。")
            self.snapshot_saved.emit()
        self.refresh_buttons()

    def _failed(self,revision,code):
        self._job = None
        self.last_error = code
        self._clear_curve()
        self.status.setText(error_message(code))
        self.status.setToolTip("技术代码："+code)
        self.refresh_buttons()

    def _progress(self,revision,stage,done,total):
        self.status.setText(f"原始曲线 {done}/{total}")

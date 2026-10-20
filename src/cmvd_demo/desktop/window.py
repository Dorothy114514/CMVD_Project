"""D2 local desktop workstation. UI never performs DICOM I/O directly."""

import numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTreeWidget, QTreeWidgetItem, QSplitter, QSlider,
    QDoubleSpinBox, QProgressBar, QHeaderView, QSpinBox,QDockWidget,QTabWidget,QScrollArea,QComboBox,
)
import pyqtgraph as pg

from .data import ReaderBackend, PlaneCache, KINDS, CONDITIONS, SUPPORTED, auto_levels, direction_label, synthetic_choices,perfusion_synthetic_choices
from .tasks import TaskRunner
from .image import PresentedImageItem
from .roi_panel import RoiPanel
from .feature_panel import FeaturePanel
from .roi_editor import RoiViewBox
from .cohort_dialog import CohortDialog

STYLE = """
QMainWindow, QWidget { background: #101b2b; color: #dae4ef; font-size: 13px; }
QLabel#brand { color: #67dbd3; font-size: 22px; font-weight: 600; }
QLabel#muted { color: #9aaec1; }
QLabel#pill { background: #253341; color: #edc58b; padding: 8px; border-radius: 5px; }
QPushButton { background: #24384d; border: 1px solid #395168; border-radius: 5px; padding: 8px 12px; }
QPushButton:hover { background: #31516a; }
QPushButton:disabled { color: #687b8e; border-color: #283b4a; }
QPushButton#primary { background: #14796f; border-color: #36afa0; }
QTreeWidget { background: #142236; border: 1px solid #2d4157; border-radius: 5px; }
QTreeWidget::item { padding: 6px 2px; }
QTreeWidget::item:selected { background: #255367; color: #ecf5fa; }
QHeaderView::section { background: #203248; color: #a9bdce; padding: 7px; border: 0; }
QDoubleSpinBox { background: #172a3c; border: 1px solid #3a5167; padding: 5px; }
QSlider::groove:horizontal { height: 5px; background: #2b4056; border-radius: 2px; }
QSlider::handle:horizontal { background: #65d3c8; width: 14px; margin: -5px 0; border-radius: 6px; }
QProgressBar { background: #172a3c; border: 0; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background: #228c83; }
"""


class BrowserWindow(QMainWindow):
    plane_shown = Signal()
    frame_painted = Signal()
    case_indexed = Signal()

    def __init__(self, backend=None):
        super().__init__()
        self.backend = backend or ReaderBackend()
        self.cache = PlaneCache()
        self.runner = TaskRunner(self)
        self.runner.result.connect(self._result)
        self.runner.failed.connect(self._failed)
        self.runner.progress.connect(self._progress)
        self.runner.idle.connect(self._idle)
        self.prefetch = TaskRunner(self)
        self.prefetch.result.connect(self._prefetched)
        self.prefetch.idle.connect(self._idle)
        self._prefetch_context = None
        self.choices, self.current_choice, self.grid = [], None, None
        self.current_pixels = None
        self.last_error = ""
        self.case_revision = 0
        self._operation = None
        self._closing = False
        self._auto_window = True
        self._needs_fit = True
        self._render_geometry = None
        self._render_photometric = None
        gray = np.arange(256, dtype=np.uint8)
        self._luts = {"MONOCHROME2": np.stack([gray]*3, axis=1),
                      "MONOCHROME1": np.stack([gray[::-1]]*3, axis=1)}
        self._playing = False
        self._advancing = False
        self.displayed_indices = None
        self.index_summary = None
        self._build_ui()
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(45)
        self.debounce.timeout.connect(self._request_plane)
        self.play_timer = QTimer(self)
        self.play_timer.setSingleShot(True)
        self.play_timer.setInterval(round(1000/self.play_fps.value()))
        self.play_timer.timeout.connect(self._next_frame)
        self.roi_panel = RoiPanel(self)
        self.feature_panel = FeaturePanel(self)
        self.cohort_dialog = CohortDialog(self)
        self.cohort_dialog.runner.idle.connect(self._idle)
        tabs = QTabWidget()
        for title,panel in (("ROI / 曲线",self.roi_panel),("D4 灌注特征",self.feature_panel)):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(panel)
            tabs.addTab(scroll,title)
        self.workspace_tabs = tabs
        self.roi_dock = QDockWidget("ROI / 曲线 / 灌注特征",self)
        self.roi_dock.setObjectName("roi-workspace")
        self.roi_dock.setWidget(tabs)
        self.roi_dock.setMinimumWidth(340)
        self.addDockWidget(Qt.RightDockWidgetArea,self.roi_dock)
        self.resizeDocks([self.roi_dock],[360],Qt.Horizontal)
        self.menuBar().addMenu("视图").addAction(self.roi_dock.toggleViewAction())
        demo_menu = self.menuBar().addMenu("合成演示")
        demo_menu.addAction("方向与播放体模",self.open_synthetic)
        demo_menu.addAction("D4首过灌注样例（完全合成）",self.open_synthetic_perfusion)
        self.menuBar().addMenu("队列").addAction("D5队列管理", self.open_cohort)
        demo_menu.addAction("D5队列管理与合成演示", self.open_cohort)
        self.plane_shown.connect(self.roi_panel.on_frame)

    def open_cohort(self):
        self.cohort_dialog.show()
        self.cohort_dialog.raise_()
        self.cohort_dialog.activateWindow()

    def background_runners(self):
        return (self.runner, self.prefetch, self.roi_panel.runner, self.feature_panel.runner, self.cohort_dialog.runner)

    def _build_ui(self):
        self.setWindowTitle("CMVD · PET/MR 研究工作站")
        self.resize(1380, 900)
        self.setMinimumSize(1040, 740)
        self.setStyleSheet(STYLE)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(22, 18, 22, 14)
        layout.setSpacing(12)
        header = QHBoxLayout()
        brand = QLabel("CMVD  /  PET·MR")
        brand.setObjectName("brand")
        header.addWidget(brand)
        header.addWidget(QLabel("研究影像浏览"))
        header.addStretch()
        self.model_label = QLabel("模型未配置")
        self.model_label.setObjectName("pill")
        header.addWidget(self.model_label)
        layout.addLayout(header)
        toolbar = QHBoxLayout()
        self.open_button = QPushButton("打开病例目录")
        self.open_button.setObjectName("primary")
        self.open_button.clicked.connect(self._choose_directory)
        self.synthetic_button = QPushButton("合成体模")
        self.synthetic_button.clicked.connect(self.open_synthetic)
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.clicked.connect(self.cancel_current)
        self.cancel_button.setEnabled(False)
        for button in (self.open_button, self.synthetic_button, self.cancel_button):
            toolbar.addWidget(button)
        self.case_label = QLabel("尚未加载病例")
        toolbar.addWidget(self.case_label)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_layout.addWidget(QLabel("序列与采集候选"))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["序列", "对象数"])
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.currentItemChanged.connect(self._selection_changed)
        self.tree.itemActivated.connect(self._selection_changed)
        left_layout.addWidget(self.tree)
        hint = QLabel("选择单个已解压病例目录。\n序列用途与静息/负荷均需人工核对。")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        left_layout.addWidget(hint)
        splitter.addWidget(left)
        viewer = QWidget()
        viewer_layout = QVBoxLayout(viewer)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.series_label = QLabel("打开病例或合成体模以开始")
        self.series_label.setFont(QFont("Microsoft YaHei UI", 12))
        viewer_layout.addWidget(self.series_label)
        tools = QHBoxLayout()
        tools.addWidget(QLabel("鼠标工具"))
        self.roi_tool = QComboBox()
        for label,key in (("编辑ROI控制点","points"),("移动整个ROI","contour"),("平移图像","pan")):
            self.roi_tool.addItem(label,key)
        self.roi_tool.setCurrentIndex(2)
        self.roi_tool.currentIndexChanged.connect(self._roi_tool_changed)
        tools.addWidget(self.roi_tool)
        self.roi_undo_button = QPushButton("撤销")
        self.roi_redo_button = QPushButton("重做")
        self.roi_undo_button.setToolTip("撤销ROI轮廓更改（Ctrl+Z）；曲线和复核需重做")
        self.roi_redo_button.setToolTip("重做ROI轮廓更改（Ctrl+Y / Ctrl+Shift+Z）")
        self.roi_undo_button.clicked.connect(lambda:self.roi_panel.undo())
        self.roi_redo_button.clicked.connect(lambda:self.roi_panel.redo())
        tools.addWidget(self.roi_undo_button)
        tools.addWidget(self.roi_redo_button)
        tools.addWidget(QLabel("滚轮缩放 · 中键平移"))
        tools.addStretch()
        viewer_layout.addLayout(tools)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("#080f1b")
        self.view = RoiViewBox(lockAspect=True, enableMenu=False)
        self.graphics.addItem(self.view)
        self.view.invertY(True)
        self.image = PresentedImageItem(axisOrder="row-major")
        self.image.frame_painted.connect(self._painted)
        self.view.addItem(self.image)
        self.graphics.setMinimumHeight(320)
        viewer_layout.addWidget(self.graphics, 1)
        self.direction = QLabel("方向：—")
        self.direction.setObjectName("muted")
        viewer_layout.addWidget(self.direction)
        self.cursor_label = QLabel("指针：—")
        self.cursor_label.setObjectName("muted")
        viewer_layout.addWidget(self.cursor_label)
        self.graphics.scene().sigMouseMoved.connect(self._mouse_moved)
        self.time_label, self.slice_label = QLabel("时间/相位：—"), QLabel("层面：—")
        self.time_slider, self.slice_slider = QSlider(Qt.Horizontal), QSlider(Qt.Horizontal)
        self.time_slider.valueChanged.connect(self._position_changed)
        self.slice_slider.valueChanged.connect(self._position_changed)
        for slider in (self.time_slider, self.slice_slider):
            slider.sliderPressed.connect(self._stop_play)
            slider.sliderReleased.connect(self._seek_released)
        for label, slider in ((self.time_label, self.time_slider), (self.slice_label, self.slice_slider)):
            viewer_layout.addWidget(label)
            viewer_layout.addWidget(slider)
            slider.setEnabled(False)
        controls = QHBoxLayout()
        self.play_button = QPushButton("逐帧播放")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._toggle_play)
        self.fit_button = QPushButton("适应视图")
        self.fit_button.clicked.connect(self._fit)
        self.auto_button = QPushButton("自动窗位")
        self.auto_button.clicked.connect(self.reset_levels)
        for button in (self.play_button, self.fit_button, self.auto_button):
            controls.addWidget(button)
        controls.addWidget(QLabel("浏览帧率"))
        self.play_fps = QSpinBox()
        self.play_fps.setRange(1, 30)
        self.play_fps.setValue(15)
        self.play_fps.setSuffix(" fps")
        self.play_fps.valueChanged.connect(self._fps_changed)
        controls.addWidget(self.play_fps)
        controls.addStretch()
        viewer_layout.addLayout(controls)
        controls = QHBoxLayout()
        controls.addStretch()
        self.level_center, self.level_width = QDoubleSpinBox(), QDoubleSpinBox()
        for box in (self.level_center, self.level_width):
            box.setDecimals(3)
            box.setRange(-1e12, 1e12)
            box.setMinimumWidth(120)
            box.valueChanged.connect(self._manual_levels)
        self.level_width.setMinimum(.001)
        controls.addWidget(QLabel("窗位"))
        controls.addWidget(self.level_center)
        controls.addWidget(QLabel("窗宽"))
        controls.addWidget(self.level_width)
        viewer_layout.addLayout(controls)
        self.metadata = QLabel("尚无影像")
        self.metadata.setWordWrap(True)
        viewer_layout.addWidget(self.metadata)
        self.qc_label = QLabel("原始采集平面显示；尚未进行方向人工验收。")
        self.qc_label.setObjectName("pill")
        self.qc_label.setWordWrap(True)
        viewer_layout.addWidget(self.qc_label)
        splitter.addWidget(viewer)
        splitter.setSizes([345, 990])
        layout.addWidget(splitter, 1)
        bottom = QHBoxLayout()
        self.status = QLabel("就绪 · 本地只读浏览")
        bottom.addWidget(self.status, 1)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        bottom.addWidget(self.progress)
        layout.addLayout(bottom)
        foot = QLabel("研究用途 · 当前无诊断输出  |  显示窗位不改变定量像素  |  左键按工具编辑 / 中键平移 / 滚轮缩放")
        foot.setObjectName("muted")
        layout.addWidget(foot)
        self.setCentralWidget(central)

    def _roi_tool_changed(self):
        self.view.set_roi_tool(self.roi_tool.currentData())
        if hasattr(self,"roi_panel"):
            self.roi_panel._overlay_key = None
            self.roi_panel.on_frame()

    def set_roi_tool(self, tool):
        self.roi_tool.setCurrentIndex(self.roi_tool.findData(tool))

    def _choose_directory(self):
        path = QFileDialog.getExistingDirectory(self, "选择单个已解压病例目录（仅本地读取）")
        if path:
            self.open_case(path)

    def _stop_play(self):
        self._playing = False
        self.play_timer.stop()
        self.play_button.setText("逐帧播放")

    def _clear_image(self):
        self.image.clear()
        self.current_pixels = None
        self.displayed_indices = None
        self.cursor_label.setText("指针：—")
        self._render_geometry = None
        self._render_photometric = None
        if hasattr(self,"roi_panel"):
            self.roi_panel.suspend()

    def _cancel_prefetch(self):
        self.prefetch.invalidate()
        self._prefetch_context = None

    def _reset_case(self):
        self.feature_panel.reset_case()
        self.roi_panel.reset_case()
        self.runner.invalidate()
        self._cancel_prefetch()
        self.debounce.stop()
        self._stop_play()
        self.case_revision += 1
        self.cache.clear()
        self.choices, self.current_choice, self.grid = [], None, None
        self.index_summary = None
        self.last_error = ""
        self.tree.clear()
        self._clear_image()
        self.series_label.setText("尚未选择序列")
        self.metadata.setText("尚无影像")
        self.direction.setText("方向：—")
        self.time_label.setText("时间/相位：—")
        self.slice_label.setText("层面：—")
        self.qc_label.setText("原始采集平面显示；尚未进行方向人工验收。")
        for control in (self.time_slider, self.slice_slider, self.play_button):
            control.setEnabled(False)

    def open_case(self, path):
        self._reset_case()
        self.case_label.setText("本地病例 · 正在索引")
        self._operation = ("index",)
        self._busy("正在读取头字段…")
        self.runner.submit(lambda progress, cancel: self.backend.index(path, progress, cancel))

    def open_synthetic(self):
        self._reset_case()
        self.case_label.setText("完全合成 · 无患者数据")
        self._populate(synthetic_choices(), {"indexed_objects": 60, "series_count": 1, "issues": []})

    def open_synthetic_perfusion(self):
        self._reset_case()
        self._populate(perfusion_synthetic_choices(), {"indexed_objects":14,"series_count":2,"issues":[]})

    def _populate(self, choices, summary):
        self.choices, self.index_summary = list(choices), summary
        groups = {}
        preferred = None
        for choice in choices:
            supported = choice.kind in SUPPORTED or choice.kind == "synthetic"
            group_name = CONDITIONS[choice.condition] if supported else "暂不支持显示"
            if group_name not in groups:
                groups[group_name] = QTreeWidgetItem(self.tree, [group_name, ""])
                groups[group_name].setFlags(groups[group_name].flags() & ~Qt.ItemIsSelectable)
            item = QTreeWidgetItem(groups[group_name], [choice.label, str(choice.count)])
            item.setData(0, Qt.UserRole, choice)
            if not supported:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            elif preferred is None or (choice.kind == "pet_dynamic" and choice.condition == "rest"):
                preferred = item
        self.tree.expandAll()
        issue_count = len(summary.get("issues", []))
        synthetic = bool(choices and all(getattr(c.source,"synthetic_id",None) for c in choices))
        self.case_label.setText(f"{'合成体模' if synthetic else '本地病例'} · "
                                f"{summary['series_count']} 序列 / {summary['indexed_objects']} 对象 / {issue_count} 问题")
        self.status.setText("索引结束" if not issue_count else f"索引有 {issue_count} 个问题；当前列表只含成功索引对象")
        if preferred is not None:
            self.tree.setCurrentItem(preferred)
        self.case_indexed.emit()

    def _selection_changed(self, item, previous):
        if item is None:
            return
        choice = item.data(0, Qt.UserRole)
        if choice is None or (choice.kind not in SUPPORTED and choice.kind != "synthetic"):
            return
        self.debounce.stop()
        self._stop_play()
        self._cancel_prefetch()
        self.roi_panel.detach_grid()
        self.current_choice, self.grid = choice, None
        self.last_error = ""
        self._clear_image()
        self.series_label.setText(f"{choice.label}  /  {CONDITIONS[choice.condition]}")
        self.metadata.setText("正在检查空间与时间网格…")
        self.direction.setText("方向：—")
        for slider in (self.time_slider, self.slice_slider):
            slider.setEnabled(False)
        self.play_button.setEnabled(False)
        self._operation = ("grid", choice.key)
        self._auto_window = True
        self._needs_fit = True
        self._busy("正在组装几何/时间轴…")
        self.runner.submit(lambda progress, cancel: self.backend.grid(choice, progress, cancel))

    def select_choice(self, key):
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            for j in range(group.childCount()):
                item = group.child(j)
                choice = item.data(0, Qt.UserRole)
                if choice.key == key:
                    if self.tree.currentItem() == item:
                        self._selection_changed(item, item)
                    else:
                        self.tree.setCurrentItem(item)
                    return
        raise KeyError("choice_not_found")

    def _busy(self, message):
        self.status.setText(message)
        self.progress.setRange(0, 0)
        self.cancel_button.setEnabled(True)

    def _result(self, revision, payload):
        if self._closing or self._operation is None:
            return
        operation = self._operation
        self._operation = None
        if operation[0] == "index":
            self._populate(*payload)
        elif operation[0] == "grid":
            self.grid = payload
            self.roi_panel.set_grid(payload,self.current_choice.key)
            for slider, maximum in ((self.time_slider, payload.shape[0]-1), (self.slice_slider, payload.shape[1]-1)):
                slider.blockSignals(True)
                slider.setRange(0, maximum)
                slider.setValue(0)
                slider.blockSignals(False)
                slider.setEnabled(maximum > 0)
            self.play_button.setEnabled(payload.shape[0] > 1)
            self._request_plane()
        elif operation[0] == "plane":
            _, key, t, z = operation
            self.cache.put(key, payload)
            self.current_pixels = payload
            self.displayed_indices = (t, z)
            self.last_error = ""
            self.image.new_frame()
            self.image.setImage(payload, autoLevels=False)
            dy, dx = self.grid.pixel_spacing_mm
            rows, columns = payload.shape
            geometry = (rows, columns, dy, dx)
            if geometry != self._render_geometry:
                self.image.setRect(QRectF(-dx/2, -dy/2, columns*dx, rows*dy))
                self._render_geometry = geometry
            photometric = getattr(self.grid, "photometric", None)
            if photometric is None:
                photometric = self.grid.cells[t][z].photometric
            if photometric != self._render_photometric:
                self.image.setLookupTable(self._luts[photometric])
                self._render_photometric = photometric
            if self._auto_window:
                self._set_levels(*auto_levels(payload))
                self._auto_window = False
            else:
                self._apply_levels()
            if self._needs_fit:
                self._fit()
                self._needs_fit = False
            self._describe(t, z)
            self.status.setText("影像已显示 · 定量值保持原单位")
            self.plane_shown.emit()
            self._start_prefetch(t, z)

    def _position_changed(self):
        if self.grid is not None:
            if not self._advancing:
                self._stop_play()
            self.runner.invalidate()
            self._cancel_prefetch()
            self._operation = None
            # Keep image, cursor and metadata on the last accepted frame until
            # replacement. Throttle preview requests; do not postpone forever.
            if not self.debounce.isActive():
                self.debounce.start()
            self._busy(self._pending_message())
            self.roi_panel.navigation_pending()

    def _pending_message(self):
        t, z = self.time_slider.value(), self.slice_slider.value()
        message = f"正在加载目标帧 {t+1} / 层面 {z+1}"
        if self.displayed_indices is not None:
            old_t, old_z = self.displayed_indices
            message += f" · 当前仍显示帧 {old_t+1} / 层面 {old_z+1}"
        return message

    def _seek_released(self):
        self.debounce.stop()
        self._request_plane()

    def _request_plane(self):
        if self.grid is None or self._closing:
            return
        t, z = self.time_slider.value(), self.slice_slider.value()
        grid = self.grid
        key = (self.case_revision, self.current_choice.key, t, z)
        cached = self.cache.get(key)
        self._cancel_prefetch()
        self._operation = ("plane", key, t, z)
        self._busy(self._pending_message())
        self.runner.submit(lambda progress, cancel: self.backend.plane(grid, t, z, cached, progress, cancel))

    def _start_prefetch(self, t, z):
        if self.grid is None or self.current_pixels is None or self._closing:
            return
        grid = self.grid
        limit = min(4, max(0, self.cache.max_bytes//self.current_pixels.nbytes-1))
        axis_count = grid.shape[0] if grid.shape[0] > 1 else grid.shape[1]
        limit = min(limit, axis_count-1)
        pending = []
        for offset in range(1, limit+1):
            tt, zz = ((t+offset) % axis_count, z) if grid.shape[0] > 1 else (t, (z+offset) % axis_count)
            key = (self.case_revision, self.current_choice.key, tt, zz)
            if self.cache.get(key) is None:
                pending.append((key, tt, zz))
        if not pending:
            return
        context = self._prefetch_context = (self.case_revision, self.current_choice.key, t, z)
        def load(progress, cancel):
            return [(key, self.backend.plane(grid, tt, zz, None, progress, cancel)) for key, tt, zz in pending]
        self.prefetch.submit(lambda progress, cancel: (context, load(progress, cancel)))

    def _prefetched(self, revision, payload):
        context, planes = payload
        if self._closing or context != self._prefetch_context:
            return
        for key, plane in planes:
            self.cache.put(key, plane)

    def _describe(self, t, z):
        grid = self.grid
        times = grid.times_s
        cardiac = "cardiac" in grid.timing_basis
        label = "相位序号" if cardiac else "时间帧"
        clock = "无时间轴" if times is None else f"{'触发' if cardiac else '参考/相对时间'} {times[z,t]:.3f} s"
        duration = "" if grid.frame_durations_s is None else f" · 帧长 {grid.frame_durations_s[z,t]:.3f} s"
        self.time_label.setText(f"{label} {t+1}/{grid.shape[0]} · {clock}{duration}")
        self.slice_label.setText(f"层面 {z+1}/{grid.shape[1]}")
        orientation = np.asarray(grid.orientation_lps)
        self.direction.setText(f"采集平面方向：上 {direction_label(-orientation[3:])}   下 {direction_label(orientation[3:])}   "
                               f"左 {direction_label(-orientation[:3])}   右 {direction_label(orientation[:3])}   · LPS")
        if hasattr(grid, "cells"):
            header = grid.cells[t][z]
            units = header.units or "MR 原始信号强度（未标定）"
            thickness = header.slice_thickness_mm
        else:
            units, thickness = "合成任意单位", None
        dy, dx = grid.pixel_spacing_mm
        thickness_text = "" if thickness is None else f" · 层厚 {thickness:g} mm"
        centers = getattr(grid, "slice_center_distances_mm", [])
        center_text = "" if len(centers) == 0 else f" · 层中心距 {float(centers[0]):g} mm"
        self.metadata.setText(f"{grid.shape[3]} × {grid.shape[2]} · 行/列间距 {dy:g}/{dx:g} mm{thickness_text}{center_text} · {units}")
        notes = ["原始采集平面；方向人工核对待完成"]
        if getattr(grid,"synthetic_id",None):
            notes = ["完全合成体模；左上亮块用于方向检查，无诊断意义"]
        if cardiac:
            notes.append("心动相位尚未同步，跨层相同序号不代表同时刻")
        if grid.timing_basis == "pet_frame_reference":
            notes.append("PET参考时刻保留原值；不等同于简单帧中点")
        if self.index_summary and self.index_summary.get("issues"):
            notes.append(f"病例索引含 {len(self.index_summary['issues'])} 个问题")
        self.qc_label.setText("；".join(notes))

    def _set_levels(self, low, high):
        for box, value in ((self.level_center, (low+high)/2), (self.level_width, high-low)):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self._apply_levels()

    def _apply_levels(self):
        center, width = self.level_center.value(), self.level_width.value()
        self.image.setLevels((center-width/2, center+width/2))

    def _manual_levels(self):
        self._auto_window = False
        self._apply_levels()

    def reset_levels(self):
        if self.current_pixels is not None:
            self._set_levels(*auto_levels(self.current_pixels))

    def _fit(self):
        if self.image.image is not None:
            self.view.autoRange(padding=.04)

    def _toggle_play(self):
        if self._playing:
            self._stop_play()
        elif self.grid is not None and self.grid.shape[0] > 1:
            self._playing = True
            self.play_button.setText("暂停逐帧")
            self._arm_playback()

    def _fps_changed(self, fps):
        self.play_timer.setInterval(round(1000/fps))

    def _painted(self, serial):
        self.frame_painted.emit()
        self._arm_playback()

    def _arm_playback(self):
        requested = (self.time_slider.value(), self.slice_slider.value())
        if (self._playing and self.displayed_indices == requested and self.current_pixels is not None
                and self.image.painted_serial == self.image.serial and not self.runner.busy
                and not self.debounce.isActive() and not self.play_timer.isActive()):
            self.play_timer.start()

    def _next_frame(self):
        if not self._playing or self.grid is None:
            return
        if not self.isVisible() or self.isMinimized():
            self.play_timer.start()
            return
        if (self.current_pixels is None or self.image.painted_serial != self.image.serial
                or self.runner.busy or self.debounce.isActive()):
            return
        self._advancing = True
        try:
            self.time_slider.setValue((self.time_slider.value()+1) % self.grid.shape[0])
        finally:
            self._advancing = False
        self.debounce.stop()
        self._request_plane()

    def _mouse_moved(self, point):
        if self.current_pixels is None or self.grid is None or self.displayed_indices is None:
            return
        local = self.image.mapFromScene(point)
        column, row = int(np.floor(local.x())), int(np.floor(local.y()))
        if 0 <= row < self.current_pixels.shape[0] and 0 <= column < self.current_pixels.shape[1]:
            t, z = self.displayed_indices
            x, y, zz = self.grid.pixel_to_lps(z, row, column)
            self.cursor_label.setText(f"行 {row} / 列 {column} · 值 {self.current_pixels[row,column]:.5g} · "
                                      f"LPS ({x:.2f}, {y:.2f}, {zz:.2f}) mm")
        else:
            self.cursor_label.setText("指针：影像范围外")

    def _progress(self, revision, stage, done, total):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        names = {"index": "头字段索引", "enumerate": "文件枚举", "pixels": "像素读取"}
        self.status.setText(f"{names.get(stage, '读取')} {done}" + (f"/{total}" if total else ""))

    def _failed(self, revision, code):
        self.last_error = code
        if self._operation and self._operation[0] == "index":
            self.case_label.setText("本地病例 · 索引未完成")
        self._operation = None
        self._clear_image()
        self._cancel_prefetch()
        self._stop_play()
        self.status.setText(f"读取未完成 · {code} · 双击序列重试或打开病例")
        self.qc_label.setText("当前无有效影像；不会输出诊断。错误不代表阴性。")

    def _idle(self):
        if not self.runner.busy and not self.debounce.isActive():
            self.progress.setRange(0, 1)
            self.progress.setValue(1 if self.current_pixels is not None else 0)
            self.cancel_button.setEnabled(False)
        self._arm_playback()
        if self._closing and not any(r.busy for r in self.background_runners()):
            QTimer.singleShot(0, self.close)

    def cancel_current(self):
        self.runner.invalidate()
        self._cancel_prefetch()
        self.debounce.stop()
        self._stop_play()
        if self._operation and self._operation[0] == "index":
            self.case_label.setText("本地病例 · 已取消索引")
        self._operation = None
        self._clear_image()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.cancel_button.setEnabled(False)
        self.status.setText("已取消 · 双击序列重试或打开病例")
        if self.grid is None:
            self.metadata.setText("加载未完成")

    def closeEvent(self, event):
        self._closing = True
        self.cohort_dialog.cancel()
        self.cohort_dialog.hide()
        self.debounce.stop()
        self._stop_play()
        self.runner.invalidate()
        self._cancel_prefetch()
        self._operation = None
        self._clear_image()
        self.cache.clear()
        if any(r.busy for r in self.background_runners()):
            self.hide()
            event.ignore()
        else:
            event.accept()

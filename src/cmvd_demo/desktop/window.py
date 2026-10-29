"""D2 local desktop workstation. UI never performs DICOM I/O directly."""

import numpy as np
from pathlib import Path
from PySide6.QtCore import Qt, QTimer, QRectF, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTreeWidget, QTreeWidgetItem, QSplitter, QSlider,
    QDoubleSpinBox, QProgressBar, QHeaderView, QSpinBox,QDockWidget,QTabWidget,QScrollArea,
    QStackedWidget, QMessageBox, QProgressDialog,
)
import pyqtgraph as pg

from .data import ReaderBackend, PlaneCache, KINDS, CONDITIONS, SUPPORTED, auto_levels, direction_label, synthetic_choices,perfusion_synthetic_choices
from .tasks import TaskRunner
from .image import PresentedImageItem
from .roi_panel import RoiPanel
from .feature_panel import FeaturePanel
from .roi_editor import RoiViewBox
from .cohort_dialog import CohortDialog
from .work_controller import WorkController
from .curve_link import CurveLink
from .sequence_panel import SequencePanel
from .comparison_panel import ComparisonController
from .research_controller import ResearchController
from .presentation import PRODUCT_NAME, PRODUCT_SUBTITLE, PRODUCT_VERSION, WINDOW_TITLE, error_message
from cmvd_demo.paths import workspace_root
from .unsaved import UnsavedWork
from .dialogs import fit_window
from .diagnostics import record_error
from .select_box import SelectBox

STYLE = """
QMainWindow, QWidget { background: #101b2b; color: #dae4ef; font-family: 'Microsoft YaHei UI'; font-size: 13px; }
QLabel#brand { color: #ecf5fa; font-size: 23px; font-weight: 600; }
QLabel#subtitle { color: #85b9c6; font-size: 13px; }
QLabel#section { color: #c6d9e7; font-size: 14px; font-weight: 600; padding: 5px 0; }
QLabel#empty-title { color: #ecf5fa; font-size: 21px; font-weight: 600; }
QWidget#empty-view, QWidget#empty-view QLabel { background: #080f1b; }
QLabel#research-badge { background: #183c43; color: #78d3cd; padding: 7px 12px; border-radius: 5px; }
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
QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit { background: #172a3c; border: 1px solid #3a5167; padding: 6px; border-radius: 4px; }
QPlainTextEdit { background: #111f30; border: 1px solid #2d4157; border-radius: 4px; padding: 5px; }
QTabWidget::pane { border: 0; border-top: 1px solid #304658; }
QTabBar::tab { background: #17283b; color: #94aebf; padding: 10px 18px; border: 0; }
QTabBar::tab:selected { background: #203c4c; color: #8ce0d7; border-bottom: 2px solid #59c5bb; }
QDockWidget::title { background: #162638; padding: 8px; }
QMenuBar { background: #101b2b; padding: 3px 10px; }
QMenuBar::item { padding: 5px 10px; }
QMenuBar::item:selected, QMenu::item:selected { background: #255367; }
QMenu { background: #17283b; border: 1px solid #395168; padding: 5px; }
QMenu::item { padding: 8px 24px; }
QSlider::groove:horizontal { height: 5px; background: #2b4056; border-radius: 2px; }
QSlider::handle:horizontal { background: #65d3c8; width: 14px; margin: -5px 0; border-radius: 6px; }
QProgressBar { background: #172a3c; border: 0; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background: #228c83; }
"""


class BrowserWindow(QMainWindow):
    plane_shown = Signal()
    frame_painted = Signal()
    case_indexed = Signal()
    case_changing = Signal()

    def __init__(self, backend=None, *, unsaved_prompt=None):
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
        self._close_approved = False
        self._pending_index = False
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
        for title,panel in (("轮廓与曲线",self.roi_panel),("灌注分析",self.feature_panel)):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(panel)
            tabs.addTab(scroll,title)
        self.workspace_tabs = tabs
        self.roi_dock = QDockWidget("影像分析",self)
        self.roi_dock.setObjectName("roi-workspace")
        self.roi_dock.setWidget(tabs)
        self.roi_dock.setMinimumWidth(360)
        self.addDockWidget(Qt.RightDockWidgetArea,self.roi_dock)
        self.resizeDocks([self.roi_dock],[380],Qt.Horizontal)
        self.file_menu = self.menuBar().addMenu("文件")
        self.file_menu.addAction("打开病例目录…", self._choose_directory)
        self.file_menu.addAction("打开工作目录", self.open_workspace)
        self.view_menu = self.menuBar().addMenu("视图")
        self.view_menu.addAction(self.roi_dock.toggleViewAction())
        self.study_menu = self.menuBar().addMenu("研究数据")
        self.study_menu.addAction("研究队列与患者划分", self.open_cohort)
        self.demo_menu = self.menuBar().addMenu("合成演示")
        self.synthetic_demo_action = self.demo_menu.addAction("影像浏览与轮廓练习（完全合成）",self.open_synthetic)
        self.perfusion_demo_action = self.demo_menu.addAction("MR首过灌注分析（完全合成）",self.open_synthetic_perfusion)
        self.cohort_demo_action = self.demo_menu.addAction("研究队列示例（完全合成）", self.open_cohort_demo)
        self.help_menu = self.menuBar().addMenu("帮助")
        self.help_menu.addAction("操作指引", self.show_usage)
        self.help_menu.addAction("关于"+PRODUCT_NAME, self.show_about)
        self.plane_shown.connect(self.roi_panel.on_frame)
        self.work = WorkController(self)
        self.curve_link = CurveLink(self)
        self.sequences = SequencePanel(self)
        self.comparison = ComparisonController(self)
        self.research = ResearchController(self)
        self.unsaved = UnsavedWork(self, unsaved_prompt)
        self._index_dialog = QProgressDialog('正在核对新病例，当前工作保留至读取成功。','取消读取',0,0,self)
        self._index_dialog.setWindowTitle('读取病例')
        self._index_dialog.setWindowModality(Qt.WindowModal)
        self._index_dialog.setAutoClose(False)
        self._index_dialog.setAutoReset(False)
        self._index_dialog.canceled.connect(self.cancel_current)
        # QProgressDialog owns a delayed force-show timer. hide() alone leaves
        # it armed, including directly after construction with an infinite range.
        self._index_dialog.reset()
        self._index_dialog.hide()
        fit_window(self,1380,900)
        self.workspace_tabs.currentChanged.connect(lambda index: self.roi_panel.drawing.cancel() if index!=0 else None)
        self.roi_dock.visibilityChanged.connect(lambda visible: self.roi_panel.drawing.cancel() if not visible else None)

    def open_cohort(self):
        self.cohort_dialog.show()
        self.cohort_dialog.raise_()
        self.cohort_dialog.activateWindow()

    def open_cohort_demo(self):
        self.open_cohort()
        self.cohort_dialog.run_action("synthetic")

    def show_usage(self):
        QMessageBox.information(self, "操作指引",
            "1. 打开单个已解压病例目录，选择影像序列。\n"
            "2. 用时间、层面滑块浏览；滚轮缩放，中键平移。\n"
            "3. 在“轮廓与曲线”勾画心肌、血池，计算并保存。\n"
            "4. 在“灌注分析”核对静息/负荷与分析窗口。\n\n"
            "练习请从“合成演示”菜单进入；保存结果可从“文件→打开工作目录”查看。")

    def show_about(self):
        QMessageBox.information(self, PRODUCT_NAME,
            f"{PRODUCT_NAME}  {PRODUCT_VERSION}\n{PRODUCT_SUBTITLE}\n\n"
            "用于影像浏览、轮廓勾画、灌注分析与研究队列整理。\n"
            "研究预览版；当前尚无可用诊断模型。")

    def open_workspace(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(workspace_root())))

    def background_runners(self):
        runners = (self.runner, self.prefetch, self.roi_panel.runner, self.feature_panel.runner, self.cohort_dialog.runner)
        if hasattr(self,"work"):runners+=(self.work.runner,)
        if hasattr(self,"sequences"):runners+=(self.sequences.runner,)
        if hasattr(self,"comparison"):runners+=tuple(p.runner for p in self.comparison.panes.values())
        if hasattr(self,"research"):runners+=(self.research.runner,)
        return runners

    def _build_ui(self):
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1380, 900)
        self.setMinimumSize(760, 480)
        self.setStyleSheet(STYLE)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 14, 18, 12)
        layout.setSpacing(10)
        header = QHBoxLayout()
        brand = QLabel(PRODUCT_NAME)
        brand.setObjectName("brand")
        header.addWidget(brand)
        subtitle = QLabel(PRODUCT_SUBTITLE)
        subtitle.setObjectName("subtitle")
        header.addWidget(subtitle)
        header.addStretch()
        research_badge = QLabel("研究预览")
        research_badge.setObjectName("research-badge")
        header.addWidget(research_badge)
        self.model_label = QLabel("模型未配置")
        self.model_label.setObjectName("pill")
        header.addWidget(self.model_label)
        layout.addLayout(header)
        toolbar = QHBoxLayout()
        self.open_button = QPushButton("打开病例目录")
        self.open_button.setObjectName("primary")
        self.open_button.clicked.connect(self._choose_directory)
        self.save_work_button = QPushButton("保存工作")
        self.save_work_button.clicked.connect(lambda:self.work.save())
        self.restore_work_button = QPushButton("恢复工作…")
        self.restore_work_button.clicked.connect(lambda:self.work.choose_restore())
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.clicked.connect(self.cancel_current)
        self.cancel_button.setEnabled(False)
        for button in (self.open_button, self.save_work_button, self.restore_work_button, self.cancel_button):
            toolbar.addWidget(button)
        self.case_label = QLabel("尚未加载病例")
        toolbar.addWidget(self.case_label)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left.setMinimumWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 10, 0)
        series_heading = QLabel("影像序列")
        series_heading.setObjectName("section")
        left_layout.addWidget(series_heading)
        self.sequence_filter=SelectBox()
        for label,key in (("全部序列","all"),("可尝试浏览","readable"),("有问题 / 提示","problems"),
                          ("仅索引","index_only"),("未预览","pending"),("已预览","previewed")):
            self.sequence_filter.addItem(label,key)
        left_layout.addWidget(self.sequence_filter)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["序列", "影像数", "状态"])
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tree.currentItemChanged.connect(self._selection_changed)
        self.tree.itemActivated.connect(self._selection_changed)
        left_layout.addWidget(self.tree)
        self.sequence_button=QPushButton("序列状态与问题清单")
        left_layout.addWidget(self.sequence_button)
        self.compare_button=QPushButton("静息／负荷对照查看");left_layout.addWidget(self.compare_button)
        self.research_button=QPushButton("研究记录导出与核对");left_layout.addWidget(self.research_button)
        hint = QLabel("选择单个已解压病例目录。\n序列用途与静息/负荷均需人工核对。")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        left_layout.addWidget(hint)
        splitter.addWidget(left)
        viewer = QWidget()
        viewer_layout = QVBoxLayout(viewer)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.series_label = QLabel("打开病例目录以开始浏览")
        self.series_label.setObjectName("section")
        viewer_layout.addWidget(self.series_label)
        tools = QHBoxLayout()
        tools.addWidget(QLabel("鼠标工具"))
        self.roi_tool = SelectBox()
        self.roi_tool.setAccessibleName("鼠标工具")
        for label,key in (("编辑轮廓点","points"),("移动整个轮廓","contour"),("平移影像","pan")):
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
        self.viewer_stack = QStackedWidget()
        self.empty_view = QWidget()
        self.empty_view.setObjectName("empty-view")
        empty_layout = QVBoxLayout(self.empty_view)
        empty_layout.setContentsMargins(24, 24, 24, 24)
        empty_layout.addStretch()
        self.empty_title = QLabel("打开病例，开始影像评阅")
        self.empty_title.setObjectName("empty-title")
        self.empty_description = QLabel("选择单个已解压病例目录。\n浏览PET/MR序列，勾画心肌与血池，保存曲线与分析结果。")
        self.empty_description.setObjectName("muted")
        for label in (self.empty_title, self.empty_description):
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignCenter)
            empty_layout.addWidget(label)
        self.empty_open_button = QPushButton("打开病例目录")
        self.empty_open_button.setObjectName("primary")
        self.empty_open_button.clicked.connect(self._choose_directory)
        empty_layout.addSpacing(14)
        empty_layout.addWidget(self.empty_open_button, 0, Qt.AlignHCenter)
        empty_layout.addStretch()
        self.viewer_stack.addWidget(self.empty_view)
        self.viewer_stack.addWidget(self.graphics)
        viewer_layout.addWidget(self.viewer_stack, 1)
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
        self.auto_button = QPushButton("自动窗宽窗位")
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
        splitter.setSizes([245, 900])
        layout.addWidget(splitter, 1)
        bottom = QHBoxLayout()
        self.status = QLabel("就绪 · 本地只读浏览")
        self.status.setWordWrap(True)
        bottom.addWidget(self.status, 1)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        bottom.addWidget(self.progress)
        layout.addLayout(bottom)
        foot = QLabel("研究用途 · 当前无诊断输出  |  窗宽窗位仅影响显示，原始定量值保持不变")
        foot.setObjectName("muted")
        layout.addWidget(foot)
        # Keep buttons legible and all controls reachable when the screen is small.
        central.setMinimumSize(1000,780)
        viewport=QScrollArea()
        viewport.setWidgetResizable(True)
        viewport.setWidget(central)
        self.setCentralWidget(viewport)

    def _roi_tool_changed(self):
        if hasattr(self,"roi_panel"):
            self.roi_panel.drawing.cancel()
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
        self.viewer_stack.setCurrentWidget(self.empty_view)
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
        if self._pending_index:
            self._finish_index()
        self.case_changing.emit()
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
        if hasattr(self,'unsaved'):
            self.unsaved.mark_saved()

    def open_case(self, path):
        path=Path(path)
        if not path.is_dir():
            self.last_error='case_directory_missing'
            self.status.setText(error_message(self.last_error)+' 当前工作已保留。')
            self.status.setToolTip('技术代码：'+self.last_error)
            return
        self.unsaved.request('打开另一病例',lambda:self._begin_index(path))

    def _begin_index(self,path):
        self._stop_play()
        self.debounce.stop()
        self._cancel_prefetch()
        self.runner.invalidate()
        self.cache.clear()
        self.roi_panel.runner.invalidate();self.roi_panel._job=None
        self.feature_panel.runner.invalidate();self.feature_panel._job=None
        self.comparison.cancel()
        self.work.cancel()
        self.research.cancel()
        self.last_error=''
        self._pending_index=True
        self._index_previous_image=(self.current_pixels,self.displayed_indices,self.viewer_stack.currentWidget())
        self.current_pixels=self.displayed_indices=None
        self.viewer_stack.setCurrentWidget(self.empty_view)
        self.empty_title.setText("正在读取病例")
        self.empty_description.setText("正在检查新病例；当前工作将在成功索引后才替换。失败或取消保留原工作。")
        self._operation = ("index",)
        self._busy("正在读取头字段…")
        self._index_dialog.reset()
        self._index_dialog.setRange(0,0)
        self._index_dialog.setValue(0)
        self._index_dialog.show()
        self.runner.submit(lambda progress, cancel: self.backend.index(path, progress, cancel))

    def _finish_index(self):
        self._pending_index=False
        self._index_dialog.reset()
        self._index_dialog.hide()
        if hasattr(self,'_index_previous_image'):
            self.current_pixels,self.displayed_indices,previous=self._index_previous_image
            del self._index_previous_image
            self.viewer_stack.setCurrentWidget(previous)

    def open_synthetic(self):
        self.unsaved.request('切换合成演示',self._open_synthetic)

    def _open_synthetic(self):
        self._reset_case()
        self.case_label.setText("完全合成 · 无患者数据")
        self._populate(synthetic_choices(), {"indexed_objects": 60, "series_count": 1, "issues": []})

    def open_synthetic_perfusion(self):
        self.unsaved.request('切换合成演示',self._open_synthetic_perfusion)

    def _open_synthetic_perfusion(self):
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
            if supported and (preferred is None or (choice.kind == "pet_dynamic" and choice.condition == "rest")):
                preferred = item
        self.tree.expandAll()
        issue_count = len(summary.get("issues", []))
        synthetic = bool(choices and all(getattr(c.source,"synthetic_id",None) for c in choices))
        self.case_label.setText(f"{'合成体模' if synthetic else '本地病例'} · "
                                f"{summary['series_count']} 序列 / {summary['indexed_objects']} 对象 / {issue_count} 问题")
        self.status.setText("索引结束" if not issue_count else f"索引有 {issue_count} 个问题；当前列表只含成功索引对象")
        self.sequences.populate(choices,summary)
        if preferred is not None:
            self.tree.setCurrentItem(preferred)
        else:
            self.empty_title.setText("暂无可预览序列")
            self.empty_description.setText("请检查左侧序列类型。仅索引的影像暂不支持显示，可用独立查看器核对。")
        self.case_indexed.emit()

    def _selection_changed(self, item, previous):
        if item is None:
            return
        choice = item.data(0, Qt.UserRole)
        if choice is None:
            return
        if self._operation is not None:
            self.sequences.cancelled()
        self.debounce.stop()
        self._stop_play()
        self._cancel_prefetch()
        self.roi_panel.detach_grid()
        self.current_choice, self.grid = choice, None
        self.last_error = ""
        self._clear_image()
        self.empty_title.setText("正在准备影像")
        self.empty_description.setText("正在核对序列的空间与时间信息。")
        self.series_label.setText(f"{choice.label}  /  {CONDITIONS[choice.condition]}")
        self.metadata.setText("正在检查空间与时间网格…")
        self.direction.setText("方向：—")
        self.time_label.setText("时间/相位：—")
        self.slice_label.setText("层面：—")
        for slider in (self.time_slider, self.slice_slider):
            slider.setEnabled(False)
        self.play_button.setEnabled(False)
        if choice.kind not in SUPPORTED and choice.kind!='synthetic':
            self.runner.invalidate();self._operation=None
            self.empty_title.setText("此序列目前仅索引")
            self.empty_description.setText("此类影像暂不支持显示，可用独立查看器核对。\n仅索引不代表源文件损坏；详情见“序列状态与问题清单”。")
            self.metadata.setText(KINDS.get(choice.kind,"其他对象")+" · "+str(choice.count)+"个对象")
            self.status.setText("仅索引 · 尚未尝试像素显示，未进行完整性或临床审核")
            self.sequences.refresh()
            return
        self._operation = ("grid", choice.key)
        self._auto_window = True
        self._needs_fit = True
        self._busy("正在组装几何/时间轴…")
        self.sequences.begin(choice.key,'grid')
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
        self.status.setToolTip("")
        self.status.setText(message)
        self.progress.setRange(0, 0)
        self.cancel_button.setEnabled(True)

    def _result(self, revision, payload):
        if self._closing or self._operation is None:
            return
        operation = self._operation
        self._operation = None
        if operation[0] == "index":
            if not payload[0]:
                self._failed(revision,'case_no_indexed_objects')
                return
            self._finish_index()
            self._reset_case()
            self._populate(*payload)
        elif operation[0] == "grid":
            self._accept_grid(payload)
            self._request_plane()
        elif operation[0] == "plane":
            _, key, t, z = operation
            self._accept_plane(payload, t, z)

    def _accept_grid(self, payload):
        self.grid = payload
        self.sequences.grid_ok(payload)
        self.roi_panel.set_grid(payload,self.current_choice.key)
        for slider, maximum in ((self.time_slider, payload.shape[0]-1), (self.slice_slider, payload.shape[1]-1)):
            slider.blockSignals(True)
            slider.setRange(0, maximum)
            slider.setValue(0)
            slider.blockSignals(False)
            slider.setEnabled(maximum > 0)
        self.play_button.setEnabled(payload.shape[0] > 1)

    def _accept_plane(self, payload, t, z):
        key = (self.case_revision, self.current_choice.key, t, z)
        self.cache.put(key, payload)
        self.current_pixels = payload
        self.displayed_indices = (t, z)
        self.last_error = ""
        self.image.new_frame()
        self.viewer_stack.setCurrentWidget(self.graphics)
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
        self.sequences.begin(self.current_choice.key,'pixels',[t,z])
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
        if self._pending_index:
            self._operation=None
            self._finish_index()
            self.last_error=code
            self.status.setText(error_message(code)+' 当前工作已保留。')
            self.status.setToolTip('技术代码：'+code)
            self.progress.setRange(0,1)
            self.cancel_button.setEnabled(False)
            return
        self.last_error = code
        self.sequences.failure(self._operation,code)
        if self._operation and self._operation[0] == "index":
            self.case_label.setText("本地病例 · 索引未完成")
        self._operation = None
        self._clear_image()
        self._cancel_prefetch()
        self._stop_play()
        message = error_message(code)
        self.empty_title.setText("此序列暂未能显示")
        self.empty_description.setText(message+"\n可双击序列重试，或选择其他序列。")
        self.status.setText(message)
        self.status.setToolTip("技术代码："+code)
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
        if self._pending_index:
            self.runner.invalidate()
            self._operation=None
            self._finish_index()
            self.status.setText('新病例读取已取消，当前工作已保留。')
            self.cancel_button.setEnabled(False)
            return
        if hasattr(self,"work") and self.work.busy:
            self.work.cancel()
            return
        self.sequences.cancelled()
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
        self.empty_title.setText("已取消影像加载")
        self.empty_description.setText("可双击左侧序列重新加载，或打开其他病例。")
        if self.grid is None:
            self.metadata.setText("加载未完成")

    def closeEvent(self, event):
        if self._pending_index:
            self.cancel_current()
        if not self._closing and not self._close_approved and self.unsaved.dirty():
            event.ignore()
            self.unsaved.request('退出工作站',self._approve_close)
            return
        self._closing = True
        self.comparison.cancel()
        self.research.cancel();self.research.dialog.hide()
        self.sequences.cancel()
        self.sequences.dialog.hide()
        self.work.cancel()
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

    def _approve_close(self):
        self._close_approved=True
        QTimer.singleShot(0,self.close)

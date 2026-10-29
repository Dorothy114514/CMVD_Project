"""D4 manual condition selection, MR preview and local PET workstation bridge."""

import copy
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
                              QPushButton, QLineEdit, QSpinBox, QCheckBox,
                              QPlainTextEdit, QFileDialog, QGroupBox)

from cmvd_demo.io import ReadCancelled
from cmvd_demo.roi import document_hash
from cmvd_demo.paths import workspace_root
from cmvd_demo.perfusion import (FeatureError, mr_features, pair_mr, pet_request, import_pet,
                                compare_reference, build_bundle, case_token_for_choices,
                                load_json, save_features, save_request)
from .tasks import TaskRunner
from .presentation import error_message, unit_label
from .curve_link import linked_plot
from .select_box import SelectBox


class FeaturePanel(QWidget):
    task_done = Signal()
    parameters_changed = Signal()

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.runner = TaskRunner(self)
        self.runner.result.connect(self._result)
        self.runner.failed.connect(self._failed)
        self.runner.idle.connect(self.refresh_buttons)
        self.runner.idle.connect(window._idle)
        self.output_root = workspace_root()/"outputs"/"features"
        self.mr_slots, self.pet_slots = {}, {}
        self.candidate = self.pet = self.mr_reference = self.saved_output = None
        self.last_error = ""
        self.single_saved_output = None
        self._job = None
        self._candidate_context = None
        layout = QVBoxLayout(self)
        notice = QLabel("灌注分析 · 研究预览\n先在“轮廓与曲线”计算心肌、血池曲线，再核对采集状态与分析窗口。")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.curve_controls = QHBoxLayout()
        layout.addLayout(self.curve_controls)
        self.curve_plot = linked_plot(self)
        layout.addWidget(self.curve_plot)
        self.curve_position = QLabel("当前曲线：—")
        self.curve_position.setWordWrap(True)
        layout.addWidget(self.curve_position)
        self.curve_hint = QLabel("先计算原始曲线，再选择基线与首过区间。")
        self.curve_hint.setWordWrap(True)
        layout.addWidget(self.curve_hint)
        parameters = QGroupBox("采集信息与分析窗口")
        form = QFormLayout(parameters)
        self.condition = SelectBox()
        self.condition.setAccessibleName("已核对的采集状态")
        self.condition.addItem("请选择已核对状态", None)
        self.condition.setToolTip("核对采集协议后再选择静息或负荷；待核对状态不能用于计算分析。")
        self.condition.addItem("静息", "rest")
        self.condition.addItem("负荷", "stress")
        self.protocol = QLineEdit()
        self.protocol.setMaxLength(80)
        self.protocol.setPlaceholderText("协议代码（无患者身份信息）")
        self.baseline_start,self.baseline, self.start, self.end = QSpinBox(),QSpinBox(), QSpinBox(), QSpinBox()
        for spin, value in ((self.baseline_start,1),(self.baseline, 3), (self.start, 4), (self.end, 20)):
            spin.setRange(1, 10000)
            spin.setValue(value)
        form.addRow("采集状态", self.condition)
        form.addRow("可比较协议代码", self.protocol)
        form.addRow("基线起始帧（从1计）",self.baseline_start)
        form.addRow("基线帧数", self.baseline)
        form.addRow("首过起始帧（从1计）", self.start)
        form.addRow("首过结束帧（含）", self.end)
        layout.addWidget(parameters)
        self.window_review = QCheckBox("已按曲线核对基线与首过窗口")
        layout.addWidget(self.window_review)
        mr_heading = QLabel("MR 心肌灌注")
        mr_heading.setObjectName("section")
        layout.addWidget(mr_heading)
        self.mr_button = QPushButton("计算当前MR平面")
        self.mr_button.clicked.connect(self.calculate_mr)
        self.capture_mr_button = QPushButton("保留MR结果至静息 / 负荷")
        self.capture_mr_button.clicked.connect(self.capture_mr)
        self.capture_pet_button = QPushButton("保留PET曲线至静息 / 负荷")
        self.capture_pet_button.clicked.connect(self.capture_pet)
        for button in (self.mr_button, self.capture_mr_button):
            layout.addWidget(button)
        self.single_save_button = QPushButton("保存当前MR分析")
        self.single_save_button.setToolTip("独立保存当前单状态MR结果、轮廓和曲线，不需要静息/负荷配对。")
        self.single_save_button.clicked.connect(lambda:self.window.work.save(single=True))
        self.single_save_button.setEnabled(False)
        layout.addWidget(self.single_save_button)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(110)
        layout.addWidget(self.summary, 1)
        self.pair_summary = QLabel("MR：尚未保留静息/负荷配对。")
        self.pair_summary.setWordWrap(True)
        self.pair_summary.setObjectName("pill")
        layout.addWidget(self.pair_summary)
        self.pair_review = QCheckBox("已确认MR两状态及同一解剖平面")
        self.protocol_review = QCheckBox("已确认MR剂量、序列及协议可比较")
        layout.addWidget(self.pair_review)
        layout.addWidget(self.protocol_review)
        pet_heading = QLabel("PET 工作站结果")
        pet_heading.setObjectName("section")
        layout.addWidget(pet_heading)
        layout.addWidget(self.capture_pet_button)
        row = QHBoxLayout()
        self.request_button = QPushButton("导出PET结果模板")
        self.request_button.clicked.connect(self.export_request)
        self.import_button = QPushButton("导入PET工作站结果")
        self.import_button.clicked.connect(self.choose_pet)
        row.addWidget(self.request_button)
        row.addWidget(self.import_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.reference_template_button = QPushButton("导出MR参考模板")
        self.reference_template_button.clicked.connect(self.export_reference_request)
        self.reference_button = QPushButton("核对MR参考结果")
        self.reference_button.clicked.connect(self.choose_reference)
        row.addWidget(self.reference_template_button)
        row.addWidget(self.reference_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.save_button = QPushButton("保存分析与质控结果")
        self.save_button.clicked.connect(self.save)
        self.cancel_button = QPushButton("取消特征任务")
        self.cancel_button.clicked.connect(self.cancel)
        row.addWidget(self.save_button)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)
        self.status = QLabel("每次只配对一个MR平面；不会自动平均不同平面。\n无PET结果或MR参考时仍可保存预览，缺项明确标记。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        for control in (self.baseline_start, self.baseline, self.start, self.end):
            control.valueChanged.connect(self.window_parameters_changed)
        self.condition.currentIndexChanged.connect(self.analysis_settings_changed)
        self.protocol.textChanged.connect(self.analysis_settings_changed)
        self.window_review.toggled.connect(self.analysis_settings_changed)
        self.pair_review.toggled.connect(self._pair_changed)
        self.protocol_review.toggled.connect(self._pair_changed)
        window.roi_panel.state_changed.connect(self.context_changed)
        window.roi_panel.curve_ready.connect(self.context_changed)
        window.plane_shown.connect(self.context_changed)
        self.refresh_buttons()

    def _context(self):
        p = self.window.roi_panel
        return (self.window.case_revision, p.key, document_hash(p.document) if p.document else None,
                self.window.displayed_indices[1] if self.window.displayed_indices else None)

    def invalidate_candidate(self, *args):
        self.runner.invalidate()
        self._job = None
        self.candidate = None
        self._candidate_context = None
        self.summary.clear()
        self.refresh_buttons()

    def window_parameters_changed(self, *args):
        self.window._stop_play()
        previous = self.window_review.blockSignals(True)
        self.window_review.setChecked(False)
        self.window_review.blockSignals(previous)
        self.analysis_settings_changed()

    def analysis_settings_changed(self, *args):
        p = self.window.roi_panel
        z = self.window.displayed_indices[1] if self.window.displayed_indices else None
        removed = [c for c,slot in self.mr_slots.items() if slot['key']==p.key and slot['feature']['slice_index']==z]
        for condition in removed:
            del self.mr_slots[condition]
        if removed:
            self.mr_reference = None
            self.pair_review.setChecked(False)
            self.protocol_review.setChecked(False)
            self._update_pair_summary()
        self.single_saved_output = self.saved_output = None
        self.invalidate_candidate()
        self.parameters_changed.emit()

    def _pair_changed(self, *args):
        self.mr_reference = None
        self.runner.invalidate()
        self._job = None
        self._update_pair_summary()
        self.refresh_buttons()

    def reset_case(self):
        self.single_saved_output = None
        self.invalidate_candidate()
        self.mr_slots.clear()
        self.pet_slots.clear()
        self.pet = self.mr_reference = self.saved_output = None
        self.summary.clear()
        self.status.setText("新病例：尚无特征或配对。")
        self.condition.setCurrentIndex(0)
        self.pair_review.setChecked(False)
        self.protocol_review.setChecked(False)
        self.window_review.setChecked(False)
        self._update_pair_summary()
        self.refresh_buttons()

    def context_changed(self):
        context = self._context()
        if ((self.candidate is not None and self._candidate_context != context)
                or (self._job is not None and self._job[1] != context)):
            self.invalidate_candidate()
        p = self.window.roi_panel
        removed = False
        for slots in (self.mr_slots, self.pet_slots):
            for condition, record in list(slots.items()):
                doc = p.documents.get(record["key"])
                if (doc is None or document_hash(doc) != document_hash(record["document"])
                        or (p.invalid_edit and record["key"]==p.key)):
                    del slots[condition]
                    removed = True
        if removed:
            self.pet = self.mr_reference = None
            self.summary.clear()
            self.status.setText("ROI或复核状态已修改；相关保留结果失效，请重新计算/导入。")
            self._update_pair_summary()
        if not p.result:
            self.runner.invalidate()
            self._job = None
            if self.candidate is not None:
                self.invalidate_candidate()
            blocked = self.window_review.blockSignals(True)
            self.window_review.setChecked(False)
            self.window_review.blockSignals(blocked)
        self.refresh_buttons()

    def refresh_buttons(self):
        p = self.window.roi_panel
        ready = p.result is not None and p._ready() and not p.invalid_edit and not self.runner.busy
        selected = self.condition.currentData() is not None
        valid = not hasattr(self.window,'curve_link') or self.window.curve_link.valid_window()
        self.mr_button.setEnabled(ready and selected and p.grid.kind == "mr_perfusion" and valid)
        self.capture_pet_button.setEnabled(ready and selected and p.grid.kind == "pet_dynamic")
        self.capture_mr_button.setEnabled(self.candidate is not None and not self.runner.busy)
        for button in (self.request_button, self.import_button):
            button.setEnabled(len(self.pet_slots) == 2 and not self.runner.busy)
        for button in (self.reference_button, self.reference_template_button):
            button.setEnabled(len(self.mr_slots) == 2 and not self.runner.busy)
        self.save_button.setEnabled((len(self.mr_slots) == 2 or self.pet is not None) and not self.runner.busy)
        self.cancel_button.setEnabled(self.runner.busy)
        if hasattr(self.window,"work"):
            self.window.work.refresh()
        if hasattr(self.window,'comparison'):self.window.comparison.refresh()

    def _record(self):
        p = self.window.roi_panel
        if not p._ready() or p.result is None or p.invalid_edit:
            raise FeatureError("feature_current_curves_required")
        return {"document": copy.deepcopy(p.document), "curves": copy.deepcopy(p.result),
                "grid": p.grid, "key": p.key}

    def _submit(self, kind, function):
        self.window._stop_play()
        self._job = kind, self._context()
        self.last_error = ""
        self.saved_output = None
        self.runner.submit(function)
        self.refresh_buttons()

    def calculate_mr(self):
        try:
            record = self._record()
            args = (record["document"], record["curves"], case_token_for_choices(self.window.choices),
                    self.condition.currentData(), self.protocol.text())
            params = dict(baseline_start=self.baseline_start.value()-1,baseline_frames=self.baseline.value(), window_start=self.start.value()-1,
                          window_end=self.end.value()-1, window_reviewed=self.window_review.isChecked())
            self.candidate = None
            self._candidate_context = self._context()
            self._submit("mr", lambda progress, cancel: mr_features(*args, **params))
            self.status.setText("正在计算MR特征…")
        except FeatureError as error:
            self._failed(0, error.code)

    def capture_mr(self):
        if self.candidate is None or self._candidate_context != self._context():
            return
        record = self._record()
        record["feature"] = copy.deepcopy(self.candidate)
        self.mr_slots[self.candidate["condition"]] = record
        self.mr_reference = None
        self.pair_review.setChecked(False)
        self.protocol_review.setChecked(False)
        self.status.setText("MR已保留："+"、".join("静息" if c == "rest" else "负荷" for c in self.mr_slots)+"；配对审核需重新确认。")
        self._update_pair_summary()
        self.refresh_buttons()

    def capture_pet(self):
        try:
            condition = self.condition.currentData()
            if condition not in ("rest", "stress"):
                raise FeatureError("feature_condition_required")
            record = self._record()
            if record["grid"].kind != "pet_dynamic":
                raise FeatureError("feature_wrong_branch")
            self.pet_slots[condition] = record
            self.pet = None
            self.status.setText("PET本地曲线已保留："+"、".join("静息" if c == "rest" else "负荷" for c in self.pet_slots)+"；工作站结果需另行导入。")
            self.refresh_buttons()
        except FeatureError as error:
            self._failed(0, error.code)

    def _tacs(self):
        return {k: (v["document"], v["curves"]) for k, v in self.pet_slots.items()}

    def _mr_pair(self):
        if len(self.mr_slots) != 2:
            raise FeatureError("feature_mr_pair_required")
        return pair_mr(self.mr_slots["rest"]["feature"], self.mr_slots["stress"]["feature"],
                       pairing_reviewed=self.pair_review.isChecked(),
                       protocol_comparable=self.protocol_review.isChecked())

    def _update_pair_summary(self):
        self.pair_summary.setToolTip("")
        if len(self.mr_slots) != 2:
            self.pair_summary.setText(f"MR：已保留{len(self.mr_slots)}/2个状态，尚未配对。")
            return
        try:
            pair = self._mr_pair()
            ratio = pair["measurements"]["mr_mpri"]["value"]
            self.pair_summary.setText(f"MR灌注储备指数（MPRI，半定量）: {ratio:.6g}\n"+
                                     ("预览结果，配对/分支质控待完成；非绝对MBF。" if pair["qc"]["warnings"]
                                      else "本地配对检查通过；参考与临床验证另记。"))
        except FeatureError as error:
            self.pair_summary.setText("MR暂不能配对："+error_message(error.code))
            self.pair_summary.setToolTip("技术代码："+error.code)

    def export_request(self):
        try:
            request = pet_request(case_token_for_choices(self.window.choices), self._tacs())
            records = list(self.pet_slots.values())
            def work(progress, cancel):
                self._check_records(records, cancel)
                return save_request(self.output_root, request, cancel=cancel)
            self._submit("saved", work)
        except FeatureError as error:
            self._failed(0, error.code)

    def choose_pet(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择已审核PET工作站结果", str(self.output_root), "工作站结果 (*.json)")
        if path:
            self.load_pet(path)

    def load_pet(self, path):
        tacs, case = copy.deepcopy(self._tacs()), case_token_for_choices(self.window.choices)
        self.pet = None
        records = list(self.pet_slots.values())
        def work(progress, cancel):
            self._check_records(records, cancel)
            return import_pet(load_json(path), case, tacs)
        self._submit("pet", work)

    def export_reference_request(self):
        try:
            pair = self._mr_pair()
            request = {"schema_version": 1, "feature_sha256": document_hash(pair), "reference_id": "REPLACE",
                       "reviewer_code": "REPLACE", "reviewed": False, "method": pair["method"],
                       "measurements": {k: {"value": None, "unit": v["unit"], "atol": None, "rtol": None}
                                        for k, v in pair["measurements"].items()}}
            self._submit("saved", lambda progress, cancel: save_request(self.output_root, request,
                         name="mr_reference_request.json", cancel=cancel))
        except FeatureError as error:
            self._failed(0, error.code)

    def choose_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择已审核MR参考结果", str(self.output_root), "分析结果 (*.json)")
        if path:
            self.load_reference(path)

    def load_reference(self, path):
        try:
            pair = self._mr_pair()
            self.mr_reference = None
            def work(progress, cancel):
                ref = load_json(path)
                return ref, compare_reference(pair, ref)
            self._submit("reference", work)
        except FeatureError as error:
            self._failed(0, error.code)

    def save(self):
        try:
            pair = self._mr_pair() if self.mr_slots else None
            bundle = build_bundle(case_token_for_choices(self.window.choices), mr=pair,
                                  pet=self.pet, mr_reference=self.mr_reference)
            records = list(self.mr_slots.values())+list(self.pet_slots.values())
            # Snapshot the source set on the GUI thread; never read widgets in a worker.
            def work(progress, cancel):
                self._check_records(records, cancel)
                return save_features(self.output_root, bundle, cancel=cancel)
            self._submit("saved", work)
        except FeatureError as error:
            self._failed(0, error.code)

    @staticmethod
    def _check_records(records, cancel):
        for record in records:
            if cancel():
                raise ReadCancelled()
            grid = record["grid"]
            if hasattr(grid, "cells"):
                for row in grid.cells:
                    for h in row:
                        if cancel():
                            raise ReadCancelled()
                        info = h.path.stat()
                        if (info.st_size, info.st_mtime_ns) != h.fingerprint:
                            raise FeatureError("source_changed_since_index")

    def cancel(self):
        self.runner.invalidate()
        self._job = None
        self.candidate = None
        self.status.setText("特征任务已取消；未发布部分结果。")
        self.refresh_buttons()

    def present_mr(self, value):
        keys = (("心肌峰值增强","myocardium.peak_enhancement"),
                ("心肌达峰时间（相对窗口起点）","myocardium.ttp_from_window_origin"),
                ("心肌最大上升斜率","myocardium.max_upslope"),("心肌/血池斜率比","normalized_upslope"))
        self.summary.setPlainText("\n".join(f"{label}: {value['measurements'][k]['value']:.6g} {unit_label(value['measurements'][k]['unit'])}"
                                           for label,k in keys))
        warnings = value["qc"]["warnings"]
        self.status.setText("MR预览完成；"+(f"有{len(warnings)}项质控待核对，不能作为已审核输入。" if warnings
                                           else "本地检查通过，参考未验证。"))

    def present_pet(self, value):
        names = {"rest_mbf":"静息心肌血流量（MBF）","stress_mbf":"负荷心肌血流量（MBF）","pet_mfr":"PET心肌血流储备（MFR）"}
        self.summary.setPlainText("\n".join(f"{names[k]}: {v['value']:.6g} {v['unit']}" for k, v in value["measurements"].items()))
        pending = len(value["qc"]["warnings"])
        self.status.setText("PET结果与本地曲线来源一致；MBF算法仍需独立验证。"+
                            (f"另有{pending}项质控待核对。" if pending else ""))

    def _result(self, revision, value):
        job, self._job = self._job, None
        if job is None or job[1] != self._context():
            return
        self.status.setToolTip("")
        if job[0] == "mr":
            self.candidate = value
            self._candidate_context = self._context()
            self.present_mr(value)
        elif job[0] == "pet":
            self.pet = value
            self.present_pet(value)
        elif job[0] == "reference":
            ref, check = value
            self.mr_reference = ref
            self.status.setText("MR参考工程容差比较："+("通过" if check["passed"] else "失败，禁止作为已核对输入"))
        elif job[0] == "saved":
            self.saved_output = value
            self.status.setText("分析结果已保存，可从“文件→打开工作目录”查看。")
        self.refresh_buttons()
        self.task_done.emit()

    def _failed(self, revision, code):
        self._job = None
        self.last_error = code
        self.status.setText(error_message(code)+" 未完成结果不会作为阴性或零值。")
        self.status.setToolTip("技术代码："+code)
        self.refresh_buttons()

"""Background work persistence and verified, source-bound GUI restoration."""
import copy
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog

from cmvd_demo.paths import workspace_root
from cmvd_demo.roi import document_hash
from cmvd_demo.perfusion import case_token_for_choices
from cmvd_demo.work import KIND, require, save_record, restore_record
from .tasks import TaskRunner
from .presentation import error_message
from .data import CONDITIONS


class WorkController(QObject):
    finished = Signal()

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.output_root = workspace_root()/"outputs/work_records"
        self.runner = TaskRunner(self)
        self.runner.result.connect(self._result)
        self.runner.failed.connect(self._failed)
        self.runner.idle.connect(self.refresh)
        self.runner.idle.connect(window._idle)
        self.runner.progress.connect(self._progress)
        self.saved_output = self.restored_output = None
        self.last_output = None
        self.last_error = ""
        self._job = None
        self._applying = False
        self._locked = False
        menu = window.file_menu
        menu.addSeparator()
        self.save_action = menu.addAction("保存工作记录", self.save)
        self.restore_action = menu.addAction("恢复工作记录…", self.choose_restore)
        self.cancel_action = menu.addAction("取消保存或恢复", self.cancel)
        self.output_action = menu.addAction("打开最近工作记录目录", self.open_output)
        window.case_changing.connect(self.context_changed)
        window.plane_shown.connect(self.refresh)
        window.roi_panel.state_changed.connect(self.context_changed)
        window.roi_panel.curve_ready.connect(self.context_changed)
        window.feature_panel.task_done.connect(self.refresh)
        for runner in (window.runner, window.roi_panel.runner, window.feature_panel.runner):
            runner.idle.connect(self.refresh)
        for control in (window.time_slider, window.slice_slider):
            control.valueChanged.connect(self.context_changed)
        f = window.feature_panel
        for control in (f.condition, f.baseline_start, f.baseline, f.start, f.end):
            signal = control.currentIndexChanged if control is f.condition else control.valueChanged
            signal.connect(self.context_changed)
        f.protocol.textChanged.connect(self.context_changed)
        for control in (f.window_review, f.pair_review, f.protocol_review):
            control.toggled.connect(self.context_changed)
        self.refresh()

    @property
    def busy(self):
        return self._job is not None or self.runner.busy

    def ready(self):
        w, p, f = self.window, self.window.roi_panel, self.window.feature_panel
        return (not w._closing and not w._pending_index and p._ready() and not p.invalid_edit
                and not any(r.busy for r in (w.runner, p.runner, f.runner)) and not w.debounce.isActive())

    def refresh(self):
        w = self.window
        external=hasattr(w,'research') and w.research.busy
        ready = self.ready() and not self.busy and not external
        self.save_action.setEnabled(ready)
        w.save_work_button.setEnabled(ready)
        self.restore_action.setEnabled(not self.busy and not external and not w._closing)
        w.restore_work_button.setEnabled(not self.busy and not external and not w._closing)
        self.cancel_action.setEnabled(self.busy)
        self.output_action.setEnabled(self.last_output is not None and not self.busy)
        f = w.feature_panel
        f.single_save_button.setEnabled(ready and f.candidate is not None and f._candidate_context == f._context())

    def capture(self, *, single=False):
        w, p, f = self.window, self.window.roi_panel, self.window.feature_panel
        require(self.ready(), "work_wait_current_tasks")
        active = w.current_choice.key
        entries = {key:{"document":copy.deepcopy(doc),"curves":{}} for key,doc in p.documents.items()
                   if key == active or doc["tracks"]}
        def add_curve(key, doc, result):
            if key not in entries:
                entries[key] = {"document":copy.deepcopy(doc),"curves":{}}
            require(document_hash(entries[key]["document"]) == document_hash(doc), "work_stale_result")
            z = str(result["slice_index"])
            previous = entries[key]["curves"].get(z)
            require(previous is None or document_hash(previous) == document_hash(result), "work_stale_result")
            entries[key]["curves"][z] = copy.deepcopy(result)
        if p.result is not None:
            add_curve(active, p.document, p.result)
        state = {"settings":{
            "condition":f.condition.currentData(),"protocol":f.protocol.text(),
            "baseline_start":f.baseline_start.value(),"baseline_frames":f.baseline.value(),
            "window_start":f.start.value(),"window_end":f.end.value(),
            "window_reviewed":f.window_review.isChecked(),"pairing_reviewed":f.pair_review.isChecked(),
            "protocol_comparable":f.protocol_review.isChecked()},
            "candidate":None,"mr_slots":{},"pet_slots":{},"pet":copy.deepcopy(f.pet),
            "mr_reference":copy.deepcopy(f.mr_reference)}
        if f.candidate is not None:
            require(f._candidate_context == f._context(), "work_stale_result")
            state["candidate"] = {"key":active,"value":copy.deepcopy(f.candidate)}
        for family in ("mr_slots","pet_slots"):
            for condition, record in getattr(f, family).items():
                add_curve(record["key"], record["document"], record["curves"])
                state[family][condition] = copy.deepcopy({k:v for k,v in record.items() if k != "grid"})
        if single:
            require(state["candidate"] is not None and w.grid.kind == "mr_perfusion", "work_current_mr_required")
            entries = {active:entries[active]}
            entries[active]["curves"] = {str(w.displayed_indices[1]):copy.deepcopy(p.result)}
            state.update(mr_slots={},pet_slots={},pet=None,mr_reference=None)
            state["settings"].update(pairing_reviewed=False,protocol_comparable=False)
        return {"schema_version":1,"kind":KIND,"scope":"single_mr" if single else "case_work",
                "case_token":case_token_for_choices(w.choices),"active_key":active,"entries":entries,
                "view":{"time":w.displayed_indices[0],"slice":w.displayed_indices[1],
                    "levels":[w.level_center.value(),w.level_width.value()],"view_range":w.view.viewRange(),
                    "fps":w.play_fps.value(),"tab":w.workspace_tabs.currentIndex(),"role":p.role.currentData(),
                    "scope":p.scope.currentIndex(),"tool":w.roi_tool.currentData(),
                    "scroll":[w.workspace_tabs.widget(i).verticalScrollBar().value() for i in range(2)]},
                "feature_state":state}

    @staticmethod
    def _signature(record):
        return document_hash({"entries":record["entries"],"feature_state":record["feature_state"],
                              "active_key":record["active_key"],"position":[record["view"][k] for k in ("time","slice")]})

    def _lock(self, enabled):
        self._locked = enabled
        self.window.centralWidget().setEnabled(not enabled)
        self.window.roi_dock.setEnabled(not enabled)

    def save(self, checked=False, *, single=False):
        if self.busy or hasattr(self.window,'research') and self.window.research.busy:
            return
        try:
            self.window._stop_play()
            record = self.capture(single=single)
            self.saved_output = None
            self.last_error = ""
            self._job = ("save",self.window.case_revision,single,self._signature(record))
            self._lock(True)
            choices = tuple(self.window.choices)
            self.runner.submit(lambda progress,cancel: save_record(self.output_root,record,choices,
                self.window.backend,progress=progress,cancel=cancel))
            self.window.status.setText("正在复核并保存工作记录，可从“文件”菜单取消…")
            self.refresh()
        except Exception as error:
            self._failed(0, getattr(error,"code","work_invalid_record"))

    def choose_restore(self):
        if self.busy:
            return
        path,_ = QFileDialog.getOpenFileName(self.window,"选择工作记录（请先打开对应病例或合成演示）",
            str(self.output_root),"工作记录 (work.json bundle.json)")
        if path:
            self.restore(path)

    def restore(self, path):
        self.window.unsaved.request('恢复记录并替换当前工作',lambda:self._restore_confirmed(path))

    def _restore_confirmed(self, path):
        if self.busy or hasattr(self.window,'research') and self.window.research.busy:
            return
        try:
            w = self.window
            require(w.choices and w.grid is not None and w.current_pixels is not None, "work_open_case_first")
            require(not w._closing and w.displayed_indices == (w.time_slider.value(),w.slice_slider.value())
                    and not w.roi_panel.drawing.active
                    and not w.debounce.isActive()
                    and not any(r.busy for r in (w.runner,w.roi_panel.runner,w.feature_panel.runner)), "work_wait_current_tasks")
            self.window._stop_play()
            self.last_error = ""
            self.restored_output = None
            self._job = ("restore",self.window.case_revision,False,None)
            self._lock(True)
            choices = tuple(self.window.choices)
            self.runner.submit(lambda progress,cancel: restore_record(path,choices,self.window.backend,
                progress=progress,cancel=cancel))
            self.window.status.setText("正在核对来源并复算工作记录，可从“文件”菜单取消…")
            self._restore_path = Path(path)
            self.refresh()
        except Exception as error:
            self._failed(0, getattr(error,"code","work_invalid_record"))

    def _result(self, revision, value):
        job = self._job
        if job is None or job[1] != self.window.case_revision or self.window._closing:
            return
        try:
            if job[0] == "save":
                require(self._signature(self.capture(single=job[2])) == job[3], "work_stale_result")
                self.saved_output = value
                self.last_output = value
                if job[2]:
                    self.window.feature_panel.single_saved_output = value
                    self.window.feature_panel.status.setText("当前MR分析已独立保存，含轮廓、曲线和参数；仍为研究记录。")
                self.window.status.setText("工作记录已保存，可从“文件→打开工作目录”查看 work_records。")
            else:
                self._applying = True
                self._apply(value)
                self.restored_output = self._restore_path if self._restore_path.is_dir() else self._restore_path.parent
                self.last_output = self.restored_output
                self.window.status.setText("工作记录已恢复，曲线和分析已复算；复核状态按原记录保留。")
            self._job = None
            self._applying = False
            self._lock(False)
            self.refresh()
            if job[0]=='restore' or (job[0]=='save' and not job[2]):
                self.window.unsaved.mark_saved()
            self.finished.emit()
        except Exception as error:
            self._applying = False
            self._failed(revision, getattr(error,"code","work_invalid_record"))

    def _apply(self, checked):
        w, p, f = self.window, self.window.roi_panel, self.window.feature_panel
        record, view = checked["record"], checked["record"]["view"]
        active = record["active_key"]
        w.runner.invalidate()
        w._cancel_prefetch()
        w.debounce.stop()
        w._operation = None
        w._clear_image()
        f.reset_case()
        p.reset_case()
        p.documents.update({key:copy.deepcopy(entry["document"]) for key,entry in record["entries"].items()})
        w.current_choice = next(c for c in w.choices if c.key == active)
        previous = w.tree.blockSignals(True)
        try:
            for i in range(w.tree.topLevelItemCount()):
                group = w.tree.topLevelItem(i)
                for j in range(group.childCount()):
                    item = group.child(j)
                    if item.data(0, Qt.UserRole).key == active:
                        w.tree.setCurrentItem(item)
        finally:
            w.tree.blockSignals(previous)
        w.series_label.setText(w.current_choice.label+" / "+CONDITIONS[w.current_choice.condition])
        w._accept_grid(checked["grids"][active])
        for slider,key in ((w.time_slider,"time"),(w.slice_slider,"slice")):
            old = slider.blockSignals(True)
            slider.setValue(view[key])
            slider.blockSignals(old)
        w._auto_window = w._needs_fit = False
        center,width = view["levels"]
        w._set_levels(center-width/2,center+width/2)
        w._accept_plane(checked["pixels"],view["time"],view["slice"])
        curve = checked["curves"][active].get(str(view["slice"]))
        if curve is not None:
            p.present_curves(copy.deepcopy(curve))
        state, settings = record["feature_state"], record["feature_state"]["settings"]
        controls = (f.condition,f.protocol,f.baseline_start,f.baseline,f.start,f.end,f.window_review,
                    f.pair_review,f.protocol_review,p.role,p.scope,w.roi_tool,w.play_fps)
        blocked = [control.blockSignals(True) for control in controls]
        try:
            f.condition.setCurrentIndex(f.condition.findData(settings["condition"]))
            f.protocol.setText(settings["protocol"])
            for control,key in ((f.baseline_start,"baseline_start"),(f.baseline,"baseline_frames"),
                                (f.start,"window_start"),(f.end,"window_end")):
                control.setValue(settings[key])
            f.window_review.setChecked(settings["window_reviewed"])
            f.pair_review.setChecked(settings["pairing_reviewed"])
            f.protocol_review.setChecked(settings["protocol_comparable"])
            p.role.setCurrentIndex(p.role.findData(view["role"]))
            p.scope.setCurrentIndex(view["scope"])
            w.roi_tool.setCurrentIndex(w.roi_tool.findData(view["tool"]))
            w.play_fps.setValue(view["fps"])
        finally:
            for control,old in zip(controls,blocked):
                control.blockSignals(old)
        w._fps_changed(view["fps"])
        w.set_roi_tool(view["tool"])
        w.view.set_roi_tool(view["tool"])
        p._overlay_key = None
        p.on_frame()
        for family in ("mr_slots","pet_slots"):
            setattr(f,family,{condition:dict(copy.deepcopy(slot),grid=checked["grids"][slot["key"]])
                             for condition,slot in state[family].items()})
        f.pet, f.mr_reference = copy.deepcopy(state["pet"]), copy.deepcopy(state["mr_reference"])
        f.last_error = ""
        p.saved_output = None
        f.candidate = copy.deepcopy(state["candidate"]["value"]) if state["candidate"] else None
        f._candidate_context = f._context() if f.candidate else None
        f.summary.clear()
        if f.candidate:
            f.present_mr(f.candidate)
        elif f.pet:
            f.present_pet(f.pet)
        f._update_pair_summary()
        p.status.setText("轮廓与原始曲线已恢复，撤销历史从本次恢复后开始记录。")
        f.status.setText("已从工作记录复算恢复；审核状态未升级，当前无诊断输出。")
        f.refresh_buttons()
        p.refresh_buttons()
        w.curve_link.refresh()
        w.play_fps.setValue(view["fps"])
        w.view.setRange(xRange=view["view_range"][0],yRange=view["view_range"][1],padding=0)
        w.workspace_tabs.setCurrentIndex(view["tab"])
        epoch = w.case_revision
        def scroll():
            if not w._closing and w.case_revision == epoch:
                for i,value in enumerate(view["scroll"]):
                    w.workspace_tabs.widget(i).verticalScrollBar().setValue(value)
        QTimer.singleShot(0,scroll)

    def context_changed(self, *args):
        if self._job is not None and not self._applying:
            self.cancel()
        self.refresh()

    def open_output(self):
        if self.last_output is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output.resolve())))

    def cancel(self):
        had_job=self._job is not None
        self.runner.invalidate()
        self._job = None
        self._lock(False)
        if not self.window._closing:
            self.window.status.setText("已取消保存或恢复，旧任务结果不会显示。")
        self.refresh()
        if had_job:
            self.finished.emit()

    def _failed(self, revision, code):
        self.last_error = code
        self._job = None
        self._lock(False)
        self.window.status.setText(error_message(code))
        self.window.status.setToolTip("技术代码："+code)
        self.refresh()
        self.finished.emit()

    def _progress(self, revision, stage, done, total):
        self.window.status.setText(f"正在复核工作记录 {done}/{total} · 可从文件菜单取消")

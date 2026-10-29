"""Background research export, integrity review and normal work restoration."""
from pathlib import Path
from PySide6.QtCore import QObject,QUrl,Qt,QEvent
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QFileDialog,QMessageBox
from cmvd_demo.paths import workspace_root
from cmvd_demo.roi import document_hash
from cmvd_demo.io import DicomReadError
from cmvd_demo.research_record import save_research,load_research
from .tasks import TaskRunner
from .presentation import PRODUCT_VERSION,error_message
from .dialogs import TaskDialog, scroll_layout, fit_window

GUIDE=('1. 打开原病例；在“静息／负荷对照查看”手动选择两侧序列，独立核对帧、层面、方向及单位。\n'
       '2. 如需曲线和MR测量，先在主窗口计算并分别保留静息/负荷结果，再打开对照。\n'
       '3. 时间联动默认关闭；开启后只按各自起点后的时间选最近采样，超范围不跳转，不证明生理同步。\n'
       '4. 导出当前病例研究记录，打开摘要与CSV核对字段/状态/单位；保留完整目录。\n'
       '5. 核对导出文件完整性；先打开原病例，再从记录恢复工作，比较ROI、曲线、参数与分析。\n'
       '6. 测试取消、换病例和错误目录；旧结果不得显示或覆盖。核对/恢复成功不等于专业或临床审核。\n'
       '完整清单：文件→打开工作目录→docs/USAGE_CHECKLIST.md，可用记事本打开。')


class ResearchController(QObject):
    def __init__(self,w):
        super().__init__(w);self.w=w;self.output_root=workspace_root()/'outputs/research_records'
        self.runner=TaskRunner(self);self.runner.result.connect(self._result);self.runner.failed.connect(self._failed)
        self.runner.idle.connect(self.refresh);self.runner.idle.connect(w._idle)
        self._job=None;self.last_error='';self.last_output=None;self.data=None
        self.dialog=TaskDialog(w);self.dialog.setWindowTitle('研究记录导出与核对')
        self.dialog.dismissing.connect(lambda: self.cancel() if self.busy else None)
        layout=scroll_layout(self.dialog)
        fit_window(self.dialog,850,460)
        intro=QLabel('导出当前病例的可读摘要、JSON/CSV、序列问题、对照位置及可恢复工作记录。\n研究草稿，未临床验证；不复制影像，不自动放行训练。')
        intro.setWordWrap(True);layout.addWidget(intro)
        row=QHBoxLayout()
        self.export_button=QPushButton('导出当前病例研究记录');self.export_button.clicked.connect(self.export)
        self.verify_button=QPushButton('核对既有导出…');self.verify_button.clicked.connect(self.choose_verify)
        self.cancel_button=QPushButton('取消任务');self.cancel_button.clicked.connect(self.cancel)
        for button in (self.export_button,self.verify_button,self.cancel_button):row.addWidget(button)
        layout.addLayout(row)
        self.message=QLabel('尚未导出或核对记录。核对只检查文件完整性，不重读原影像或执行临床审核。')
        self.message.setTextFormat(Qt.PlainText)
        self.message.setWordWrap(True);layout.addWidget(self.message,1)
        row=QHBoxLayout()
        self.summary_button=QPushButton('打开摘要');self.summary_button.clicked.connect(self.open_summary)
        self.folder_button=QPushButton('打开导出目录');self.folder_button.clicked.connect(self.open_output)
        self.restore_button=QPushButton('从此记录恢复工作');self.restore_button.clicked.connect(self.restore)
        self.guide_button=QPushButton('使用测试步骤');self.guide_button.clicked.connect(self.guide)
        for button in (self.summary_button,self.folder_button,self.restore_button,self.guide_button):row.addWidget(button)
        layout.addLayout(row)
        w.research_button.clicked.connect(self.open)
        w.study_menu.addAction('研究记录导出与核对',self.open)
        w.help_menu.addAction('对照查看与研究记录操作核对',self.guide)
        w.case_changing.connect(self.reset);w.plane_shown.connect(self.refresh)
        w.roi_panel.state_changed.connect(self.context_changed);w.roi_panel.curve_ready.connect(self.context_changed)
        w.work.finished.connect(self.refresh)
        for runner in (w.runner,w.roi_panel.runner,w.feature_panel.runner):runner.idle.connect(self.refresh)
        self.refresh()

    @property
    def busy(self):return self._job is not None or self.runner.busy

    def signature(self):
        return document_hash({'work':self.w.work.capture(),'review':self.w.sequences.model.snapshot(),'comparison':self.w.comparison.snapshot()})

    def refresh(self,*args):
        w=self.w;ready=w.work.ready() and not w.work.busy and not w.comparison.busy and not w.sequences.runner.busy and not w.roi_panel.drawing.active
        self.export_button.setEnabled(ready and not self.busy)
        self.verify_button.setEnabled(not self.busy and not w.work.busy)
        self.cancel_button.setEnabled(self.busy)
        for button in (self.summary_button,self.folder_button):button.setEnabled(self.last_output is not None and not self.busy)
        self.restore_button.setEnabled(self.last_output is not None and self.data is not None and not self.busy and bool(w.choices) and not w.work.busy)
        w.work.refresh()

    def _lock(self,locked):
        self.w.centralWidget().setEnabled(not locked);self.w.roi_dock.setEnabled(not locked)
        self.w.comparison.dialog.setEnabled(not locked);self.w.sequences.dialog.setEnabled(not locked)

    def open(self):self.refresh();self.dialog.show();self.dialog.raise_();self.dialog.activateWindow()

    def eventFilter(self,obj,event):
        if obj is self.dialog and event.type()==QEvent.Close and self.busy:self.cancel()
        return super().eventFilter(obj,event)

    def export(self):
        if self.busy or self.w.work.busy:return
        try:
            if self.w.comparison.busy or self.w.roi_panel.drawing.active:raise DicomReadError('research_wait_comparison')
            self.w._stop_play();work=self.w.work.capture();review=self.w.sequences.model.snapshot();comparison=self.w.comparison.snapshot()
            signature=self.signature();self.last_error='';self.last_output=None;self.data=None
            self._job=('save',self.w.case_revision,signature,None);self._lock(True)
            choices=tuple(self.w.choices);backend=self.w.backend
            def job(progress,cancel):
                root=save_research(self.output_root,work,review,comparison,choices,backend,PRODUCT_VERSION,progress=progress,cancel=cancel)
                return root,load_research(root,cancel=cancel)
            self.runner.submit(job);self.message.setText('正在复核来源、曲线与分析，导出点击时刻的研究记录…');self.refresh()
        except Exception as error:self._failed(0,getattr(error,'code','research_invalid_record'))

    def choose_verify(self):
        path,_=QFileDialog.getOpenFileName(self.dialog,'选择完整研究记录内的research.json或bundle.json',str(self.output_root),'研究记录 (research.json bundle.json)')
        if path:self.verify(path)

    def verify(self,path):
        if self.busy or self.w.work.busy:return
        self.last_error='';self.last_output=None;self.data=None
        self._job=('verify',self.w.case_revision,None,Path(path))
        self.runner.submit(lambda progress,cancel:load_research(path,cancel=cancel))
        self.message.setText('正在核对完成标记、文件哈希和CSV/摘要一致性；不读取影像…');self.refresh()

    def _result(self,revision,value):
        job,self._job=self._job,None
        if job is None or job[1]!=self.w.case_revision or self.w._closing:self._lock(False);return
        try:
            if job[0]=='save':
                if self.signature()!=job[2]:raise DicomReadError('research_stale_result')
                self.last_output,self.data=value
                self.w.unsaved.mark_saved()
            else:
                path=job[3];self.last_output=path if path.is_dir() else path.parent;self.data=value
            self.message.setText(('研究记录已导出并完成文件核对。' if job[0]=='save' else '文件完整性及派生CSV/摘要核对通过。')+
                '\n目录：'+self.last_output.name+'\n序列状态条目：'+str(len(self.data['sequence_review']['sequences']))+
                '；问题/提示：'+str(len(self.data['sequence_review']['issues']))+
                '\n可打开summary.html和CSV；从记录恢复需先打开原病例。问题处置与对照位置为快照，不会自动恢复或升级审核。')
            self._lock(False);self.refresh()
        except Exception as error:self._failed(0,getattr(error,'code','research_invalid_record'))

    def _failed(self,revision,code):
        self._job=None;self.last_error=code;self.last_output=None;self.data=None
        self._lock(False);self.message.setText(error_message(code));self.refresh()
        self.message.setToolTip('技术代码：'+code)

    def cancel(self):
        self.runner.invalidate();self._job=None;self._lock(False);self.refresh()
        self.message.setText('已取消；未发布旧任务结果。中断目录保留且不记为完整记录。')

    def reset(self):
        self.cancel();self.last_output=None;self.data=None;self.last_error='';self.refresh()

    def context_changed(self,*args):
        if self._job is not None and self._job[0]=='save':self.cancel()
        self.refresh()

    def open_summary(self):
        if self.last_output:QDesktopServices.openUrl(QUrl.fromLocalFile(str((self.last_output/'summary.html').resolve())))

    def open_output(self):
        if self.last_output:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output.resolve())))

    def restore(self):
        if self.last_output and self.data and not self.busy:
            self.w.work.restore(self.last_output/self.data['work_relative'])

    def guide(self):QMessageBox.information(self.dialog,'对照查看与研究记录操作核对',GUIDE)

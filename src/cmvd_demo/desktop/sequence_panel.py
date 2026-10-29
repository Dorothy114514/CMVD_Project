"""Current-case status, bounded issue ledger and advisory report linking."""
from PySide6.QtCore import QObject,Qt,Signal
from PySide6.QtGui import QColor,QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,
                              QTableWidget,QTableWidgetItem,QHeaderView,QFileDialog,QAbstractItemView)

from cmvd_demo.paths import workspace_root
from cmvd_demo.perfusion import case_token_for_choices
from cmvd_demo.sequence_review import SequenceReview,STATES,DISPOSITIONS,save_review,load_history
from .data import KINDS
from .presentation import error_message
from .tasks import TaskRunner
from .dialogs import TaskDialog, scroll_layout, fit_window
from .select_box import SelectBox

STAGES={'header_read':'头字段读取','metadata_parse':'头字段解析','index':'病例索引','grid':'空间/时间组装',
        'pixels':'像素读取','capability':'显示支持范围','history':'历史报告提示','qc':'质控待核对'}


def reason(row):
    if row['code']=='indexed_only':return '此类对象目前仅索引，暂未实现显示。'
    if row['code']=='historical_unverified_report':return '历史报告已关联；当前像素和处理结果未由该报告证实。'
    if row['code'] in ('condition_requires_review','qc_pending'):return '采集状态或分支条件需要人工核对。'
    if row['code']=='phase_synchronization_not_performed':return '心脏电影各层相位未同步，不能视为同步三维心动周期。'
    if row['code']=='read_failed_unclassified':return '头字段读取未完成，当前无法确认对象所属序列；请核对文件格式与完整导出。'
    if row['code']=='metadata_parse_failed':return '头字段解析未通过，当前无法确认对象所属序列；请核对原始导出。'
    if row['code']=='unknown_read_failure':return '读取未完成，暂无可确认的具体原因；请保留记录核对原始导出。'
    if row['code']=='pixel_decode_failed':return '像素解码未完成，请核对支持格式和原始导出。'
    return error_message(row['code'])


def advice(row):
    if row['stage']=='capability':return '可用独立查看器核对；不代表源文件损坏。'
    if row['origin']=='history':return '历史技术标识匹配，仍需核对当前原件；不自动放行。'
    if row['stage']=='qc':return '按采集方案人工核对；此处不执行专业审核。'
    if row['sequence_key'] is None and row['stage'] in ('header_read','metadata_parse','index'):
        return '无法自动定位到序列；保留文件标识与记录，交医院核源/补导出。'
    if row['code'] in ('duplicate_sop_instance','zero_rescale_slope','declared_frame_count_mismatch',
                       'declared_slice_count_mismatch','missing_spatial_time_cell','inconsistent_geometry'):
        return '保留原件与此记录，交医院核源/补导出；不强行拼接。'
    return '可定位后重试；若仍失败，保留记录核对原始导出。'


class SequencePanel(QObject):
    changed=Signal()

    def __init__(self,window):
        super().__init__(window)
        self.w=window;self.model=SequenceReview()
        self.output_root=workspace_root()/'outputs/sequence_reviews'
        self.runner=TaskRunner(self)
        self.runner.result.connect(self._result);self.runner.failed.connect(self._failed)
        self.runner.idle.connect(self.refresh);self.runner.idle.connect(window._idle)
        self.saved_output=None;self.last_error='';self._job=None;self._locate_target=None
        self.dialog=TaskDialog(window);self.dialog.setWindowTitle('序列状态与问题清单')
        self.dialog.dismissing.connect(self.cancel)
        layout=scroll_layout(self.dialog)
        fit_window(self.dialog,1060,620)
        self.summary=QLabel('先打开单个病例目录。');self.summary.setWordWrap(True);layout.addWidget(self.summary)
        row=QHBoxLayout()
        self.scope=SelectBox();self.scope.addItems(['整个病例','当前序列'])
        self.visibility=SelectBox();self.visibility.addItems(['当前问题与提示','包含本次重试历史'])
        for combo in (self.scope,self.visibility):combo.currentIndexChanged.connect(self.refresh);row.addWidget(combo)
        self.locate_button=QPushButton('定位 / 重试序列');self.locate_button.clicked.connect(self.locate);row.addWidget(self.locate_button)
        layout.addLayout(row)
        self.table=QTableWidget(0,6)
        self.table.setHorizontalHeaderLabels(['定位','阶段','原因','建议','来源','处置/重试'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        for col in (2,3):self.table.horizontalHeader().setSectionResizeMode(col,QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._selected)
        self.table.itemDoubleClicked.connect(lambda _:self.locate())
        layout.addWidget(self.table,1)
        self.details=QLabel('选择一条问题查看完整原因和建议。');self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.PlainText);layout.addWidget(self.details)
        row=QHBoxLayout()
        self.disposition=SelectBox()
        for key,text in DISPOSITIONS.items():self.disposition.addItem(text,key)
        self.mark_button=QPushButton('记录本地处置');self.mark_button.clicked.connect(self.mark)
        row.addWidget(self.disposition);row.addWidget(self.mark_button)
        self.import_button=QPushButton('关联既有报告…');self.import_button.clicked.connect(self.choose_history)
        self.export_button=QPushButton('导出状态与问题');self.export_button.clicked.connect(self.export)
        self.cancel_button=QPushButton('取消报告任务');self.cancel_button.clicked.connect(self.cancel)
        self.folder_button=QPushButton('打开导出目录');self.folder_button.clicked.connect(self.open_output)
        for button in (self.import_button,self.export_button,self.cancel_button,self.folder_button):row.addWidget(button)
        layout.addLayout(row)
        self.message=QLabel('已预览只代表实际显示过的帧；未做全帧或临床验证。索引失败对象可能无法归属到序列。')
        self.message.setWordWrap(True);layout.addWidget(self.message)
        self._visible=[];self._table_signature=None
        window.sequence_filter.currentIndexChanged.connect(self.refresh)
        window.case_changing.connect(self.reset)
        window.sequence_button.clicked.connect(self.open)
        window.study_menu.addAction('序列状态与问题清单',self.open)
        window.plane_shown.connect(self._plane)
        self.refresh()

    def reset(self):
        self.cancel();self.model=SequenceReview();self.saved_output=None;self.last_error='';self._locate_target=None
        self.message.setText('已预览只代表实际显示过的帧；未做全帧或临床验证。索引失败对象可能无法归属到序列。')
        self.refresh()

    def populate(self,choices,summary):
        token=case_token_for_choices(choices) if choices else None
        self.model.populate(choices,summary,token);self.refresh()

    def begin(self,key,stage,frame=None):self.model.begin(key,stage,frame);self.refresh()

    def grid_ok(self,grid):
        self.model.grid_ok(self.w.current_choice.key,grid.shape,getattr(grid,'qc_warnings',()))
        self.refresh()

    def _plane(self):
        if self.w.current_choice and self.w.displayed_indices:
            self.model.plane_ok(self.w.current_choice.key,*self.w.displayed_indices)
            self.refresh()
            target,self._locate_target=self._locate_target,None
            if target and target[:2]==(self.w.case_revision,self.w.current_choice.key):
                t,z=target[2]
                if 0<=t<self.w.grid.shape[0] and 0<=z<self.w.grid.shape[1]:
                    self.w.time_slider.setValue(t);self.w.slice_slider.setValue(z)
                else:self.message.setText('原记录的帧/层面超出当前网格范围，未跳转；请核对原报告。')

    def failure(self,operation,code):
        stage=operation[0] if operation else 'pixels'
        key=None if stage=='index' else (self.w.current_choice.key if self.w.current_choice else None)
        frame=[operation[2],operation[3]] if operation and stage=='plane' else None
        self.model.failure(key,'pixels' if stage=='plane' else stage,code,frame);self.refresh()

    def cancelled(self):
        operation=self.w._operation
        key=self.w.current_choice.key if self.w.current_choice else None
        stage=operation[0] if operation else 'pixels'
        self.model.cancelled(key,'pixels' if stage=='plane' else stage);self._locate_target=None;self.refresh()

    def refresh(self,*args):
        w=self.w;counts={state:0 for state in STATES}
        for row in self.model.rows.values():counts[row['state']]+=1
        self.summary.setText((' / '.join(f'{STATES[k]} {v}' for k,v in counts.items() if v) or '尚无序列，请先打开单个病例目录')+
            f"；记录问题/提示 {len(self.model.issues)}，省略 {self.model.omitted_issues}。未做全帧或临床验证。")
        issue_counts={}
        for issue in self.model.issues:
            if issue['active']:issue_counts[issue['sequence_key']]=issue_counts.get(issue['sequence_key'],0)+1
        mode=w.sequence_filter.currentData()
        for i in range(w.tree.topLevelItemCount()):
            group=w.tree.topLevelItem(i);visible=0
            for j in range(group.childCount()):
                item=group.child(j);choice=item.data(0,Qt.UserRole)
                row=self.model.rows.get(choice.key)
                if row is None:continue
                issue_count=issue_counts.get(choice.key,0)
                item.setText(2,STATES[row['state']])
                item.setForeground(2,QColor('#edbd75' if row['state'] in ('failed','partial_failure') else '#99cec5'))
                item.setToolTip(0,f"序列标识：{choice.key}\n已记录显示位置：{len(row['frames'])}（最多2048）\n"+
                    f"关联提示：{issue_count}；采集状态为候选，未专业审核。")
                show=(mode=='all' or mode=='problems' and bool(issue_count) or mode==row['state']
                      or mode=='readable' and row['state']!='index_only')
                # Keep a loading selection visible without changing its reading request.
                item.setHidden(not show);visible+=int(show)
            group.setHidden(not visible)
        signature=(self.scope.currentIndex(),self.visibility.currentIndex(),
            self.w.current_choice.key if self.scope.currentIndex() and self.w.current_choice else None,
            tuple((id(r),r['active'],r['disposition']) for r in self.model.issues))
        if signature!=self._table_signature:
            self._table_signature=signature;self._refresh_table()
        busy=self.runner.busy or self._job is not None
        self.export_button.setEnabled(bool(self.model.index_complete or self.model.issues or self.model.events) and not busy)
        self.import_button.setEnabled(bool(self.model.rows) and not busy)
        self.cancel_button.setEnabled(busy);self.folder_button.setEnabled(self.saved_output is not None)
        self._selected();self.changed.emit()

    def _refresh_table(self):
        selected=self.table.currentRow()
        previous=self._visible[selected] if 0<=selected<len(self._visible) else None
        self._visible=[r for r in self.model.issues if (r['active'] or self.visibility.currentIndex()==1)
            and (self.scope.currentIndex()==0 or self.w.current_choice and r['sequence_key']==self.w.current_choice.key)]
        self.table.blockSignals(True);self.table.setRowCount(len(self._visible))
        for index,row in enumerate(self._visible):
            key=row['sequence_key']
            loc=key or ('未归属序列 / '+row['file_token'] if row['file_token'] else '病例级')
            if row['frame'] is not None:loc+=f" · 帧{row['frame'][0]+1}/层{row['frame'][1]+1}"
            origin='历史提示（未核当前像素）' if row['origin']=='history' else '当前会话'
            state='本次重试通过，历史保留' if not row['active'] else DISPOSITIONS[row['disposition']]
            for col,text in enumerate((loc,STAGES.get(row['stage'],'读取'),reason(row),advice(row),origin,state)):
                item=QTableWidgetItem(text);item.setToolTip(text+'\n技术代码：'+row['code']);self.table.setItem(index,col,item)
        if previous in self._visible:self.table.selectRow(self._visible.index(previous))
        self.table.blockSignals(False)
    def _selected(self):
        index=self.table.currentRow();valid=0<=index<len(self._visible)
        if valid:
            row=self._visible[index]
            self.details.setText('原因：'+reason(row)+'\n建议：'+advice(row)+'\n技术代码：'+row['code'])
        else:self.details.setText('选择一条问题查看完整原因和建议。')
        self.locate_button.setEnabled(valid and self._visible[index]['sequence_key'] in self.model.rows
                                      and not self.w.work.busy)
        self.mark_button.setEnabled(valid)

    def open(self):
        self.refresh();self.dialog.show();self.dialog.raise_();self.dialog.activateWindow()

    def locate(self):
        index=self.table.currentRow()
        if 0<=index<len(self._visible) and self._visible[index]['sequence_key'] in self.model.rows:
            if self.w.work.busy:return
            issue=self._visible[index]
            self.w.sequence_filter.setCurrentIndex(0)
            self.w.select_choice(issue['sequence_key'])
            self._locate_target=(self.w.case_revision,issue['sequence_key'],issue['frame']) if issue['frame'] else None

    def mark(self):
        index=self.table.currentRow()
        if 0<=index<len(self._visible):
            self._visible[index]['disposition']=self.disposition.currentData();self.model.revision+=1;self.refresh()
            self.message.setText('本地处置已记录；不代表文件修复或专业审核通过，请导出保存记录。')

    def export(self):
        if self.runner.busy:return
        self.last_error='';self.saved_output=None
        snapshot=self.model.snapshot();self._job=('save',self.w.case_revision)
        self.runner.submit(lambda progress,cancel:save_review(self.output_root,snapshot,cancel=cancel));self.refresh()
        self.message.setText('正在保存点击时刻的状态与问题记录…')

    def choose_history(self):
        path,_=QFileDialog.getOpenFileName(self.dialog,'选择逐包检查报告或既有review.json',str(workspace_root()/'outputs'),'JSON (*.json)')
        if path:self.import_history(path)

    def import_history(self,path):
        if self.runner.busy:return
        keys={c.key for c in self.w.choices}
        token=self.model.case_token
        self.last_error='';self._job=('import',self.w.case_revision)
        self.runner.submit(lambda progress,cancel:load_history(path,keys,token,cancel=cancel));self.refresh()

    def _result(self,revision,value):
        job,self._job=self._job,None
        if job is None or job[1]!=self.w.case_revision or self.w._closing:return
        if job[0]=='save':
            self.saved_output=value;self.message.setText('状态与问题记录已导出；请保留完整目录，记录不含患者源路径。')
        else:
            self.model.issue(None,'history','historical_unverified_report',origin='history',source_sha256=value['source_sha256'])
            for row in value['rows']:
                self.model.issue(row['sequence_key'],'history',row['code'],file_token=row['file_token'],
                                 frame=row.get('frame'),origin='history',source_sha256=value['source_sha256'])
            self.model.event(None,'history','imported')
            self.message.setText('历史报告仅按病例/序列技术标识关联，未验证当前像素。'+
                ('报告曾使用临时去重索引；桌面仍执行原严格读取规则。' if value['source_type']=='transient_index' else '不改变当前读取状态或审核。'))
        self.refresh()

    def _failed(self,revision,code):
        self._job=None;self.last_error=code;self.message.setText(error_message(code));self.refresh()

    def cancel(self):self.runner.invalidate();self._job=None;self.refresh()

    def open_output(self):
        if self.saved_output:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.saved_output.resolve())))

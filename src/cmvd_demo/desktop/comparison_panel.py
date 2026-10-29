"""Two independent, bounded, read-only viewers using the existing reader."""
import copy
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject,Qt,QTimer,QRectF,Signal,QEvent
from PySide6.QtWidgets import (QDialog,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QSlider,
    QPushButton,QCheckBox,QDoubleSpinBox,QTableWidget,QTableWidgetItem,QHeaderView)
from cmvd_demo.io import DicomReadError
from cmvd_demo.roi import binding_for,document_hash
from cmvd_demo.roi.model import contour_at
from cmvd_demo.comparison import relative_target,link_allowed,plane_time
from cmvd_demo.perfusion.labels import measurement_label
from .data import SUPPORTED,CONDITIONS,auto_levels,direction_label,PlaneCache
from .tasks import TaskRunner
from .presentation import error_message,unit_label
from .dialogs import TaskDialog, scroll_layout, fit_window
from .select_box import SelectBox


class ComparePane(QWidget):
    accepted=Signal()

    def __init__(self,owner,condition):
        super().__init__(owner.dialog)
        self.owner=owner;self.w=owner.w;self.condition=condition
        self.runner=TaskRunner(self);self.runner.result.connect(self._result);self.runner.failed.connect(self._failed)
        self.runner.idle.connect(self.w._idle);self.runner.idle.connect(owner.refresh)
        self.grid=self.binding=self.displayed=self.pixels=None;self.key=None;self._job=None
        self.last_error='';self.cache=PlaneCache(16*1024**2);self._auto=True;self._fit=True
        self.debounce=QTimer(self);self.debounce.setSingleShot(True);self.debounce.setInterval(40)
        self.debounce.timeout.connect(self.request_plane)
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        layout.addWidget(QLabel('静息侧（手动选择）' if condition=='rest' else '负荷侧（手动选择）'))
        self.selector=SelectBox();self.selector.currentIndexChanged.connect(self.select);layout.addWidget(self.selector)
        self.graphics=pg.GraphicsLayoutWidget();self.graphics.setMinimumHeight(200)
        self.view=self.graphics.addViewBox(lockAspect=True,enableMenu=False,invertY=True)
        self.image=pg.ImageItem(axisOrder='row-major');self.view.addItem(self.image)
        self.overlays={}
        for role,boundary,color in (('myocardium','outer_rc','#ffc45e'),('myocardium','inner_rc','#ffc45e'),('blood_pool','outer_rc','#62dbea')):
            line=pg.PlotDataItem(pen=pg.mkPen(color,width=2));line.setParentItem(self.image);line.hide()
            self.overlays[role,boundary]=line
        layout.addWidget(self.graphics,1)
        self.label=QLabel('先打开病例并选择序列。');self.label.setWordWrap(True);layout.addWidget(self.label)
        self.time=QSlider(Qt.Horizontal);self.slice=QSlider(Qt.Horizontal)
        for label,control in (('时间 / 相位',self.time),('层面（独立选择）',self.slice)):
            layout.addWidget(QLabel(label));layout.addWidget(control);control.setEnabled(False)
            control.valueChanged.connect(self.navigate);control.sliderReleased.connect(self.request_plane)
        row=QHBoxLayout();self.center=QDoubleSpinBox();self.width=QDoubleSpinBox()
        for control in (self.center,self.width):control.setDecimals(3);control.setRange(-1e12,1e12);control.setMaximumWidth(145)
        self.width.setMinimum(.001)
        row.addWidget(QLabel('窗位'));row.addWidget(self.center);row.addWidget(QLabel('窗宽'));row.addWidget(self.width)
        self.auto_button=QPushButton('自动窗位');self.auto_button.clicked.connect(self.auto);row.addWidget(self.auto_button)
        self.retry=QPushButton('重试');self.retry.clicked.connect(self.select);row.addWidget(self.retry);layout.addLayout(row)
        for control in (self.center,self.width):control.valueChanged.connect(self.levels)
        self.plot=pg.PlotWidget(background='#080f1b',enableMenu=False);self.plot.setMaximumHeight(130)
        for menu in (self.plot.getPlotItem().ctrlMenu,self.plot.getViewBox().menu):
            if menu is not None:menu.setParent(self.plot,menu.windowFlags())
        self.plot.addLegend();self.curve_lines={r:self.plot.plot(pen=pg.mkPen(c,width=2),name=n)
            for r,c,n in (('myocardium','#ffc45e','心肌'),('blood_pool','#62dbea','血池'))}
        self.cursor=pg.InfiniteLine(angle=90,movable=False,pen='#a8bbc9');self.plot.addItem(self.cursor);self.cursor.hide()
        layout.addWidget(self.plot)
        self.curve_label=QLabel('未保留此层曲线；可先在主窗口计算并保留。');self.curve_label.setWordWrap(True);layout.addWidget(self.curve_label)

    @property
    def busy(self):return self.runner.busy or self._job is not None or self.debounce.isActive()

    def clear(self):
        self.runner.invalidate();self.debounce.stop();self._job=None
        self.grid=self.binding=self.displayed=self.pixels=None;self.key=None;self.last_error='';self.cache.clear()
        self.image.clear();self.label.setText('请选择序列；尚无有效影像。')
        for control in (self.time,self.slice):control.setEnabled(False)
        self.clear_curves()
        for line in self.overlays.values():line.hide()

    def clear_curves(self):
        for line in self.curve_lines.values():line.setData([],[])
        self.cursor.hide();self.curve_label.setText('未保留此层曲线；可先在主窗口计算并保留。')

    def populate(self,choices,preferred=None):
        self.clear();self.selector.blockSignals(True);self.selector.clear();self.selector.addItem('请选择对应状态的序列',None)
        for c in choices:self.selector.addItem(c.label+' · '+CONDITIONS[c.condition],c.key)
        self.selector.setCurrentIndex(max(0,self.selector.findData(preferred)));self.selector.blockSignals(False)
        if preferred:self.select()

    def select(self,*args):
        key=self.selector.currentData();self.clear();self.key=key
        if key is None:self.owner.refresh();return
        choice=next((c for c in self.w.choices if c.key==key),None)
        if choice is None:return
        self._auto=self._fit=True;self._job=('grid',self.w.case_revision,key)
        self.label.setText('正在核对空间、时间与来源…')
        backend=self.w.backend
        def work(progress,cancel):
            grid=backend.grid(choice,progress,cancel)
            return grid,binding_for(grid)
        self.runner.submit(work);self.owner.refresh()

    def _result(self,revision,payload):
        job,self._job=self._job,None
        if job is None or job[1]!=self.w.case_revision or job[2]!=self.key or self.w._closing:return
        if job[0]=='grid':
            self.grid,self.binding=payload
            self.w.sequences.model.grid_ok(self.key,self.grid.shape,getattr(self.grid,'qc_warnings',()))
            for control,n in ((self.time,self.grid.shape[0]),(self.slice,self.grid.shape[1])):
                control.blockSignals(True);control.setRange(0,n-1);control.setValue(0);control.blockSignals(False);control.setEnabled(n>1)
            self.request_plane();return
        t,z=job[3:];self.pixels=payload;self.displayed=(t,z);self.last_error=''
        self.cache.put((self.key,t,z),payload)
        self.image.setImage(payload,autoLevels=False)
        dy,dx=self.grid.pixel_spacing_mm;rows,cols=payload.shape
        self.image.setRect(QRectF(-dx/2,-dy/2,cols*dx,rows*dy))
        photo=getattr(self.grid,'photometric',None) or self.grid.cells[t][z].photometric
        self.image.setLookupTable(self.w._luts[photo])
        if self._auto:self.auto();self._auto=False
        else:self.levels()
        if self._fit:self.view.autoRange(padding=.03);self._fit=False
        self.w.sequences.model.plane_ok(self.key,t,z);self.w.sequences.refresh()
        self.describe();self.owner.refresh();self.accepted.emit()

    def describe(self):
        if self.displayed is None:return
        t,z=self.displayed;value=plane_time(self.binding,t,z)
        orientation=self.grid.orientation_lps
        direction='上 '+direction_label(-np.asarray(orientation[3:]))+' / 左 '+direction_label(-np.asarray(orientation[:3]))
        center=self.grid.pixel_to_lps(z,(self.grid.shape[2]-1)/2,(self.grid.shape[3]-1)/2)
        self.label.setText(f'已显示帧 {t+1}/{self.grid.shape[0]} · 层 {z+1}/{self.grid.shape[1]} · '+
            ('无可靠时间轴' if value is None else f'参考时间 {value:.4g} s')+'\n'+direction+' · LPS · '+unit_label(', '.join(self.binding['units']))+
            '\n层面中心 LPS ('+', '.join(f'{v:.2f}' for v in center)+') mm')

    def navigate(self,*args):
        if self.grid is None:return
        self.runner.invalidate();self._job=None
        self.describe();self.label.setText(self.label.text()+f'\n正在加载目标帧 {self.time.value()+1} / 层 {self.slice.value()+1}；图像与ROI保留已显示位置。')
        if not self.debounce.isActive():self.debounce.start()
        self.owner.link_from(self);self.owner.refresh()

    def request_plane(self):
        self.debounce.stop()
        if self.grid is None or self.w._closing:return
        t,z=self.time.value(),self.slice.value();key=self.key;grid=self.grid;cached=self.cache.get((key,t,z));backend=self.w.backend
        self._job=('plane',self.w.case_revision,key,t,z)
        self.runner.submit(lambda progress,cancel:backend.plane(grid,t,z,cached,progress,cancel));self.owner.refresh()

    def _failed(self,revision,code):
        job=self._job
        if self.key and job:
            self.w.sequences.model.failure(self.key,'grid' if job[0]=='grid' else 'pixels',code,
                None if job[0]=='grid' else list(job[3:]))
            self.w.sequences.refresh()
        self._job=None;self.last_error=code;self.pixels=self.displayed=None;self.image.clear()
        for line in self.overlays.values():line.hide()
        self.clear_curves();self.label.setText(error_message(code)+'\n可重试此序列；另一侧可独立查看。')
        self.label.setToolTip('技术代码：'+code);self.owner.refresh()

    def auto(self):
        if self.pixels is None:return
        lo,hi=auto_levels(self.pixels)
        for control,value in ((self.center,(lo+hi)/2),(self.width,hi-lo)):
            control.blockSignals(True);control.setValue(value);control.blockSignals(False)
        self.levels()

    def levels(self,*args):
        if self.pixels is not None:self.image.setLevels((self.center.value()-self.width.value()/2,self.center.value()+self.width.value()/2))

    def content(self):
        p,f=self.w.roi_panel,self.w.feature_panel
        if self.displayed is None or self.binding is None:return None,None
        doc=p.documents.get(self.key)
        if doc is None or doc['binding']!=self.binding or (p.key==self.key and p.invalid_edit):return None,None
        z=self.displayed[1];result=None
        if p.key==self.key and p.result and p.result['slice_index']==z:result=p.result
        for slots in (f.mr_slots,f.pet_slots):
            for record in slots.values():
                if record['key']==self.key and record['curves']['slice_index']==z:result=record['curves']
        if result and result['document_sha256']!=document_hash(doc):result=None
        return doc,result

    def refresh_content(self):
        doc,result=self.content()
        for line in self.overlays.values():line.hide()
        if doc and self.displayed:
            t,z=self.displayed
            for track in doc['tracks']:
                if track['slice_index']!=z:continue
                for boundary,vertices in contour_at(track,t).items():
                    line=self.overlays.get((track['role'],boundary))
                    if vertices is not None and line is not None:
                        pts=np.asarray(vertices);pts=np.vstack([pts,pts[0]])
                        line.setData(pts[:,1]+.5,pts[:,0]+.5);line.show()
        if result is None:self.clear_curves();return
        for line in self.curve_lines.values():line.setData([],[])
        for curve in result['curves']:self.curve_lines[curve['role']].setData(curve['time_s'],curve['mean'])
        self.plot.setLabel('bottom','原始采样时间',units='s')
        self.plot.setLabel('left','原始均值',units=unit_label(result['curves'][0]['units']))
        self.cursor.setValue(self.grid.times_s[self.displayed[1],self.displayed[0]]);self.cursor.show()
        self.curve_label.setText('只读原始曲线；未插值、未自动配准，审核状态未升级。')

    def snapshot(self):
        if self.key is None:return None
        if self.busy:raise DicomReadError('research_wait_comparison')
        if self.displayed is None:return None
        doc,result=self.content();t,z=self.displayed
        return {'key':self.key,'binding':copy.deepcopy(self.binding),'time':t,'slice':z,
            'levels':[self.center.value(),self.width.value()],
            'roi_document_sha256':document_hash(doc) if doc and doc['tracks'] else None,
            'curves_sha256':document_hash(result) if result else None}


class ComparisonController(QObject):
    def __init__(self,w):
        super().__init__(w);self.w=w;self.epoch=None;self._linking=False;self._filling=False
        self.dialog=TaskDialog(w);self.dialog.setWindowTitle('静息／负荷对照查看')
        self.dialog.dismissing.connect(self.cancel)
        layout=scroll_layout(self.dialog)
        fit_window(self.dialog,1280,900)
        self.notice=QLabel('本病例只读对照；两侧状态、解剖层面及协议须人工核对。未自动配准、未作生理同步。')
        self.notice.setWordWrap(True);layout.addWidget(self.notice)
        self.link=QCheckBox('联动相对时间（最近采样，不插值；层面独立）');self.link.toggled.connect(lambda _:self.refresh())
        layout.addWidget(self.link)
        row=QHBoxLayout();self.panes={}
        for condition in ('rest','stress'):
            pane=ComparePane(self,condition);self.panes[condition]=pane;row.addWidget(pane)
        layout.addLayout(row,1)
        self.table=QTableWidget(0,3);self.table.setHorizontalHeaderLabels(['已保留MR测量','静息','负荷'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);self.table.setFixedHeight(150)
        self.table.verticalHeader().setDefaultSectionSize(25)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers);layout.addWidget(self.table)
        self.status=QLabel('可先在主窗口计算并分别保留静息、负荷结果，再查看曲线和测量。');self.status.setWordWrap(True);layout.addWidget(self.status)
        w.compare_button.clicked.connect(self.open);w.study_menu.addAction('静息／负荷对照查看',self.open)
        w.case_changing.connect(self.reset);w.roi_panel.state_changed.connect(self.refresh)
        w.roi_panel.curve_ready.connect(self.refresh);w.feature_panel.task_done.connect(self.refresh)
        w.work.finished.connect(self.refresh)

    @property
    def busy(self):return any(p.busy for p in self.panes.values())

    def reset(self):
        self.epoch=None;self.link.setChecked(False)
        for pane in self.panes.values():pane.populate([])
        self.table.setRowCount(0);self._table_signature=None

    def eventFilter(self,obj,event):
        if obj is self.dialog and event.type()==QEvent.Close:self.cancel()
        return super().eventFilter(obj,event)

    def open(self):
        if self.w.work.busy or (hasattr(self.w,'research') and self.w.research.busy):return
        if self.epoch!=self.w.case_revision:
            self._filling=True;choices=[c for c in self.w.choices if c.kind in SUPPORTED or c.kind=='synthetic']
            kind=self.w.current_choice.kind if self.w.current_choice else None
            for state,pane in self.panes.items():
                preferred=next((c.key for c in choices if c.condition==state and c.kind==kind),None)
                if preferred is None:preferred=next((c.key for c in choices if c.condition==state),None)
                pane.populate(choices,preferred)
            self.epoch=self.w.case_revision;self._filling=False
        self.refresh();self.dialog.show();self.dialog.raise_();self.dialog.activateWindow()

    def link_from(self,pane):
        if not self.link.isChecked() or self._linking:return
        other=self.panes['stress' if pane.condition=='rest' else 'rest']
        if not link_allowed(pane.binding,other.binding):
            self.status.setText('当前类型、单位或时间轴不支持联动；两侧继续独立显示。');return
        try:
            target=relative_target(pane.grid.times_s[pane.slice.value()],other.grid.times_s[other.slice.value()],pane.time.value())
            self._linking=True;other.time.setValue(target)
            self.status.setText('按各自首帧起点的相对时间选择最近采样；不代表心动相位、解剖位置或采集生理同步。')
        except DicomReadError as error:self.status.setText(error_message(error.code))
        finally:self._linking=False

    def refresh(self,*args):
        if self._filling:return
        left,right=(self.panes[s] for s in ('rest','stress'))
        message='本病例只读对照；状态、解剖层面及协议须人工核对。未自动配准、未作生理同步。'
        if left.key and left.key==right.key:message+=' 两侧选择同一序列，不能视为静息/负荷配对。'
        elif left.binding and right.binding and (left.binding['kind']!=right.binding['kind'] or left.binding['units']!=right.binding['units']):
            message+=' 两侧类型或单位不同，仅独立显示，不作信号强度定量比较。'
        self.notice.setText(message)
        for pane in self.panes.values():pane.refresh_content()
        # Keep table objects stable during frame navigation; only rebuild on value/source changes.
        slots={}
        for state,pane in self.panes.items():
            slot=self.w.feature_panel.mr_slots.get(state);doc,result=pane.content()
            if slot and result and slot['key']==pane.key and slot['curves']==result and slot['document']==doc:slots[state]=slot
        signature=document_hash({s:slot['feature'] for s,slot in slots.items()})
        if signature!=getattr(self,'_table_signature',None):
            priority=['normalized_upslope','myocardium.max_upslope','myocardium.peak_enhancement','myocardium.ttp_from_window_origin']
            names=sorted({n for slot in slots.values() for n in slot['feature']['measurements']},key=lambda n:(priority.index(n) if n in priority else len(priority),n))
            self._table_signature=signature
            self.table.setRowCount(len(names))
            for i,name in enumerate(names):
                self.table.setItem(i,0,QTableWidgetItem(measurement_label(name)))
                for col,state in ((1,'rest'),(2,'stress')):
                    v=slots.get(state,{}).get('feature',{}).get('measurements',{}).get(name)
                    self.table.setItem(i,col,QTableWidgetItem(f"{v['value']:.6g} {unit_label(v['unit'])}" if v else '未保留 / 来源不匹配'))
        if hasattr(self.w,'research'):self.w.research.refresh()

    def snapshot(self):return {'linked_time':self.link.isChecked(),'views':{s:p.snapshot() for s,p in self.panes.items()}}

    def cancel(self):
        for pane in self.panes.values():pane.clear()
        self.dialog.hide();self.epoch=None

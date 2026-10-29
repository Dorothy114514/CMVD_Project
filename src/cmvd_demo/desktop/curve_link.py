"""Source-bound curve seeking and manual MR windows shared by both curve plots."""
import numpy as np
from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel
import pyqtgraph as pg

from cmvd_demo.roi import document_hash
from cmvd_demo.perfusion import FeatureError
from cmvd_demo.perfusion.windows import sample_times, nearest_frame, span_frames, analysis_frames
from .presentation import error_message, unit_label
from .select_box import SelectBox


TOOLS = (('定位影像帧','seek'),('选基线（两次点击）','baseline'),('选首过（两次点击）','first_pass'))


def linked_plot(parent):
    plot = pg.PlotWidget(parent=parent,background='#080f1b')
    for menu in (plot.getPlotItem().ctrlMenu, plot.getViewBox().menu):
        menu.setParent(plot,menu.windowFlags())
    plot.setMinimumHeight(210)
    plot.setLabel('bottom','时间',units='s')
    plot.getAxis('bottom').enableAutoSIPrefix(False)
    plot.getAxis('left').enableAutoSIPrefix(False)
    plot.showGrid(x=True,y=True,alpha=.15)
    plot.addLegend()
    return plot


class CurveLink(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.w,self.p,self.f = window,window.roi_panel,window.feature_panel
        self._updating = False
        self._curve_hash = None
        self._selection = None
        self._drag_context = None
        self._combos = []
        for host,layout in ((self.p,self.p.curve_controls),(self.f,self.f.curve_controls)):
            combo = SelectBox()
            for label,key in TOOLS: combo.addItem(label,key)
            combo.currentIndexChanged.connect(lambda index,c=combo:self.change_tool(c.currentData()))
            layout.addWidget(combo)
            self._combos.append(combo)
        self.p.curve_tool,self.f.curve_tool = self._combos
        self.plots = (self.p.plot,self.f.curve_plot)
        self.regions = []
        self.anchors = []
        for plot in self.plots:
            pair = []
            for kind,color in (('baseline',(60,180,150,45)),('first_pass',(240,178,65,45))):
                region = pg.LinearRegionItem(values=[0,1],brush=color,pen=pg.mkPen(color[:3],width=2),swapMode='block')
                region.setZValue(5)
                region.sigRegionChanged.connect(lambda item,k=kind:self.preview_region(k,item))
                region.sigRegionChangeFinished.connect(lambda item,k=kind:self.finish_region(k,item))
                region.hide()
                pair.append(region)
            self.regions.append(pair)
            anchor = pg.InfiniteLine(angle=90,movable=False,pen=pg.mkPen('#cb9bff',width=2,style=Qt.DashLine))
            anchor.setZValue(12);anchor.hide()
            self.anchors.append(anchor)
            plot.scene().sigMouseClicked.connect(lambda event,chart=plot:self.click(chart,event))
        self.cursor = pg.InfiniteLine(angle=90,movable=False,pen=pg.mkPen('#d1e5f4',width=2))
        self.pending = pg.InfiniteLine(angle=90,movable=False,pen=pg.mkPen('#f29494',style=Qt.DashLine))
        self.cursor.setZValue(10);self.pending.setZValue(11)
        self.cursor.hide();self.pending.hide()
        self.p.state_changed.connect(self.refresh)
        self.p.curve_ready.connect(self.refresh)
        window.plane_shown.connect(self.refresh)
        window.case_changing.connect(self.clear)
        window.time_slider.valueChanged.connect(self.navigation)
        window.slice_slider.valueChanged.connect(self.navigation)
        self.f.parameters_changed.connect(self.parameters_changed)
        self.f.runner.idle.connect(self.refresh)
        self.p.runner.idle.connect(self.refresh)
        window.runner.idle.connect(self.refresh)
        window.work.finished.connect(self.refresh)
        window.work.runner.idle.connect(self.refresh)
        self.refresh()

    def context(self):
        p,w = self.p,self.w
        if (p.result is None or p.document is None or p.grid is not w.grid or w.current_pixels is None
                or w.displayed_indices is None or p.result['slice_index']!=w.displayed_indices[1]
                or p.result['document_sha256']!=document_hash(p.document)):
            return None
        return (w.case_revision,p.key,p.result['document_sha256'],p.result['slice_index'],document_hash(p.result))

    def times(self):
        if self.context() is None:
            raise FeatureError('feature_current_curves_required')
        t = sample_times(self.p.grid.times_s[self.w.displayed_indices[1]])
        if not all(np.array_equal(curve['time_s'],t) for curve in self.p.result['curves']):
            raise FeatureError('feature_curve_binding_mismatch')
        return t

    def interactive(self):
        return (not self.w._closing and not self.w.work.busy and not self.p.invalid_edit
                and not self.p.drawing.active and self.context() is not None
                and self.w.slice_slider.value()==self.w.displayed_indices[1])

    def is_mr(self):
        return self.p.grid is not None and self.p.grid.kind=='mr_perfusion'

    def windows(self):
        return analysis_frames(len(self.times()),self.f.baseline_start.value(),self.f.baseline.value(),
                               self.f.start.value(),self.f.end.value())

    def valid_window(self):
        try:
            self.windows()
            return self.is_mr()
        except FeatureError:
            return False

    def change_tool(self, tool):
        if tool!='seek': self.w._stop_play()
        self._selection = None
        for anchor in self.anchors: anchor.hide()
        for combo in self._combos:
            old = combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(tool))
            combo.blockSignals(old)
        self._tell('点击曲线定位影像帧；绿色为基线，金色为首过，可拖动边线调整。' if tool=='seek'
                   else '依次点击区间两端；按实际采集时间吸附至帧，第二次点击后应用。')

    def _tell(self, message):
        self.f.curve_hint.setText(message)
        self.p.curve_hint.setText(message)

    def clear(self):
        self._selection = self._drag_context = None
        self._curve_hash = None
        self.f.curve_plot.clear()
        self.cursor.hide();self.pending.hide()
        for anchor in self.anchors: anchor.hide()
        self.f.curve_position.setText('当前曲线：—')
        for pair in self.regions:
            for region in pair: region.hide()
        self._tell('先计算当前层原始曲线，再点击定位或手动选择分析区间。')

    def refresh(self):
        if self._updating:
            return
        self._updating = True
        try:
            context = self.context()
            if context is None:
                self.clear()
                for combo in self._combos: combo.setEnabled(False)
                return
            times = self.times()
            if context!=self._curve_hash:
                self._selection = self._drag_context = None
                self._curve_hash = context
                self.f.curve_plot.clear()
                for curve in self.p.result['curves']:
                    role = curve['role']
                    self.f.curve_plot.plot(times,curve['mean'],pen=pg.mkPen('#f6bd60' if role=='myocardium' else '#63d8eb',width=2),
                                           symbol='o',symbolSize=4,name='心肌' if role=='myocardium' else '血池')
                self.f.curve_plot.setLabel('left','原始均值',units=unit_label(self.p.result['curves'][0]['units']))
            if not self.interactive(): self._selection = None
            for plot,pair,anchor in zip(self.plots,self.regions,self.anchors):
                if anchor not in plot.getPlotItem().items: plot.addItem(anchor,ignoreBounds=True)
                anchor.setVisible(self._selection is not None)
                if self._selection is not None: anchor.setValue(times[self._selection[2]])
                for region in pair:
                    if region not in plot.getPlotItem().items: plot.addItem(region,ignoreBounds=True)
                    region.setBounds((times[0],times[-1]))
                    region.setMovable(self.interactive() and self.is_mr())
                    region.hide()
            for item in (self.cursor,self.pending):
                if item not in self.f.curve_plot.getPlotItem().items: self.f.curve_plot.addItem(item,ignoreBounds=True)
            self.cursor.setValue(times[self.w.displayed_indices[0]])
            if self.p._cursor is not None: self.p._cursor.setZValue(10)
            self.cursor.show()
            target = self.w.time_slider.value()
            waiting = target!=self.w.displayed_indices[0]
            self.pending.setVisible(waiting)
            if waiting: self.pending.setValue(times[target])
            for combo in self._combos:
                combo.setEnabled(self.interactive())
                for i in (1,2): combo.model().item(i).setEnabled(self.is_mr())
            if self.is_mr():
                try:
                    b0,b1,a0,a1 = self.windows()
                    for pair in self.regions:
                        for region,lo,hi in ((pair[0],b0,b1),(pair[1],a0,a1)):
                            region.setRegion((times[lo],times[hi]));region.show()
                    self._tell(f'基线 {b0+1}–{b1+1}帧（{times[b0]:.4g}–{times[b1]:.4g}s）；首过 {a0+1}–{a1+1}帧（{times[a0]:.4g}–{times[a1]:.4g}s）。修改后请重新计算。')
                except FeatureError:
                    self._tell('分析区间无效：基线至少2帧，且结束早于首过起点；首过至少2帧且在当前序列范围内。请用帧号或两次点选设置。')
            else:
                self.change_tool('seek')
                self._tell('点击曲线定位当前层影像；此类曲线不启用MR基线/首过分析窗口。')
            t = self.w.displayed_indices[0]
            self.f.curve_position.setText(f'已显示帧 {t+1}/{len(times)} · {times[t]:.4g} s'+
                (f'；正在加载帧 {target+1}，游标保留已显示位置' if waiting else ''))
        except FeatureError as error:
            self.clear()
            self._tell(error_message(error.code))
        finally:
            self._updating = False

    def navigation(self):
        self._selection = self._drag_context = None
        self.refresh()

    def parameters_changed(self):
        self._selection = self._drag_context = None
        self.refresh()

    def click(self, plot, event):
        if event.button()!=Qt.LeftButton or not self.interactive(): return
        vb = plot.getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()): return
        position = vb.mapSceneToView(event.scenePos()).x()
        times = self.times()
        if not times[0]<=position<=times[-1]: return
        index = nearest_frame(times,position)
        mode = self._combos[0].currentData()
        if mode=='seek':
            self.w._stop_play()
            self.w.time_slider.setValue(index)
            return
        if not self.is_mr(): return
        context = self.context()
        if self._selection is None or self._selection[:2]!=(context,mode):
            self._selection = context,mode,index
            for anchor in self.anchors:
                anchor.setValue(times[index]);anchor.show()
            self._tell(f'已选起点：帧{index+1}，{times[index]:.4g}s；再点一次选择另一端。')
        else:
            first = self._selection[2]
            self._selection = None
            for anchor in self.anchors: anchor.hide()
            self.apply_span('baseline' if mode=='baseline' else 'first_pass',(times[first],times[index]))

    def preview_region(self, kind, region):
        if self._updating: return
        if not self.interactive() or not self.is_mr():
            self.refresh();return
        self.w._stop_play()
        self._selection = None
        self._drag_context = self.context()
        self._tell('正在选择基线区间；松开后按采样帧应用。' if kind=='baseline' else '正在选择首过区间；松开后按采样帧应用。')

    def finish_region(self, kind, region):
        if self._updating: return
        context,self._drag_context = self._drag_context,None
        if context is not None and context==self.context() and self.interactive():
            self.apply_span(kind,region.getRegion())
        else:
            self.refresh()

    def apply_span(self, kind, span):
        if not self.interactive() or not self.is_mr(): return False
        try:
            times = self.times()
            a,b = span_frames(times,span)
            b0,bn,s,e = self.f.baseline_start.value(),self.f.baseline.value(),self.f.start.value(),self.f.end.value()
            if kind=='baseline': b0,bn = a+1,b-a+1
            else: s,e = a+1,b+1
            other_valid = (1<=s<e<=len(times)) if kind=='baseline' else (1<=b0 and bn>=2 and b0+bn-1<=len(times))
            if other_valid:
                analysis_frames(len(times),b0,bn,s,e)
            controls = (self.f.baseline_start,self.f.baseline,self.f.start,self.f.end)
            if [control.value() for control in controls]==[b0,bn,s,e]:
                self.refresh()
                return True
            old = [control.blockSignals(True) for control in controls]
            try:
                for control,value in zip(controls,(b0,bn,s,e)): control.setValue(value)
            finally:
                for control,blocked in zip(controls,old): control.blockSignals(blocked)
            self.f.window_parameters_changed()
            self.refresh()
            return True
        except FeatureError as error:
            self.refresh()
            self._tell(error_message(error.code)+' 本次选择未应用，请重新选择。')
            return False

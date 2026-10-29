"""Guard destructive user actions; only a successful full save permits continuation."""
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QMessageBox
from cmvd_demo.roi import document_hash


class UnsavedWork(QObject):
    def __init__(self, window, prompt=None):
        super().__init__(window)
        self.w = window
        self.prompt = prompt
        self.pending = None
        self.saved_signature = self.signature()
        window.work.finished.connect(self._saved)

    def signature(self):
        p, f = self.w.roi_panel, self.w.feature_panel
        documents = {k:v for k,v in p.documents.items() if v['tracks']}
        slots = {family:{s:{k:v for k,v in slot.items() if k != 'grid'}
                         for s,slot in getattr(f, family).items()}
                 for family in ('mr_slots', 'pet_slots')}
        data = {'documents':documents, **slots, 'pet':f.pet,
                'reference':f.mr_reference, 'candidate':f.candidate}
        if documents or f.candidate or any(slots.values()) or f.pet:
            data['settings'] = [f.condition.currentData(), f.protocol.text(),
                f.baseline_start.value(), f.baseline.value(), f.start.value(), f.end.value(),
                f.window_review.isChecked(), f.pair_review.isChecked(), f.protocol_review.isChecked()]
        return document_hash(data)

    def dirty(self):
        p = self.w.roi_panel
        return p.drawing.active or p.invalid_edit or self.signature() != self.saved_signature

    def mark_saved(self):
        self.saved_signature = self.signature()

    def ask(self, action):
        if self.prompt is not None:
            return self.prompt(action)
        dialog=QMessageBox(QMessageBox.Question,'保存当前研究工作',
            '当前轮廓或分析有未保存修改。'+action+'前是否保存？\n'
            '未完成勾画须先完成或取消；单项分析保存不代替整个病例工作保存。',
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,self.w)
        for choice,text in ((QMessageBox.Save,'保存工作'),(QMessageBox.Discard,'放弃修改'),(QMessageBox.Cancel,'取消操作')):
            dialog.button(choice).setText(text)
        dialog.setDefaultButton(QMessageBox.Cancel)
        dialog.setEscapeButton(QMessageBox.Cancel)
        answer=dialog.exec()
        dialog.deleteLater()
        return answer

    def request(self, action, callback):
        if self.pending is not None:
            self.w.status.setText('正在保存当前工作，请等待或取消保存。')
            return
        if not self.dirty():
            callback()
            return
        self.w._stop_play()
        answer = self.ask(action)
        if answer == QMessageBox.Discard:
            callback()
        elif answer == QMessageBox.Save:
            if not self.w.work.ready() or self.w.work.busy or self.w.research.busy:
                self.w.status.setText('尚不能保存：请先完成或取消勾画，等待当前影像/曲线任务结束，再重试。当前工作已保留。')
                return
            self.pending = callback
            self.w.work.save()

    def _saved(self):
        callback, self.pending = self.pending, None
        if callback is not None and not self.w.work.last_error and self.w.work.saved_output and not self.dirty():
            # Leave the work result callback before changing the case or closing widgets.
            epoch,signature=self.w.case_revision,self.saved_signature
            def continue_if_current():
                if self.w.case_revision==epoch and self.signature()==signature and not self.dirty():
                    callback()
            QTimer.singleShot(0, continue_if_current)

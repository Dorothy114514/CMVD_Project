"""Native regressions for actual QA findings; no patient data or source mutation."""
import copy
import errno
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from test_desktop import APP, until, dispose_window, discard_unsaved
from PySide6.QtCore import Qt,QTimer,QEventLoop
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox,QScrollArea
from cmvd_demo.desktop.window import BrowserWindow
from cmvd_demo.desktop.diagnostics import error_code, record_error
from cmvd_demo.io import DicomReadError,ReadCancelled
from cmvd_demo.roi import document_hash


class ProductFixTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.answer=QMessageBox.Cancel;self.asked=[]
        def answer(action):self.asked.append(action);return self.answer
        self.w=BrowserWindow(unsaved_prompt=answer)
        self.w.work.output_root=self.root/'work';self.w.research.output_root=self.root/'research'
        self.w.sequences.output_root=self.root/'sequence'
        self.w.show();self.w.open_synthetic_perfusion()
        until(lambda:self.w.current_pixels is not None)
        until(lambda:not self.w.runner.busy)

    def tearDown(self):
        self.w.unsaved.prompt=discard_unsaved;self.w.unsaved.pending=None
        self.w._close_approved=True
        dispose_window(self.w);self.temp.cleanup()

    def roi(self):
        self.w.roi_panel.add_button.click()
        self.assertTrue(self.w.unsaved.dirty())
        return document_hash(self.w.roi_panel.document)

    def assert_no_delayed_progress(self,w):
        seen=[];loop=QEventLoop();timer=QTimer();timer.setInterval(25)
        timer.timeout.connect(lambda:seen.append(w._index_dialog.isVisible()))
        timer.start();QTimer.singleShot(w._index_dialog.minimumDuration()+1200,loop.quit)
        loop.exec();timer.stop()
        self.assertFalse(any(seen),'progress dialog reappeared with no index operation')
        self.assertFalse(w._pending_index)

    def test_empty_start_idle_does_not_auto_show_progress_or_read_old_case(self):
        other=BrowserWindow(unsaved_prompt=discard_unsaved)
        try:
            other.show()
            with patch.object(other.backend,'index') as index:
                self.assert_no_delayed_progress(other)
                index.assert_not_called()
            self.assertFalse(other.choices);self.assertIsNone(other.current_pixels)
        finally:dispose_window(other)

    def test_successful_index_stops_delayed_progress_timer(self):
        from cmvd_demo.desktop.data import perfusion_synthetic_choices
        with patch.object(self.w.backend,'index',return_value=(perfusion_synthetic_choices(),{'indexed_objects':14,'series_count':2,'issues':[]})):
            self.w.open_case(self.root)
            until(lambda:not self.w.runner.busy and not self.w._pending_index)
        self.assertIsNotNone(self.w.current_pixels)
        self.assert_no_delayed_progress(self.w)

    def test_failed_index_stops_delayed_progress_timer(self):
        with patch.object(self.w.backend,'index',side_effect=DicomReadError('unknown_read_failure')):
            self.w.open_case(self.root);until(lambda:not self.w.runner.busy)
        self.assertEqual(self.w.last_error,'unknown_read_failure')
        self.assert_no_delayed_progress(self.w)

    def test_cancelled_index_stops_delayed_progress_timer(self):
        started,release=Event(),Event();self.addCleanup(release.set)
        def slow(*a):started.set();release.wait(3);raise ReadCancelled()
        with patch.object(self.w.backend,'index',side_effect=slow):
            self.w.open_case(self.root);until(started.is_set)
            self.w.cancel_current();release.set();until(lambda:not self.w.runner.busy)
        self.assert_no_delayed_progress(self.w)

    def test_cancel_switch_preserves_roi_and_case(self):
        before=self.roi();epoch=self.w.case_revision
        self.w.open_synthetic()
        self.assertEqual(self.w.case_revision,epoch)
        self.assertEqual(document_hash(self.w.roi_panel.document),before)
        self.assertTrue(self.asked)

    def test_actual_default_prompt_cancel_is_safe(self):
        before=self.roi();self.w.unsaved.prompt=None
        def respond():
            dialog=APP.activeModalWidget()
            self.assertIsInstance(dialog,QMessageBox)
            dialog.button(QMessageBox.Cancel).click()
        QTimer.singleShot(0,respond)
        self.w.open_synthetic()
        self.assertEqual(document_hash(self.w.roi_panel.document),before)

    def test_cancel_close_keeps_window_and_work_visible(self):
        self.roi();self.w.close()
        self.assertTrue(self.w.isVisible());self.assertFalse(self.w._closing)

    def test_discard_close_drains_normally(self):
        self.roi();self.answer=QMessageBox.Discard;self.w.close()
        until(lambda:self.w._closing and not self.w.isVisible())

    def test_save_then_switch_only_after_success(self):
        self.roi();epoch=self.w.case_revision;self.answer=QMessageBox.Save
        self.w.open_synthetic()
        self.assertEqual(self.w.case_revision,epoch)
        until(lambda:self.w.case_revision>epoch,15000)
        self.assertIsNotNone(self.w.work.saved_output)
        self.assertTrue((self.w.work.saved_output/'bundle.json').is_file())
        self.assertFalse(self.w.unsaved.dirty())

    def test_save_failure_aborts_switch_and_logs_safe_code(self):
        from cmvd_demo.desktop import work_controller,diagnostics
        before=self.roi();epoch=self.w.case_revision;self.answer=QMessageBox.Save
        with patch.object(diagnostics,'workspace_root',return_value=self.root), patch.object(work_controller,'save_record',side_effect=PermissionError('SYNTHETIC_SENSITIVE_DO_NOT_LOG')):
            self.w.open_synthetic();until(lambda:not self.w.work.busy)
        self.assertEqual(self.w.case_revision,epoch)
        self.assertEqual(document_hash(self.w.roi_panel.document),before)
        self.assertIsNone(self.w.unsaved.pending)
        self.assertEqual(self.w.work.last_error,'filesystem_permission_denied')
        logs=list((self.root/'outputs/runtime_logs').glob('*events.jsonl'));self.assertEqual(len(logs),1)
        text=logs[0].read_text(encoding='utf-8');self.assertNotIn('SYNTHETIC_SENSITIVE',text)
        self.assertIn('filesystem_permission_denied',text)

    def test_cancel_save_never_resumes_switch(self):
        from cmvd_demo.desktop import work_controller
        before=self.roi();epoch=self.w.case_revision;self.answer=QMessageBox.Save
        started,release=Event(),Event();self.addCleanup(release.set)
        def slow(*a,**kw):started.set();release.wait(3);raise ReadCancelled()
        with patch.object(work_controller,'save_record',side_effect=slow):
            self.w.open_synthetic();until(started.is_set);self.w.work.cancel();release.set()
            until(lambda:not self.w.work.busy)
        self.assertEqual(self.w.case_revision,epoch)
        self.assertEqual(document_hash(self.w.roi_panel.document),before)
        self.assertIsNone(self.w.unsaved.pending)

    def test_save_then_exit_persists_before_close(self):
        self.roi();self.answer=QMessageBox.Save;self.w.close()
        until(lambda:self.w._closing,15000)
        self.assertTrue((self.w.work.saved_output/'bundle.json').is_file())

    def test_missing_directory_keeps_unsaved_work_without_prompt(self):
        before=self.roi();pixels=self.w.current_pixels.copy();self.w.open_case(self.root/'absent')
        self.assertEqual(self.w.last_error,'case_directory_missing')
        self.assertEqual(document_hash(self.w.roi_panel.document),before)
        self.assertTrue((self.w.current_pixels==pixels).all());self.assertFalse(self.asked)

    def test_index_failure_restores_entire_old_work(self):
        before=self.roi();pixels=self.w.current_pixels.copy();epoch=self.w.case_revision
        self.answer=QMessageBox.Discard
        with patch.object(self.w.backend,'index',side_effect=DicomReadError('unknown_read_failure')):
            self.w.open_case(self.root);until(lambda:not self.w.runner.busy)
        self.assertFalse(self.w._pending_index);self.assertEqual(self.w.case_revision,epoch)
        self.assertEqual(document_hash(self.w.roi_panel.document),before)
        self.assertTrue((self.w.current_pixels==pixels).all());self.assertTrue(self.w.unsaved.dirty())

    def test_index_cancel_keeps_old_work_and_rejects_late_result(self):
        before=self.roi();self.answer=QMessageBox.Discard
        started,release=Event(),Event();self.addCleanup(release.set)
        def slow(*a):started.set();release.wait(3);return [],{'indexed_objects':0,'series_count':0,'issues':[]}
        with patch.object(self.w.backend,'index',side_effect=slow):
            self.w.open_case(self.root);until(started.is_set)
            self.assertIsNone(self.w.current_pixels)
            self.w.cancel_current();release.set();until(lambda:not self.w.runner.busy)
        self.assertEqual(document_hash(self.w.roi_panel.document),before)
        self.assertIsNotNone(self.w.current_pixels);self.assertFalse(self.w._index_dialog.isVisible())

    def test_empty_index_is_not_committed(self):
        self.roi();self.answer=QMessageBox.Discard;epoch=self.w.case_revision
        with patch.object(self.w.backend,'index',return_value=([],{'indexed_objects':0,'series_count':0,'issues':[]})):
            self.w.open_case(self.root);until(lambda:not self.w.runner.busy)
        self.assertEqual(self.w.case_revision,epoch);self.assertEqual(self.w.last_error,'case_no_indexed_objects')

    def test_draft_save_choice_cannot_silently_discard_draft(self):
        p=self.w.roi_panel;p.draw_button.click();self.assertTrue(p.drawing.active)
        self.answer=QMessageBox.Save;epoch=self.w.case_revision;self.w.open_synthetic()
        self.assertEqual(self.w.case_revision,epoch);self.assertTrue(p.drawing.active)
        self.assertIsNone(self.w.unsaved.pending)

    def test_restore_cancel_keeps_unsaved_edit(self):
        self.roi();self.w.work.save();until(lambda:not self.w.work.busy,15000)
        run=self.w.work.saved_output;self.assertFalse(self.w.unsaved.dirty())
        self.w.roi_panel.remove_track();before=document_hash(self.w.roi_panel.document)
        self.w.work.restore(run);self.assertEqual(document_hash(self.w.roi_panel.document),before)
        self.assertIsNone(self.w.work.restored_output)

    def test_browsing_does_not_mark_saved_roi_dirty(self):
        self.roi();self.w.work.save();until(lambda:not self.w.work.busy,15000)
        self.w.time_slider.setValue(3);until(lambda:self.w.displayed_indices==(3,0))
        self.w.level_center.setValue(90);self.assertFalse(self.w.unsaved.dirty())

    def test_comparison_escape_cancels_pending_frame(self):
        c=self.w.comparison;c.open();until(lambda:all(p.displayed is not None for p in c.panes.values()))
        p=c.panes['rest'];started,release=Event(),Event();self.addCleanup(release.set)
        original=self.w.backend.plane
        def slow(*a):started.set();release.wait(3);return original(*a)
        with patch.object(self.w.backend,'plane',side_effect=slow):
            p.time.setValue(3);until(started.is_set);QTest.keyClick(c.dialog,Qt.Key_Escape)
            self.assertFalse(c.dialog.isVisible());self.assertIsNone(p._job)
            release.set();until(lambda:not p.runner.busy)
        self.assertIsNone(p.pixels);self.assertIsNone(p.displayed)

    def test_research_escape_cancels_and_unlocks(self):
        from cmvd_demo.desktop import research_controller
        r=self.w.research;r.open();started,release=Event(),Event();self.addCleanup(release.set)
        def slow(*a,**kw):started.set();release.wait(3);raise ReadCancelled()
        with patch.object(research_controller,'save_research',side_effect=slow):
            r.export();until(started.is_set);QTest.keyClick(r.dialog,Qt.Key_Escape)
            self.assertTrue(self.w.centralWidget().isEnabled());self.assertIsNone(r._job)
            release.set();until(lambda:not r.busy)
        self.assertIsNone(r.last_output)

    def test_sequence_x_and_escape_cancel_hidden_save(self):
        from cmvd_demo.desktop import sequence_panel
        for escape in (False,True):
            p=self.w.sequences;p.open();started,release=Event(),Event();self.addCleanup(release.set)
            def slow(*a,**kw):started.set();release.wait(3);raise ReadCancelled()
            with patch.object(sequence_panel,'save_review',side_effect=slow):
                p.export();until(started.is_set)
                if escape:QTest.keyClick(p.dialog,Qt.Key_Escape)
                else:p.dialog.close()
                self.assertIsNone(p._job);release.set();until(lambda:not p.runner.busy)
            self.assertIsNone(p.saved_output)

    def test_output_error_logged_and_has_specific_ui_message(self):
        from cmvd_demo.desktop import diagnostics
        blocked=self.root/'file';blocked.write_text('synthetic');r=self.w.research;r.output_root=blocked
        with patch.object(diagnostics,'workspace_root',return_value=self.root):
            r.export();until(lambda:not r.busy)
        self.assertEqual(r.last_error,'output_path_unavailable')
        self.assertIn('同名文件',r.message.text());self.assertIn(r.last_error,r.message.toolTip())
        self.assertTrue(list((self.root/'outputs/runtime_logs').glob('*events.jsonl')))
        self.assertTrue(self.w.centralWidget().isEnabled())

    def test_logging_failure_does_not_crash_error_handling(self):
        from cmvd_demo.desktop import diagnostics
        blocked=self.root/'blocked';blocked.write_text('synthetic')
        with patch.object(diagnostics,'workspace_root',return_value=blocked):
            self.assertFalse(record_error('output_disk_full','ResearchController'))
        self.assertEqual(error_code(OSError(errno.ENOSPC,'SYNTHETIC')),'output_disk_full')

    def test_scrollable_layout_preserves_full_button_text(self):
        self.w.resize(1040,600);APP.processEvents()
        self.assertIsInstance(self.w.centralWidget(),QScrollArea)
        self.assertGreater(self.w.centralWidget().horizontalScrollBar().maximum(),0)
        for b in (self.w.play_button,self.w.fit_button,self.w.auto_button):
            self.assertGreaterEqual(b.width(),b.minimumSizeHint().width())
        c=self.w.comparison;c.open();APP.processEvents();c.dialog.resize(900,600);APP.processEvents()
        self.assertLessEqual(c.dialog.height(),600)
        self.assertTrue(c.dialog.findChildren(QScrollArea))


if __name__=='__main__':unittest.main()

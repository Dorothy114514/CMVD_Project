from test_desktop import discard_unsaved
"""Native Qt save/reopen, rejected restores, cancellation and worker lifetime."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from test_desktop import APP,until,dispose_window
from cmvd_demo.desktop.window import BrowserWindow
from cmvd_demo.roi import document_hash
from cmvd_demo.work import load_record


class WorkDesktopTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.windows=[]

    def tearDown(self):
        for window in self.windows:dispose_window(window)

    def window(self):
        w=BrowserWindow(unsaved_prompt=discard_unsaved)
        self.windows.append(w)
        w.work.output_root=self.root
        w.show()
        w.open_synthetic_perfusion()
        until(lambda:w.current_pixels is not None)
        return w

    def calculate(self,w,condition='rest'):
        w.select_choice('synthetic-perfusion-'+condition)
        until(lambda:w.current_pixels is not None and w.current_choice.key=='synthetic-perfusion-'+condition)
        p,f=w.roi_panel,w.feature_panel
        for role in (0,1):
            p.role.setCurrentIndex(role)
            QTest.mouseClick(p.add_button,Qt.LeftButton)
        QTest.mouseClick(p.calculate_button,Qt.LeftButton)
        until(lambda:p.result is not None or bool(p.last_error))
        self.assertFalse(p.last_error)
        f.condition.setCurrentIndex(1 if condition=='rest' else 2)
        f.protocol.setText('SYNTH-WORK')
        f.baseline.setValue(2)
        f.start.setValue(3)
        f.end.setValue(7)
        f.calculate_mr()
        until(lambda:f.candidate is not None or bool(f.last_error))
        self.assertFalse(f.last_error)

    def wait(self,w):
        until(lambda:not w.work.busy,timeout=15000)
        self.assertFalse(w.work.last_error,w.work.last_error)

    def test_single_analysis_button_saves_and_reopens_in_another_window(self):
        w=self.window()
        self.calculate(w)
        w.time_slider.setValue(4)
        until(lambda:w.displayed_indices==(4,0))
        w.level_center.setValue(50)
        w.level_width.setValue(120)
        w.workspace_tabs.setCurrentIndex(1)
        expected=copy.deepcopy(w.feature_panel.candidate)
        doc=document_hash(w.roi_panel.document)
        QTest.mouseClick(w.feature_panel.single_save_button,Qt.LeftButton)
        self.wait(w)
        run=w.work.saved_output
        self.assertIsNotNone(run)
        self.assertEqual(load_record(run)['scope'],'single_mr')
        restored=self.window()
        restored.work.restore(run/'work.json')
        self.wait(restored)
        self.assertEqual(document_hash(restored.roi_panel.document),doc)
        self.assertEqual(restored.feature_panel.candidate,expected)
        self.assertEqual(restored.displayed_indices,(4,0))
        self.assertEqual(restored.level_center.value(),50)
        self.assertEqual(restored.level_width.value(),120)
        self.assertEqual(restored.workspace_tabs.currentIndex(),1)
        self.assertIsNotNone(restored.roi_panel.result)
        self.assertFalse(any(restored.roi_panel.document['reviews'].values()))
        self.assertEqual(restored.feature_panel.mr_slots,{})
        self.assertTrue(restored.centralWidget().isEnabled())

    def test_all_case_documents_pair_and_candidate_survive_restore(self):
        w=self.window()
        for condition in ('rest','stress'):
            self.calculate(w,condition)
            w.feature_panel.capture_mr()
        from test_perfusion import reference
        w.feature_panel.mr_reference=reference(w.feature_panel._mr_pair())
        expected=w.work.capture()
        w.save_work_button.click()
        self.wait(w)
        run=w.work.saved_output
        other=self.window()
        other.work.restore(run)
        self.wait(other)
        self.assertEqual(other.current_choice.key,'synthetic-perfusion-stress')
        self.assertEqual(set(other.roi_panel.documents),set(expected['entries']))
        self.assertEqual(set(other.feature_panel.mr_slots),{'rest','stress'})
        self.assertEqual(other.feature_panel.mr_reference,expected['feature_state']['mr_reference'])
        self.assertAlmostEqual(other.feature_panel._mr_pair()['measurements']['mr_mpri']['value'],2.)
        other.time_slider.setValue(2)
        until(lambda:other.displayed_indices==(2,0))
        np.testing.assert_array_equal(other.current_pixels,other.grid.read_plane(2,0))

    def test_pet_workstation_result_and_local_tacs_restore_from_original_generated_dicoms(self):
        from test_perfusion_desktop import PerfusionDesktopTests
        w=self.window()
        self.w,self.f=w,w.feature_panel
        self.f.output_root=self.root/'features'
        path=PerfusionDesktopTests.prepare_pet(self)
        self.f.load_pet(path)
        until(lambda:self.f.pet is not None or bool(self.f.last_error))
        self.assertFalse(self.f.last_error)
        expected=copy.deepcopy(self.f.pet)
        source=w.grid.cells[0][0].path.parent
        w.work.save()
        self.wait(w)
        run=w.work.saved_output
        other=self.window()
        other.open_case(source)
        until(lambda:other.current_pixels is not None)
        other.work.restore(run)
        self.wait(other)
        self.assertEqual(other.feature_panel.pet,expected)
        self.assertEqual(set(other.feature_panel.pet_slots),{'rest','stress'})
        self.assertAlmostEqual(other.feature_panel.pet['measurements']['pet_mfr']['value'],3.)
        self.assertFalse(other.feature_panel.pet['qc']['independent_mbf_validation'])
        other.roi_panel.remove_track()
        self.assertIsNone(other.feature_panel.pet)

    def test_tampered_restore_keeps_existing_work_untouched(self):
        w=self.window()
        self.calculate(w)
        w.work.save()
        self.wait(w)
        run=w.work.saved_output
        expected=w.work.capture()
        with (run/'work.json').open('a') as stream:stream.write(' ')
        w.work.restore(run)
        until(lambda:not w.work.busy)
        self.assertEqual(w.work.last_error,'work_file_changed')
        self.assertEqual(w.roi_panel.document,expected['entries'][expected['active_key']]['document'])
        self.assertIsNotNone(w.feature_panel.candidate)
        self.assertTrue(w.centralWidget().isEnabled())

    def test_cancelled_background_save_has_no_late_publication(self):
        w=self.window()
        self.calculate(w)
        started,release=Event(),Event()
        self.addCleanup(release.set)
        def delayed(*args,**kwargs):
            started.set()
            release.wait(3)
            return self.root/'MUST-NOT-PUBLISH'
        with patch('cmvd_demo.desktop.work_controller.save_record',side_effect=delayed):
            w.work.save()
            until(started.is_set)
            self.assertFalse(w.centralWidget().isEnabled())
            w.work.cancel_action.trigger()
            release.set()
            until(lambda:not w.work.busy)
        self.assertIsNone(w.work.saved_output)
        self.assertTrue(w.centralWidget().isEnabled())

    def test_case_change_and_close_drain_sixth_channel(self):
        w=self.window()
        self.calculate(w)
        started,release=Event(),Event()
        self.addCleanup(release.set)
        def delayed(*args,**kwargs):
            started.set()
            release.wait(3)
            return self.root/'MUST-NOT-PUBLISH'
        with patch('cmvd_demo.desktop.work_controller.save_record',side_effect=delayed):
            w.work.save()
            until(started.is_set)
            w.open_synthetic()
            w.close()
            self.assertTrue(w.work.runner.busy)
            self.assertFalse(w.isVisible())
            release.set()
            until(lambda:not any(r.busy for r in w.background_runners()))
        self.assertIsNone(w.work.saved_output)

    def test_empty_window_requires_explicit_original_case(self):
        w=BrowserWindow(unsaved_prompt=discard_unsaved)
        self.windows.append(w)
        w.show()
        w.work.restore(self.root/'not-opened')
        self.assertEqual(w.work.last_error,'work_open_case_first')
        self.assertFalse(w.work.busy)

    def test_wrong_case_restore_does_not_change_current_image(self):
        w=self.window()
        self.calculate(w)
        w.work.save()
        self.wait(w)
        run=w.work.saved_output
        w.open_synthetic()
        until(lambda:w.current_pixels is not None)
        pixels=w.current_pixels.copy()
        choice=w.current_choice.key
        w.work.restore(run)
        until(lambda:not w.work.busy)
        self.assertEqual(w.work.last_error,'work_case_mismatch')
        self.assertEqual(w.current_choice.key,choice)
        np.testing.assert_array_equal(w.current_pixels,pixels)

    def test_cancel_restore_never_applies_late_result(self):
        w=self.window()
        self.calculate(w)
        w.work.save()
        self.wait(w)
        run=w.work.saved_output
        old=document_hash(w.roi_panel.document)
        started,release=Event(),Event()
        self.addCleanup(release.set)
        from cmvd_demo.desktop import work_controller
        original=work_controller.restore_record
        def delayed(*args,**kwargs):
            value=original(*args,**kwargs)
            started.set()
            release.wait(3)
            return value
        with patch.object(work_controller,'restore_record',side_effect=delayed):
            w.work.restore(run)
            until(started.is_set)
            w.work.cancel_action.trigger()
            release.set()
            until(lambda:not w.work.busy)
        self.assertIsNone(w.work.restored_output)
        self.assertEqual(document_hash(w.roi_panel.document),old)
        self.assertTrue(w.roi_dock.isEnabled())

    def test_restored_results_invalidate_when_roi_is_edited(self):
        w=self.window()
        self.calculate(w)
        w.work.save()
        self.wait(w)
        w.work.restore(w.work.saved_output)
        self.wait(w)
        w.roi_panel.role.setCurrentIndex(1)
        w.roi_panel.remove_track()
        self.assertIsNone(w.roi_panel.result)
        self.assertIsNone(w.feature_panel.candidate)
        self.assertFalse(w.feature_panel.single_save_button.isEnabled())

    def test_valid_work_can_recover_an_invalid_current_edit(self):
        w=self.window()
        self.calculate(w)
        w.work.save()
        self.wait(w)
        run=w.work.saved_output
        expected=document_hash(w.roi_panel.document)
        w.roi_panel.invalid_edit=True
        w.roi_panel.last_error='roi_self_intersection'
        w.feature_panel.last_error='feature_stale_curves'
        w.work.restore(run)
        self.wait(w)
        self.assertFalse(w.roi_panel.invalid_edit)
        self.assertFalse(w.roi_panel.last_error)
        self.assertFalse(w.feature_panel.last_error)
        self.assertEqual(document_hash(w.roi_panel.document),expected)


if __name__=='__main__':unittest.main()

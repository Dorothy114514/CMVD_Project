from test_desktop import discard_unsaved
"""Actual nonuniform times, native chart gestures, invalidation and persistence."""
import copy
from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest

import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtTest import QTest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cmvd_demo.desktop.window import BrowserWindow
from cmvd_demo.perfusion import FeatureError
from cmvd_demo.perfusion.windows import nearest_frame,span_frames,analysis_frames
from cmvd_demo.roi import document_hash
from test_desktop import APP,until,dispose_window
from test_roi_editing import drag


class WindowMappingTests(unittest.TestCase):
    def test_nearest_nonuniform_sample_and_earlier_tie(self):
        times=[0,1,2,4,7,8,10]
        self.assertEqual(nearest_frame(times,5.8),4)
        self.assertEqual(nearest_frame(times,5.5),3)
        self.assertEqual(span_frames(times,[9.9,2.1]),(2,6))
        self.assertEqual(analysis_frames(7,1,2,3,7),(0,1,2,6))

    def test_single_sample_overlap_and_bad_times_rejected(self):
        for times in ([0,0,1],[0,float('nan'),2],[1]):
            with self.assertRaises(FeatureError):nearest_frame(times,0)
        with self.assertRaises(FeatureError):span_frames([0,1,5],[.9,1.1])
        for args in ((7,1,3,3,7),(7,1,2,3,8),(7,1,1,3,7),(7,1,2,3,3)):
            with self.assertRaises(FeatureError):analysis_frames(*args)


class CurveLinkTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.w=BrowserWindow(unsaved_prompt=discard_unsaved)
        self.windows=[self.w]
        self.w.work.output_root=Path(self.temp.name)
        self.w.show();self.w.activateWindow()
        self.w.open_synthetic_perfusion()
        until(lambda:self.w.current_pixels is not None and not self.w.runner.busy)
        self.p,self.f,self.link=self.w.roi_panel,self.w.feature_panel,self.w.curve_link
        self.curves()
        self.f.condition.setCurrentIndex(1)
        self.f.protocol.setText('SYNTH-CURVE')
        self.f.baseline.setValue(2);self.f.start.setValue(3);self.f.end.setValue(7)
        self.w.workspace_tabs.setCurrentIndex(1)
        self.w.workspace_tabs.widget(1).ensureWidgetVisible(self.f.curve_plot)
        APP.processEvents()

    def tearDown(self):
        for window in self.windows:dispose_window(window)
        self.temp.cleanup()

    def curves(self):
        for role in (0,1):self.p.role.setCurrentIndex(role);self.p.add_reference()
        self.p.calculate()
        until(lambda:self.p.result is not None)

    def position(self,time,plot=None,fraction=.7):
        plot=plot or self.f.curve_plot
        vb=plot.getViewBox()
        low,high=vb.viewRange()[1]
        return plot.mapFromScene(vb.mapViewToScene(QPointF(time,low+(high-low)*fraction)))

    def click(self,time,plot=None):
        plot=plot or self.f.curve_plot
        QTest.mouseClick(plot.viewport(),Qt.LeftButton,Qt.NoModifier,self.position(time,plot))
        APP.processEvents()

    def values(self):
        return [c.value() for c in (self.f.baseline_start,self.f.baseline,self.f.start,self.f.end)]

    def calculate(self):
        self.f.calculate_mr()
        until(lambda:self.f.candidate is not None or bool(self.f.last_error))
        self.assertFalse(self.f.last_error)

    def test_regions_use_sample_times_and_numeric_inputs_update_both_plots(self):
        for pair in self.link.regions:
            self.assertEqual(pair[0].getRegion(),(0.,1.))
            self.assertEqual(pair[1].getRegion(),(2.,10.))
            self.assertTrue(pair[0].isVisible())
        self.f.end.setValue(6)
        self.assertTrue(all(pair[1].getRegion()==(2.,8.) for pair in self.link.regions))

    def test_native_curve_click_seeks_image_and_slider_updates_cursor(self):
        self.click(5.8)
        until(lambda:self.w.displayed_indices==(4,0))
        self.assertEqual(self.link.cursor.value(),7.)
        self.assertEqual(self.p._cursor.value(),7.)
        self.w.time_slider.setValue(3)
        until(lambda:self.w.displayed_indices==(3,0))
        self.assertEqual(self.link.cursor.value(),4.)
        np.testing.assert_array_equal(self.w.current_pixels,self.w.grid.read_plane(3,0))

    def test_native_two_click_window_selection_and_review_reset(self):
        self.calculate();self.f.window_review.setChecked(True)
        self.f.curve_tool.setCurrentIndex(2)
        self.click(3.9);self.click(8.1)
        self.assertEqual(self.values(),[1,2,4,6])
        self.assertIsNone(self.f.candidate)
        self.assertFalse(self.f.window_review.isChecked())
        self.assertEqual(self.p.curve_tool.currentData(),'first_pass')
        self.assertTrue(all(pair[1].getRegion()==(4.,8.) for pair in self.link.regions))

    def test_native_drag_baseline_edge_snaps_and_does_not_change_plot_range(self):
        self.f.start.setValue(4)
        view=copy.deepcopy(self.f.curve_plot.getViewBox().viewRange())
        widget=self.f.curve_plot.viewport()
        events=[]
        for i,region in enumerate(self.link.regions[1]):
            region.sigRegionChanged.connect(lambda item,j=i:events.append(('move',j,item.getRegion())))
            region.sigRegionChangeFinished.connect(lambda item,j=i:events.append(('end',j,item.getRegion())))
        drag(widget,self.position(1.),self.position(2.05))
        self.assertEqual(self.values(),[1,3,4,7],repr(events))
        self.assertTrue(all(pair[0].getRegion()==(0.,2.) for pair in self.link.regions))
        np.testing.assert_allclose(self.f.curve_plot.getViewBox().viewRange(),view)

    def test_unchanged_selection_keeps_existing_review_and_candidate(self):
        self.f.window_review.setChecked(True);self.calculate()
        expected=document_hash(self.f.candidate)
        self.assertTrue(self.link.apply_span('first_pass',(2.,10.)))
        self.assertTrue(self.f.window_review.isChecked())
        self.assertEqual(document_hash(self.f.candidate),expected)

    def test_window_editing_pauses_playback(self):
        self.w.play_button.click()
        self.assertTrue(self.w._playing)
        self.f.curve_tool.setCurrentIndex(1)
        self.assertFalse(self.w._playing)
        self.w.play_button.click()
        self.assertTrue(self.w._playing)
        self.f.start.setValue(4)
        self.assertFalse(self.w._playing)

    def test_series_change_clears_current_window_review_but_keeps_retained_state(self):
        self.f.window_review.setChecked(True);self.calculate();self.f.capture_mr()
        self.w.select_choice('synthetic-perfusion-stress')
        until(lambda:self.w.current_pixels is not None)
        self.assertFalse(self.f.window_review.isChecked())
        self.assertIn('rest',self.f.mr_slots)

    def test_slow_seek_keeps_displayed_cursor_and_accepts_final_click(self):
        started,release=Event(),Event()
        self.addCleanup(release.set)
        original=self.w.backend.plane
        def delayed(grid,t,z,cached,progress,cancel):
            if t==3 and cached is None:
                started.set();release.wait(3)
            return original(grid,t,z,cached,progress,cancel)
        self.w.cache.clear();self.w.backend.plane=delayed
        self.click(4.)
        until(started.is_set)
        self.assertEqual(self.w.displayed_indices,(0,0))
        self.assertEqual(self.link.cursor.value(),0.)
        self.assertTrue(self.link.pending.isVisible())
        self.click(8.)
        release.set()
        until(lambda:self.w.displayed_indices==(5,0))
        self.assertEqual(self.link.cursor.value(),8.)
        self.assertFalse(self.link.pending.isVisible())

    def test_overlap_and_single_frame_selection_leave_fields_unchanged(self):
        original=self.values()
        self.assertFalse(self.link.apply_span('baseline',(0.,4.)))
        self.assertEqual(self.values(),original)
        self.assertFalse(self.link.apply_span('first_pass',(4.,4.)))
        self.assertEqual(self.values(),original)
        self.assertIn('未应用',self.f.curve_hint.text())

    def test_parameter_change_removes_current_retained_state_and_reference(self):
        self.calculate();self.f.capture_mr()
        self.w.select_choice('synthetic-perfusion-stress')
        until(lambda:self.w.current_pixels is not None)
        self.curves();self.f.condition.setCurrentIndex(2)
        self.calculate();self.f.capture_mr()
        self.f.mr_reference={'SYNTHETIC':'must be invalidated'}
        self.f.pair_review.setChecked(True);self.f.protocol_review.setChecked(True)
        self.f.window_review.setChecked(True)
        # Review edits invalidate only this active plane's retained state.
        self.assertEqual(set(self.f.mr_slots),{'rest'})
        self.assertIsNone(self.f.mr_reference)
        self.assertFalse(self.f.pair_review.isChecked())
        self.f.start.setValue(4)
        self.assertFalse(self.f.window_review.isChecked())

    def test_late_analysis_is_discarded_after_chart_selection(self):
        started,release=Event(),Event()
        self.addCleanup(release.set)
        def delayed(progress,cancel):
            started.set();release.wait(3);return {'SYNTHETIC':'must not publish'}
        self.f._submit('mr',delayed)
        until(started.is_set)
        self.assertTrue(self.link.apply_span('first_pass',(4.,10.)))
        release.set();until(lambda:not self.f.runner.busy)
        self.assertIsNone(self.f.candidate)

    def test_roi_edit_clears_both_curves_and_regions(self):
        self.p.remove_track()
        self.assertIsNone(self.p.result)
        self.assertTrue(all(not region.isVisible() for pair in self.link.regions for region in pair))
        self.assertFalse(self.f.curve_tool.isEnabled())
        self.assertFalse(self.link.cursor.isVisible())

    def test_selection_first_point_cannot_leak_to_different_series(self):
        self.f.curve_tool.setCurrentIndex(1);self.click(.1)
        self.assertIsNotNone(self.link._selection)
        self.assertTrue(all(anchor.isVisible() for anchor in self.link.anchors))
        self.w.select_choice('synthetic-perfusion-stress')
        until(lambda:self.w.current_pixels is not None)
        self.assertIsNone(self.link._selection)
        self.assertFalse(self.f.curve_tool.isEnabled())

    def test_roi_plot_uses_the_same_click_tool(self):
        self.w.workspace_tabs.setCurrentIndex(0)
        self.w.workspace_tabs.widget(0).ensureWidgetVisible(self.p.plot)
        APP.processEvents()
        self.click(7.9,self.p.plot)
        until(lambda:self.w.displayed_indices==(5,0))
        self.assertEqual(self.link.cursor.value(),8.)

    def test_non_mr_curves_seek_without_mr_analysis_regions(self):
        self.w.open_synthetic()
        until(lambda:self.w.current_pixels is not None)
        self.curves()
        self.assertTrue(all(not region.isVisible() for pair in self.link.regions for region in pair))
        self.assertFalse(self.f.curve_tool.model().item(1).isEnabled())
        self.assertFalse(self.f.mr_button.isEnabled())

    def test_native_invalid_region_drag_restores_the_saved_interval(self):
        before=self.values()
        drag(self.f.curve_plot.viewport(),self.position(1.),self.position(4.1))
        self.assertEqual(self.values(),before)
        self.assertEqual(self.link.regions[1][0].getRegion(),(0.,1.))

    def test_work_restore_rebuilds_both_windows_without_resetting_saved_review(self):
        self.assertTrue(self.link.apply_span('first_pass',(4.,10.)))
        self.f.window_review.setChecked(True);self.calculate()
        expected=document_hash(self.f.candidate)
        self.w.work.save();until(lambda:not self.w.work.busy)
        self.assertFalse(self.w.work.last_error)
        record=self.w.work.saved_output
        other=BrowserWindow(unsaved_prompt=discard_unsaved);self.windows.append(other);other.show()
        other.open_synthetic_perfusion();until(lambda:other.current_pixels is not None)
        other.work.restore(record);until(lambda:not other.work.busy)
        self.assertFalse(other.work.last_error)
        self.assertEqual(document_hash(other.feature_panel.candidate),expected)
        self.assertTrue(other.feature_panel.window_review.isChecked())
        self.assertTrue(all(pair[1].getRegion()==(4.,10.) for pair in other.curve_link.regions))

    def test_invalid_defaults_can_be_fixed_by_selecting_both_intervals(self):
        self.f.end.setValue(20)
        self.assertFalse(self.f.mr_button.isEnabled())
        self.assertTrue(self.link.apply_span('baseline',(0.,1.)))
        self.assertTrue(self.link.apply_span('first_pass',(2.,10.)))
        self.assertTrue(self.f.mr_button.isEnabled())


if __name__=='__main__':unittest.main()

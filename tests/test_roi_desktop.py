"""Native Qt ROI editing, persistence and stale-result boundaries."""

import copy
from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest

import numpy as np
from PySide6.QtCore import QPointF,Qt
from PySide6.QtTest import QTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from cmvd_demo.desktop.window import BrowserWindow
from cmvd_demo.roi import contour_mask,document_hash,set_track
from test_desktop import APP,until,dispose_window


class RoiDesktopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.w = BrowserWindow()
        self.w.roi_panel.output_root = Path(self.temp.name)
        self.w.show()
        self.w.open_synthetic()
        until(lambda:self.w.current_pixels is not None and not self.w.prefetch.busy)
        self.p = self.w.roi_panel

    def tearDown(self):
        dispose_window(self.w)

    def add_pair(self):
        self.p.role.setCurrentIndex(0)
        QTest.mouseClick(self.p.add_button,Qt.LeftButton)
        self.p.role.setCurrentIndex(1)
        QTest.mouseClick(self.p.add_button,Qt.LeftButton)
        self.assertEqual(len(self.p.document["tracks"]),2)

    def calculate(self):
        QTest.mouseClick(self.p.calculate_button,Qt.LeftButton)
        until(lambda:self.p.result is not None or bool(self.p.last_error))
        self.assertEqual(self.p.last_error,"")
        return self.p.result

    def test_reference_polygons_align_with_image_pixel_centres_and_lps(self):
        self.add_pair()
        track = self.p.document["tracks"][1]
        r,c = track["contour"]["outer_rc"][0]
        item = self.p.overlays["blood_pool"][0]
        point = item.mapToItem(self.w.image,item.getLocalHandlePositions()[0][1])
        np.testing.assert_allclose([point.y()-.5,point.x()-.5],[r,c])
        physical = self.w.image.mapToView(point)
        np.testing.assert_allclose([physical.x(),physical.y()],[c,r*2])

    def test_moving_roi_updates_reference_not_display_normalized_values(self):
        self.add_pair()
        before = np.asarray(self.p.document["tracks"][1]["contour"]["outer_rc"])
        item = self.p.overlays["blood_pool"][0]
        item.setPos(item.pos()+QPointF(1,2))
        after = np.asarray(self.p.document["tracks"][1]["contour"]["outer_rc"])
        np.testing.assert_allclose(after,before+[2,1])
        self.assertFalse(self.p.invalid_edit)
        result = self.calculate()
        mask = contour_mask(self.p.document["tracks"][1]["contour"],self.w.grid.shape[-2:])
        expected = [self.w.grid.read_plane(t,0)[mask].mean(dtype=np.float64) for t in range(12)]
        np.testing.assert_allclose(result["curves"][1]["mean"],expected)
        self.assertFalse(result["qc"]["feature_input_ready"])

    def test_frame_override_is_local_and_can_be_reset(self):
        self.add_pair()
        base = copy.deepcopy(self.p.document["tracks"][1]["contour"])
        self.w.time_slider.setValue(3)
        until(lambda:self.w.displayed_indices==(3,0))
        self.p.scope.setCurrentIndex(1)
        item = self.p.overlays["blood_pool"][0]
        item.setPos(item.pos()+QPointF(1,0))
        track = self.p.document["tracks"][1]
        self.assertEqual(track["contour"],base)
        self.assertEqual(track["overrides"][0]["frame_index"],3)
        QTest.mouseClick(self.p.reset_frame_button,Qt.LeftButton)
        self.assertEqual(self.p.document["tracks"][1]["overrides"],[])

    def test_reference_mode_cannot_silently_replace_independent_frame_contour(self):
        self.add_pair()
        self.w.time_slider.setValue(2)
        until(lambda:self.w.displayed_indices==(2,0))
        self.p.scope.setCurrentIndex(1)
        item = self.p.overlays["blood_pool"][0]
        item.setPos(item.pos()+QPointF(1,0))
        self.p.scope.setCurrentIndex(0)
        self.assertFalse(self.p.overlays["blood_pool"][0].isEnabled())
        self.p.scope.setCurrentIndex(1)
        self.assertTrue(self.p.overlays["blood_pool"][0].isEnabled())

    def test_snapshot_reload_reproduces_curve_and_rejects_other_source(self):
        self.add_pair()
        result = copy.deepcopy(self.calculate())
        original = document_hash(self.p.document)
        QTest.mouseClick(self.p.save_button,Qt.LeftButton)
        until(lambda:self.p.saved_output is not None)
        output = self.p.saved_output
        QTest.mouseClick(self.p.remove_button,Qt.LeftButton)
        self.assertIsNone(self.p.result)
        self.p.load_roi(output/"roi.json")
        until(lambda:not self.p.runner.busy)
        self.assertEqual(original,document_hash(self.p.document))
        self.assertEqual(result,self.calculate())
        # A different geometry/time binding must never replace the current ROI.
        import json
        data = json.loads((output/"roi.json").read_text())
        data["binding"]["source_signature"] = "f"*64
        bad = Path(self.temp.name)/"wrong-source.json"
        bad.write_text(json.dumps(data))
        self.p.load_roi(bad)
        until(lambda:bool(self.p.last_error))
        self.assertEqual(self.p.last_error,"roi_source_binding_mismatch")
        self.assertEqual(original,document_hash(self.p.document))
        self.assertIsNone(self.p.result)

    def test_layer_and_case_switch_hide_previous_overlay_and_curve(self):
        self.add_pair()
        self.calculate()
        self.w.slice_slider.setValue(1)
        until(lambda:self.w.displayed_indices==(0,1))
        self.assertEqual(self.p.overlays,{})
        self.assertIsNone(self.p.result)
        self.w.slice_slider.setValue(0)
        until(lambda:self.w.displayed_indices==(0,0))
        self.assertEqual(len(self.p.overlays),2)
        self.w.open_synthetic()
        self.assertEqual(self.p.documents,{} if self.p.document is None else {self.p.key:self.p.document})
        until(lambda:self.w.current_pixels is not None)
        self.assertEqual(self.p.document["tracks"],[])
        self.assertEqual(self.p.overlays,{})

    def test_invalid_drag_blocks_save_and_curve_until_repaired(self):
        self.add_pair()
        item = self.p.overlays["blood_pool"][0]
        item.setPos(item.pos()+QPointF(200,0))
        self.assertTrue(self.p.invalid_edit)
        self.assertFalse(self.p.save_button.isEnabled())
        self.assertFalse(self.p.calculate_button.isEnabled())
        item.setPos(QPointF(0,0))
        self.assertFalse(self.p.invalid_edit)
        self.assertTrue(self.p.calculate_button.isEnabled())

    def test_roi_edit_invalidates_curve_and_local_review_flags(self):
        self.add_pair()
        self.p.orientation_review.setChecked(True)
        self.p.motion_review.setChecked(True)
        self.calculate()
        item = self.p.overlays["blood_pool"][0]
        item.setPos(item.pos()+QPointF(1,0))
        self.assertIsNone(self.p.result)
        self.assertFalse(any(self.p.document["reviews"].values()))
        self.assertFalse(self.p.orientation_review.isChecked())

    def test_cancelled_curve_job_and_case_change_discard_late_result(self):
        self.add_pair()
        started,release = Event(),Event()
        self.addCleanup(release.set)
        original = self.w.backend.plane
        def slow(*args):
            started.set()
            release.wait(3)
            return original(*args)
        self.w.backend.plane = slow
        self.p.calculate()
        until(started.is_set)
        self.w.open_synthetic()
        self.assertIsNone(self.p.result)
        release.set()
        until(lambda:not self.p.runner.busy and self.w.current_pixels is not None)
        self.assertIsNone(self.p.result)
        self.assertEqual(self.p.document["tracks"],[])

    def test_editor_objects_reused_with_added_vertex_and_different_frame_topologies(self):
        self.add_pair()
        item = self.p.overlays["blood_pool"][0]
        segment = item.segments[0]
        endpoints = [handle["item"].pos() for handle in segment.handles]
        item.segmentClicked(segment,pos=(endpoints[0]+endpoints[1])*.5)
        item.stateChanged()
        self.assertFalse(self.p.invalid_edit)
        self.assertEqual(len(self.p.document["tracks"][1]["contour"]["outer_rc"]),9)
        contour = copy.deepcopy(self.p.document["tracks"][1]["contour"])
        contour["outer_rc"] = [contour["outer_rc"][i] for i in (0,3,5,7)]
        self.p._replace_document(set_track(self.p.document,"blood_pool",0,1,contour,frame_only=True))
        editor_id = id(self.p.overlays["blood_pool"][0])
        for i in range(24):
            frame = i%2
            self.w.time_slider.setValue(frame)
            until(lambda:self.w.displayed_indices==(frame,0))
            self.assertEqual(id(self.p.overlays["blood_pool"][0]),editor_id)
            self.assertEqual(len(self.p.overlays["blood_pool"][0].getHandles()),9 if frame==0 else 4)
        self.assertLessEqual(len(self.p._overlay_pool),3)


if __name__=="__main__":
    unittest.main()

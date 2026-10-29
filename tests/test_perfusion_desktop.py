from test_desktop import discard_unsaved
"""Native Qt D4 tests with entirely generated signal truth."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest
import numpy as np
from pydicom.uid import generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from cmvd_demo.desktop.data import Choice
from cmvd_demo.desktop.window import BrowserWindow
from cmvd_demo.roi import new_document, set_track
from test_desktop import APP, until, dispose_window
from test_perfusion import PerfusionGrid, reference
import test_dicom_reader as reader_fixtures
from cmvd_demo.perfusion import load_json


class PerfusionDesktopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.w = BrowserWindow(unsaved_prompt=discard_unsaved)
        self.f = self.w.feature_panel
        self.f.output_root = Path(self.temp.name)
        self.w.show()
        self.w._reset_case()
        choices = [Choice(c, "SYNTH-D4-"+c, "mr_perfusion", "unknown", 7, PerfusionGrid(c))
                   for c in ("rest", "stress")]
        self.w._populate(choices, {"indexed_objects": 14, "series_count": 2, "issues": []})
        until(lambda: self.w.current_pixels is not None)
        self.f.protocol.setText("synthetic-protocol")
        self.f.baseline.setValue(2)
        self.f.start.setValue(3)
        self.f.end.setValue(7)

    def tearDown(self):
        dispose_window(self.w)
        self.temp.cleanup()

    def curves(self, condition):
        self.w.select_choice(condition)
        until(lambda: self.w.current_pixels is not None and self.w.current_choice.key == condition)
        p = self.w.roi_panel
        doc = new_document(p.grid)
        doc["purpose"] = "engineering_fixture"
        for role, row in (("myocardium", 2), ("blood_pool", 20)):
            doc = set_track(doc, role, 0, 0, {"outer_rc": [[row, 3], [row, 8], [row+4, 8], [row+4, 3]], "inner_rc": None})
        p._replace_document(doc)
        p.on_frame()
        p.calculate()
        until(lambda: p.result is not None or bool(p.last_error))
        self.assertFalse(p.last_error)
        self.f.condition.setCurrentIndex(1 if condition == "rest" else 2)

    def capture(self, condition):
        self.curves(condition)
        self.f.calculate_mr()
        until(lambda: self.f.candidate is not None or bool(self.f.last_error))
        self.assertFalse(self.f.last_error)
        self.f.capture_mr()

    def test_default_phantom_cannot_be_mislabeled_mr(self):
        self.w.open_synthetic()
        until(lambda: self.w.current_pixels is not None)
        self.assertFalse(self.f.mr_button.isEnabled())
        self.assertFalse(self.f.capture_pet_button.isEnabled())

    def test_mr_pair_native_save_and_reference_comparison(self):
        self.capture("rest")
        self.capture("stress")
        self.f.pair_review.setChecked(True)
        self.f.protocol_review.setChecked(True)
        pair = self.f._mr_pair()
        self.assertEqual(pair["measurements"]["mr_mpri"]["value"], 2)
        self.assertIn("MR灌注储备指数（MPRI，半定量）: 2",self.f.pair_summary.text())
        ref = reference(pair)
        path = Path(self.temp.name)/"reference.json"
        path.write_text(json.dumps(ref), encoding="utf-8")
        self.f.load_reference(path)
        until(lambda: self.f.mr_reference is not None or bool(self.f.last_error))
        self.assertFalse(self.f.last_error)
        self.f.save()
        until(lambda: self.f.saved_output is not None or bool(self.f.last_error))
        self.assertFalse(self.f.last_error)
        result = json.loads((self.f.saved_output/"features.json").read_text())
        self.assertTrue(result["mr_reference_check"]["passed"])
        self.assertIn("pet_branch_missing", result["qc"]["warnings"])
        self.assertFalse(result["qc"]["model_input_ready"])

    def test_roi_edit_invalidates_retained_branch(self):
        self.capture("rest")
        self.assertIn("rest", self.f.mr_slots)
        doc = copy.deepcopy(self.w.roi_panel.document)
        doc["tracks"][0]["contour"]["outer_rc"][0][0] += .1
        self.w.roi_panel._replace_document(doc)
        self.assertNotIn("rest", self.f.mr_slots)
        self.assertIsNone(self.f.candidate)

    def test_invalid_roi_edit_and_undo_do_not_restore_retained_features(self):
        self.capture("rest")
        panel=self.w.roi_panel
        panel.role.setCurrentIndex(1)
        before=copy.deepcopy(panel.document["tracks"])
        panel.overlays["blood_pool"][0].setPos(200,0)
        self.assertTrue(panel.invalid_edit)
        self.assertNotIn("rest",self.f.mr_slots)
        panel.undo()
        self.assertEqual(panel.document["tracks"],before)
        self.assertFalse(panel.invalid_edit)
        self.assertNotIn("rest",self.f.mr_slots)
        self.assertIsNone(panel.result)

    def test_new_case_clears_all_retained_results(self):
        self.capture("rest")
        self.w.open_synthetic()
        until(lambda: self.w.current_pixels is not None)
        self.assertFalse(self.f.mr_slots)
        self.assertIsNone(self.f.pet)
        self.assertIsNone(self.f.mr_reference)

    def test_parameter_edit_invalidates_candidate(self):
        self.curves("rest")
        self.f.calculate_mr()
        until(lambda: self.f.candidate is not None)
        self.f.end.setValue(6)
        self.assertIsNone(self.f.candidate)
        self.assertFalse(self.f.capture_mr_button.isEnabled())

    def test_same_plane_refresh_does_not_cancel_valid_pending_feature_task(self):
        self.curves("rest")
        started, release = Event(), Event()
        def blocked(progress,cancel):
            started.set()
            release.wait(3)
            return Path(self.temp.name)
        self.f._submit("saved",blocked)
        until(started.is_set)
        self.f.context_changed()
        self.assertIsNotNone(self.f._job)
        release.set()
        until(lambda:self.f.saved_output is not None)

    def test_bad_analysis_window_has_safe_error(self):
        self.curves("rest")
        self.f.end.setValue(100)
        self.f.calculate_mr()
        until(lambda: bool(self.f.last_error))
        self.assertEqual(self.f.last_error, "feature_invalid_analysis_window")
        self.assertIsNone(self.f.candidate)

    def test_cancel_discards_late_worker_and_close_waits(self):
        started, release = Event(), Event()
        def blocked(progress, cancel):
            started.set()
            release.wait(3)
            return {"unused": True}
        self.f._submit("mr", blocked)
        until(started.is_set)
        self.f.cancel()
        self.w.close()
        release.set()
        until(lambda: not self.f.runner.busy)
        self.assertIsNone(self.f.candidate)

    def prepare_pet(self):
        writer = reader_fixtures.ReaderTests()
        writer.setUp()
        self.addCleanup(writer.doCleanups)
        for condition in ("rest", "stress"):
            uid = generate_uid()
            for t in range(3):
                pixels = np.full((32,40), 10+5*t, dtype=np.uint16)
                writer.write(condition+str(t)+".dcm", series=uid, t=t, z=0,
                             SeriesDescription=condition+" dynamic", NumberOfSlices=1,
                             Rows=32, Columns=40, PixelData=pixels.tobytes())
        self.w.open_case(writer.root)
        until(lambda:self.w.current_pixels is not None or bool(self.w.last_error))
        self.assertFalse(self.w.last_error)
        for condition, index in (("rest",1),("stress",2)):
            choice = next(c for c in self.w.choices if c.condition==condition)
            self.w.select_choice(choice.key)
            until(lambda:self.w.current_pixels is not None)
            p = self.w.roi_panel
            doc = new_document(p.grid)
            doc["purpose"] = "engineering_fixture"
            for role,row in (("myocardium",2),("blood_pool",20)):
                doc = set_track(doc,role,0,0,{"outer_rc":[[row,3],[row,8],[row+4,8],[row+4,3]],"inner_rc":None})
            p._replace_document(doc)
            p.on_frame()
            p.calculate()
            until(lambda:p.result is not None or bool(p.last_error))
            self.assertFalse(p.last_error)
            self.f.condition.setCurrentIndex(index)
            self.f.capture_pet()
        self.f.export_request()
        until(lambda:self.f.saved_output is not None or bool(self.f.last_error))
        self.assertFalse(self.f.last_error)
        request = load_json(self.f.saved_output/"pet_request.json")
        for key in ("method_id","software_version","protocol_id","tracer","reference_id","reviewer_code"):
            request[key] = "SYNTHETIC-TEST"
        request["branches"]["rest"]["mbf"] = .8
        request["branches"]["stress"]["mbf"] = 2.4
        request["reported_mfr"] = 3
        request["qc_passed"] = True
        path = Path(self.temp.name)/"synthetic-workstation.json"
        path.write_text(json.dumps(request),encoding="utf-8")
        return path

    def test_pet_native_template_import_export_and_roi_invalidation(self):
        path = self.prepare_pet()
        self.f.load_pet(path)
        until(lambda:self.f.pet is not None or bool(self.f.last_error))
        self.assertFalse(self.f.last_error)
        self.assertAlmostEqual(self.f.pet["measurements"]["pet_mfr"]["value"],3)
        self.f.save()
        until(lambda:self.f.saved_output is not None or bool(self.f.last_error))
        self.assertFalse(self.f.last_error)
        result = load_json(self.f.saved_output/"features.json")
        self.assertIn("mr_branch_missing",result["qc"]["warnings"])
        self.w.roi_panel.remove_track()
        self.assertIsNone(self.f.pet)

    def test_pet_changed_source_rejected_before_workstation_import(self):
        path = self.prepare_pet()
        header = self.f.pet_slots["rest"]["grid"].cells[0][0]
        with header.path.open("ab") as f:
            f.write(b"synthetic-source-change")
        self.f.load_pet(path)
        until(lambda:bool(self.f.last_error))
        self.assertEqual(self.f.last_error,"source_changed_since_index")
        self.assertIsNone(self.f.pet)


if __name__ == "__main__":
    unittest.main()

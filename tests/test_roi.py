"""Known-value ROI masks/curves and controlled snapshot boundaries."""

import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from cmvd_demo.roi import RoiError,binding_for,new_document,validate_document,contour_mask,set_track,extract_curves,save_snapshot,load_document,document_hash
from cmvd_demo.io import ReadCancelled


class Grid:
    synthetic_id = "ROI_TEST_GRID"
    shape = (4,2,16,20)
    kind = "mr_perfusion"
    orientation_lps = (0,1,0,-1,0,0)
    pixel_spacing_mm = (2.,3.)
    timing_basis = "relative_acquisition"
    frame_durations_s = None
    def __init__(self):
        self.times_s = np.array([[0,.7,2.1,4.8],[.05,1.3,2.4,5.5]])
    def pixel_to_lps(self,z,row,column):
        return np.array([10-2*row,20+3*column,30+10*z],dtype=float)
    def read_plane(self,t,z):
        rows,cols = np.mgrid[:16,:20]
        return (100*t+1000*z+rows*10+cols).astype(np.float32)


def rectangle(r0,c0,r1,c1):
    return {"outer_rc":[[r0,c0],[r0,c1],[r1,c1],[r1,c0]],"inner_rc":None}


class RoiTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid()

    def doc(self,role="myocardium",z=0,contour=None):
        return set_track(new_document(self.grid),role,z,0,contour or rectangle(3,4,6,8))

    def test_mask_uses_row_column_pixel_centres_including_outer_boundary(self):
        mask = contour_mask(rectangle(3,4,6,8),(16,20))
        self.assertEqual(int(mask.sum()),20)
        self.assertTrue(mask[3,4] and mask[6,8])
        self.assertFalse(mask[4,3] or mask[7,8])

    def test_annular_exclusion_and_hole_boundary(self):
        contour = rectangle(2,2,10,10)
        contour["inner_rc"] = rectangle(4,4,8,8)["outer_rc"]
        mask = contour_mask(contour,(16,20))
        self.assertEqual(int(mask.sum()),56)
        self.assertFalse(mask[4,4] or mask[6,6])

    def test_self_intersection_outside_and_nonfinite_are_rejected(self):
        bad = [rectangle(-1,2,4,5),rectangle(1,1,2,2)]
        bad[1]["outer_rc"] = [[1,1],[7,7],[1,7],[5,1]]
        bad.append(rectangle(1,float("nan"),5,8))
        for contour in bad:
            with self.subTest(contour=repr(contour)):
                with self.assertRaises(RoiError):
                    contour_mask(contour,(16,20))

    def test_duplicate_vertex_empty_mask_and_zero_area_are_rejected(self):
        for vertices in ([[1,1],[1,1],[3,2]],[[.1,.1],[.1,.4],[.4,.4]],[[1,1],[2,2],[3,3]]):
            with self.assertRaises(RoiError):
                contour_mask({"outer_rc":vertices,"inner_rc":None},(16,20))

    def test_hole_must_be_strictly_inside_outer(self):
        for inner in (rectangle(0,0,3,3),rectangle(2,2,4,4)):
            contour = rectangle(2,2,10,10)
            contour["inner_rc"] = inner["outer_rc"]
            with self.assertRaises(RoiError):
                contour_mask(contour,(16,20))

    def test_known_mean_float64_std_area_units_and_nonuniform_times(self):
        result = extract_curves(self.grid,self.doc(z=1),1)
        curve = result["curves"][0]
        expected = np.array([self.grid.read_plane(t,1)[3:7,4:9].mean(dtype=np.float64) for t in range(4)])
        np.testing.assert_array_equal(curve["mean"],expected)
        np.testing.assert_allclose(curve["std"],[self.grid.read_plane(0,1)[3:7,4:9].std(dtype=np.float64)]*4)
        self.assertEqual(curve["pixel_count"],[20]*4)
        self.assertEqual(curve["area_mm2"],[120.]*4)
        self.assertEqual(curve["units"],"SYNTHETIC_AU")
        np.testing.assert_array_equal(curve["time_s"],self.grid.times_s[1])
        self.assertFalse(result["qc"]["feature_input_ready"])
        self.assertFalse(result["qc"]["clinical_validated"])

    def test_mean_accumulates_in_float64(self):
        doc = self.doc(contour=rectangle(3.5,3.5,4.5,6.5))
        pixels = np.zeros((16,20),dtype=np.float32)
        pixels[4,4:7] = [1e8,1,-1e8]
        result = extract_curves(self.grid,doc,0,read_plane=lambda t,z:pixels)
        np.testing.assert_allclose(result["curves"][0]["mean"],[1/3]*4)

    def test_explicit_frame_motion_correction_reproduces_moving_signal(self):
        doc = self.doc(contour=rectangle(3.5,2.5,5.5,4.5))
        def moving(t,z):
            pixels = np.zeros((16,20),dtype=np.float32)
            pixels[4:6,3+t:5+t] = 10
            return pixels
        fixed = extract_curves(self.grid,doc,0,read_plane=moving)
        self.assertLess(fixed["curves"][0]["mean"][-1],10)
        for t in range(1,4):
            doc = set_track(doc,"myocardium",0,t,rectangle(3.5,2.5+t,5.5,4.5+t),frame_only=True)
        corrected = extract_curves(self.grid,doc,0,read_plane=moving)
        self.assertEqual(corrected["curves"][0]["mean"],[10.]*4)
        self.assertFalse(corrected["qc"]["temporal_interpolation"])
        self.assertFalse(corrected["qc"]["image_resampling"])

    def test_frame_override_does_not_interpolate_other_frames(self):
        doc = set_track(self.doc(),"myocardium",0,2,rectangle(8,9,10,11),frame_only=True)
        curve = extract_curves(self.grid,doc,0)["curves"][0]
        self.assertEqual(curve["pixel_count"],[20,20,9,20])

    def test_geometry_edit_resets_reviews(self):
        doc = self.doc()
        doc["reviews"] = {key:True for key in doc["reviews"]}
        updated = set_track(doc,"myocardium",0,1,rectangle(2,3,6,7))
        self.assertFalse(any(updated["reviews"].values()))
        self.assertTrue(all(doc["reviews"].values()))

    def test_pair_and_manual_review_required_for_feature_qc(self):
        doc = self.doc(contour=rectangle(3,4,6,8))
        doc = set_track(doc,"blood_pool",0,0,rectangle(9,12,11,14))
        doc["reviews"] = {key:True for key in doc["reviews"]}
        result = extract_curves(self.grid,doc,0)
        self.assertTrue(result["qc"]["feature_input_ready"])
        self.assertFalse(result["qc"]["clinical_validated"])

    def test_overlapping_myocardium_and_blood_are_rejected(self):
        doc = set_track(self.doc(),"blood_pool",0,0,rectangle(4,5,5,6))
        with self.assertRaisesRegex(RoiError,"roi_myocardium_blood_overlap"):
            extract_curves(self.grid,doc,0)

    def test_unsupported_time_axis_or_static_does_not_invent_timing(self):
        self.grid.times_s = None
        with self.assertRaisesRegex(RoiError,"roi_time_series_required"):
            extract_curves(self.grid,self.doc(),0)

    def test_nonmonotonic_times_or_invalid_durations_are_rejected(self):
        self.grid.times_s[0] = [0,.7,.6,4.8]
        with self.assertRaisesRegex(RoiError,"roi_invalid_time_axis"):
            extract_curves(self.grid,self.doc(),0)
        self.grid = Grid()
        self.grid.frame_durations_s = np.array([[1,1,-1,1],[1,1,1,1]])
        with self.assertRaisesRegex(RoiError,"roi_invalid_frame_duration"):
            extract_curves(self.grid,self.doc(),0)

    def test_empty_branch_or_invalid_pixels_fail_without_partial_curve(self):
        with self.assertRaisesRegex(RoiError,"roi_no_tracks_on_plane"):
            extract_curves(self.grid,new_document(self.grid),0)
        with self.assertRaisesRegex(RoiError,"roi_invalid_pixels"):
            extract_curves(self.grid,self.doc(),0,read_plane=lambda t,z:np.full((16,20),np.nan))

    def test_cancelled_curve_job_returns_no_partial_success(self):
        done = []
        with self.assertRaises(ReadCancelled):
            extract_curves(self.grid,self.doc(),0,progress=lambda *args:done.append(1),cancel=lambda:bool(done))
        self.assertEqual(len(done),1)

    def test_snapshot_reload_and_csv_reproduce_values_without_overwrite(self):
        doc = self.doc()
        result = extract_curves(self.grid,doc,0)
        with tempfile.TemporaryDirectory() as directory:
            first = save_snapshot(directory,doc,result)
            before = (first/"roi.json").read_bytes()
            loaded = load_document(first/"roi.json",binding_for(self.grid))
            self.assertEqual(document_hash(doc),document_hash(loaded))
            self.assertEqual(result,extract_curves(self.grid,loaded,0))
            second = save_snapshot(directory,loaded,result)
            self.assertNotEqual(first,second)
            self.assertEqual(before,(first/"roi.json").read_bytes())
            with (first/"curves.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([float(r["mean"]) for r in rows],result["curves"][0]["mean"])
            self.assertTrue(json.loads((first/"bundle.json").read_text())["completed"])

    def test_different_image_time_geometry_or_source_rejects_loaded_roi(self):
        doc = self.doc()
        with tempfile.TemporaryDirectory() as directory:
            output = save_snapshot(directory,doc)
            for change in ("source_signature","pixel_spacing_mm","times_s_by_slice"):
                binding = copy.deepcopy(doc["binding"])
                binding[change] = "different"
                with self.assertRaisesRegex(RoiError,"roi_source_binding_mismatch"):
                    load_document(output/"roi.json",binding)

    def test_duplicate_tracks_and_invalid_frame_indices_are_rejected(self):
        doc = self.doc()
        doc["tracks"].append(copy.deepcopy(doc["tracks"][0]))
        with self.assertRaisesRegex(RoiError,"roi_duplicate_track"):
            validate_document(doc)
        with self.assertRaises(RoiError):
            set_track(self.doc(),"myocardium",0,100,rectangle(3,4,6,8),frame_only=True)

    def test_stale_curve_snapshot_is_rejected(self):
        doc = self.doc()
        result = extract_curves(self.grid,doc,0)
        doc["reviews"]["motion_and_contours"] = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RoiError,"roi_stale_curve_result"):
                save_snapshot(directory,doc,result)
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_cine_trigger_axis_is_preserved_and_not_perfusion_ready(self):
        self.grid.kind = "cine_stack"
        self.grid.timing_basis = "cardiac_trigger_per_slice"
        curve = extract_curves(self.grid,self.doc(z=1),1)
        self.assertIn("not_a_dynamic_perfusion_branch",curve["qc"]["warnings"])
        self.assertEqual(curve["curves"][0]["time_s"],self.grid.times_s[1].tolist())

    def test_engineering_fixture_cannot_become_reviewed_feature_input(self):
        doc = set_track(self.doc(),"blood_pool",0,0,rectangle(9,12,11,14))
        doc["purpose"] = "engineering_fixture"
        doc["reviews"] = {key:True for key in doc["reviews"]}
        result = extract_curves(self.grid,doc,0)
        self.assertFalse(result["qc"]["feature_input_ready"])
        self.assertIn("engineering_fixture_not_anatomical_annotation",result["qc"]["warnings"])

    def test_native_pet_binding_keeps_units_durations_and_hides_raw_identity(self):
        import test_dicom_reader as fixtures
        from cmvd_demo.io import build_grid
        fixture = fixtures.ReaderTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        grid = build_grid(fixture.pet())
        doc = set_track(new_document(grid),"myocardium",2,0,rectangle(0,0,1,1))
        result = extract_curves(grid,doc,2)
        curve = result["curves"][0]
        self.assertEqual(curve["units"],"BQML")
        self.assertEqual(curve["frame_duration_s"],[10.,20.,20.])
        np.testing.assert_array_equal(curve["mean"],[45,245,445])
        text = json.dumps(doc)+json.dumps(result)
        for private in (fixture.study,fixture.reference,str(fixture.root),"SYNTHETIC^ONLY","SYNTH-READER"):
            self.assertNotIn(private,text)

    def test_failed_export_leaves_no_completed_marker(self):
        doc = self.doc()
        result = extract_curves(self.grid,doc,0)
        original = Path.open
        def fail_csv(path,*args,**kwargs):
            if path.name=="curves.csv":
                raise OSError("synthetic_export_failure")
            return original(path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path,"open",fail_csv):
                with self.assertRaises(OSError):
                    save_snapshot(directory,doc,result)
            runs = list(Path(directory).iterdir())
            self.assertEqual(len(runs),1)
            self.assertTrue((runs[0]/"roi.json").exists())
            self.assertFalse((runs[0]/"bundle.json").exists())

    def test_malformed_oversized_or_extra_field_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"roi.json"
            path.write_text("{invalid}")
            with self.assertRaises(RoiError):
                load_document(path,binding_for(self.grid))
            doc = self.doc()
            doc["unexpected_field"] = "ignored instructions are not schema data"
            path.write_text(json.dumps(doc))
            with self.assertRaises(RoiError):
                load_document(path,binding_for(self.grid))
            path.write_bytes(b" "*(4*1024**2+1))
            with self.assertRaisesRegex(RoiError,"roi_file_too_large"):
                load_document(path,binding_for(self.grid))


if __name__=="__main__":
    unittest.main()

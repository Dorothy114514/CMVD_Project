"""D4 known-value nonuniform curves, provenance rejection and snapshot failures."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from cmvd_demo.desktop.data import PhantomGrid
from cmvd_demo.roi import new_document, set_track, extract_curves, document_hash
from cmvd_demo.io import ReadCancelled
from cmvd_demo.perfusion import (FeatureError, mr_features, pair_mr, pet_request, import_pet,
                                compare_reference, build_bundle, save_features, load_json, save_request)

CASE = "a"*64


class PerfusionGrid(PhantomGrid):
    shape = (7, 1, 32, 40)
    kind = "mr_perfusion"
    def __init__(self, condition="rest"):
        self.synthetic_id = "D4_SYNTH_"+condition
        self.times_s = np.array([[0., 1., 2., 4., 7., 8., 10.]])
        self.condition = condition
    def read_plane(self, t, z):
        myo = 10+np.array([0, 0, 2, 10, 16, 12, 6])[t]*(2 if self.condition == "stress" else 1)
        blood = 20+np.array([0, 0, 8, 24, 28, 20, 8])[t]
        data = np.empty((32, 40), dtype=np.float32)
        data[:16], data[16:] = myo, blood
        return data


def fixture(condition="rest", kind="mr_perfusion"):
    grid = PerfusionGrid(condition)
    grid.kind = kind
    if kind == "pet_dynamic":
        grid.frame_durations_s = np.ones((1, 7))
    doc = new_document(grid)
    for role, row in (("myocardium", 2), ("blood_pool", 20)):
        doc = set_track(doc, role, 0, 0, {"outer_rc": [[row, 3], [row, 8], [row+4, 8], [row+4, 3]], "inner_rc": None})
    doc["purpose"] = "engineering_fixture"
    result = extract_curves(grid, doc, 0)
    # Explicit handcrafted unit fixtures, never presented as real patient data.
    units = "BQML" if kind == "pet_dynamic" else "MR_SIGNAL_AU"
    doc["binding"]["units"] = [units]
    for curve in result["curves"]:
        curve["units"] = units
    result["document_sha256"] = document_hash(doc)
    return doc, result


def mr(condition="rest", **changes):
    doc, result = fixture(condition)
    params = dict(baseline_frames=2, window_start=2, window_end=6, window_reviewed=True)
    params.update(changes)
    return mr_features(doc, result, CASE, condition, "protocol-v1", **params)


def pair():
    return pair_mr(mr(), mr("stress"), pairing_reviewed=True, protocol_comparable=True)


def tacs():
    return {c: fixture(c, "pet_dynamic") for c in ("rest", "stress")}


def workstation(local=None):
    result = pet_request(CASE, local or tacs())
    for key in ("method_id", "software_version", "protocol_id", "tracer", "reference_id", "reviewer_code"):
        result[key] = "SYNTH-TEST"
    result["branches"]["rest"]["mbf"] = 0.8
    result["branches"]["stress"]["mbf"] = 2.4
    result["qc_passed"] = True
    result["reported_mfr"] = 3.
    return result


def reference(feature):
    return {"schema_version": 1, "feature_sha256": document_hash(feature), "reference_id": "SYNTH-REF",
            "reviewer_code": "TEST", "reviewed": True, "method": feature["method"],
            "measurements": {k: {"value": v["value"], "unit": v["unit"], "atol": 1e-10, "rtol": 0.}
                             for k, v in feature["measurements"].items()}}


class PerfusionTests(unittest.TestCase):
    def test_known_nonuniform_peak_ttp_upslope_auc_and_units(self):
        values = mr()["measurements"]
        expected = {"myocardium.baseline": 10, "myocardium.peak": 26, "myocardium.peak_enhancement": 16,
                    "myocardium.peak_time": 7, "myocardium.ttp_from_window_origin": 6,
                    "myocardium.max_upslope": 4, "blood_pool.max_upslope": 8,
                    "myocardium.signed_enhancement_auc": 84, "blood_pool.signed_enhancement_auc": 166,
                    "normalized_upslope": .5}
        for key, value in expected.items():
            self.assertAlmostEqual(values[key]["value"], value)
        self.assertEqual(values["normalized_upslope"]["unit"], "1")
        self.assertEqual(values["myocardium.max_upslope"]["unit"], "MR_SIGNAL_AU/s")

    def test_pair_ratio_two_and_full_reproducible_input(self):
        result = pair()
        self.assertAlmostEqual(result["measurements"]["mr_mpri"]["value"], 2)
        restored = json.loads(json.dumps(result))
        r = restored["rest"]
        computed = mr_features(r["roi_document"], r["input_curves"], CASE, "rest", "protocol-v1",
                               baseline_frames=2, window_start=2, window_end=6, window_reviewed=True)
        self.assertEqual(r, computed)

    def test_explicit_windows_required(self):
        for args in (dict(baseline_frames=1), dict(window_start=0), dict(window_end=7),
                     dict(window_start=6), dict(baseline_frames=True), dict(window_end=2)):
            with self.subTest(args=args), self.assertRaises(FeatureError):
                mr(**args)

    def test_baseline_interval_can_explicitly_skip_initial_transient(self):
        doc, curves = fixture()
        for c in curves["curves"]:
            c["mean"][0] = 1000
        with self.assertRaisesRegex(FeatureError,"feature_no_positive_enhancement"):
            mr_features(doc,curves,CASE,"rest","P",baseline_frames=2,window_start=3,window_end=6)
        result = mr_features(doc,curves,CASE,"rest","P",baseline_start=1,baseline_frames=2,
                             window_start=3,window_end=6)
        self.assertEqual(result["parameters"]["baseline_start"],1)
        self.assertEqual(result["measurements"]["myocardium.baseline"]["value"],11)

    def test_cine_synthetic_and_pet_are_not_mr_perfusion(self):
        doc, curves = fixture()
        for kind in ("cine_stack", "synthetic", "pet_dynamic"):
            doc["binding"]["kind"] = kind
            curves["document_sha256"] = document_hash(doc)
            with self.assertRaisesRegex(FeatureError, "feature_wrong_branch"):
                mr_features(doc, curves, CASE, "rest", "P", baseline_frames=2, window_start=2, window_end=6)

    def test_stale_curve_and_missing_roi_rejected(self):
        for change in ("stale", "missing"):
            doc, curves = fixture()
            if change == "stale":
                curves["document_sha256"] = "0"*64
            else:
                curves["curves"].pop()
            with self.assertRaises(FeatureError):
                mr_features(doc, curves, CASE, "rest", "P", baseline_frames=2, window_start=2, window_end=6)

    def test_nonfinite_duplicate_time_wrong_units_and_nonpositive_signal_rejected(self):
        for change in ("nan", "time", "unit", "baseline", "flat", "denominator"):
            doc, curves = fixture()
            if change == "nan":
                curves["curves"][0]["mean"][2] = float("nan")
            elif change == "time":
                curves["curves"][0]["time_s"][2] = 1
            elif change == "unit":
                doc["binding"]["units"] = ["BQML"]
            elif change == "baseline":
                curves["curves"][0]["mean"][:2] = [0, 0]
            elif change == "flat":
                curves["curves"][0]["mean"] = [10]*7
            else:
                curves["curves"][1]["mean"] = [20]*7
            curves["document_sha256"] = document_hash(doc)
            with self.subTest(change=change), self.assertRaises(FeatureError):
                mr_features(doc, curves, CASE, "rest", "P", baseline_frames=2, window_start=2, window_end=6)

    def test_condition_and_anonymous_protocol_required(self):
        doc, curves = fixture()
        for condition, protocol in (("unknown", "P"), ("rest", ""), ("rest", "C:/patient")):
            with self.assertRaises(FeatureError):
                mr_features(doc, curves, CASE, condition, protocol, baseline_frames=2, window_start=2, window_end=6)

    def test_peak_at_window_end_is_explicit_warning(self):
        result = mr(window_end=4)
        self.assertIn("myocardium_peak_at_window_end", result["qc"]["warnings"])

    def test_fixture_and_missing_reviews_never_become_ready(self):
        result = mr(window_reviewed=False)
        self.assertFalse(result["qc"]["local_qc_passed"])
        self.assertFalse(result["qc"]["clinical_validated"])
        self.assertIn("baseline_and_first_pass_window_not_reviewed", result["qc"]["warnings"])
        self.assertFalse(build_bundle(CASE, mr=pair())["qc"]["perfusion_input_ready"])

    def test_pair_rejects_same_source_patient_method_units_and_orientation(self):
        for change in ("same", "patient", "method", "unit", "orientation", "condition", "denominator"):
            r, s = mr(), mr("stress")
            if change == "same":
                s["source_binding"] = copy.deepcopy(r["source_binding"])
            elif change == "patient":
                s["case_token"] = "b"*64
            elif change == "method":
                s["protocol_id"] = "another"
            elif change == "unit":
                s["source_binding"]["units"] = ["BQML"]
            elif change == "orientation":
                s["source_binding"]["orientation_lps"] = [0, 1, 0, -1, 0, 0]
            elif change == "condition":
                s["condition"] = "rest"
            else:
                r["measurements"]["normalized_upslope"]["value"] = 0
            with self.subTest(change=change), self.assertRaises(FeatureError):
                pair_mr(r, s)

    def test_pair_manual_protocol_and_plane_review_remain_explicit(self):
        p = pair_mr(mr(), mr("stress"))
        self.assertIn("same_anatomical_plane_and_conditions_not_reviewed", p["qc"]["warnings"])
        self.assertIn("dose_sequence_and_protocol_comparability_not_reviewed", p["qc"]["warnings"])

    def test_pet_template_is_empty_and_import_uses_workstation_not_tac_amplitude(self):
        local = tacs()
        self.assertIsNone(pet_request(CASE, local)["branches"]["rest"]["mbf"])
        result = import_pet(workstation(local), CASE, local)
        self.assertAlmostEqual(result["measurements"]["pet_mfr"]["value"], 3)
        self.assertEqual(result["measurements"]["rest_mbf"]["unit"], "mL/min/g")
        self.assertFalse(result["qc"]["independent_mbf_validation"])
        self.assertFalse(result["qc"]["local_qc_passed"])

    def test_pet_wrong_unit_patient_binding_qc_number_and_reported_ratio_rejected(self):
        local = tacs()
        changes = (lambda p: p.update(unit="BQML"), lambda p: p.update(case_token="b"*64),
                   lambda p: p.update(qc_passed=False), lambda p: p.update(qc_passed=1),
                   lambda p: p.update(reported_mfr=1), lambda p: p.update(method_id="REPLACE"),
                   lambda p: p.update(mfr_absolute_tolerance=-1),
                   lambda p: p["branches"]["rest"].update(mbf=0),
                   lambda p: p["branches"]["rest"].update(mbf=True),
                   lambda p: p["branches"]["rest"]["binding"].update(slice_index=1),
                   lambda p: p.update(extra="forbidden"))
        for index, change in enumerate(changes):
            payload = workstation(local)
            change(payload)
            with self.subTest(index=index), self.assertRaises(FeatureError):
                import_pet(payload, CASE, local)

    def test_pet_frame_duration_and_same_source_required(self):
        local = tacs()
        local["rest"][0]["binding"]["frame_durations_s_by_slice"] = None
        local["rest"][1]["document_sha256"] = document_hash(local["rest"][0])
        with self.assertRaises(FeatureError):
            pet_request(CASE, local)
        local = tacs()
        local["stress"] = local["rest"]
        with self.assertRaisesRegex(FeatureError, "feature_same_source_pair"):
            pet_request(CASE, local)

    def test_reference_known_ratio_and_prespecified_tolerances(self):
        p = pair()
        ref = reference(p)
        ref["measurements"]["mr_mpri"]["value"] = 2.001
        ref["measurements"]["mr_mpri"]["atol"] = .002
        self.assertTrue(compare_reference(p, ref)["passed"])
        ref["measurements"]["mr_mpri"]["atol"] = .0001
        self.assertFalse(compare_reference(p, ref)["passed"])

    def test_reference_wrong_unit_source_review_fields_or_tolerance_rejected(self):
        p = pair()
        changes = (lambda r: r.update(feature_sha256="0"*64), lambda r: r.update(reviewed=False),
                   lambda r: r["measurements"]["mr_mpri"].update(unit="s^-1"),
                   lambda r: r["measurements"]["mr_mpri"].update(atol=-1),
                   lambda r: r["measurements"].pop("mr_mpri"), lambda r: r.update(method="other"))
        for change in changes:
            ref = reference(p)
            change(ref)
            with self.assertRaises(FeatureError):
                compare_reference(p, ref)

    def test_bundle_missing_reference_and_cross_case_gates(self):
        p = pair()
        pet = import_pet(workstation(), CASE, tacs())
        bundle = build_bundle(CASE, mr=p, pet=pet, mr_reference=reference(p))
        self.assertFalse(bundle["qc"]["model_input_ready"])
        self.assertFalse(bundle["qc"]["perfusion_input_ready"])
        self.assertFalse(bundle["qc"]["clinical_validated"])
        with self.assertRaises(FeatureError):
            build_bundle("b"*64, mr=p)

    def test_snapshot_reloads_canonically_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = build_bundle(CASE, mr=pair())
            one, two = save_features(root, bundle), save_features(root, bundle)
            self.assertNotEqual(one, two)
            self.assertEqual(load_json(one/"features.json"), bundle)
            self.assertEqual(load_json(one/"bundle.json")["features_sha256"], document_hash(bundle))
            self.assertTrue((one/"features.csv").exists())

    def test_snapshot_failure_and_cancellation_leave_no_completion(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("cmvd_demo.perfusion.storage.csv.DictWriter", side_effect=OSError("fixture_failure")):
                with self.assertRaises(OSError):
                    save_features(root, build_bundle(CASE, mr=pair()))
            run = next(Path(root).iterdir())
            self.assertFalse((run/"bundle.json").exists())
            with self.assertRaises(ReadCancelled):
                save_features(root, build_bundle(CASE, mr=pair()), cancel=lambda: True)
            self.assertEqual(len(list(Path(root).iterdir())), 1)

    def test_json_duplicate_nonfinite_and_size_limit_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"test.json"
            for raw in ('{"a":1,"a":2}', '{"a":NaN}', 'broken'):
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(FeatureError):
                    load_json(path)
            with patch("cmvd_demo.perfusion.storage.MAX_BYTES", 2):
                with self.assertRaises(FeatureError):
                    load_json(path)


if __name__ == "__main__":
    unittest.main()

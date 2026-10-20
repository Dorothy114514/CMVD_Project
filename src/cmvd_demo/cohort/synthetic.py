"""Quarantined all-synthetic D5 forms and known-value D4 engineering snapshots."""

import numpy as np

from cmvd_demo.roi import new_document, set_track, extract_curves, document_hash
from cmvd_demo.perfusion import mr_features, pair_mr, pet_request, import_pet, build_bundle, save_features
from cmvd_demo.perfusion.storage import new_run, write_json, load_json
from .inputs import SCHEMAS, check_cancel
from .storage import write_csv


class _KnownGrid:
    shape = (7, 1, 32, 40)
    orientation_lps = (1, 0, 0, 0, 1, 0)
    pixel_spacing_mm = (2., 1.)
    timing_basis = "synthetic_relative_acquisition"
    photometric = "MONOCHROME2"

    def __init__(self, rid, condition, kind):
        self.synthetic_id = f"D5-{rid}-{condition}-{kind}"
        self.condition, self.kind = condition, kind
        self.times_s = np.array([[0., 1., 2., 4., 7., 8., 10.]])
        self.frame_durations_s = np.ones((1, 7)) if kind == "pet_dynamic" else None

    def pixel_to_lps(self, z, row, column):
        return np.array([column, 2*row, 10*z], dtype=float)

    def read_plane(self, t, z):
        myo = 10+np.array([0, 0, 2, 10, 16, 12, 6])[t]*(2 if self.condition == "stress" else 1)
        blood = 20+np.array([0, 0, 8, 24, 28, 20, 8])[t]
        pixels = np.empty((32, 40), dtype=np.float32)
        pixels[:16], pixels[16:] = myo, blood
        return pixels


def _curves(rid, condition, kind):
    grid = _KnownGrid(rid, condition, kind)
    doc = new_document(grid)
    for role, r in (("myocardium", 2), ("blood_pool", 20)):
        doc = set_track(doc, role, 0, 0, {"outer_rc": [[r, 3], [r, 8], [r+4, 8], [r+4, 3]], "inner_rc": None})
    doc["purpose"] = "engineering_fixture"
    doc["reviews"] = {key: True for key in doc["reviews"]}  # Known fixture geometry only.
    result = extract_curves(grid, doc, 0)
    # Handcrafted unit fixtures, not calibrated measurements from real images.
    unit = "BQML" if kind == "pet_dynamic" else "MR_SIGNAL_AU"
    doc["binding"]["units"] = [unit]
    for curve in result["curves"]:
        curve["units"] = unit
    result["document_sha256"] = document_hash(doc)
    result["qc"]["warnings"] = ["engineering_fixture_not_anatomical_annotation"]
    result["qc"]["feature_input_ready"] = False
    return doc, result


def synthetic_bundle(rid, case_token):
    branches = []
    for condition in ("rest", "stress"):
        doc, curves = _curves(rid, condition, "mr_perfusion")
        branches.append(mr_features(doc, curves, case_token, condition, "SYNTH-PROTOCOL", baseline_frames=2,
                                    window_start=2, window_end=6, window_reviewed=True))
    mr = pair_mr(*branches, pairing_reviewed=True, protocol_comparable=True)
    ref = {"schema_version": 1, "feature_sha256": document_hash(mr), "reference_id": "SYNTH-KNOWN-VALUES",
           "reviewer_code": "SYNTH", "reviewed": True, "method": mr["method"],
           "measurements": {k: {"value": v, "unit": "1", "atol": 1e-10, "rtol": 0.}
                            for k, v in {"rest_normalized_upslope": .5, "stress_normalized_upslope": 1., "mr_mpri": 2.}.items()}}
    local = {condition: _curves(rid, condition, "pet_dynamic") for condition in ("rest", "stress")}
    request = pet_request(case_token, local)
    for key in ("method_id", "software_version", "protocol_id", "tracer", "reference_id", "reviewer_code"):
        request[key] = "SYNTH-ONLY"
    request["branches"]["rest"]["mbf"], request["branches"]["stress"]["mbf"] = .8, 2.4
    request["reported_mfr"], request["qc_passed"] = 3., True
    return build_bundle(case_token, mr=mr, pet=import_pet(request, case_token, local), mr_reference=ref)


def create_synthetic_project(workspace, root, *, progress=None, cancel=None):
    check_cancel(cancel)
    run = new_run(root)
    project = load_json(workspace/"configs/cohort_project.template.json")
    project["mode"] = "synthetic_fixture"
    rows = {kind: [] for kind in SCHEMAS}
    # Six balanced development patients, one independent patient, one extra
    # examination of patient 1. All identifiers and values are synthetic.
    for number in range(1, 9):
        check_cancel(cancel)
        if progress:
            progress("cohort_synthetic", number, 8)
        rid = f"SYNTH-E{number:03}"
        p = 1 if number == 8 else number
        patient = f"SYNTH-P{p:03}"
        token = document_hash(["D5-synthetic-case", rid])
        rows["manifest"].append(dict(ResearchID=rid, DICOMWritten="28", ProcessingErrors="0"))
        rows["associations"].append(dict(research_id=rid, patient_key=patient, case_token=token,
            index_exam="0" if number == 8 else "1", identity_reviewed="1", used_in_development="0" if p == 7 else "1",
            reviewer_code="SYNTH", source_reference="SYNTH-ASSOCIATION"))
        if number != 8:
            rows["partitions"].append(dict(patient_key=patient, partition="independent_test" if p == 7 else "development",
                reviewer_code="SYNTH", source_reference="SYNTH-PARTITION", exclusion_reason=""))
        if p == 7 or number == 8:
            continue  # Held out case intentionally has no clinical/label/features.
        rows["clinical"].append(dict(patient_id=rid, age_years=str(40+p), hypertension=str(p % 2), diabetes="0", smoking="0",
            rest_heart_rate_bpm=str(60+p), ldl_c_mmol_l="2.2", measurement_time_reference="SYNTH-INDEX", source_reference="SYNTH-CLINICAL"))
        rows["labels"].append(dict(patient_id=rid, cmvd_label=str(p % 2), label_source="hospital_reported_physician_diagnosis",
            label_definition_version="SYNTH-DEFINITION", diagnosis_time_reference="SYNTH-INDEX", reviewer_code="SYNTH", source_reference="SYNTH-LABEL"))
        bundle = synthetic_bundle(rid, token)
        saved = save_features(run/"feature_fixtures", bundle, cancel=cancel)
        rows["perfusion"].append(dict(research_id=rid, features_path=str(saved/"features.json"), features_sha256=document_hash(bundle),
            measurement_reviewed="1", reviewer_code="SYNTH", source_reference="SYNTH-MEASUREMENTS"))
    for kind, fields in SCHEMAS.items():
        check_cancel(cancel)
        write_csv(run/(kind+".csv"), list(fields), rows[kind])
        project["inputs"][kind] = str(run/(kind+".csv"))
    policy = project["policy"]
    policy["reported_case_count"] = 8
    for key in policy:
        if key.endswith("_reviewed"):
            policy[key] = True
        elif key.endswith("_reference"):
            policy[key] = "SYNTH-REFERENCE"
    policy["reviewer_code"], policy["label_definition_version"] = "SYNTH", "SYNTH-DEFINITION"
    write_json(run/"cohort_project.json", project)
    check_cancel(cancel)
    write_json(run/"bundle.json", {"completed": True, "kind": "synthetic_cohort_fixture", "real_cases_created": 0})
    return run/"cohort_project.json"

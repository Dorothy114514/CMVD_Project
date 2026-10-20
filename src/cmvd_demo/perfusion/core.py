"""D4 numerical contracts, independent of Qt and diagnostic thresholds.

The MR algorithm is an explicit engineering definition, not a fitted MBF model.
PET MBF numbers originate in an external workstation, never in a TAC ratio.
"""

import copy
import re

import numpy as np

from cmvd_demo.roi import RoiError, validate_document, document_hash


class FeatureError(RoiError):
    pass


MR_METHOD = "mr_baseline_subtracted_adjacent_upslope_v1"


def require(ok, code):
    if not ok:
        raise FeatureError(code)


def number(value, *, positive=False):
    require(type(value) in (int, float), "feature_number_required")
    try:
        valid = bool(np.isfinite(value)) and (not positive or value > 0)
    except (TypeError, OverflowError):
        valid = False
    require(valid, "feature_nonfinite_or_nonpositive")
    return float(value)


def token(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value),
            "feature_opaque_token_required")
    return value


def signature(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value), "feature_signature_required")
    return value


def case_token_for_choices(choices):
    """Temporary local association, NOT a replacement research patient ID."""
    ids = set()
    for choice in choices:
        sources = choice.source if isinstance(choice.source, tuple) else (choice.source,)
        for source in sources:
            if hasattr(source, "objects"):
                ids.update(h.study_uid for h in source.objects)
            elif getattr(source, "synthetic_id", None):
                ids.add("synthetic:"+source.synthetic_id)
    require(bool(ids), "feature_case_context_required")
    return document_hash(sorted(ids))


def measurement(value, unit):
    return {"value": number(float(value)), "unit": unit}


def curve_input(document, result, kind):
    validate_document(document)
    require(document["binding"]["kind"] == kind, "feature_wrong_branch")
    require(isinstance(result, dict) and result.get("schema_version") == 1
            and result.get("method") == "pixel_centre_polygon_mean_v1"
            and result.get("document_sha256") == document_hash(document), "feature_stale_curves")
    z = result.get("slice_index")
    shape = document["binding"]["shape"]
    require(type(z) is int and 0 <= z < shape[1], "feature_invalid_plane")
    curves = result.get("curves")
    require(isinstance(curves, list) and len(curves) == 2
            and {c.get("role") for c in curves} == {"myocardium", "blood_pool"}, "feature_roi_pair_required")
    output = {}
    for curve in curves:
        require(curve.get("slice_index") == z and curve.get("units") in document["binding"]["units"],
                "feature_curve_binding_mismatch")
        try:
            t, y = np.asarray(curve["time_s"], dtype=float), np.asarray(curve["mean"], dtype=float)
        except (KeyError, ValueError, TypeError):
            raise FeatureError("feature_invalid_curve") from None
        require(t.shape == y.shape == (shape[0],) and len(t) >= 3 and np.isfinite(t).all()
                and np.isfinite(y).all() and np.all(np.diff(t) > 0), "feature_invalid_curve")
        expected = document["binding"]["times_s_by_slice"]
        require(expected is not None and np.array_equal(t, expected[z]), "feature_curve_binding_mismatch")
        expected_durations = document["binding"]["frame_durations_s_by_slice"]
        durations = curve.get("frame_duration_s")
        require((durations is None and expected_durations is None) or
                (durations is not None and expected_durations is not None
                 and np.array_equal(durations, expected_durations[z])), "feature_curve_binding_mismatch")
        require(curve["units"] == curves[0]["units"], "feature_units_mismatch")
        output[curve["role"]] = (t, y)
    return output


def local_warnings(document, result):
    warnings = list(result["qc"]["warnings"])
    if document["binding"]["units"] == ["SYNTHETIC_AU"]:
        warnings.append("synthetic_signal_not_research_feature")
    if document["purpose"] == "engineering_fixture":
        warnings.append("engineering_fixture_not_research_feature")
    for review in ("orientation_and_coverage", "motion_and_contours"):
        if not document["reviews"][review]:
            warnings.append(review+"_not_reviewed")
    return list(dict.fromkeys(warnings))


def mr_features(document, result, case_token, condition, protocol_id, *, baseline_frames,
                window_start, window_end, window_reviewed=False, baseline_start=0):
    signature(case_token)
    require(condition in ("rest", "stress"), "feature_condition_required")
    token(protocol_id)
    require(type(window_reviewed) is bool, "feature_review_boolean_required")
    curves = curve_input(document, result, "mr_perfusion")
    require(document["binding"]["units"] in (["MR_SIGNAL_AU"], ["SYNTHETIC_AU"]), "feature_mr_units")
    n = document["binding"]["shape"][0]
    require(all(type(i) is int for i in (baseline_start, baseline_frames, window_start, window_end))
            and 0 <= baseline_start and 2 <= baseline_frames
            and baseline_start+baseline_frames <= window_start < window_end < n, "feature_invalid_analysis_window")
    values, warnings = {}, local_warnings(document, result)
    if not window_reviewed:
        warnings.append("baseline_and_first_pass_window_not_reviewed")
    unit = document["binding"]["units"][0]
    # Include one point before the window for its first finite difference.
    origin = window_start-1
    for role, (t, y) in curves.items():
        baseline = float(np.mean(y[baseline_start:baseline_start+baseline_frames], dtype=np.float64))
        require(baseline > 0, "feature_nonpositive_baseline")
        peak = window_start+int(np.argmax(y[window_start:window_end+1]))
        slopes = np.diff(y[origin:peak+1])/np.diff(t[origin:peak+1])
        upslope = float(np.max(slopes))
        require(upslope > 0 and y[peak] > baseline, "feature_no_positive_enhancement")
        area = np.trapezoid(y[origin:window_end+1]-baseline, t[origin:window_end+1])
        prefix = role+"."
        for key, value, units in (
            ("baseline", baseline, unit), ("peak", y[peak], unit),
            ("peak_enhancement", y[peak]-baseline, unit),
            ("relative_peak_enhancement", (y[peak]-baseline)/baseline, "1"),
            ("peak_time", t[peak], "s"),
            ("ttp_from_window_origin", t[peak]-t[origin], "s"),
            ("max_upslope", upslope, unit+"/s"), ("signed_enhancement_auc", area, unit+"*s")):
            values[prefix+key] = measurement(value, units)
        if peak == window_end:
            warnings.append(role+"_peak_at_window_end")
        if peak == window_start:
            warnings.append(role+"_peak_at_window_start")
    normalized = values["myocardium.max_upslope"]["value"]/values["blood_pool.max_upslope"]["value"]
    values["normalized_upslope"] = measurement(normalized, "1")
    return {"schema_version": 1, "kind": "mr_plane_features", "case_token": case_token,
            "condition": condition, "protocol_id": protocol_id, "method": MR_METHOD,
            "source_binding": copy.deepcopy(document["binding"]), "slice_index": result["slice_index"],
            "roi_document_sha256": document_hash(document), "curves_sha256": document_hash(result),
            "roi_document": copy.deepcopy(document), "input_curves": copy.deepcopy(result),
            "parameters": {"baseline_start":baseline_start,"baseline_frames": baseline_frames, "window_start": window_start,
                           "window_end": window_end, "window_origin_index": origin,
                           "window_reviewed": window_reviewed, "temporal_interpolation": False},
            "measurements": values,
            "qc": {"warnings": warnings, "local_qc_passed": not warnings,
                   "reference_validated": False, "clinical_validated": False}}


def pair_mr(rest, stress, *, pairing_reviewed=False, protocol_comparable=False):
    require(type(pairing_reviewed) is bool and type(protocol_comparable) is bool, "feature_review_boolean_required")
    require(rest["kind"] == stress["kind"] == "mr_plane_features"
            and rest["condition"] == "rest" and stress["condition"] == "stress", "feature_condition_pair")
    require(rest["case_token"] == stress["case_token"], "feature_case_mismatch")
    require(rest["method"] == stress["method"] and rest["protocol_id"] == stress["protocol_id"], "feature_method_mismatch")
    require(rest["source_binding"]["units"] == stress["source_binding"]["units"], "feature_units_mismatch")
    require(rest["source_binding"]["source_signature"] != stress["source_binding"]["source_signature"], "feature_same_source_pair")
    a = np.asarray(rest["source_binding"]["orientation_lps"])
    b = np.asarray(stress["source_binding"]["orientation_lps"])
    # Geometry screen only; same anatomical level still requires explicit review.
    require(np.allclose(a, b, atol=0.02, rtol=0), "feature_plane_orientation_mismatch")
    r = number(rest["measurements"]["normalized_upslope"]["value"], positive=True)
    s = number(stress["measurements"]["normalized_upslope"]["value"], positive=True)
    warnings = []
    if not rest["qc"]["local_qc_passed"] or not stress["qc"]["local_qc_passed"]:
        warnings.append("mr_branch_qc_pending")
    if not pairing_reviewed:
        warnings.append("same_anatomical_plane_and_conditions_not_reviewed")
    if not protocol_comparable:
        warnings.append("dose_sequence_and_protocol_comparability_not_reviewed")
    return {"schema_version": 1, "kind": "mr_plane_pair", "case_token": rest["case_token"],
            "method": MR_METHOD, "rest": copy.deepcopy(rest), "stress": copy.deepcopy(stress),
            "reviews": {"pairing_reviewed": pairing_reviewed, "protocol_comparable": protocol_comparable},
            "measurements": {"rest_normalized_upslope": measurement(r, "1"),
                             "stress_normalized_upslope": measurement(s, "1"),
                             "mr_mpri": measurement(s/r, "1")},
            "qc": {"warnings": warnings, "local_qc_passed": not warnings,
                   "reference_validated": False, "clinical_validated": False}}


def tac_binding(document, result):
    curve_input(document, result, "pet_dynamic")
    require(document["binding"]["units"] == ["BQML"], "feature_pet_tac_units")
    durations = document["binding"]["frame_durations_s_by_slice"]
    require(durations is not None, "feature_pet_frame_duration_required")
    for row in durations:
        require(all(number(v, positive=True) > 0 for v in row), "feature_pet_frame_duration_required")
    return {"source_signature": document["binding"]["source_signature"],
            "roi_document_sha256": document_hash(document), "curves_sha256": document_hash(result),
            "slice_index": result["slice_index"]}


def pet_request(case_token, local_tacs):
    signature(case_token)
    require(set(local_tacs) == {"rest", "stress"}, "feature_pet_pair_required")
    branches = {}
    for condition in ("rest", "stress"):
        doc, result = local_tacs[condition]
        branches[condition] = {"binding": tac_binding(doc, result), "mbf": None}
    require(branches["rest"]["binding"]["source_signature"] != branches["stress"]["binding"]["source_signature"],
            "feature_same_source_pair")
    return {"schema_version": 1, "kind": "pet_workstation_result", "case_token": case_token,
            "unit": "mL/min/g", "region": "global_lv", "method_id": "REPLACE",
            "software_version": "REPLACE", "protocol_id": "REPLACE", "tracer": "REPLACE",
            "reference_id": "REPLACE", "reviewer_code": "REPLACE", "qc_passed": False,
            "reported_mfr": None, "mfr_absolute_tolerance": 0.01, "branches": branches}


def import_pet(payload, case_token, local_tacs):
    expected = pet_request(case_token, local_tacs)
    require(isinstance(payload, dict) and set(payload) == set(expected), "feature_pet_schema")
    require(payload["schema_version"] == 1 and payload["kind"] == expected["kind"], "feature_pet_schema")
    require(payload["case_token"] == case_token, "feature_case_mismatch")
    require(payload["unit"] == "mL/min/g" and payload["region"] == "global_lv", "feature_pet_unit_or_region")
    for name in ("method_id", "software_version", "protocol_id", "tracer", "reference_id", "reviewer_code"):
        require(token(payload[name]) != "REPLACE", "feature_pet_provenance_incomplete")
    require(payload["qc_passed"] is True, "feature_pet_workstation_qc_failed")
    require(isinstance(payload["branches"], dict) and set(payload["branches"]) == {"rest", "stress"}, "feature_pet_schema")
    values, warnings = {}, []
    for condition in ("rest", "stress"):
        branch = payload["branches"][condition]
        require(isinstance(branch, dict) and set(branch) == {"binding", "mbf"}
                and branch["binding"] == expected["branches"][condition]["binding"], "feature_pet_source_mismatch")
        values[condition+"_mbf"] = measurement(number(branch["mbf"], positive=True), "mL/min/g")
        if local_warnings(*local_tacs[condition]):
            warnings.append(condition+"_local_tac_review_pending")
    ratio = number(values["stress_mbf"]["value"]/values["rest_mbf"]["value"], positive=True)
    tolerance = number(payload["mfr_absolute_tolerance"])
    require(0 <= tolerance <= 0.1, "feature_pet_ratio_tolerance")
    if payload["reported_mfr"] is not None:
        reported = number(payload["reported_mfr"], positive=True)
        require(abs(reported-ratio) <= tolerance+1e-12, "feature_pet_reported_mfr_mismatch")
    values["pet_mfr"] = measurement(ratio, "1")
    return {"schema_version": 1, "kind": "pet_workstation_features", "case_token": case_token,
            "method": "reviewed_workstation_import_v1", "workstation_result": copy.deepcopy(payload),
            "workstation_sha256": document_hash(payload), "measurements": values,
            "local_tacs": {k: {"roi_document": copy.deepcopy(v[0]), "input_curves": copy.deepcopy(v[1])}
                           for k, v in local_tacs.items()},
            "qc": {"warnings": warnings, "local_qc_passed": not warnings,
                   "workstation_qc_reported": True, "source_binding_checked": True,
                   "independent_mbf_validation": False, "clinical_validated": False}}


def compare_reference(feature, reference):
    """Prespecified engineering comparison. Does not grant clinical validation."""
    require(isinstance(reference, dict) and set(reference) ==
            {"schema_version", "feature_sha256", "reference_id", "reviewer_code", "reviewed",
             "method", "measurements"}, "feature_reference_schema")
    require(reference["schema_version"] == 1 and reference["feature_sha256"] == document_hash(feature),
            "feature_reference_binding")
    require(reference["method"] == feature["method"], "feature_reference_method")
    require(token(reference["reference_id"]) != "REPLACE"
            and token(reference["reviewer_code"]) != "REPLACE", "feature_reference_not_reviewed")
    require(reference["reviewed"] is True, "feature_reference_not_reviewed")
    require(set(reference["measurements"]) == set(feature["measurements"]), "feature_reference_fields")
    checks = {}
    for key, measured in feature["measurements"].items():
        ref = reference["measurements"][key]
        require(isinstance(ref, dict) and set(ref) == {"value", "unit", "atol", "rtol"}, "feature_reference_schema")
        require(ref["unit"] == measured["unit"], "feature_reference_units")
        v, a, r = number(ref["value"]), number(ref["atol"]), number(ref["rtol"])
        require(a >= 0 and 0 <= r <= 1, "feature_reference_tolerance")
        error, limit = abs(measured["value"]-v), a+r*abs(v)
        checks[key] = {"absolute_error": error, "allowed_error": limit, "passed": error <= limit+1e-12}
    return {"feature_sha256": document_hash(feature), "reference": copy.deepcopy(reference),
            "checks": checks, "passed": all(c["passed"] for c in checks.values()), "clinical_validated": False}


def build_bundle(case_token, *, mr=None, pet=None, mr_reference=None):
    signature(case_token)
    require(mr is not None or pet is not None, "feature_empty_bundle")
    for branch, kind in ((mr, "mr_plane_pair"), (pet, "pet_workstation_features")):
        if branch is not None:
            require(branch.get("kind") == kind, "feature_bundle_branch")
            require(branch["case_token"] == case_token, "feature_case_mismatch")
    warnings = []
    for name, branch in (("mr", mr), ("pet", pet)):
        if branch is None:
            warnings.append(name+"_branch_missing")
        elif not branch["qc"]["local_qc_passed"]:
            warnings.append(name+"_local_qc_pending")
    check = None
    if mr_reference is not None:
        require(mr is not None, "feature_reference_binding")
        check = compare_reference(mr, mr_reference)
    if check is None or not check["passed"]:
        warnings.append("mr_reference_missing_or_failed")
    return {"schema_version": 1, "kind": "perfusion_feature_bundle", "case_token": case_token,
            "mr": copy.deepcopy(mr), "pet": copy.deepcopy(pet), "mr_reference_check": check,
            "qc": {"warnings": warnings, "perfusion_input_ready": not warnings,
                   "model_input_ready": False, "clinical_validated": False,
                   "diagnosis_available": False},
            "limitations": ["PET MBF is imported, not computed from images locally",
                            "MR is exploratory per-plane semiquantitation, not absolute MBF",
                            "Local TAC binding is not an independent MBF reference comparison",
                            "Clinical fields, labels and trained model are not included"]}

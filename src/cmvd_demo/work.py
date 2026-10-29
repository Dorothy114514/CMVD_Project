"""Append-only image work records, verified against the original reader.

No Qt, source paths, raw identifiers, automatic repair or model admission.
"""
import copy
import csv
import hashlib
import io
from pathlib import Path
import re

import numpy as np

from cmvd_demo.io import DicomReadError, ReadCancelled
from cmvd_demo.roi import binding_for, validate_document, document_hash, extract_curves
from cmvd_demo.perfusion import mr_features, pair_mr, import_pet, compare_reference, case_token_for_choices
from cmvd_demo.perfusion.core import curve_input
from cmvd_demo.perfusion.storage import load_json, write_json, new_run

KIND = "image_work_record"
MAX_ENTRIES = 64


def require(ok, code="work_invalid_record"):
    if not ok:
        raise DicomReadError(code)


def check_cancel(cancel):
    if cancel and cancel():
        raise ReadCancelled()


def equal(a, b):
    require(document_hash(a) == document_hash(b), "work_recalculation_mismatch")


def check_sources(grids, cancel=None):
    for grid in grids.values():
        for row in getattr(grid, "cells", ()):
            for header in row:
                check_cancel(cancel)
                info = header.path.stat()
                require((info.st_size, info.st_mtime_ns) == header.fingerprint, "source_changed_since_index")


def verify_mr(feature, documents, curves, case_token):
    require(isinstance(feature, dict), "work_invalid_analysis")
    key = feature["key"]
    value = feature["value"]
    require(key in documents, "work_analysis_source_missing")
    doc = documents[key]
    result = curves[key].get(str(value["slice_index"]))
    require(result is not None, "work_analysis_curves_missing")
    equal(value["roi_document"], doc)
    equal(value["input_curves"], result)
    params = value["parameters"]
    calculated = mr_features(doc, result, case_token, value["condition"], value["protocol_id"],
        baseline_start=params["baseline_start"], baseline_frames=params["baseline_frames"],
        window_start=params["window_start"], window_end=params["window_end"],
        window_reviewed=params["window_reviewed"])
    equal(calculated, value)
    return calculated


def _validate_view(view, shape):
    fields = {"time", "slice", "levels", "view_range", "fps", "tab", "role", "scope", "tool", "scroll"}
    require(isinstance(view, dict) and set(view) == fields, "work_invalid_view")
    for key, limit in (("time", shape[0]), ("slice", shape[1]), ("tab", 2), ("scope", 2)):
        require(type(view[key]) is int and 0 <= view[key] < limit, "work_invalid_view")
    require(type(view["fps"]) is int and 1 <= view["fps"] <= 30, "work_invalid_view")
    require(view["role"] in ("myocardium", "blood_pool") and view["tool"] in ("points", "contour", "pan"), "work_invalid_view")
    levels, ranges = np.asarray(view["levels"], dtype=float), np.asarray(view["view_range"], dtype=float)
    require(levels.shape == (2,) and np.isfinite(levels).all() and abs(levels[0]) <= 1e12
            and .001 <= levels[1] <= 1e12, "work_invalid_view")
    require(ranges.shape == (2, 2) and np.isfinite(ranges).all() and np.all(np.abs(ranges) <= 1e12)
            and np.all(ranges[:,1] > ranges[:,0]), "work_invalid_view")
    require(isinstance(view["scroll"], list) and len(view["scroll"]) == 2
            and all(type(x) is int and 0 <= x <= 1000000 for x in view["scroll"]), "work_invalid_view")


def _validate_settings(settings):
    fields = {"condition", "protocol", "baseline_start", "baseline_frames", "window_start", "window_end",
              "window_reviewed", "pairing_reviewed", "protocol_comparable"}
    require(isinstance(settings, dict) and set(settings) == fields, "work_invalid_settings")
    require(settings["condition"] in (None, "rest", "stress"), "work_invalid_settings")
    require(isinstance(settings["protocol"], str) and len(settings["protocol"]) <= 80, "work_invalid_settings")
    for key in ("baseline_start", "baseline_frames", "window_start", "window_end"):
        require(type(settings[key]) is int and 1 <= settings[key] <= 10000, "work_invalid_settings")
    for key in ("window_reviewed", "pairing_reviewed", "protocol_comparable"):
        require(type(settings[key]) is bool, "work_invalid_settings")


def _verify_record(record, choices, backend, *, progress=None, cancel=None):
    """Rebuild only saved grids and recompute saved curves/analyses before use."""
    check_cancel(cancel)
    required = {"schema_version", "kind", "scope", "case_token", "active_key", "entries", "view", "feature_state"}
    require(isinstance(record, dict) and set(record) == required)
    require(record["schema_version"] == 1 and record["kind"] == KIND
            and record["scope"] in ("case_work", "single_mr"))
    require(bool(choices), "work_open_case_first")
    require(record["case_token"] == case_token_for_choices(choices), "work_case_mismatch")
    available = {choice.key: choice for choice in choices}
    entries = record["entries"]
    require(isinstance(entries, dict) and 1 <= len(entries) <= MAX_ENTRIES)
    require(record["active_key"] in entries, "work_active_source_missing")
    grids, documents, curves = {}, {}, {}
    for key, entry in sorted(entries.items()):
        check_cancel(cancel)
        require(isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", key) and key in available,
                "work_series_missing")
        require(isinstance(entry, dict) and set(entry) == {"document", "curves"})
        grid = backend.grid(available[key], progress or (lambda *a: None), cancel or (lambda: False))
        grids[key] = grid
        check_sources({key:grid}, cancel)
        doc = validate_document(entry["document"], binding_for(grid))
        require(not getattr(grid, "synthetic_id", None) or doc["purpose"] == "engineering_fixture", "work_synthetic_purpose")
        documents[key], curves[key] = doc, {}
        require(isinstance(entry["curves"], dict) and len(entry["curves"]) <= grid.shape[1])
        for z, stored in sorted(entry["curves"].items()):
            require(isinstance(z, str) and z.isdigit() and str(int(z)) == z and 0 <= int(z) < grid.shape[1])
            actual = extract_curves(grid, doc, int(z), cancel=cancel, progress=progress,
                read_plane=lambda t, zz, g=grid: backend.plane(g, t, zz, None,
                    progress or (lambda *a: None), cancel or (lambda: False)))
            equal(actual, stored)
            curves[key][z] = actual
    active = record["active_key"]
    _validate_view(record["view"], grids[active].shape)
    state = record["feature_state"]
    require(isinstance(state, dict) and set(state) == {"settings", "candidate", "mr_slots", "pet_slots", "pet", "mr_reference"})
    _validate_settings(state["settings"])
    mr = {}
    for family in ("mr_slots", "pet_slots"):
        require(isinstance(state[family], dict) and set(state[family]) <= {"rest", "stress"})
        for condition, slot in sorted(state[family].items()):
            required_slot = {"key", "document", "curves", "feature"} if family == "mr_slots" else {"key", "document", "curves"}
            require(isinstance(slot, dict) and set(slot) == required_slot)
            key = slot["key"]
            require(key in documents, "work_analysis_source_missing")
            equal(slot["document"], documents[key])
            stored = curves[key].get(str(slot["curves"]["slice_index"]))
            require(stored is not None, "work_analysis_curves_missing")
            equal(slot["curves"], stored)
            if family == "mr_slots":
                require(slot["feature"]["condition"] == condition, "work_invalid_analysis")
                mr[condition] = verify_mr({"key":key,"value":slot["feature"]}, documents, curves, record["case_token"])
            else:
                curve_input(documents[key], stored, "pet_dynamic")
    candidate = state["candidate"]
    if candidate is not None:
        require(candidate["key"] == active and candidate["value"]["slice_index"] == record["view"]["slice"], "work_invalid_analysis")
        verify_mr(candidate, documents, curves, record["case_token"])
        params, settings = candidate["value"]["parameters"], state["settings"]
        require(candidate["value"]["condition"] == settings["condition"]
                and candidate["value"]["protocol_id"] == settings["protocol"], "work_invalid_settings")
        for key in ("baseline_start", "window_start", "window_end"):
            require(params[key]+1 == settings[key], "work_invalid_settings")
        for key in ("baseline_frames", "window_reviewed"):
            require(params[key] == settings[key], "work_invalid_settings")
    if record["scope"] == "single_mr":
        require(candidate is not None and len(entries) == 1 and not state["pet_slots"]
                and state["pet"] is None and state["mr_reference"] is None, "work_invalid_single_analysis")
    if state["pet"] is not None:
        tacs = {c:(slot["document"],slot["curves"]) for c,slot in state["pet_slots"].items()}
        equal(import_pet(state["pet"]["workstation_result"], record["case_token"], tacs), state["pet"])
    if state["mr_reference"] is not None:
        require(set(mr) == {"rest", "stress"}, "work_invalid_reference")
        pair = pair_mr(mr["rest"], mr["stress"], pairing_reviewed=state["settings"]["pairing_reviewed"],
                       protocol_comparable=state["settings"]["protocol_comparable"])
        compare_reference(pair, state["mr_reference"])
    view = record["view"]
    pixels = backend.plane(grids[active], view["time"], view["slice"], None,
                           progress or (lambda *a: None), cancel or (lambda: False))
    check_sources(grids, cancel)
    check_cancel(cancel)
    return {"record":copy.deepcopy(record), "grids":grids, "curves":curves, "pixels":pixels}


def verify_record(record, choices, backend, *, progress=None, cancel=None):
    try:
        return _verify_record(record, choices, backend, progress=progress, cancel=cancel)
    except DicomReadError:
        raise
    except OSError:
        raise DicomReadError("work_source_unavailable") from None
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, RecursionError):
        raise DicomReadError("work_invalid_record") from None


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measurements_csv(record):
    state = record["feature_state"]
    analyses = [state["candidate"]["value"]] if state["candidate"] else []
    analyses += [slot["feature"] for _,slot in sorted(state["mr_slots"].items())]
    unique = {document_hash(value):value for value in analyses}
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["state","measurement","value","unit","local_qc_passed","clinical_validated"])
    writer.writeheader()
    for _, value in sorted(unique.items()):
        for name, item in sorted(value["measurements"].items()):
            writer.writerow(dict(state=value["condition"],measurement=name,**item,
                local_qc_passed=value["qc"]["local_qc_passed"],clinical_validated=False))
    return stream.getvalue()


def save_record(root, record, choices, backend, *, progress=None, cancel=None):
    check_cancel(cancel)
    run = new_run(root)
    write_json(run/"attempt.json", {"completed":False,"kind":KIND,"clinical_validated":False})
    checked = verify_record(record, choices, backend, progress=progress, cancel=cancel)
    check_cancel(cancel)
    write_json(run/"work.json", record)
    entry = record["entries"][record["active_key"]]
    write_json(run/"roi.json", entry["document"])
    current = entry["curves"].get(str(record["view"]["slice"]))
    if current is not None:
        write_json(run/"curves.json", current)
    with (run/"measurements.csv").open("x", encoding="utf-8", newline="") as stream:
        stream.write(_measurements_csv(record))
    check_sources(checked["grids"], cancel)
    check_cancel(cancel)
    hashes = {p.name:_sha(p) for p in sorted(run.iterdir()) if p.is_file()}
    write_json(run/"bundle.json", {"completed":True,"schema_version":1,"kind":KIND,"scope":record["scope"],
               "files_sha256":hashes,"model_input_ready":False,"clinical_validated":False})
    return run


def _load_record(path):
    path = Path(path)
    require(path.is_dir() or path.name in ("work.json", "bundle.json"), "work_record_file_required")
    root = path if path.is_dir() else path.parent
    marker = load_json(root/"bundle.json")
    require(marker.get("kind") == KIND and marker.get("schema_version") == 1
            and marker.get("completed") is True, "work_completion_required")
    expected = {"attempt.json","work.json","roi.json","measurements.csv"}
    hashes = marker.get("files_sha256", {})
    require(isinstance(hashes, dict) and set(hashes) in (expected, expected|{"curves.json"}), "work_file_set")
    for name, sha in sorted(hashes.items()):
        file = root/name
        require(not file.is_symlink() and file.stat().st_size <= 8*1024**2
                and _sha(file) == sha, "work_file_changed")
    record = load_json(root/"work.json")
    require(marker.get("scope") == record.get("scope") and marker.get("model_input_ready") is False
            and marker.get("clinical_validated") is False, "work_invalid_record")
    entry = record["entries"][record["active_key"]]
    equal(load_json(root/"roi.json"), entry["document"])
    curve = entry["curves"].get(str(record["view"]["slice"]))
    require((curve is not None) == ("curves.json" in hashes), "work_file_set")
    if curve is not None:
        equal(load_json(root/"curves.json"), curve)
    with (root/"measurements.csv").open(encoding="utf-8", newline="") as stream:
        require(stream.read() == _measurements_csv(record), "work_recalculation_mismatch")
    return record


def load_record(path):
    try:
        return _load_record(path)
    except DicomReadError:
        raise
    except OSError:
        raise DicomReadError("work_completion_required") from None
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, RecursionError):
        raise DicomReadError("work_invalid_record") from None


def restore_record(path, choices, backend, *, progress=None, cancel=None):
    record = load_record(path)
    checked = verify_record(record, choices, backend, progress=progress, cancel=cancel)
    equal(load_record(path), record)
    check_cancel(cancel)
    return checked

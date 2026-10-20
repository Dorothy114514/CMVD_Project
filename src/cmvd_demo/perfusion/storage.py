"""Controlled append-only D4 snapshots; completion marker is written last."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from cmvd_demo.io import ReadCancelled
from cmvd_demo.roi import document_hash
from .core import FeatureError, require


MAX_BYTES = 8*1024**2


def load_json(path):
    path = Path(path)
    require(path.stat().st_size <= MAX_BYTES, "feature_file_too_large")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "feature_duplicate_json_key")
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(FeatureError("feature_nonfinite_json")))
    except (ValueError, UnicodeError, RecursionError):
        raise FeatureError("feature_invalid_json") from None


def write_json(path, value):
    raw = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
    require(len(raw.encode("utf-8")) <= MAX_BYTES, "feature_file_too_large")
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(raw)


def new_run(root):
    run = Path(root)/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    return run


def save_request(root, request, *, name="pet_request.json", cancel=None):
    require(name in ("pet_request.json", "mr_reference_request.json"), "feature_request_filename")
    if cancel is not None and cancel():
        raise ReadCancelled()
    run = new_run(root)
    write_json(run/name, request)
    if cancel is not None and cancel():
        raise ReadCancelled()
    write_json(run/"bundle.json", {"completed": True, "kind": "reference_request_only", "measurements_available": False})
    return run


def save_features(root, bundle, *, cancel=None):
    def check():
        if cancel is not None and cancel():
            raise ReadCancelled()
    check()
    run = new_run(root)
    write_json(run/"features.json", bundle)
    rows = []
    mr, pet = bundle["mr"], bundle["pet"]
    for branch, data in (("pet", pet), ("mr_pair", mr),
                         ("mr_rest", mr["rest"] if mr else None),
                         ("mr_stress", mr["stress"] if mr else None)):
        if data is not None:
            for name, item in data["measurements"].items():
                rows.append({"branch": branch, "feature": name, **item,
                             "local_qc_passed": data["qc"]["local_qc_passed"],
                             "perfusion_input_ready": bundle["qc"]["perfusion_input_ready"],
                             "clinical_validated": False})
    check()
    with (run/"features.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["branch", "feature", "value", "unit", "local_qc_passed",
                                                    "perfusion_input_ready", "clinical_validated"])
        writer.writeheader()
        writer.writerows(rows)
    check()
    write_json(run/"bundle.json", {"schema_version": 1, "completed": True,
                                   "features_sha256": document_hash(bundle),
                                   "perfusion_input_ready": bundle["qc"]["perfusion_input_ready"],
                                   "model_input_ready": False, "clinical_validated": False})
    return run

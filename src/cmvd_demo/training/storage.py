"""Append-only JSON model package; no pickle loading or independent evaluation."""

import copy
import hashlib
import importlib.metadata
from pathlib import Path
import platform

import numpy as np

from cmvd_demo.cohort.inputs import check_cancel
from cmvd_demo.cohort.storage import write_csv
from cmvd_demo.perfusion.storage import new_run, load_json, write_json
from cmvd_demo.roi import document_hash
from .data import require, load_training_data
from .core import train_nested, validate_contract, model_probabilities


FILES = {"report.json", "models.json", "fold_audit.json", "oof_predictions.csv",
         "training_contract.json", "feature_contract.json", "provenance.json"}


def code_hashes(workspace):
    roots = [workspace/"src/cmvd_demo/training", workspace/"src/cmvd_demo/cohort"]
    paths = sorted(p for root in roots for p in root.glob("*.py"))
    return {p.relative_to(workspace).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def train_snapshot(workspace, snapshot, root, *, allow_synthetic=False, contract=None, progress=None, cancel=None):
    workspace = Path(workspace)
    feature_path, training_path = workspace/"configs/cohort_contract.json", workspace/"configs/training_contract.json"
    feature_contract = load_json(feature_path)
    contract = copy.deepcopy(contract if contract is not None else load_json(training_path))
    data = load_training_data(snapshot, feature_contract, allow_synthetic=allow_synthetic, cancel=cancel)
    validate_contract(contract, feature_contract)
    code = code_hashes(workspace)
    feature_bytes_hash = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    training_bytes_hash = hashlib.sha256(training_path.read_bytes()).hexdigest()
    check_cancel(cancel)
    run = new_run(root)
    # Publish explicit started evidence so interrupted fitting is recognizable.
    write_json(run/"attempt.json", {"completed": False, "mode": data.audit["mode"],
               "source_cohort_bundle_sha256": data.snapshot_sha256, "training_contract_sha256": document_hash(contract)})
    try:
        result = train_nested(data, contract, progress=progress, cancel=cancel)
        data.verify_sources(cancel)
        write_json(run/"training_contract.json", contract)
        write_json(run/"feature_contract.json", feature_contract)
        write_json(run/"models.json", result["models"])
        write_json(run/"fold_audit.json", result["fold_audit"])
        write_json(run/"report.json", {**result["report"], "completed": True})
        write_csv(run/"oof_predictions.csv", ["variant", "research_id", "patient_key", "outer_fold", "cmvd_label",
                                             "probability_positive", "predicted_label"], result["oof_predictions"])
        write_json(run/"provenance.json", {"python": platform.python_version(),
            "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn")},
            "code_sha256": code, "source_cohort_bundle_sha256": data.snapshot_sha256,
            "feature_config_bytes_sha256": feature_bytes_hash, "training_config_bytes_sha256": training_bytes_hash,
            "clinical_validated": False})
        # Actual disk roundtrip, not just the in-memory JSON export.
        loaded = load_json(run/"models.json")
        for name, model in loaded.items():
            columns = [feature_contract["feature_order"].index(k) for k in model["feature_order"]]
            require(np.array_equal(model_probabilities(model, data.x[:, columns]),
                                   model_probabilities(result["models"][name], data.x[:, columns])), "training_disk_roundtrip_mismatch")
        data.verify_sources(cancel)
        require(code_hashes(workspace) == code and hashlib.sha256(feature_path.read_bytes()).hexdigest() == feature_bytes_hash
                and hashlib.sha256(training_path.read_bytes()).hexdigest() == training_bytes_hash, "training_code_or_config_changed")
        check_cancel(cancel)
        files = {name: hashlib.sha256((run/name).read_bytes()).hexdigest() for name in sorted(FILES | {"attempt.json"})}
        check_cancel(cancel)
        write_json(run/"bundle.json", {"schema_version": 1, "kind": "frozen_logistic_model_package", "completed": True,
                   "mode": data.audit["mode"], "files_sha256": files,
                   "source_cohort_bundle_sha256": data.snapshot_sha256, "clinical_validated": False})
    except BaseException as error:
        from cmvd_demo.io import DicomReadError
        write_json(run/"failure.json", {"completed": False, "code": error.code if isinstance(error, DicomReadError) else type(error).__name__})
        raise
    return run


def load_model_package(run, *, allow_synthetic=False):
    run = Path(run)
    marker = load_json(run/"bundle.json")
    require(marker.get("kind") == "frozen_logistic_model_package" and marker.get("completed") is True
            and marker.get("mode") in ("research", "synthetic_fixture")
            and set(marker.get("files_sha256", {})) == FILES | {"attempt.json"}, "training_complete_model_package_required")
    require(marker["mode"] != "synthetic_fixture" or allow_synthetic is True, "training_synthetic_explicit_opt_in_required")
    for name, expected in marker["files_sha256"].items():
        require(hashlib.sha256((run/name).read_bytes()).hexdigest() == expected, "training_model_package_tampered")
    feature, contract, models, report = (load_json(run/name) for name in (
        "feature_contract.json", "training_contract.json", "models.json", "report.json"))
    validate_contract(contract, feature)
    require(set(models) == set(contract["variants"]) and report.get("completed") is True
            and report.get("mode") == marker["mode"] and report.get("primary_model") == contract["primary_model"],
            "training_model_package_binding")
    rules = feature["clinical_fields"] | feature["perfusion_fields"]
    for name, model in models.items():
        fields = contract["variants"][name]
        require(model.get("feature_order") == fields and model.get("mode") == marker["mode"]
                and model.get("feature_contract_sha256") == document_hash(feature) == report.get("feature_contract_sha256")
                and model.get("training_contract_sha256") == document_hash(contract) == report.get("training_contract_sha256")
                and model.get("source_cohort_bundle_sha256") == marker["source_cohort_bundle_sha256"] == report.get("source_cohort_bundle_sha256")
                and model.get("feature_units") == {k: rules[k]["unit"] for k in fields}
                and model.get("feature_rules") == {k: rules[k] for k in fields}
                and model.get("threshold") == contract["threshold"] and model.get("classes") == [0, 1]
                and model.get("C") in contract["candidate_C"], "training_model_package_binding")
        # Validates parameter shapes/finiteness before the model is exposed.
        model_probabilities(model, np.zeros((1, len(fields))))
    return {"models": models, "report": report, "feature_contract": feature, "training_contract": contract}

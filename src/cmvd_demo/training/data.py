"""D6 accepts only a complete, frozen D5 snapshot; independent rows stay out."""

import csv
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import numpy as np

from cmvd_demo.io import DicomReadError
from cmvd_demo.cohort.inputs import opaque, check_cancel
from cmvd_demo.cohort.storage import load_frozen
from cmvd_demo.perfusion.storage import load_json
from cmvd_demo.roi import document_hash


class TrainingError(DicomReadError):
    pass


def require(ok, code):
    if not ok:
        raise TrainingError(code)


def csv_rows(path, fields):
    require(path.stat().st_size <= 16*1024**2, "training_csv_size_limit")
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            require(reader.fieldnames == fields, "training_csv_columns_or_order")
            rows = list(reader)
        require(len(rows) <= 10000 and all(None not in r and all(v is not None for v in r.values()) for r in rows),
                "training_csv_row_structure")
        return rows
    except (csv.Error, UnicodeError):
        raise TrainingError("training_csv_invalid") from None


def validate_values(x, fields, contract):
    x = np.asarray(x, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] == len(fields) and np.isfinite(x).all(), "training_nonfinite_or_missing_features")
    rules = contract["clinical_fields"] | contract["perfusion_fields"]
    for j, key in enumerate(fields):
        rule, col = rules[key], x[:, j]
        require(("allowed" not in rule or np.isin(col, rule["allowed"]).all())
                and ("min" not in rule or (col >= rule["min"]).all())
                and ("max" not in rule or (col <= rule["max"]).all())
                and ("min_exclusive" not in rule or (col > rule["min_exclusive"]).all()), "training_feature_range")
    return x


def validate_folds(folds, patients, *, n_splits):
    all_patients = set(patients)
    require(len(folds) == n_splits, "training_outer_fold_count")
    seen = []
    for index, fold in enumerate(folds):
        train, valid = fold.get("train_patients", []), fold.get("validation_patients", [])
        require(fold.get("fold") == index and len(set(train)) == len(train) and len(set(valid)) == len(valid)
                and train and valid and not set(train) & set(valid)
                and set(train) | set(valid) == all_patients, "training_patient_fold_leakage_or_coverage")
        seen.extend(valid)
    require(len(seen) == len(patients) and set(seen) == all_patients, "training_oof_coverage")


@dataclass(frozen=True)
class TrainingData:
    x: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    research_ids: tuple = field(repr=False)
    patients: tuple = field(repr=False)
    folds: tuple = field(repr=False)
    cv_policy: dict = field(repr=False)
    feature_contract: dict = field(repr=False)
    audit: dict = field(repr=False)
    snapshot_sha256: str
    sources: tuple = field(repr=False)

    @property
    def synthetic(self):
        return self.audit["mode"] == "synthetic_fixture"

    def verify_sources(self, cancel=None):
        for path, expected in self.sources:
            check_cancel(cancel)
            require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                    "training_source_changed")


def load_training_data(run, feature_contract, *, allow_synthetic=False, cancel=None):
    check_cancel(cancel)
    run = Path(run)
    marker_hash = hashlib.sha256((run/"bundle.json").read_bytes()).hexdigest()
    plan = load_frozen(run)
    audit, marker = load_json(run/"audit.json"), load_json(run/"bundle.json")
    require(audit.get("mode") in ("research", "synthetic_fixture"), "training_cohort_mode")
    synthetic = audit["mode"] == "synthetic_fixture"
    require(not synthetic or allow_synthetic is True, "training_synthetic_explicit_opt_in_required")
    require(audit.get("partitions_frozen") is True and audit.get("development_matrix_ready") is True
            and audit.get("freeze_ready") is True and not audit.get("issues")
            and plan.get("folds") == audit.get("folds"), "training_cohort_not_ready")
    require(synthetic or (audit.get("training_inputs_ready") is True and audit.get("training_prerequisites_ready") is True
                         and audit.get("perfusion_usage") == "reviewed_D4_snapshots"), "training_research_gate_blocked")
    require(all(plan.get(k) == audit.get(k) for k in ("input_sha256", "project_sha256", "contract_sha256"))
            and plan.get("population") == audit.get("population"), "training_cohort_plan_binding")
    require(audit.get("contract_sha256") == document_hash(feature_contract)
            and audit.get("contract_id") == feature_contract["contract_id"]
            and audit.get("feature_order") == feature_contract["feature_order"], "training_feature_contract_mismatch")
    fields = feature_contract["feature_order"]
    units = {k: r["unit"] for k, r in (feature_contract["clinical_fields"] | feature_contract["perfusion_fields"]).items()}
    require(audit.get("feature_units") == units, "training_feature_units_mismatch")
    require(opaque(audit.get("label_definition_version")) and opaque(audit.get("label_definition_reference")),
            "training_label_definition_required")
    patient_plan = plan.get("patients", [])
    patients_seen, exams_seen, eligible = set(), set(), {}
    for item in patient_plan:
        patient, visits, rid = item.get("patient_key"), item.get("research_ids", []), item.get("index_research_id")
        require(opaque(patient) and patient not in patients_seen and visits and len(set(visits)) == len(visits)
                and all(opaque(v) and v not in exams_seen for v in visits), "training_duplicate_patient_or_exam")
        require(item.get("partition") in ("development", "independent_test", "excluded"), "training_partition_unknown")
        patients_seen.add(patient)
        exams_seen.update(visits)
        require((patient.startswith("SYNTH-") and all(v.startswith("SYNTH-") for v in visits)) if synthetic
                else not (patient.startswith("SYNTH-") or any(v.startswith("SYNTH-") for v in visits)), "training_identifier_mode_mismatch")
        if item["partition"] != "excluded":
            require(rid in visits, "training_index_exam_missing")
        if item["partition"] == "development":
            eligible[rid] = patient
    features = csv_rows(run/"features.csv", ["research_id"]+fields)
    labels = csv_rows(run/"labels.csv", ["research_id", "cmvd_label"])
    ids = [r["research_id"] for r in features]
    label_map = {r["research_id"]: r["cmvd_label"] for r in labels}
    require(ids and len(set(ids)) == len(ids) and len(label_map) == len(labels)
            and set(ids) == set(label_map) and set(ids) <= set(eligible), "training_development_rows_mismatch_or_independent_leakage")
    require(len(ids) == audit["development_included_count"] and all(label_map[rid] in ("0", "1") for rid in ids),
            "training_label_or_row_count")
    try:
        x = validate_values([[float(row[k]) for k in fields] for row in features], fields, feature_contract)
    except TrainingError:
        raise
    except (ValueError, OverflowError):
        raise TrainingError("training_nonfinite_or_missing_features") from None
    y = np.asarray([int(label_map[rid]) for rid in ids], dtype=np.int64)
    patients = tuple(eligible[rid] for rid in ids)
    require(len(set(patients)) == len(ids) and set(y) == {0, 1}, "training_duplicate_patient_or_single_class")
    require(plan["folds"].get("ready") is True, "training_folds_not_ready")
    validate_folds(plan["folds"]["folds"], patients, n_splits=plan["cv_policy"]["n_splits"])
    sources = tuple((run/name, digest) for name, digest in marker["files_sha256"].items())+((run/"bundle.json", marker_hash),)
    data = TrainingData(x, y, tuple(ids), patients, tuple(plan["folds"]["folds"]), plan["cv_policy"], feature_contract, audit, marker_hash, sources)
    data.verify_sources(cancel)
    return data

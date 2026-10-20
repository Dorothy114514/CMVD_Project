"""Read-only controlled CSV projections. Never infer IDs or labels from paths."""

import csv
import hashlib
from pathlib import Path
import re

from cmvd_demo.io import DicomReadError, ReadCancelled


class CohortError(DicomReadError):
    pass


SCHEMAS = {
    "manifest": ("ResearchID", "DICOMWritten", "ProcessingErrors"),
    "associations": ("research_id", "patient_key", "case_token", "index_exam", "identity_reviewed",
                     "used_in_development", "reviewer_code", "source_reference"),
    "clinical": ("patient_id", "age_years", "hypertension", "diabetes", "smoking",
                 "rest_heart_rate_bpm", "ldl_c_mmol_l", "measurement_time_reference", "source_reference"),
    "labels": ("patient_id", "cmvd_label", "label_source", "label_definition_version",
               "diagnosis_time_reference", "reviewer_code", "source_reference"),
    "perfusion": ("research_id", "features_path", "features_sha256", "measurement_reviewed",
                  "reviewer_code", "source_reference"),
    "partitions": ("patient_key", "partition", "reviewer_code", "source_reference", "exclusion_reason"),
    "missingness": ("patient_id", "field", "reason"),
}


def check_cancel(cancel):
    if cancel is not None and cancel():
        raise ReadCancelled()


def require(ok, code):
    if not ok:
        raise CohortError(code)


def opaque(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value) is not None


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def resolve_input(workspace, value):
    require(isinstance(value, str) and bool(value.strip()), "cohort_input_path_required")
    p = Path(value)
    return p if p.is_absolute() else Path(workspace)/p


def read_table(path, kind, *, cancel=None):
    """Only projected columns escape this function, even for legacy manifests."""
    path = Path(path)
    require(path.stat().st_size <= 16*1024**2, "cohort_csv_too_large")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(text.splitlines(), strict=True)
        fields = reader.fieldnames
        require(fields is not None and len(fields) == len(set(fields)), "cohort_csv_duplicate_or_missing_header")
        require(set(SCHEMAS[kind]) <= set(fields), "cohort_csv_required_columns_missing")
        # Extra candidate clinical/legacy manifest columns are intentionally not
        # projected; strict schemas prevent a typo becoming an unused new field.
        if kind not in ("manifest", "clinical"):
            require(set(fields) == set(SCHEMAS[kind]), "cohort_csv_unknown_columns")
        rows = []
        for row in reader:
            check_cancel(cancel)
            require(None not in row and all(row[k] is not None for k in fields), "cohort_csv_row_width")
            require(len(rows) < 10000, "cohort_csv_row_limit")
            rows.append({key: row[key].strip() for key in SCHEMAS[kind]})
        return rows, hashlib.sha256(raw).hexdigest()
    except (UnicodeError, csv.Error):
        raise CohortError("cohort_csv_invalid_utf8_or_structure") from None


def indexed(rows, field, *, mode, composite=None):
    result = {}
    for row in rows:
        key = row[field]
        require(opaque(key), "cohort_invalid_association_identifier")
        if mode == "research":
            require(not key.startswith("SYNTH-"), "cohort_synthetic_identifier_in_research")
        identity = (key, row[composite]) if composite else key
        require(identity not in result, "cohort_duplicate_association_key")
        result[identity] = row
    return result

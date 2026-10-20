"""Append-only D5 artifacts. A marker publishes a complete run, never a partial."""

import copy
import csv
import hashlib
from pathlib import Path
import shutil

from cmvd_demo.perfusion.storage import new_run, write_json, load_json
from cmvd_demo.roi import document_hash
from .inputs import require, check_cancel


def write_csv(path, fields, rows):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_sources(cohort, cancel):
    for path, expected in cohort["_sources"]:
        check_cancel(cancel)
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                "cohort_source_changed_recheck_required")


def save_cohort(root, cohort, *, freeze=False, cancel=None):
    check_cancel(cancel)
    require(type(freeze) is bool, "cohort_freeze_boolean")
    require(not freeze or cohort["report"]["freeze_ready"], "cohort_partition_freeze_blocked")
    verify_sources(cohort, cancel)
    run = new_run(root)
    report = copy.deepcopy(cohort["report"])
    report["partitions_frozen"] = freeze
    report["training_inputs_ready"] = bool(freeze and report["training_prerequisites_ready"])
    write_json(run/"audit.json", report)
    cases = [{**row, "reasons": "|".join(row["reasons"])} for row in cohort["cases"]]
    write_csv(run/"case_status.csv", ["research_id", "patient_key", "partition", "included", "reasons"], cases)
    fields = report["feature_order"]
    write_csv(run/"features.csv", ["research_id"]+fields,
              [{"research_id": s["research_id"], **{key: s["features"][key] for key in fields}}
               for s in cohort["samples"]])
    write_csv(run/"labels.csv", ["research_id", "cmvd_label"],
              [{"research_id": s["research_id"], "cmvd_label": s["label"]} for s in cohort["samples"]])
    write_csv(run/"missingness.csv", ["research_id", "field", "reason"], cohort["missingness"])
    # Audit plans are distinct from frozen plans. Only the latter is a D6 input.
    plan = {"schema_version": 1, "kind": "patient_partition_plan", "mode": report["mode"],
            "frozen": freeze, "patients": cohort["patient_plan"], "audit_sha256": document_hash(report),
            "input_sha256": report["input_sha256"], "project_sha256": report["project_sha256"],
            "contract_sha256": report["contract_sha256"],
            "cv_policy": {key: cohort["policy"][key] for key in ("n_splits", "random_seed")},
            "folds": report["folds"]}
    write_json(run/("partitions.json" if freeze else "partition_plan.json"), plan)
    check_cancel(cancel)
    verify_sources(cohort, cancel)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir() if p.is_file()}
    check_cancel(cancel)
    write_json(run/"bundle.json", {"schema_version": 1, "kind": "cohort_snapshot", "completed": True,
                                   "mode": report["mode"], "partitions_frozen": freeze, "files_sha256": files})
    return run


def load_frozen(run):
    """Validate all published artifacts before a later stage uses a partition."""
    run = Path(run)
    marker = load_json(run/"bundle.json")
    require(marker.get("kind") == "cohort_snapshot" and marker.get("completed") is True
            and marker.get("partitions_frozen") is True, "cohort_frozen_snapshot_required")
    expected_names = {"audit.json", "case_status.csv", "features.csv", "labels.csv", "missingness.csv", "partitions.json"}
    require(set(marker.get("files_sha256", {})) == expected_names, "cohort_snapshot_file_set")
    for name, expected in marker["files_sha256"].items():
        require(hashlib.sha256((run/name).read_bytes()).hexdigest() == expected, "cohort_snapshot_tampered")
    plan, report = load_json(run/"partitions.json"), load_json(run/"audit.json")
    require(plan.get("frozen") is True and plan.get("audit_sha256") == document_hash(report)
            and plan.get("mode") == marker.get("mode") == report.get("mode"), "cohort_partition_binding")
    return plan


def export_templates(workspace, root, *, cancel=None):
    """Produce blank supplemental forms; never produce or replace the old census."""
    workspace = Path(workspace)
    check_cancel(cancel)
    run = new_run(root)
    filenames = {"associations": "patient_associations.csv", "clinical": "clinical_features.csv",
                 "labels": "diagnosis_labels.csv", "perfusion": "perfusion_registry.csv",
                 "partitions": "patient_partitions.csv", "missingness": "clinical_missingness.csv"}
    project = load_json(workspace/"configs/cohort_project.template.json")
    for kind, filename in filenames.items():
        check_cancel(cancel)
        # New run directory is exclusive; no existing file is overwritten.
        with (workspace/"examples/templates"/filename).open("rb") as source, (run/filename).open("xb") as target:
            shutil.copyfileobj(source, target)
        project["inputs"][kind] = str((run/filename).resolve())
    project["inputs"]["manifest"] = str((workspace/"metadata/full_case_manifest.csv").resolve())
    write_json(run/"cohort_project.json", project)
    check_cancel(cancel)
    write_json(run/"bundle.json", {"completed": True, "kind": "blank_cohort_forms", "real_cases_created": 0})
    return run

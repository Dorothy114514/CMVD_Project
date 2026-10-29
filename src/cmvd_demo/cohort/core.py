"""D5 patient association and complete-case development matrix; no model fit."""

import hashlib
import math
from pathlib import Path

from cmvd_demo.io import DicomReadError
from cmvd_demo.perfusion import load_json
from cmvd_demo.roi import document_hash
from .inputs import (CohortError, SCHEMAS, require, opaque, digest, resolve_input,
                     read_table, indexed, check_cancel)
from .perfusion import verified_ratios
from .protocol import population_status


PARTITIONS = {"development", "independent_test", "excluded"}


def reviewed(row):
    return bool(row) and opaque(row.get("reviewer_code")) and opaque(row.get("source_reference"))


def clinical_values(row, contract):
    values, reasons = {}, []
    for field, rule in contract["clinical_fields"].items():
        text = row.get(field, "") if row else ""
        if not text:
            reasons.append("clinical_missing:"+field)
            continue
        try:
            value = float(text)
            valid = math.isfinite(value)
            valid &= "allowed" not in rule or value in rule["allowed"]
            valid &= "min" not in rule or value >= rule["min"]
            valid &= "max" not in rule or value <= rule["max"]
            valid &= "min_exclusive" not in rule or value > rule["min_exclusive"]
        except (ValueError, OverflowError):
            valid = False
        if valid:
            values[field] = value
        else:
            reasons.append("clinical_invalid:"+field)
    if not row or not opaque(row.get("measurement_time_reference")) or not opaque(row.get("source_reference")):
        reasons.append("clinical_provenance_missing")
    return values, reasons


def make_folds(samples, n_splits=3, seed=20261020):
    """Stable stratification of development patients only; never reduce folds."""
    require(type(n_splits) is int and 2 <= n_splits <= 10 and type(seed) is int,
            "cohort_fold_policy")
    require(len({s["patient_key"] for s in samples}) == len(samples), "cohort_fold_duplicate_patient")
    require(all(s["partition"] == "development" and s["label"] in (0, 1) for s in samples),
            "cohort_fold_non_development_patient")
    groups = [[s for s in samples if s["label"] == label] for label in (0, 1)]
    if min(map(len, groups)) < n_splits:
        return {"ready": False, "reason": "insufficient_patients_per_class", "folds": []}
    assignments = {}
    for group in groups:
        ordered = sorted(group, key=lambda s: hashlib.sha256(f'{seed}:{s["patient_key"]}'.encode()).hexdigest())
        for index, sample in enumerate(ordered):
            assignments[sample["patient_key"]] = index % n_splits
    all_patients = set(assignments)
    folds = []
    for index in range(n_splits):
        valid = {p for p, f in assignments.items() if f == index}
        folds.append({"fold": index, "train_patients": sorted(all_patients-valid),
                      "validation_patients": sorted(valid)})
    return {"ready": True, "reason": None, "folds": folds}


def assemble_cohort(workspace, project, *, progress=None, cancel=None):
    """Read explicit local CSVs, preserving original IDs and independent isolation.

    Missing files are an audit result, not permission to invent a manifest.
    All diagnostics are fixed codes; input free text is never echoed.
    """
    workspace = Path(workspace)
    require(isinstance(project, dict) and type(project.get("schema_version")) is int and project["schema_version"] in (1, 2)
            and project.get("mode") in ("research", "synthetic_fixture"), "cohort_project_schema")
    mode = project["mode"]
    inputs, policy = project.get("inputs"), project.get("policy")
    require(isinstance(inputs, dict) and set(inputs) == set(SCHEMAS) and isinstance(policy, dict),
            "cohort_project_schema")
    contract_path = workspace/"configs/cohort_contract.json"
    contract = load_json(contract_path)
    issues, tables, fingerprints, sources = [], {}, {}, []
    sources.append((contract_path, hashlib.sha256(contract_path.read_bytes()).hexdigest()))
    for kind in SCHEMAS:
        check_cancel(cancel)
        if progress:
            progress("cohort_inputs", len(tables), len(SCHEMAS))
        path = resolve_input(workspace, inputs[kind])
        try:
            rows, fingerprint = read_table(path, kind, cancel=cancel)
            key = {"manifest": "ResearchID", "associations": "research_id", "clinical": "patient_id",
                   "labels": "patient_id", "perfusion": "research_id", "partitions": "patient_key",
                   "missingness": "patient_id"}[kind]
            table = indexed(rows, key, mode=mode, composite="field" if kind == "missingness" else None)
            tables[kind] = table
            fingerprints[kind] = fingerprint
            sources.append((path, fingerprint))
        except FileNotFoundError:
            tables[kind] = {}
            issues.append("input_missing:"+kind)
        except (CohortError, OSError) as error:
            tables[kind] = {}
            code = error.code if isinstance(error, CohortError) else "unreadable"
            issues.append("input_invalid:"+kind+":"+code)
    manifest, assoc = tables["manifest"], tables["associations"]
    require(type(policy.get("reported_case_count")) is int and policy["reported_case_count"] > 0,
            "cohort_reported_case_count_required")
    if len(manifest) != policy["reported_case_count"]:
        issues.append("reported_manifest_count_mismatch")
    for flag, ref in (("reconciliation_reviewed", "reconciliation_reference"),
                      ("clinical_schema_reviewed", "clinical_schema_reference"),
                      ("exclusions_reviewed", "exclusions_reference")):
        if policy.get(flag) is not True or not opaque(policy.get(ref)):
            issues.append("policy_pending:"+flag)
    if not opaque(policy.get("reviewer_code")):
        issues.append("policy_pending:reviewer_code")
    if not opaque(policy.get("label_definition_version")) or not opaque(policy.get("label_definition_reference")):
        issues.append("policy_pending:label_definition")
    for kind, table in tables.items():
        if kind in ("manifest", "partitions"):
            continue
        keys = {key[0] if isinstance(key, tuple) else key for key in table}
        if not keys <= set(manifest):
            issues.append("unmatched_research_id:"+kind)
    for (_, field), note in tables["missingness"].items():
        if field not in contract["clinical_fields"] or not opaque(note["reason"]):
            issues.append("invalid_missingness_field_or_reason")
    groups, token_owners = {}, {}
    for rid, row in assoc.items():
        if rid not in manifest:
            continue
        if not opaque(row["patient_key"]) or (mode == "research" and row["patient_key"].startswith("SYNTH-")):
            issues.append("invalid_patient_key")
            continue
        if mode == "synthetic_fixture" and not (rid.startswith("SYNTH-") and row["patient_key"].startswith("SYNTH-")):
            issues.append("fixture_requires_synthetic_identifiers")
        groups.setdefault(row["patient_key"], []).append(rid)
        if digest(row["case_token"]):
            owner = token_owners.setdefault(row["case_token"], rid)
            if owner != rid:
                issues.append("case_token_multiple_exams")
    if set(tables["partitions"]) != set(groups):
        issues.append("partition_patient_coverage_mismatch")
    patient_plan, identity_ok = [], bool(manifest) and len(assoc) == len(manifest)
    for patient, visits in sorted(groups.items()):
        check_cancel(cancel)
        assignment = tables["partitions"].get(patient, {})
        partition = assignment.get("partition", "unassigned")
        if partition not in PARTITIONS or not reviewed(assignment):
            issues.append("partition_missing_or_unreviewed")
            identity_ok = False
        if partition == "excluded" and not opaque(assignment.get("exclusion_reason")):
            issues.append("exclusion_reason_missing")
        selected = [rid for rid in visits if assoc[rid]["index_exam"] == "1"]
        if partition != "excluded" and len(selected) != 1:
            issues.append("index_exam_not_unique")
            identity_ok = False
        for rid in visits:
            a = assoc[rid]
            if (not reviewed(a) or a["identity_reviewed"] != "1" or not digest(a["case_token"])
                    or a["used_in_development"] not in ("0", "1") or a["index_exam"] not in ("0", "1")):
                issues.append("association_not_reviewed")
                identity_ok = False
            if partition == "independent_test" and a["used_in_development"] != "0":
                issues.append("independent_patient_used_or_unknown_development")
                identity_ok = False
        patient_plan.append({"patient_key": patient, "partition": partition,
                             "research_ids": sorted(visits), "index_research_id": selected[0] if len(selected) == 1 else None})
    statuses, samples, missingness = [], [], []
    plans = {p["patient_key"]: p for p in patient_plan}
    schema_ready = policy.get("clinical_schema_reviewed") is True and opaque(policy.get("clinical_schema_reference"))
    for index, (rid, original) in enumerate(sorted(manifest.items())):
        check_cancel(cancel)
        if progress:
            progress("cohort_cases", index+1, len(manifest))
        a = assoc.get(rid, {})
        plan = plans.get(a.get("patient_key"), {})
        partition = plan.get("partition", "unassigned")
        reasons, features, label = [], {}, None
        # Do not inspect held-out clinical values, labels or feature snapshots.
        if partition == "independent_test":
            reasons = ["reserved_independent_not_loaded"]
        elif partition == "excluded":
            reasons = ["explicit_patient_exclusion"]
        elif not a or not plan:
            reasons = ["patient_association_missing"]
        elif plan["index_research_id"] != rid:
            reasons = ["non_index_exam_or_ambiguous_selection"]
        elif partition != "development":
            reasons = ["patient_partition_unassigned"]
        else:
            if not identity_ok:
                reasons.append("cohort_identity_review_pending")
            try:
                require(int(original["DICOMWritten"]) > 0 and int(original["ProcessingErrors"]) == 0,
                        "manifest_processing_failed_or_empty")
            except (ValueError, CohortError):
                reasons.append("manifest_processing_failed_or_empty")
            features, clinical_errors = clinical_values(tables["clinical"].get(rid), contract)
            reasons.extend(clinical_errors)
            for error in clinical_errors:
                if error.startswith("clinical_missing:"):
                    field = error.split(":", 1)[1]
                    note = tables["missingness"].get((rid, field), {}).get("reason", "")
                    missingness.append({"research_id": rid, "field": field,
                                        "reason": note if opaque(note) else "unspecified"})
            if not schema_ready:
                reasons.append("clinical_schema_not_confirmed")
            diagnosis = tables["labels"].get(rid, {})
            if (diagnosis.get("cmvd_label") not in ("0", "1")
                    or diagnosis.get("label_source") != "hospital_reported_physician_diagnosis"
                    or diagnosis.get("label_definition_version") != policy.get("label_definition_version")
                    or not opaque(policy.get("label_definition_reference"))
                    or not reviewed(diagnosis) or not opaque(diagnosis.get("diagnosis_time_reference"))):
                reasons.append("physician_label_missing_invalid_or_unreviewed")
            else:
                label = int(diagnosis["cmvd_label"])
            registry = tables["perfusion"].get(rid, {})
            if registry.get("measurement_reviewed") != "1" or not reviewed(registry):
                reasons.append("perfusion_measurement_review_missing")
            else:
                try:
                    ratios, feature_sources = verified_ratios(resolve_input(workspace, registry["features_path"]),
                        registry["features_sha256"], a["case_token"], mode=mode, cancel=cancel)
                    features.update(ratios)
                    sources.extend(feature_sources)
                except (DicomReadError, OSError, ValueError) as error:
                    # Cancellation must never become an ordinary exclusion.
                    from cmvd_demo.io import ReadCancelled
                    if isinstance(error, ReadCancelled):
                        raise
                    reasons.append(error.code if isinstance(error, DicomReadError) else "perfusion_snapshot_unreadable")
            if not reasons:
                samples.append({"research_id": rid, "patient_key": a["patient_key"], "partition": "development",
                                "features": features, "label": label})
        statuses.append({"research_id": rid, "patient_key": a.get("patient_key", "") if plan else "",
                         "partition": partition, "included": not reasons, "reasons": reasons})
    folds = make_folds(samples, policy.get("n_splits", 3), policy.get("random_seed", 20261020))
    population, population_issues, population_freezable = population_status(project, patient_plan, len(samples))
    issues = sorted(set(issues+population_issues))
    freeze_blockers = [i for i in issues if not i.startswith(("input_missing:clinical", "input_missing:labels",
        "input_missing:perfusion", "input_missing:missingness", "policy_pending:clinical_schema", "policy_pending:label_definition"))]
    freeze_ready = identity_ok and not freeze_blockers and population_freezable
    report = {"schema_version": 1, "kind": "cohort_audit", "mode": mode,
              "contract_id": contract["contract_id"], "contract_sha256": document_hash(contract),
              "project_sha256": document_hash(project), "input_sha256": fingerprints,
              "population": population,
              "label_definition_version": policy.get("label_definition_version"),
              "label_definition_reference": policy.get("label_definition_reference"),
              "reported_case_count": policy["reported_case_count"], "manifest_case_count": len(manifest),
              "associated_patient_count": len(groups), "development_included_count": len(samples),
              "reserved_independent_patients": sum(p["partition"] == "independent_test" for p in patient_plan),
              "issues": issues, "freeze_ready": freeze_ready,
              "development_matrix_ready": bool(samples) and not issues and folds["ready"],
              "training_prerequisites_ready": mode == "research" and bool(samples) and not issues and folds["ready"] and freeze_ready,
              "training_inputs_ready": False, "partitions_frozen": False,
              "model_trained": False, "clinical_validated": False,
              "feature_order": contract["feature_order"], "feature_units": {
                  key: value["unit"] for key, value in (contract["clinical_fields"] | contract["perfusion_fields"]).items()},
              "missing_policy": contract["missing_policy"], "folds": folds,
              "perfusion_usage": "synthetic_fixture_values_only" if mode == "synthetic_fixture" else "reviewed_D4_snapshots",
              "limitations": ["Candidate fields require hospital confirmation; no imputation, scaling or model training",
                  "Independent patient measurements and labels are excluded from development exports",
                  "D4 snapshot recomputation does not re-read pixels or establish clinical validity"]}
    return {"report": report, "cases": statuses, "samples": samples, "missingness": missingness,
            "patient_plan": patient_plan, "policy": dict(policy), "_sources": sources}

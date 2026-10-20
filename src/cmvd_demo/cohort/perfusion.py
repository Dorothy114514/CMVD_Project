"""Recompute completed D4 snapshots before admitting their two ratios."""

import hashlib
from pathlib import Path

from cmvd_demo.perfusion import mr_features, pair_mr, import_pet, build_bundle, load_json
from cmvd_demo.roi import document_hash
from .inputs import CohortError, require, digest, check_cancel


def verified_ratios(path, expected_hash, case_token, *, mode="research", cancel=None):
    path = Path(path)
    require(path.name == "features.json", "cohort_completed_feature_snapshot_required")
    check_cancel(cancel)
    marker_path = path.with_name("bundle.json")
    before = [(p, hashlib.sha256(p.read_bytes()).hexdigest()) for p in (path, marker_path)]
    marker, stored = load_json(marker_path), load_json(path)
    require(digest(expected_hash) and marker.get("completed") is True
            and marker.get("features_sha256") == expected_hash == document_hash(stored),
            "cohort_feature_hash_or_completion")
    require(stored.get("case_token") == case_token, "cohort_feature_case_association")
    require(stored.get("mr") is not None and stored.get("pet") is not None,
            "cohort_perfusion_branch_missing")
    try:
        mr = stored["mr"]
        branches = []
        for condition in ("rest", "stress"):
            check_cancel(cancel)
            item = mr[condition]
            p = item["parameters"]
            rebuilt = mr_features(item["roi_document"], item["input_curves"], case_token,
                                  condition, item["protocol_id"],
                                  baseline_start=p["baseline_start"], baseline_frames=p["baseline_frames"],
                                  window_start=p["window_start"], window_end=p["window_end"],
                                  window_reviewed=p["window_reviewed"])
            require(rebuilt == item, "cohort_feature_recomputation_mismatch")
            branches.append(rebuilt)
        rebuilt_mr = pair_mr(*branches, **mr["reviews"])
        require(rebuilt_mr == mr, "cohort_feature_recomputation_mismatch")
        pet = stored["pet"]
        local = {key: (value["roi_document"], value["input_curves"])
                 for key, value in pet["local_tacs"].items()}
        rebuilt_pet = import_pet(pet["workstation_result"], case_token, local)
        ref = stored["mr_reference_check"]
        rebuilt = build_bundle(case_token, mr=rebuilt_mr, pet=rebuilt_pet,
                               mr_reference=ref["reference"] if ref else None)
        require(rebuilt == stored, "cohort_feature_recomputation_mismatch")
        if mode == "synthetic_fixture":
            # A separate fixture-only matrix exercises plumbing. Never promote
            # engineering annotations or change D4's false readiness flags.
            from cmvd_demo.perfusion.core import local_warnings
            allowed = {"engineering_fixture_not_anatomical_annotation", "engineering_fixture_not_research_feature"}
            documents = [(b["roi_document"], b["input_curves"]) for b in branches] + list(local.values())
            require(all(d["purpose"] == "engineering_fixture" and set(local_warnings(d, c)) <= allowed
                        for d, c in documents), "cohort_fixture_review_or_purpose")
            require(rebuilt["mr_reference_check"] is not None and rebuilt["mr_reference_check"]["passed"],
                    "cohort_fixture_reference_missing")
            require(all(mr["reviews"].values()) and all(b["parameters"]["window_reviewed"] for b in branches),
                    "cohort_fixture_window_or_pairing")
        else:
            require(rebuilt["qc"]["perfusion_input_ready"] is True, "cohort_perfusion_qc_pending")
        values = {}
        for key, branch in (("pet_mfr", rebuilt_pet), ("mr_mpri", rebuilt_mr)):
            item = branch["measurements"][key]
            require(item["unit"] == "1", "cohort_perfusion_unit")
            values[key] = item["value"]
    except (KeyError, TypeError, IndexError, AttributeError):
        raise CohortError("cohort_feature_structure") from None
    check_cancel(cancel)
    require(all(hashlib.sha256(p.read_bytes()).hexdigest() == sha for p, sha in before),
            "cohort_source_changed_recheck_required")
    return values, before

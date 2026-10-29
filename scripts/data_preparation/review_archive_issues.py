"""Reinspect only issue packages, with explicit byte-verified transient selection.

No source repair, deletion, persistent reader override, labels or training.
Every execution uses new report/audit directories; no automatic resumption.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from cmvd_demo.io import DicomReadError, scan_case
from cmvd_demo.preparation.archives import load_json, write_json, sha256_file, inspect_index, require
from cmvd_demo.preparation.duplicates import select_identical_sops
from cmvd_demo.preparation.qc import branch_inventory, review_status


def review(plan_path):
    plan = load_json(plan_path)
    old_root = Path(plan["report_root"])
    bundle = load_json(old_root/"bundle.json")
    require(bundle.get("completed") is True, "qc_source_run_incomplete")
    for name, digest in bundle["files_sha256"].items():
        require(Path(name).name == name and sha256_file(old_root/name) == digest, "qc_source_bundle_changed")
    old_report = load_json(old_root/"report.json")
    require(old_report.get("completed") is True, "qc_source_report_incomplete")
    targets = set(old_report["archives_with_inspection_issues"])
    require(bool(targets), "qc_no_issue_packages")
    selected = [r for r in plan["archives"] if r["archive_token"] in targets]
    require(len(selected) == len(targets), "qc_package_mapping_mismatch")
    require(all(re.fullmatch(r"ARCH-[0-9a-f]{16}", r["archive_token"]) for r in selected), "qc_invalid_package_token")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    report_root = ROOT/"outputs/archive_qc"/run_id
    restricted = ROOT/"audit/restricted/archive_qc"/run_id
    report_root.mkdir(parents=True, exist_ok=False)
    restricted.mkdir(parents=True, exist_ok=False)
    print("run_id="+run_id, flush=True)
    write_json(restricted/"scope.json", {"source_plan": str(Path(plan_path).resolve()),
               "source_bundle_sha256": sha256_file(old_root/"bundle.json"), "archives": selected})
    last = 0.0

    def progress(stage, done, total):
        nonlocal last
        now = time.monotonic()
        if now-last >= 10 or done == total:
            print(f"{stage} {done}/{total}", flush=True)
            last = now

    results = []
    for index, row in enumerate(selected, 1):
        token = row["archive_token"]
        print(f"package_begin {index}/{len(selected)} {token}", flush=True)
        before = load_json(old_root/(token+".json"))
        require(before.get("extraction_passed") is True, "qc_extraction_not_verified")
        scan = scan_case(row["output"], progress=progress)
        require(scan.enumerated_files == before["extracted_files"], "qc_source_file_count_changed")
        selection = select_identical_sops(scan, progress=progress)
        write_json(restricted/(token+".duplicate_selection.json"), {"groups": selection.groups})
        inspection = inspect_index(selection.scan, restricted, token, progress=progress)
        branches = branch_inventory(selection.scan, inspection)
        result = {"archive_token": token, "before_grid_failures": before["grid_failures"],
                  "duplicate_selection": selection.safe_summary(), "inspection": inspection,
                  "branch_inventory": branches,
                  **review_status(inspection, branches, selection.safe_summary())}
        write_json(report_root/(token+".json"), result)
        results.append(result)
        print(f'package_done {index}/{len(selected)} headers_failed={inspection["header_issue_count"]} grid_failures={len(inspection["grid_failures"])} excluded_copies={selection.safe_summary()["excluded_from_transient_index"]}', flush=True)
    report = {"schema_version": 1, "run_id": run_id, "source_run_id": plan["run_id"],
              "completed": True, "scope": "previous issue packages only; transient duplicate selection; first/last pixels",
              "attempted_packages": len(results), "all_original_packages": len(plan["archives"]),
              "enumerated_files": sum(r["inspection"]["index"]["enumerated_files"] for r in results),
              "header_issues": sum(r["inspection"]["header_issue_count"] for r in results),
              "grids_passed": sum(len(r["inspection"]["grids"]) for r in results),
              "sampled_pixel_planes": sum(r["inspection"]["sampled_pixel_planes"] for r in results),
              "before_grid_failures": sum(len(r["before_grid_failures"]) for r in results),
              "remaining_grid_failures": sum(len(r["inspection"]["grid_failures"]) for r in results),
              "remaining_grid_codes": dict(sorted(Counter(g["code"] for r in results for g in r["inspection"]["grid_failures"]).items())),
              "duplicate_groups": sum(r["duplicate_selection"]["duplicate_groups"] for r in results),
              "conflicting_sop_groups": sum(r["duplicate_selection"]["conflicting_groups"] for r in results),
              "excluded_identical_copies_from_transient_index": sum(r["duplicate_selection"]["excluded_from_transient_index"] for r in results),
              "mainline_check_passed_packages_in_transient_view": [r["archive_token"] for r in results if r["mainline_checks_passed_in_transient_view"]],
              "engineering_clear_packages_in_transient_view": [r["archive_token"] for r in results if r["engineering_clear_in_transient_view"]],
              "remaining_issue_packages": [r["archive_token"] for r in results if not r["engineering_clear_in_transient_view"]],
              "zero_slope_objects_all_indexed_series": sum(r["branch_inventory"]["zero_slope_objects_all_indexed_series"] for r in results),
              "source_files_modified_or_removed": 0, "default_reader_changed": False,
              "all_pixels_checked": False, "pixel_privacy_audit": False, "clinical_validated": False,
              "real_training_performed": False,
              "limits": ["9 issue packages are not a fresh full cohort scan", "no source frame counts or scale factors repaired",
                         "transient index is not applied by desktop or clinical cohort loaders", "header failures remain visible",
                         "first/last pixel checks do not validate every frame", "no patient identity, labels or clinical QC established"]}
    write_json(report_root/"report.json", report)
    # Completion marker last; exceptions above leave partial evidence without it.
    write_json(report_root/"bundle.json", {"schema_version": 1, "completed": True,
               "files_sha256": {p.name: sha256_file(p) for p in sorted(report_root.glob("*.json"))},
               "restricted_files_sha256": {p.name: sha256_file(p) for p in sorted(restricted.glob("*.json"))},
               "clinical_validated": False})
    print(f'completed packages={len(results)} remaining_grid_failures={report["remaining_grid_failures"]}', flush=True)
    return report


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    try:
        review(args.plan)
        return 0
    except Exception as error:
        print("not_completed="+(error.code if isinstance(error, DicomReadError) else type(error).__name__), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

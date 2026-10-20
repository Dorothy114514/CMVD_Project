"""D1 regression on two unpacked development cases; never opens archives.

All output is local controlled research material. SAMPLE-A/B are temporary
aliases, not study IDs or diagnosis labels. No patient index is generated.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cmvd_demo.io import DicomReadError, scan_case, build_grid, build_cine_stack


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def validate_case(alias, case_dir, expected, restricted):
    def progress(stage, done, total):
        print(f"{alias} {stage} {done}/{total}", flush=True)

    scan = scan_case(case_dir, progress=progress)
    result = {"alias": alias, "index": scan.safe_summary(), "grids": [], "failures": [],
              "decoded_unique_objects": 0, "max_grid_array_bytes": 0}
    for issue in scan.issues:
        restricted.append({"alias": alias, **issue.safe_summary(), "path": str(issue.path)})
    if scan.enumerated_files != expected[0] or len(scan.series) != expected[1]:
        result["failures"].append({"code": "sample_structure_changed"})

    sax = [s for s in scan.series if s.kind == "mr_cine" and "sax" in s.tokens]
    selected = [s for s in scan.series if s.kind in
                {"pet_static", "pet_dynamic", "pet_gated", "mr_perfusion", "mr_cine"} and s not in sax]
    jobs = [(s.token, lambda s=s: build_grid(s)) for s in selected]
    if sax:
        jobs.append(("cine_sax_stack", lambda: build_cine_stack(sax)))
    covered = set()
    for index, (token, build) in enumerate(jobs, 1):
        print(f"{alias} grid {index}/{len(jobs)} {token}", flush=True)
        try:
            grid = build()
            pixels = grid.materialize(progress=progress)
            entry = {"token": token, **grid.safe_summary(), "array_bytes": pixels.nbytes,
                     "scaled_pixels_sha256": hashlib.sha256(memoryview(pixels)).hexdigest(),
                     "finite_pixels": bool(np.isfinite(pixels).all())}
            if not entry["finite_pixels"]:
                raise DicomReadError("nonfinite_scaled_pixels")
            if grid.kind == "pet_dynamic" and (grid.shape[:2] != (19, 127) or
                    not np.allclose(grid.frame_durations_s.sum(axis=1), 600)):
                raise DicomReadError("sample_dynamic_protocol_changed")
            if grid.kind == "cine_stack" and grid.shape[:2] != (25, 10):
                raise DicomReadError("sample_cine_protocol_changed")
            if grid.kind == "mr_perfusion" and grid.shape[:2] != (70, 1):
                raise DicomReadError("sample_perfusion_protocol_changed")
            result["grids"].append(entry)
            result["max_grid_array_bytes"] = max(result["max_grid_array_bytes"], pixels.nbytes)
            covered.update(h.file_token for row in grid.cells for h in row)
            del pixels
        except Exception as error:
            result["failures"].append({"grid_token": token, "code": error.code if isinstance(error, DicomReadError)
                                      else "unexpected_" + type(error).__name__})
    changed = []
    for series in scan.series:
        for header in series.objects:
            try:
                stat = header.path.stat()
                if (stat.st_size, stat.st_mtime_ns) != header.fingerprint:
                    changed.append(header.file_token)
            except OSError:
                changed.append(header.file_token)
    result["source_fingerprint_changes"] = changed
    result["decoded_unique_objects"] = len(covered)
    if len(covered) != 7997 or len(result["grids"]) != 18:
        result["failures"].append({"code": "sample_pixel_coverage_changed"})
    result["indexed_not_pixel_decoded_by_kind"] = dict(Counter(
        s.kind for s in scan.series for h in s.objects if h.file_token not in covered))
    result["passed"] = not scan.issues and not result["failures"] and not changed
    print(f"{alias} passed={result['passed']} decoded={len(covered)} failures={len(result['failures'])}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", action="store_true", required=True,
                        help="only the existing one unpacked directory per source group")
    parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = ROOT / "outputs" / "reader_validation" / run_id
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"stage": "D1", "scope": "two_development_samples_only", "cases": [],
              "clinical_validation": False, "pixel_privacy_audit": False,
              "source_write_operations": 0, "archives_opened": 0,
              "limits": ["not_cohort_validation", "no_MBF_or_diagnosis",
                         "unadvertised_missing_whole_frames_or_planes_cannot_always_be_detected",
                         "cine_phase_synchronization_not_performed"]}
    restricted = []
    for alias, folder, expected in [("SAMPLE-A", "\u9633\u6027", (8126, 45)),
                                     ("SAMPLE-B", "\u9634\u6027", (8151, 50))]:
        try:
            directories = [p for p in (ROOT / "data" / folder).iterdir() if p.is_dir() and not p.is_symlink()]
            if len(directories) != 1:
                raise DicomReadError("exactly_one_unpacked_sample_required")
            report["cases"].append(validate_case(alias, directories[0], expected, restricted))
        except Exception as error:
            report["cases"].append({"alias": alias, "passed": False,
                                    "failures": [{"code": error.code if isinstance(error, DicomReadError)
                                                  else "unexpected_" + type(error).__name__}]})
    if restricted:
        (output / "restricted").mkdir()
        dump(output / "restricted" / "failures.json", restricted)
    report["elapsed_seconds"] = round(time.monotonic()-started, 2)
    report["passed"] = all(c["passed"] for c in report["cases"])
    report["completed"] = True
    dump(output / "report.json", report)
    print(f"Report: {output.relative_to(ROOT).as_posix()}/report.json", flush=True)
    print(f"passed={report['passed']} elapsed_seconds={report['elapsed_seconds']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

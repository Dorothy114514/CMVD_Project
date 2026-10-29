"""Controlled archive extraction with preflight and bounded read-only inspection.

Archive tokens identify packages, NOT patients, research IDs or labels. All raw
paths and tool output stay in audit/restricted. No existing image is modified.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import time
import uuid
import zlib

import numpy as np

from cmvd_demo.io import DicomReadError, scan_case, build_grid, build_cine_stack


class ArchiveError(DicomReadError):
    pass


def require(ok, code):
    if not ok:
        raise ArchiveError(code)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, allow_nan=False, indent=2)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def safe_member(name):
    require(isinstance(name, str) and name and not any(ord(c) < 32 for c in name), "archive_invalid_member_path")
    p = PureWindowsPath(name)
    require(not p.is_absolute() and not p.drive and not p.root, "archive_absolute_member_path")
    raw_parts = name.replace("\\", "/").split("/")
    require(all(v not in ("", ".", "..") for v in raw_parts), "archive_member_traversal")
    for part in raw_parts:
        require(not any(c in part for c in ':*?"<>|') and part.rstrip(" .") == part,
                "archive_windows_ambiguous_member")
        require(len(part.encode("utf-16-le")) <= 510, "archive_member_component_too_long")
        stem = part.split(".", 1)[0].upper()
        require(not re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³]|CONIN\$|CONOUT\$", stem),
                "archive_reserved_device_member")
    return "/".join(raw_parts)


def parse_listing(text, *, preserve_duplicate_files=False):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"^----------$", text, maxsplit=1, flags=re.M)
    require(len(parts) == 2, "archive_listing_structure")
    entries, seen = [], {}
    for block in re.split(r"\n\s*\n", parts[1].strip()):
        record = {}
        for line in block.splitlines():
            if " = " in line:
                key, value = line.split(" = ", 1)
            elif line.endswith(" ="):
                key, value = line[:-2], ""  # A legal empty field on the last record.
            else:
                raise ArchiveError("archive_listing_field_structure")
            require(key not in record, "archive_listing_duplicate_field")
            record[key] = value
        require("Path" in record and "Size" in record, "archive_listing_missing_fields")
        member = safe_member(record["Path"])
        key = member.casefold()
        require(record.get("Encrypted", "-") == "-" and record.get("Anti", "-") == "-",
                "archive_encrypted_or_anti_member")
        attrs = record.get("Attributes", "")
        require(not any(k in record for k in ("Symbolic Link", "Hard Link", "Reparse"))
                and "l" not in attrs.split(" ", 1)[-1] and "L" not in attrs,
                "archive_link_member")
        folder = record.get("Folder") == "+" or attrs.startswith("D")
        try:
            size = int(record["Size"])
        except ValueError:
            raise ArchiveError("archive_invalid_member_size") from None
        require(size >= 0 and (not folder or size == 0), "archive_invalid_member_size")
        entry = {"path": member, "folder": folder, "size": size, "crc32": record.get("CRC", "").upper()}
        if key in seen:
            previous = seen[key]
            require(preserve_duplicate_files and not folder and not previous["folder"]
                    and member == previous["path"] and re.fullmatch(r"[0-9A-F]{8}", entry["crc32"])
                    and re.fullmatch(r"[0-9A-F]{8}", previous["crc32"]), "archive_duplicate_member_path")
            previous["duplicate_name"] = entry["duplicate_name"] = True
        seen[key] = entry
        entries.append(entry)
    require(bool(entries), "archive_empty_listing")
    # A file cannot be the ancestor of another member, regardless of order.
    files = {e["path"].casefold() for e in entries if not e["folder"]}
    for e in entries:
        parts = e["path"].casefold().split("/")
        require(all("/".join(parts[:i]) not in files for i in range(1, len(parts))),
                "archive_member_file_directory_conflict")
    return entries


def beneath(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path != root and path.is_relative_to(root), "archive_output_outside_root")
    return path


def run_tool(tool, arguments, log, *, progress=None, timeout=1800):
    """Never print archive names, member names, or tool diagnostics publicly."""
    with Path(log).open("xb") as stream:
        process = subprocess.Popen([str(tool), *map(str, arguments)], stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        started, last = time.monotonic(), 0.
        while True:
            try:
                code = process.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                if progress and now-last >= 15:
                    progress("archive_tool_running", round(now-started), 0)
                    last = now
                if now-started > timeout:
                    process.terminate()  # Our own extractor only; retain partial output.
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise ArchiveError("archive_tool_timeout_partial_retained")
    require(code == 0, "archive_tool_exit_"+str(code))


def create_plan(workspace, output_base, tool, *, progress=None):
    workspace, output_base, tool = Path(workspace).resolve(), Path(output_base).resolve(), Path(tool).resolve()
    require(tool.is_file(), "archive_tool_missing")
    require(not output_base.is_relative_to(workspace), "archive_outputs_require_external_root")
    require(not output_base.is_symlink(), "archive_output_link")
    archives = sorted((workspace/"data").glob("*/*.7z"))
    require(bool(archives), "archive_inputs_missing")
    require(all(p.is_file() and not p.is_symlink() for p in archives), "archive_source_link")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    restricted = workspace/"audit/restricted/archive_extraction"/run_id
    report_dir = workspace/"outputs/archive_validation"/run_id
    target = output_base/run_id
    restricted.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    target.mkdir(parents=True, exist_ok=False)
    plan = {"schema_version": 1, "run_id": run_id, "tool": str(tool), "tool_sha256": sha256_file(tool),
            "output_root": str(target), "restricted_root": str(restricted), "report_root": str(report_dir),
            "archives": [], "preflight_completed": False, "clinical_manifest_created": False,
            "recovery_policy": "No automatic resume/overwrite of attempted archive. Retain partial output; use explicit new run for retries.",
            "source_policy": "data/ and existing unpacked samples read-only; no deidentification or relabeling",
            "inspection_scope": "all file headers; supported grid assembly; first and last plane of every supported grid, not all pixels"}
    safe = []
    for index, archive in enumerate(archives, 1):
        token = "ARCH-"+hashlib.sha256(archive.relative_to(workspace/"data").as_posix().encode()).hexdigest()[:16]
        if progress:
            progress("preflight", index, len(archives))
        row = {"archive_token": token, "source": str(archive), "source_size": archive.stat().st_size,
               "source_mtime_ns": archive.stat().st_mtime_ns,
               "output": str(target/("batch_"+str((index-1)//5+1).zfill(2))/token), "preflight_ok": False}
        try:
            row["source_sha256"] = sha256_file(archive)
            log = restricted/(token+".listing.log")
            run_tool(tool, ["l", "-slt", "-sccUTF-8", "--", archive], log, progress=progress)
            entries = parse_listing(log.read_text(encoding="utf-8-sig"), preserve_duplicate_files=True)
            require(sum(not e["folder"] for e in entries) <= 200000, "archive_file_count_budget")
            total_bytes = sum(e["size"] for e in entries)
            require(total_bytes <= 20*1024**3, "archive_uncompressed_size_budget")
            write_json(restricted/(token+".members.json"), entries)
            require((archive.stat().st_size, archive.stat().st_mtime_ns) == (row["source_size"], row["source_mtime_ns"]),
                    "archive_source_changed_during_preflight")
            row.update(preflight_ok=True, files=sum(not e["folder"] for e in entries),
                       directories=sum(e["folder"] for e in entries), uncompressed_bytes=total_bytes,
                       duplicate_name_entries=sum(e.get("duplicate_name", False) for e in entries),
                       extraction_collision_policy="auto_rename_preserve_both" if any(e.get("duplicate_name") for e in entries) else "no_overwrite")
        except (ArchiveError, OSError, UnicodeError) as error:
            row["failure"] = error.code if isinstance(error, ArchiveError) else type(error).__name__
        plan["archives"].append(row)
        safe.append({k: v for k, v in row.items() if k in
                     ("archive_token", "source_size", "source_sha256", "preflight_ok", "files", "directories", "uncompressed_bytes", "duplicate_name_entries", "extraction_collision_policy", "failure")})
    expected_bytes = sum(r.get("uncompressed_bytes", 0) for r in plan["archives"])
    files = sum(r.get("files", 0) for r in plan["archives"])
    free = shutil.disk_usage(target).free
    plan["space_ok"] = free >= expected_bytes+files*4096+20*1024**3
    plan["expected_uncompressed_bytes"], plan["free_bytes_at_plan"] = expected_bytes, free
    plan["preflight_completed"] = True
    write_json(restricted/"plan.json", plan)
    write_json(report_dir/"preflight.json", {"run_id": run_id, "completed": True, "archive_count": len(archives),
        "archives": safe, "expected_files": files, "expected_uncompressed_bytes": expected_bytes,
        "space_ok": plan["space_ok"], "all_preflight_passed": all(r["preflight_ok"] for r in safe),
        "study_ids_or_labels_generated": False})
    return restricted/"plan.json"


def verify_tree(target, members, *, resolution=None):
    expected = {e["path"].casefold(): e["size"] for e in members if not e["folder"]}
    actual = {}
    for base, dirs, names in os.walk(target, followlinks=False):
        for name in dirs+names:
            path = Path(base)/name
            stat = path.lstat()
            require(not path.is_symlink() and not (getattr(stat, "st_file_attributes", 0) & 0x400),
                    "archive_extracted_reparse_point")
            beneath(path, target)
        for name in names:
            path = Path(base)/name
            key = path.relative_to(target).as_posix().casefold()
            require(key not in actual, "archive_extracted_duplicate_path")
            actual[key] = path.stat().st_size
    duplicates = defaultdict(list)
    for entry in members:
        if entry.get("duplicate_name"):
            duplicates[entry["path"].casefold()].append(entry)
    require(set(expected) <= set(actual) and all(actual[k] == v for k, v in expected.items() if k not in duplicates),
            "archive_extracted_file_set_or_size_mismatch")
    extras = set(actual)-set(expected)
    require(len(extras) == sum(len(g)-1 for g in duplicates.values()), "archive_extracted_file_set_or_size_mismatch")
    if not duplicates:
        return len(actual), sum(actual.values())
    actual_paths = {p.relative_to(target).as_posix().casefold(): p for p in Path(target).rglob('*') if p.is_file()}
    for original, group in duplicates.items():
        candidates = [original]+sorted(k for k in extras if PureWindowsPath(k).parent == PureWindowsPath(original).parent)
        expected_crc = Counter((g["size"], g["crc32"]) for g in group)
        matched = []
        for key in candidates:
            crc = 0
            with actual_paths[key].open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024**2), b''):
                    crc = zlib.crc32(chunk, crc)
            value = (actual[key], f'{crc:08X}')
            if expected_crc[value]:
                expected_crc[value] -= 1
                matched.append(key)
                if resolution is not None:
                    resolution.append({"original_member": group[0]["path"],
                        "extracted_relative_path": actual_paths[key].relative_to(target).as_posix(),
                        "size": actual[key], "crc32": value[1]})
        require(not any(expected_crc.values()) and original in matched, "archive_duplicate_contents_not_preserved")
        extras.difference_update(matched)
    require(not extras, "archive_unexpected_extracted_files")
    return len(actual), sum(actual.values())


def inspect_case(target, restricted, token, *, progress=None):
    scan = scan_case(target, progress=progress)
    return inspect_index(scan, restricted, token, progress=progress)


def inspect_index(scan, restricted, token, *, progress=None):
    """Inspect an explicit in-memory index using the same strict reader rules."""
    result = {"archive_token": token, "index": scan.safe_summary(), "grids": [], "grid_failures": [],
              "sampled_pixel_planes": 0, "all_pixels_checked": False,
              "clinical_validated": False, "pixel_privacy_audit": False}
    if scan.issues:
        write_json(restricted/(token+".header_issues.json"), [{**i.safe_summary(), "path": str(i.path)} for i in scan.issues])
    kinds = {"pet_static", "pet_dynamic", "pet_gated", "mr_perfusion", "mr_cine"}
    jobs = [(s.token, lambda s=s: build_grid(s)) for s in scan.series if s.kind in kinds]
    sax = defaultdict(list)
    for s in scan.series:
        if s.kind == "mr_cine" and "sax" in s.tokens:
            h = s.objects[0]
            sax[(h.study_uid, h.orientation, h.rows, h.columns, h.pixel_spacing)].append(s)
    for index, group in enumerate(sax.values()):
        jobs.append(("cine_sax_group_"+str(index+1), lambda group=group: build_cine_stack(group)))
    for index, (grid_token, build) in enumerate(jobs, 1):
        if progress:
            progress("grid_inspection", index, len(jobs))
        try:
            grid = build()
            summary = {"grid_token": grid_token, **grid.safe_summary(), "pixel_checks": []}
            nt, nz, nr, nc = grid.shape
            require(nr*nc*4 <= 16*1024**2, "archive_plane_memory_budget")
            for t, z in dict.fromkeys(((0, 0), (nt-1, nz-1))):
                pixels = grid.read_plane(t, z)
                require(bool(np.isfinite(pixels).all()), "archive_nonfinite_sample_pixels")
                summary["pixel_checks"].append({"time_index": t, "slice_index": z, "shape": list(pixels.shape),
                    "sha256": hashlib.sha256(memoryview(pixels)).hexdigest()})
                result["sampled_pixel_planes"] += 1
            result["grids"].append(summary)
        except Exception as error:
            result["grid_failures"].append({"grid_token": grid_token,
                "code": error.code if isinstance(error, DicomReadError) else "unexpected_"+type(error).__name__})
    result["indexed_not_mainline_objects"] = sum(len(s.objects) for s in scan.series if s.kind not in kinds)
    result["header_issue_count"] = len(scan.issues)
    result["inspection_passed"] = not scan.issues and not result["grid_failures"] and bool(jobs)
    result["study_tokens"] = sorted({hashlib.sha256(h.study_uid.encode()).hexdigest()[:16]
        for s in scan.series for h in s.objects})
    return result


def execute_batch(plan_path, *, start, count, progress=None):
    plan = load_json(plan_path)
    require(plan.get("preflight_completed") is True and plan.get("space_ok") is True, "archive_plan_not_ready")
    tool = Path(plan["tool"])
    require(sha256_file(tool) == plan["tool_sha256"], "archive_tool_changed")
    restricted, report_dir, target_root = map(Path, (plan["restricted_root"], plan["report_root"], plan["output_root"]))
    require(type(start) is int and type(count) is int and start >= 0 and count > 0
            and start < len(plan["archives"]), "archive_batch_range")
    selected = plan["archives"][start:start+count]
    require(shutil.disk_usage(target_root).free >= sum(r.get("uncompressed_bytes", 0)+r.get("files", 0)*4096 for r in selected)+10*1024**3,
            "archive_batch_disk_space")
    # Reject the whole requested batch before changing anything if a target was
    # ever attempted. Recovery never silently reuses a partially extracted case.
    for row in selected:
        require(not Path(row["output"]).exists() and not (report_dir/(row["archive_token"]+".json")).exists(),
                "archive_existing_attempt_refuses_overwrite")
        beneath(row["output"], target_root)
    for index, row in enumerate(selected, start+1):
        token, source, target = row["archive_token"], Path(row["source"]), Path(row["output"])
        started = time.monotonic()
        if progress:
            progress("archive_begin", index, len(plan["archives"]))
        result = {"archive_token": token, "extraction_passed": False, "inspection_passed": False, "failures": []}
        try:
            require(row["preflight_ok"], "archive_preflight_failed_not_extracted")
            require((source.stat().st_size, source.stat().st_mtime_ns) == (row["source_size"], row["source_mtime_ns"])
                    and sha256_file(source) == row["source_sha256"], "archive_source_changed_before_extract")
            target.mkdir(parents=True, exist_ok=False)
            collision = "-aou" if row.get("duplicate_name_entries") else "-aos"
            run_tool(tool, ["x", collision, "-y", "-bd", "-bb0", "-bsp0", "-sccUTF-8", "-mmt=2",
                "-o"+str(target), "--", source], restricted/(token+".extract.log"), progress=progress)
            members = load_json(restricted/(token+".members.json"))
            resolution = []
            files, size = verify_tree(target, members, resolution=resolution)
            if resolution:
                write_json(restricted/(token+".duplicate_resolution.json"), resolution)
            require(sha256_file(source) == row["source_sha256"], "archive_source_changed_during_extract")
            result.update(extraction_passed=True, extracted_files=files, extracted_bytes=size,
                          compressed_source_sha256_unchanged=True, duplicate_name_entries=row.get("duplicate_name_entries", 0),
                          extraction_collision_policy=row.get("extraction_collision_policy", "no_overwrite"))
            inspection = inspect_case(target, restricted, token, progress=progress)
            result.update(inspection)
        except Exception as error:
            result["failures"].append({"code": error.code if isinstance(error, DicomReadError) else "unexpected_"+type(error).__name__})
        result["elapsed_seconds"] = round(time.monotonic()-started, 2)
        write_json(report_dir/(token+".json"), result)
        if progress:
            progress("archive_done", index, len(plan["archives"]))
    aggregate = summarize(plan)
    write_json(report_dir/(f"batch-{start:03d}-{start+len(selected):03d}.json"), aggregate)
    return aggregate


def summarize(plan):
    report_dir = Path(plan["report_root"])
    rows = [load_json(report_dir/(r["archive_token"]+".json")) for r in plan["archives"]
            if (report_dir/(r["archive_token"]+".json")).is_file()]
    studies = defaultdict(list)
    for row in rows:
        for token in row.get("study_tokens", []):
            studies[token].append(row["archive_token"])
    return {"schema_version": 1, "scope": plan["inspection_scope"], "run_id": plan["run_id"],
        "archive_count": len(plan["archives"]), "attempted": len(rows), "completed": len(rows) == len(plan["archives"]),
        "extraction_passed": sum(r["extraction_passed"] for r in rows),
        "inspection_passed": sum(r["inspection_passed"] for r in rows),
        "extracted_files": sum(r.get("extracted_files", 0) for r in rows),
        "extracted_bytes": sum(r.get("extracted_bytes", 0) for r in rows),
        "indexed_objects": sum(r.get("index", {}).get("indexed_objects", 0) for r in rows),
        "header_issues": sum(r.get("header_issue_count", 0) for r in rows),
        "grid_failures": sum(len(r.get("grid_failures", [])) for r in rows),
        "sampled_pixel_planes": sum(r.get("sampled_pixel_planes", 0) for r in rows),
        "archives_with_execution_failure": [r["archive_token"] for r in rows if r["failures"]],
        "archives_with_inspection_issues": [r["archive_token"] for r in rows if not r["inspection_passed"]],
        "shared_study_token_groups": [v for v in studies.values() if len(v) > 1],
        "limits": ["archive count is not patient count", "no clinical labels or old research IDs generated",
                   "header read failures remain unclassified", "first/last pixels only, not all pixels",
                   "no deidentification, pixel privacy review, measurement QC or clinical validation"],
        "source_write_operations": 0, "clinical_validated": False, "pixel_privacy_audit": False}

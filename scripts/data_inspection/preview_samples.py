"""Read-only inspection of the one unpacked sample in each data group.

Never opens archives or writes to data/. Reports and a few native pixel previews
stay under ignored outputs/. Free-text descriptions and patient identity are not
copied into reports. This is not a privacy audit or clinical validation.
"""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import time
import uuid
import warnings
import zlib

import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parents[2]
TAGS = [
    "Modality", "SOPClassUID", "SeriesInstanceUID", "SeriesNumber",
    "SeriesDescription", "ProtocolName", "SequenceName", "Rows", "Columns",
    "NumberOfFrames", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
    "ImageOrientationPatient", "ImagePositionPatient", "TemporalPositionIdentifier",
    "NumberOfTemporalPositions", "TriggerTime", "CardiacNumberOfImages",
    "AcquisitionNumber", "AcquisitionTime", "InstanceNumber", "FrameReferenceTime", "ActualFrameDuration", "Units",
    "DecayCorrection", "CorrectedImage", "RescaleSlope", "RescaleIntercept",
    "RadiopharmaceuticalInformationSequence", "ImageType", "BurnedInAnnotation",
]
TOKENS = ["rest", "stress", "dynamic", "dyn", "static", "gated", "nh3", "ammonia",
          "cine", "sax", "short", "2ch", "4ch", "trufi", "tfl", "perf", "first",
          "t1", "t2", "molli", "mapping", "psir", "lge", "scout", "localizer"]


def vector(ds, name):
    value = ds.get(name)
    return tuple(round(float(v), 6) for v in value) if value is not None else None


def numeric(ds, name):
    value = ds.get(name)
    return float(value) if value is not None else None


def acquisition_seconds(ds):
    value = str(ds.get("AcquisitionTime", "") or "")
    if len(value) < 6:
        return None
    return int(value[:2]) * 3600 + int(value[2:4]) * 60 + float(value[4:])


def write_png(path, array):
    """Minimal lossless serialization for local grayscale pixel inspection."""
    array = np.ascontiguousarray(array, dtype=np.uint8)
    height, width = array.shape
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    scanlines = b"".join(b"\0" + row.tobytes() for row in array)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


def inspect_sample(case_dir, sample_code, run_dir):
    paths = sorted(p for p in case_dir.rglob("*") if p.is_file())
    series = {}
    failures = Counter()
    inspection_failures = []
    started = time.monotonic()
    for index, path in enumerate(paths, 1):
        header_loaded = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=TAGS)
            header_loaded = True
            key = str(ds.get("SeriesInstanceUID", "missing"))
            if key not in series:
                description = " ".join(str(ds.get(k, "")) for k in ("SeriesDescription", "ProtocolName", "SequenceName")).lower()
                radiopharm = ds.get("RadiopharmaceuticalInformationSequence", [])
                tracer = radiopharm[0] if radiopharm else None
                series[key] = {
                    "series_number": int(ds.SeriesNumber) if ds.get("SeriesNumber") is not None else None,
                    "modality": str(ds.get("Modality", "unknown")),
                    "protocol_tokens": [token for token in TOKENS if token in description],
                    "sop_class": str(ds.get("SOPClassUID", "").name) if "SOPClassUID" in ds else None,
                    "units": str(ds.get("Units", "")),
                    "decay_correction": str(ds.get("DecayCorrection", "")),
                    "tracer_half_life_s": numeric(tracer, "RadionuclideHalfLife") if tracer is not None else None,
                    "tracer_dose_present": bool(tracer is not None and "RadionuclideTotalDose" in tracer),
                    "tracer_start_time_present": bool(tracer is not None and ("RadiopharmaceuticalStartTime" in tracer or "RadiopharmaceuticalStartDateTime" in tracer)),
                    "records": [],
                }
            ts = ds.file_meta.get("TransferSyntaxUID")
            record = {
                "path": path,
                "shape": (int(ds.get("Rows") or 0), int(ds.get("Columns") or 0)),
                "frames": int(ds.get("NumberOfFrames") or 1),
                "pixel_spacing_mm": vector(ds, "PixelSpacing"),
                "slice_thickness_mm": numeric(ds, "SliceThickness"),
                "spacing_between_slices_mm": numeric(ds, "SpacingBetweenSlices"),
                "position": vector(ds, "ImagePositionPatient"),
                "orientation": vector(ds, "ImageOrientationPatient"),
                "trigger_ms": numeric(ds, "TriggerTime"),
                "temporal_position": numeric(ds, "TemporalPositionIdentifier"),
                "declared_temporal_positions": numeric(ds, "NumberOfTemporalPositions"),
                "cardiac_images": numeric(ds, "CardiacNumberOfImages"),
                "acquisition_number": numeric(ds, "AcquisitionNumber"),
                "acquisition_clock_s": acquisition_seconds(ds),
                "instance_number": numeric(ds, "InstanceNumber"),
                "frame_reference_ms": numeric(ds, "FrameReferenceTime"),
                "frame_duration_ms": numeric(ds, "ActualFrameDuration"),
                "rescale_slope": numeric(ds, "RescaleSlope"),
                "rescale_intercept": numeric(ds, "RescaleIntercept"),
                "compressed": bool(ts.is_compressed) if ts is not None else None,
                "burned_in_annotation": str(ds.get("BurnedInAnnotation", "absent")),
            }
            series[key]["records"].append(record)
        except Exception as error:
            # Do not print raw exceptions, paths, patient names, dates or UIDs.
            if header_loaded:
                inspection_failures.append({"file_ordinal": index, "error_type": type(error).__name__})
            else:
                failures[type(error).__name__] += 1
        if index % 1000 == 0 or index == len(paths):
            print(f"{sample_code}: headers {index}/{len(paths)}, read_failures={sum(failures.values())}, inspection_failures={len(inspection_failures)}, elapsed_s={time.monotonic()-started:.1f}", flush=True)

    summaries = []
    ordered = sorted(series.values(), key=lambda s: (s["series_number"] is None, s["series_number"] or 0))
    for ordinal, item in enumerate(ordered, 1):
        records = item["records"]
        summary = {k: v for k, v in item.items() if k != "records"}
        summary["series_code"] = f"{sample_code}-S{ordinal:02d}"
        summary["file_count"] = len(records)
        summary["total_pixel_frames_declared"] = sum(r["frames"] for r in records)
        for key in ("shape", "frames", "pixel_spacing_mm", "slice_thickness_mm", "spacing_between_slices_mm", "declared_temporal_positions", "cardiac_images", "frame_duration_ms", "rescale_slope", "rescale_intercept", "compressed", "burned_in_annotation"):
            values = sorted({r[key] for r in records if r[key] is not None})
            summary[key + "_values"] = values
        for key in ("position", "orientation", "trigger_ms", "temporal_position", "acquisition_number", "frame_reference_ms"):
            values = {r[key] for r in records if r[key] is not None}
            summary[key + "_unique_count"] = len(values)
            summary[key + "_missing_count"] = sum(r[key] is None for r in records)
            if key in ("trigger_ms", "frame_reference_ms") and values:
                summary[key + "_range"] = [min(values), max(values)]
            if key != "acquisition_number":
                summary[key + "_values"] = sorted(values)
        clocks = sorted({r["acquisition_clock_s"] for r in records if r["acquisition_clock_s"] is not None})
        summary["relative_acquisition_times_s"] = [round(t-clocks[0], 6) for t in clocks] if clocks else []
        if summary["frame_reference_ms_unique_count"] > 1:
            cells = Counter((r["position"], r["frame_reference_ms"]) for r in records)
            summary["dynamic_grid"] = {
                "expected_cells": summary["position_unique_count"] * summary["frame_reference_ms_unique_count"],
                "observed_cells": len(cells),
                "duplicate_objects": sum(count-1 for count in cells.values()),
                "frame_schedule_ms": sorted({(r["frame_reference_ms"], r["frame_duration_ms"]) for r in records}),
            }
        summaries.append(summary)

    # Only representative native MR/PT objects; never render secondary captures/reports.
    selected = []
    selection_specs = [
        ("PT", lambda s: s["file_count"] > 500),
        ("PT", lambda s: s["file_count"] > 500),
        ("MR", lambda s: "cine" in s["protocol_tokens"] and "sax" in s["protocol_tokens"]),
        ("MR", lambda s: "perf" in s["protocol_tokens"] or "tfl" in s["protocol_tokens"]),
    ]
    previews = []
    for modality, predicate in selection_specs:
        candidates = [i for i, summary in enumerate(summaries) if summary["modality"] == modality and predicate(summary) and i not in selected and summary["sop_class"] in ("MR Image Storage", "Positron Emission Tomography Image Storage")]
        if not candidates:
            continue
        chosen = max(candidates, key=lambda i: summaries[i]["file_count"])
        selected.append(chosen)
        summary = summaries[chosen]
        records = ordered[chosen]["records"]
        record = records[len(records) // 2]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = pydicom.dcmread(record["path"])
            pixels = ds.pixel_array
            if pixels.ndim == 3 and pixels.shape[-1] not in (3, 4):
                pixels = pixels[pixels.shape[0] // 2]
            if pixels.ndim != 2:
                raise ValueError("Only native monochrome 2D previews supported")
            pixels = pixels.astype(np.float64)
            finite = pixels[np.isfinite(pixels)]
            lower, upper = np.percentile(finite, [1, 99.5])
            rendered = np.clip((pixels-lower) / max(upper-lower, 1e-12), 0, 1)
            filename = summary["series_code"] + ".png"
            write_png(run_dir / filename, np.rint(rendered * 255).astype(np.uint8))
            previews.append({"series_code": summary["series_code"], "file": filename, "status": "rendered", "scaling": "per-image percentile display only, not quantitative"})
        except Exception as error:
            previews.append({"series_code": summary["series_code"], "status": "failed", "error_type": type(error).__name__})
    return {"sample_code": sample_code, "files_enumerated": len(paths), "headers_summarized": sum(s["file_count"] for s in summaries), "header_read_failures_unclassified": dict(failures), "inspection_failures": inspection_failures, "series": summaries, "previews": previews}


def main():
    run_dir = ROOT / "outputs/sample_preview" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"inspector_version": 2, "scope": "two unpacked directory samples only; archives not opened", "privacy": "controlled raw study derivatives, not deidentified or pixel-privacy audited", "samples": []}
    for group_name, sample_code in (("\u9633\u6027", "SAMPLE-P"), ("\u9634\u6027", "SAMPLE-N")):
        group = ROOT / "data" / group_name
        cases = [p for p in group.iterdir() if p.is_dir()]
        if len(cases) != 1:
            raise ValueError("Expected exactly one unpacked sample per group; no case was auto-selected")
        report["samples"].append(inspect_sample(cases[0], sample_code, run_dir))
    report["scan_status"] = "completed_for_two_samples"
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Two-sample report: {run_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()

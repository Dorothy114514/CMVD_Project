"""Classic single-frame MR/PET reader; no mutation, normalization or diagnosis.

Coordinates follow DICOM LPS. Arrays use [time, slice, row, column]. A cine
stack retains each slice's actual trigger times; phase synchronization is a
later operation. Raw paths, dates, UIDs and free-text descriptions stay private
to in-memory records and are excluded from safe summaries.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from pathlib import Path
import time
from typing import Callable
import warnings

import numpy as np
import pydicom

MR = "1.2.840.10008.5.1.4.1.1.4"
PET = "1.2.840.10008.5.1.4.1.1.128"
SC = "1.2.840.10008.5.1.4.1.1.7"
ARCHIVES = {".7z", ".zip", ".rar", ".tar", ".gz"}
TOKENS = ("rest", "stress", "dynamic", "dyn", "gated", "static", "cine", "sax", "perf", "tfl", "t1", "t2", "psir", "molli")
TAGS = [
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "SOPClassUID",
    "FrameOfReferenceUID", "Modality", "SeriesNumber", "SeriesDescription",
    "ProtocolName", "SequenceName", "ImageType", "Rows", "Columns",
    "NumberOfFrames", "SamplesPerPixel", "PhotometricInterpretation",
    "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
    "SliceThickness", "SpacingBetweenSlices", "InstanceNumber",
    "TriggerTime", "NominalInterval", "CardiacNumberOfImages",
    "TemporalPositionIdentifier", "NumberOfTemporalPositions",
    "NumberOfTimeSlices", "NumberOfTimeSlots", "NumberOfSlices",
    "AcquisitionDate", "AcquisitionTime", "AcquisitionDateTime",
    "FrameReferenceTime", "ActualFrameDuration", "Units", "DecayCorrection",
    "CorrectedImage", "RescaleSlope", "RescaleIntercept",
    "RadiopharmaceuticalInformationSequence", "BurnedInAnnotation",
]


class DicomReadError(ValueError):
    """Safe, stable error code; never embeds source values or paths."""

    def __init__(self, code, file_token=None):
        self.code = code
        self.file_token = file_token
        super().__init__(code + (":" + file_token if file_token else ""))


class ReadCancelled(DicomReadError):
    def __init__(self):
        super().__init__("cancelled")


def _cancel(cancel):
    if cancel is not None and cancel():
        raise ReadCancelled()


def _token(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _number(ds, name):
    value = ds.get(name)
    if value is None or value == "":
        return None
    result = float(value)
    if not np.isfinite(result):
        raise DicomReadError("nonfinite_metadata")
    return result


def _vector(ds, name, length):
    value = ds.get(name)
    if value is None:
        return None
    result = tuple(float(v) for v in value)
    if len(result) != length or not np.isfinite(result).all():
        raise DicomReadError("invalid_geometry")
    return result


def _acquisition_time(ds):
    """Private absolute scalar for ordering; only relative times are exported."""
    dt = str(ds.get("AcquisitionDateTime", "") or "")
    date = str(ds.get("AcquisitionDate", "") or "")
    clock = str(ds.get("AcquisitionTime", "") or "")
    if dt:
        # An offset is constant within an accepted series; differing offsets are
        # not supported. Preserve offset text for consistency checks in memory.
        date, clock = dt[:8], dt[8:]
        offset = clock[clock.find("+"):] if "+" in clock else (clock[clock.find("-"):] if "-" in clock else "")
        if offset:
            clock = clock[:-len(offset)]
    else:
        offset = ""
    if len(clock) < 6:
        return None, bool(date), offset
    hour, minute, second = int(clock[:2]), int(clock[2:4]), float(clock[4:])
    if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
        raise DicomReadError("invalid_acquisition_time")
    day = datetime.strptime(date, "%Y%m%d").toordinal() if date else 0
    return day * 86400_000000 + hour * 3600_000000 + minute * 60_000000 + round(second * 1_000000), bool(date), offset


@dataclass(frozen=True)
class Issue:
    file_token: str
    stage: str
    code: str
    error_type: str
    path: Path = field(repr=False)

    def safe_summary(self):
        return {k: getattr(self, k) for k in ("file_token", "stage", "code", "error_type")}


@dataclass(frozen=True)
class ObjectHeader:
    file_token: str
    path: Path = field(repr=False)
    fingerprint: tuple = field(repr=False)
    study_uid: str = field(repr=False)
    series_uid: str = field(repr=False)
    sop_uid: str = field(repr=False)
    frame_of_reference_uid: str = field(repr=False)
    sop_class: str
    modality: str
    series_number: int | None
    tokens: tuple
    rows: int
    columns: int
    number_of_frames: int
    samples_per_pixel: int
    photometric: str
    position: tuple | None
    orientation: tuple | None
    pixel_spacing: tuple | None
    slice_thickness_mm: float | None
    declared_slice_spacing_mm: float | None
    trigger_s: float | None
    nominal_interval_s: float | None
    temporal_position: float | None
    declared_temporal_positions: float | None
    declared_cardiac_images: float | None
    declared_time_slices: float | None
    declared_time_slots: float | None
    declared_slices: float | None
    acquisition_time: int | None = field(repr=False)
    acquisition_has_date: bool
    acquisition_offset: str = field(repr=False)
    frame_reference_s: float | None
    frame_duration_s: float | None
    slope: float | None
    intercept: float | None
    units: str
    decay_correction: str
    corrected_image: tuple
    half_life_s: float | None
    injected_dose_bq: float | None = field(repr=False)
    injection_start: str | None = field(repr=False)
    transfer_syntax: str
    burned_in_annotation: str


def _header(ds, path, token):
    required = ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "SOPClassUID")
    if any(not str(ds.get(key, "") or "") for key in required):
        raise DicomReadError("missing_identity_uid", token)
    descriptions = " ".join(str(ds.get(key, "") or "") for key in ("SeriesDescription", "ProtocolName", "SequenceName")).lower()
    number = _number(ds, "SeriesNumber")
    acquisition, has_date, offset = _acquisition_time(ds)
    tracer = ds.get("RadiopharmaceuticalInformationSequence")
    tracer = tracer[0] if tracer else None
    stat = path.stat()
    ms = lambda name: _number(ds, name) / 1000 if _number(ds, name) is not None else None
    return ObjectHeader(
        file_token=token, path=path, fingerprint=(stat.st_size, stat.st_mtime_ns),
        study_uid=str(ds.StudyInstanceUID), series_uid=str(ds.SeriesInstanceUID), sop_uid=str(ds.SOPInstanceUID),
        frame_of_reference_uid=str(ds.get("FrameOfReferenceUID", "") or ""),
        sop_class=str(ds.SOPClassUID), modality=str(ds.get("Modality", "") or ""),
        series_number=int(number) if number is not None else None,
        tokens=tuple(t for t in TOKENS if t in descriptions),
        rows=int(ds.get("Rows") or 0), columns=int(ds.get("Columns") or 0),
        number_of_frames=int(ds.get("NumberOfFrames") or 1), samples_per_pixel=int(ds.get("SamplesPerPixel") or 1),
        photometric=str(ds.get("PhotometricInterpretation", "") or ""),
        position=_vector(ds, "ImagePositionPatient", 3), orientation=_vector(ds, "ImageOrientationPatient", 6),
        pixel_spacing=_vector(ds, "PixelSpacing", 2), slice_thickness_mm=_number(ds, "SliceThickness"),
        declared_slice_spacing_mm=_number(ds, "SpacingBetweenSlices"), trigger_s=ms("TriggerTime"),
        nominal_interval_s=ms("NominalInterval"), temporal_position=_number(ds, "TemporalPositionIdentifier"),
        declared_temporal_positions=_number(ds, "NumberOfTemporalPositions"),
        declared_cardiac_images=_number(ds, "CardiacNumberOfImages"),
        declared_time_slices=_number(ds, "NumberOfTimeSlices"),
        declared_time_slots=_number(ds, "NumberOfTimeSlots"), declared_slices=_number(ds, "NumberOfSlices"),
        acquisition_time=acquisition, acquisition_has_date=has_date, acquisition_offset=offset,
        frame_reference_s=ms("FrameReferenceTime"), frame_duration_s=ms("ActualFrameDuration"),
        slope=_number(ds, "RescaleSlope"), intercept=_number(ds, "RescaleIntercept"),
        units=str(ds.get("Units", "") or ""), decay_correction=str(ds.get("DecayCorrection", "") or ""),
        corrected_image=tuple(str(v) for v in ds.get("CorrectedImage", [])),
        half_life_s=_number(tracer, "RadionuclideHalfLife") if tracer is not None else None,
        injected_dose_bq=_number(tracer, "RadionuclideTotalDose") if tracer is not None else None,
        injection_start=str(tracer.get("RadiopharmaceuticalStartDateTime") or tracer.get("RadiopharmaceuticalStartTime") or "") if tracer is not None else None,
        transfer_syntax=str(ds.file_meta.get("TransferSyntaxUID", "")),
        burned_in_annotation=str(ds.get("BurnedInAnnotation", "absent") or "absent"),
    )


@dataclass(frozen=True)
class Series:
    token: str
    objects: tuple[ObjectHeader, ...] = field(repr=False)
    kind: str
    condition_candidate: str

    @property
    def tokens(self):
        return tuple(sorted({t for h in self.objects for t in h.tokens}))

    def safe_summary(self):
        return {"series_token": self.token, "kind": self.kind,
                "condition_candidate": self.condition_candidate,
                "protocol_tokens": self.tokens, "object_count": len(self.objects),
                "series_number": self.objects[0].series_number}


def _series(key, objects):
    classes = {h.sop_class for h in objects}
    tokens = {t for h in objects for t in h.tokens}
    conditions = tokens & {"rest", "stress"}
    condition = next(iter(conditions)) if len(conditions) == 1 else "unknown"
    if any(h.number_of_frames != 1 for h in objects):
        kind = "unsupported_multiframe"
    elif all(uid == SC or uid.startswith(SC + ".") for uid in classes):
        kind = "secondary_capture"
    elif classes == {PET} and all(h.modality == "PT" for h in objects):
        refs = {h.frame_reference_s for h in objects if h.frame_reference_s is not None}
        triggers = {h.trigger_s for h in objects if h.trigger_s is not None}
        kind = "pet_dynamic" if len(refs) > 1 else ("pet_gated" if len(triggers) > 1 or "gated" in tokens else "pet_static")
    elif classes == {MR} and all(h.modality == "MR" for h in objects):
        if "cine" in tokens:
            kind = "mr_cine"
        elif "perf" in tokens or (("dyn" in tokens or "dynamic" in tokens) and condition != "unknown"):
            kind = "mr_perfusion"
        else:
            kind = "mr_other"
    else:
        kind = "unsupported_sop_class"
    return Series(_token("|".join(key)), tuple(objects), kind, condition)


@dataclass(frozen=True)
class ScanResult:
    enumerated_files: int
    ignored_archives: int
    series: tuple[Series, ...]
    issues: tuple[Issue, ...]

    def safe_summary(self):
        return {"enumerated_files": self.enumerated_files, "ignored_archives": self.ignored_archives,
                "indexed_objects": sum(len(s.objects) for s in self.series),
                "series_count": len(self.series), "issues": [i.safe_summary() for i in self.issues],
                "series": [s.safe_summary() for s in self.series]}


def scan_case(case_dir, *, progress: Callable | None = None, cancel: Callable | None = None):
    """Index one explicit case directory; never opens archives or derives labels.

    progress(stage, done, total) receives counts, not identity. Files are read
    without force=True; read failures remain unclassified, not 'non-DICOM'.
    """
    case_dir = Path(case_dir).resolve()
    if not case_dir.is_dir():
        raise DicomReadError("case_directory_missing")
    paths, last_enumeration_progress = [], 0.0
    for path in case_dir.rglob("*"):
        _cancel(cancel)
        if path.is_file() and not path.is_symlink():
            paths.append(path)
        now = time.monotonic()
        if progress is not None and now-last_enumeration_progress >= 1:
            progress("enumerate", len(paths), 0)
            last_enumeration_progress = now
    paths.sort()
    _cancel(cancel)
    groups, issues = defaultdict(list), []
    ignored, last_progress = 0, 0.0
    for done, path in enumerate(paths, 1):
        _cancel(cancel)
        file_token = _token(path.relative_to(case_dir).as_posix())
        if path.suffix.lower() in ARCHIVES:
            ignored += 1
        else:
            stage = "header_read"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=TAGS)
                stage = "metadata_parse"
                header = _header(ds, path, file_token)
                groups[(header.study_uid, header.series_uid)].append(header)
            except Exception as error:
                code = error.code if isinstance(error, DicomReadError) else ("read_failed_unclassified" if stage == "header_read" else "metadata_parse_failed")
                issues.append(Issue(file_token, stage, code, type(error).__name__, path))
        now = time.monotonic()
        if progress is not None and (now-last_progress >= 1 or done == len(paths)):
            progress("index", done, len(paths))
            last_progress = now
    indexed = tuple(sorted((_series(k, v) for k, v in groups.items()), key=lambda s: (s.objects[0].series_number is None, s.objects[0].series_number or 0, s.token)))
    _cancel(cancel)
    return ScanResult(len(paths), ignored, indexed, tuple(issues))


def _validate_geometry(headers):
    first = headers[0]
    if first.orientation is None or first.position is None or first.pixel_spacing is None:
        raise DicomReadError("missing_geometry")
    orientation = np.asarray(first.orientation)
    column_direction, row_direction = orientation[:3], orientation[3:]
    if not (np.isclose(np.linalg.norm(column_direction), 1, atol=1e-4) and np.isclose(np.linalg.norm(row_direction), 1, atol=1e-4) and abs(np.dot(column_direction, row_direction)) < 1e-4):
        raise DicomReadError("invalid_orientation")
    if first.rows <= 0 or first.columns <= 0 or min(first.pixel_spacing) <= 0:
        raise DicomReadError("invalid_geometry")
    for h in headers:
        if h.position is None or h.orientation is None or h.pixel_spacing is None:
            raise DicomReadError("missing_geometry", h.file_token)
        if (h.rows, h.columns) != (first.rows, first.columns) or not np.allclose(h.orientation, first.orientation, atol=1e-5, rtol=0) or not np.allclose(h.pixel_spacing, first.pixel_spacing, atol=1e-5, rtol=0):
            raise DicomReadError("inconsistent_geometry", h.file_token)
        if h.number_of_frames != 1 or h.samples_per_pixel != 1 or h.photometric not in ("MONOCHROME1", "MONOCHROME2"):
            raise DicomReadError("unsupported_pixel_layout", h.file_token)
    normal = np.cross(column_direction, row_direction)
    return first, normal / np.linalg.norm(normal)


def _positions(headers, normal):
    origins = []
    for h in sorted(headers, key=lambda h: float(np.dot(h.position, normal))):
        if not origins or np.linalg.norm(np.asarray(h.position)-origins[-1]) > 1e-3:
            origins.append(np.asarray(h.position))
    if len(origins) > 1:
        steps = np.diff(origins, axis=0)
        projected = steps @ normal
        if np.any(projected <= 1e-3) or not np.allclose(steps, projected[:, None]*normal, atol=1e-3, rtol=0):
            raise DicomReadError("inconsistent_slice_origins")
    return np.asarray(origins)


@dataclass(frozen=True)
class ImageGrid:
    """Lazy grid, preserving per-slice timing rather than inventing synchrony."""

    kind: str
    cells: tuple = field(repr=False)  # [time][slice], ObjectHeader references
    origins_lps_mm: np.ndarray = field(repr=False)
    orientation_lps: tuple
    pixel_spacing_mm: tuple
    times_s: np.ndarray | None = field(repr=False)  # [slice,time]
    timing_basis: str
    frame_durations_s: np.ndarray | None = field(repr=False)  # [slice,time]
    qc_warnings: tuple = ()

    def __post_init__(self):
        for value in (self.origins_lps_mm, self.times_s, self.frame_durations_s):
            if value is not None:
                value.setflags(write=False)

    @property
    def shape(self):
        first = self.cells[0][0]
        return len(self.cells), len(self.cells[0]), first.rows, first.columns

    @property
    def slice_center_distances_mm(self):
        return np.linalg.norm(np.diff(self.origins_lps_mm, axis=0), axis=1)

    def pixel_to_lps(self, slice_index, row, column):
        orientation = np.asarray(self.orientation_lps)
        return self.origins_lps_mm[slice_index] + orientation[:3]*column*self.pixel_spacing_mm[1] + orientation[3:]*row*self.pixel_spacing_mm[0]

    def lps_to_pixel(self, slice_index, point):
        """Return row, column and signed distance from the selected plane (mm)."""
        delta = np.asarray(point) - self.origins_lps_mm[slice_index]
        u, v = np.asarray(self.orientation_lps[:3]), np.asarray(self.orientation_lps[3:])
        normal = np.cross(u, v)
        return np.array([delta @ v / self.pixel_spacing_mm[0], delta @ u / self.pixel_spacing_mm[1], delta @ normal / np.linalg.norm(normal)])

    def read_plane(self, time_index, slice_index, *, scaled=True, dtype=np.float32):
        return read_pixels(self.cells[time_index][slice_index], scaled=scaled, dtype=dtype)

    def materialize(self, *, dtype=np.float32, max_bytes=512*1024**2, progress=None, cancel=None):
        if np.dtype(dtype).kind != "f":
            raise DicomReadError("scaled_pixels_require_float_dtype")
        size = int(np.prod(self.shape)) * np.dtype(dtype).itemsize
        if size > max_bytes:
            raise DicomReadError("memory_budget_exceeded")
        _cancel(cancel)
        array = np.empty(self.shape, dtype=dtype)
        total, done, last = self.shape[0]*self.shape[1], 0, 0.0
        for t, row in enumerate(self.cells):
            for z, header in enumerate(row):
                _cancel(cancel)
                array[t, z] = read_pixels(header, dtype=dtype)
                done += 1
                now = time.monotonic()
                if progress is not None and (now-last >= 1 or done == total):
                    progress("pixels", done, total)
                    last = now
        _cancel(cancel)
        return array

    def safe_summary(self):
        axis = "phase_order" if "cardiac" in self.timing_basis else "frame"
        return {"kind": self.kind, "axes": [axis, "slice", "row", "column"], "shape": self.shape,
                "timing_basis": self.timing_basis, "times_s_by_slice": None if self.times_s is None else self.times_s.tolist(),
                "frame_durations_s_by_slice": None if self.frame_durations_s is None else self.frame_durations_s.tolist(),
                "pixel_spacing_mm": self.pixel_spacing_mm, "slice_center_distances_mm": self.slice_center_distances_mm.tolist(),
                "slice_thickness_mm_values": sorted({h.slice_thickness_mm for row in self.cells for h in row if h.slice_thickness_mm is not None}),
                "units": sorted({h.units or "unspecified" for row in self.cells for h in row}),
                "qc_warnings": self.qc_warnings, "phase_synchronization_performed": False if "cardiac" in self.timing_basis else None}


def build_grid(series):
    """Validate a complete grid; never fill missing cells or infer lost slices."""
    headers = series.objects
    if series.kind not in ("pet_dynamic", "pet_gated", "pet_static", "mr_cine", "mr_perfusion"):
        raise DicomReadError("series_not_supported_for_assembly")
    if len({h.sop_uid for h in headers}) != len(headers):
        raise DicomReadError("duplicate_sop_instance")
    first, normal = _validate_geometry(headers)
    if len({h.frame_of_reference_uid for h in headers}) != 1:
        raise DicomReadError("inconsistent_frame_of_reference")
    if len({(h.units, h.decay_correction, h.corrected_image, h.half_life_s, h.injected_dose_bq, h.injection_start) for h in headers}) != 1:
        raise DicomReadError("inconsistent_quantitative_metadata")
    origins = _positions(headers, normal)
    if series.kind == "pet_dynamic":
        values, basis = [h.frame_reference_s for h in headers], "pet_frame_reference"
    elif series.kind in ("pet_gated", "mr_cine"):
        values, basis = [h.trigger_s for h in headers], "cardiac_trigger"
    elif series.kind == "mr_perfusion":
        values, basis = [h.acquisition_time for h in headers], "relative_acquisition"
        if len({(h.acquisition_has_date, h.acquisition_offset) for h in headers}) != 1:
            raise DicomReadError("inconsistent_acquisition_reference")
        if all(v is not None for v in values) and not first.acquisition_has_date and max(values)-min(values) > 43200_000000:
            raise DicomReadError("ambiguous_midnight_without_date")
    else:
        values, basis = [0.0]*len(headers), "none"
    if any(v is None for v in values):
        raise DicomReadError("missing_time_axis")
    times = sorted(set(values))
    declaration_field = {"pet_dynamic": "declared_time_slices", "pet_gated": "declared_time_slots", "mr_cine": "declared_cardiac_images", "mr_perfusion": "declared_temporal_positions"}.get(series.kind)
    if declaration_field:
        advertised = {getattr(h, declaration_field) for h in headers if getattr(h, declaration_field) is not None}
        if advertised and advertised != {len(times)}:
            raise DicomReadError("declared_frame_count_mismatch")
    if first.sop_class == PET:
        advertised = {h.declared_slices for h in headers if h.declared_slices is not None}
        if advertised and advertised != {len(origins)}:
            raise DicomReadError("declared_slice_count_mismatch")
    cells = [[None]*len(origins) for _ in times]
    lookup = {value: i for i, value in enumerate(times)}
    for h, value in zip(headers, values):
        distances = np.linalg.norm(origins-np.asarray(h.position), axis=1)
        z, t = int(np.argmin(distances)), lookup[value]
        if distances[z] > 1e-3:
            raise DicomReadError("inconsistent_slice_origins", h.file_token)
        if cells[t][z] is not None:
            raise DicomReadError("duplicate_spatial_time_cell", h.file_token)
        cells[t][z] = h
    if any(h is None for row in cells for h in row):
        raise DicomReadError("missing_spatial_time_cell")
    # Detect holes supported by explicit per-file spacing or temporal counts.
    # If a whole unadvertised slice/frame is absent, metadata cannot prove it existed.
    if len(origins) > 1:
        measured = np.diff(origins, axis=0) @ normal
        advertised = {abs(h.declared_slice_spacing_mm) for h in headers if h.declared_slice_spacing_mm not in (None, 0)}
        if len(advertised) > 1 or (advertised and not np.allclose(measured, next(iter(advertised)), atol=1e-3, rtol=1e-4)):
            raise DicomReadError("slice_spacing_mismatch_or_gap")
        if not np.allclose(measured, measured[0], atol=1e-3, rtol=1e-4):
            raise DicomReadError("irregular_slice_spacing")
    durations = None
    if series.kind == "pet_dynamic":
        durations = np.asarray([[cells[t][z].frame_duration_s or 0 for t in range(len(times))] for z in range(len(origins))])
        if np.any(durations <= 0) or not np.allclose(durations, durations[0], rtol=0, atol=1e-6):
            raise DicomReadError("invalid_frame_durations")
    if basis == "relative_acquisition":
        times = [(t-times[0])/1_000000 for t in times]
    times_array = None if basis == "none" else np.tile(np.asarray(times, dtype=np.float64), (len(origins), 1))
    warnings_out = ("condition_requires_review",) if series.condition_candidate == "unknown" else ()
    return ImageGrid(series.kind, tuple(tuple(row) for row in cells), origins, first.orientation,
                     first.pixel_spacing, times_array, basis, durations, warnings_out)


def build_cine_stack(series_list):
    """Combine explicitly selected cine planes, retaining unsynchronized clocks."""
    if not series_list or any(s.kind != "mr_cine" for s in series_list):
        raise DicomReadError("cine_selection_required")
    all_headers = [h for s in series_list for h in s.objects]
    if len({h.study_uid for h in all_headers}) != 1 or len({h.frame_of_reference_uid for h in all_headers}) != 1 or not all_headers[0].frame_of_reference_uid:
        raise DicomReadError("cine_study_or_reference_mismatch")
    first, normal = _validate_geometry(all_headers)
    grids = [build_grid(s) for s in series_list]
    if any(g.shape[1] != 1 for g in grids) or len({g.shape[0] for g in grids}) != 1:
        raise DicomReadError("cine_phase_count_or_plane_mismatch")
    order = sorted(range(len(grids)), key=lambda i: float(grids[i].origins_lps_mm[0] @ normal))
    origins = _positions(all_headers, normal)
    if len(origins) != len(grids):
        raise DicomReadError("duplicate_cine_plane")
    distances = np.linalg.norm(np.diff(origins, axis=0), axis=1)
    if len(distances) > 1 and not np.allclose(distances, distances[0], atol=1e-3, rtol=1e-4):
        raise DicomReadError("irregular_cine_slice_spacing")
    advertised = {abs(h.declared_slice_spacing_mm) for h in all_headers if h.declared_slice_spacing_mm not in (None, 0)}
    if len(advertised) > 1 or (advertised and len(distances) and not np.allclose(distances, next(iter(advertised)), atol=1e-3, rtol=1e-4)):
        raise DicomReadError("cine_slice_spacing_mismatch_or_gap")
    cells = tuple(tuple(grids[i].cells[t][0] for i in order) for t in range(grids[0].shape[0]))
    times = np.vstack([grids[i].times_s[0] for i in order])
    qc = ("phase_synchronization_not_performed",)
    if not np.allclose(times, times[0], atol=1e-6, rtol=0):
        qc += ("slice_trigger_grids_differ",)
    return ImageGrid("cine_stack", cells, origins, first.orientation, first.pixel_spacing,
                     times, "cardiac_trigger_per_slice", None, qc)


def read_pixels(header, *, scaled=True, dtype=np.float32):
    """Decode one unchanged native monochrome frame; apply m*stored+b only.

    MONOCHROME1 is not inverted here: display transformations belong to the UI.
    PET units remain the encoded Units value, never MBF or SUV by implication.
    """
    if header.sop_class not in (MR, PET) or header.number_of_frames != 1 or header.samples_per_pixel != 1 or header.photometric not in ("MONOCHROME1", "MONOCHROME2"):
        raise DicomReadError("unsupported_pixel_layout", header.file_token)
    try:
        stat = header.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != header.fingerprint:
            raise DicomReadError("source_changed_since_index", header.file_token)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = pydicom.dcmread(header.path)
            pixels = ds.pixel_array
        if pixels.shape != (header.rows, header.columns):
            raise DicomReadError("pixel_shape_mismatch", header.file_token)
        if not scaled:
            stat_after = header.path.stat()
            if (stat_after.st_size, stat_after.st_mtime_ns) != header.fingerprint:
                raise DicomReadError("source_changed_during_read", header.file_token)
            return pixels
        if np.dtype(dtype).kind != "f":
            raise DicomReadError("scaled_pixels_require_float_dtype", header.file_token)
        if header.sop_class == PET and (header.slope is None or header.intercept is None or not header.units):
            raise DicomReadError("pet_rescale_or_units_missing", header.file_token)
        slope = header.slope if header.slope is not None else 1.0
        intercept = header.intercept if header.intercept is not None else 0.0
        if slope == 0:
            raise DicomReadError("zero_rescale_slope", header.file_token)
        result = (pixels.astype(np.float64)*slope + intercept).astype(dtype)
        if not np.isfinite(result).all():
            raise DicomReadError("nonfinite_scaled_pixels", header.file_token)
        stat_after = header.path.stat()
        if (stat_after.st_size, stat_after.st_mtime_ns) != header.fingerprint:
            raise DicomReadError("source_changed_during_read", header.file_token)
        return result
    except DicomReadError:
        raise
    except Exception as error:
        raise DicomReadError("pixel_decode_failed_" + type(error).__name__, header.file_token) from None

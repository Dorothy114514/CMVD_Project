from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "deidentified"
WORK = ROOT / "outputs" / "_deid_work"
SEVEN = ROOT / "tools" / "7zr.exe"
INTEGRITY = ROOT / "audit" / "restricted" / "archive_integrity_results.csv"
MANIFEST = ROOT / "metadata" / "full_case_manifest.csv"
EXCLUDED = ROOT / "audit" / "excluded_pixel_phi_risk.csv"
STATE = ROOT / "outputs" / "deidentification_state.json"

UID_NAMESPACE = uuid.UUID("e477684b-c751-4ace-92ae-17b8f83ee5ec")
DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})(.*)$")

# DICOM Basic Profile-oriented identifiers plus common vendor/clinical fields.
CLEAR_KEYWORDS = {
    "PatientBirthDate", "PatientBirthTime", "PatientAddress", "PatientTelephoneNumbers",
    "OtherPatientIDs", "OtherPatientNames", "PatientMotherBirthName", "MedicalRecordLocator",
    "EthnicGroup", "Occupation", "AdditionalPatientHistory", "PatientComments",
    "AccessionNumber", "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "ReferringPhysicianName", "ReferringPhysicianAddress", "ReferringPhysicianTelephoneNumbers",
    "PhysiciansOfRecord", "PerformingPhysicianName", "NameOfPhysiciansReadingStudy",
    "OperatorsName", "AdmittingDiagnosesDescription", "StudyID", "RequestingPhysician",
    "RequestedProcedureID", "ScheduledProcedureStepID", "ScheduledPerformingPhysicianName",
    "ProtocolName", "StationName", "DeviceSerialNumber", "IssuerOfPatientID",
    "FillerOrderNumberImagingServiceRequest", "PlacerOrderNumberImagingServiceRequest",
    "HumanPerformerName", "HumanPerformerOrganization", "VerifyingObserverName",
}

KEEP_CLASS_UIDS = {
    "SOPClassUID", "MediaStorageSOPClassUID", "TransferSyntaxUID", "ImplementationClassUID",
    "CodingSchemeUID", "ContextGroupExtensionCreatorUID", "PrivateInformationCreatorUID",
}

SECONDARY_CAPTURE_PREFIXES = {
    "1.2.840.10008.5.1.4.1.1.7",  # Secondary Capture family
}


def uid_map(value: str) -> str:
    return "2.25." + str(uuid.uuid5(UID_NAMESPACE, value).int)


def shift_date(value: str, days: int) -> str:
    match = DATE_RE.match(value or "")
    if not match:
        return value
    try:
        d = datetime.strptime("".join(match.groups()[:3]), "%Y%m%d") + timedelta(days=days)
        return d.strftime("%Y%m%d") + match.group(4)
    except ValueError:
        return value


def walk_datasets(ds: Dataset):
    yield ds
    for elem in list(ds):
        if elem.VR == "SQ" and elem.value:
            for item in elem.value:
                yield from walk_datasets(item)


def deidentify(ds: Dataset, research_id: str, day_shift: int) -> None:
    for nested in walk_datasets(ds):
        nested.remove_private_tags()
        for elem in list(nested):
            keyword = elem.keyword
            if elem.VR == "PN" or keyword in CLEAR_KEYWORDS:
                elem.value = ""
            elif elem.VR == "DA":
                if isinstance(elem.value, (list, tuple)):
                    elem.value = [shift_date(str(v), day_shift) for v in elem.value]
                else:
                    elem.value = shift_date(str(elem.value), day_shift)
            elif elem.VR == "DT":
                if isinstance(elem.value, (list, tuple)):
                    elem.value = [shift_date(str(v), day_shift) for v in elem.value]
                else:
                    elem.value = shift_date(str(elem.value), day_shift)
            elif elem.VR == "UI" and keyword not in KEEP_CLASS_UIDS:
                if isinstance(elem.value, (list, tuple)):
                    elem.value = [uid_map(str(v)) for v in elem.value]
                elif elem.value:
                    elem.value = uid_map(str(elem.value))
    ds.PatientName = research_id
    ds.PatientID = research_id
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "Basic profile; private tags removed; shifted dates and UIDs"
    if hasattr(ds, "file_meta") and getattr(ds, "SOPInstanceUID", None):
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID


def pixel_phi_risk(ds: Dataset) -> tuple[bool, str]:
    burned = str(getattr(ds, "BurnedInAnnotation", "")).upper()
    modality = str(getattr(ds, "Modality", "")).upper()
    sop = str(getattr(ds, "SOPClassUID", ""))
    if burned == "YES":
        return True, "BurnedInAnnotation=YES"
    if modality in {"OT", "SC", "NM"}:
        return True, f"conservative modality exclusion: {modality}"
    if any(sop.startswith(prefix) for prefix in SECONDARY_CAPTURE_PREFIXES):
        return True, "Secondary Capture SOP class"
    return False, ""


def safe_age(value: str) -> str:
    value = str(value or "")
    return value if re.fullmatch(r"\d{3}[DWMY]", value) else ""


def save_state(completed: list[str]) -> None:
    STATE.write_text(json.dumps({"completed": completed}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Existing deidentified data found. Refusing to overwrite; plan an isolated rebuild first.")
    for folder in (WORK, MANIFEST.parent, EXCLUDED.parent):
        folder.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    integrity = {row["ArchiveFile"]: row for row in csv.DictReader(INTEGRITY.open(encoding="utf-8-sig"))}
    archives = sorted(DATA.glob("*/*.7z"), key=lambda p: (p.parent.name, p.name))
    counters = {"阳性": 0, "阴性": 0}
    assignments = []
    for archive in archives:
        cohort = archive.parent.name
        counters[cohort] += 1
        prefix = "POS" if cohort == "阳性" else "NEG"
        assignments.append((archive, cohort, f"{prefix}-{counters[cohort]:03d}"))

    completed = []
    if STATE.exists():
        completed = json.loads(STATE.read_text(encoding="utf-8")).get("completed", [])
    manifest_rows = []
    excluded_rows = []
    if MANIFEST.exists():
        manifest_rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    if EXCLUDED.exists():
        excluded_rows = list(csv.DictReader(EXCLUDED.open(encoding="utf-8-sig")))

    for index, (archive, cohort, research_id) in enumerate(assignments, 1):
        if research_id in completed:
            print(f"[{index}/{len(assignments)}] skip completed {research_id}", flush=True)
            continue
        print(f"[{index}/{len(assignments)}] {research_id}", flush=True)
        started = time.time()
        work_case = WORK / research_id
        out_case = OUT / cohort / research_id
        if work_case.exists():
            shutil.rmtree(work_case)
        if out_case.exists():
            shutil.rmtree(out_case)
        work_case.mkdir(parents=True)
        out_case.mkdir(parents=True)
        result = subprocess.run([str(SEVEN), "x", "-y", "-bd", f"-o{work_case}", str(archive)], capture_output=True)
        if result.returncode:
            raise RuntimeError(f"extract failed for {research_id}: {result.stderr.decode(errors='replace')}")

        modalities, series_uids, shifted_dates = Counter(), set(), set()
        dicom_written = dicom_excluded = non_dicom = errors = 0
        sex = age = ""
        series_dirs: dict[str, str] = {}
        for source in work_case.rglob("*"):
            if not source.is_file():
                continue
            try:
                ds = pydicom.dcmread(source, force=False)
            except Exception:
                non_dicom += 1
                continue
            if not sex:
                sex = str(getattr(ds, "PatientSex", ""))
            if not age:
                age = safe_age(getattr(ds, "PatientAge", ""))
            risk, reason = pixel_phi_risk(ds)
            modality = str(getattr(ds, "Modality", "UNKNOWN")) or "UNKNOWN"
            modalities[modality] += 1
            if risk:
                dicom_excluded += 1
                excluded_rows.append({"ResearchID": research_id, "Cohort": cohort, "Reason": reason,
                                      "Modality": modality, "SeriesDescription": str(getattr(ds, "SeriesDescription", ""))})
                continue
            original_series_uid = str(getattr(ds, "SeriesInstanceUID", "NO_SERIES"))
            if original_series_uid not in series_dirs:
                series_dirs[original_series_uid] = f"SERIES-{len(series_dirs)+1:03d}"
            series_uids.add(original_series_uid)
            day_shift = (int(hashlib.sha256(research_id.encode()).hexdigest()[:8], 16) % 731) - 365
            try:
                deidentify(ds, research_id, day_shift)
                for keyword in ("StudyDate", "SeriesDate", "AcquisitionDate"):
                    value = str(getattr(ds, keyword, ""))
                    if value:
                        shifted_dates.add(value)
                target_dir = out_case / series_dirs[original_series_uid]
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"IMG-{dicom_written+1:06d}.dcm"
                ds.save_as(target, enforce_file_format=True)
                dicom_written += 1
            except Exception:
                errors += 1

        shutil.rmtree(work_case)
        meta = integrity.get(archive.name, {})
        manifest_rows = [r for r in manifest_rows if r.get("ResearchID") != research_id]
        manifest_rows.append({
            "ResearchID": research_id, "CohortLabel": cohort,
            "LabelMeaning": "directory label only; clinical endpoint not documented",
            "IntegrityStatus": meta.get("TestStatus", ""), "ArchiveBytes": meta.get("CompressedBytes", ""),
            "ArchiveInternalFiles": meta.get("Files", ""), "ArchiveUncompressedBytes": meta.get("UncompressedBytes", ""),
            "DICOMWritten": dicom_written, "DICOMExcludedPixelPHIRisk": dicom_excluded,
            "NonDICOMFiles": non_dicom, "ProcessingErrors": errors, "SeriesWritten": len(series_dirs),
            "ModalitiesInSource": ";".join(f"{k}:{v}" for k, v in sorted(modalities.items())),
            "Sex": sex, "AgeDICOM": age, "ShiftedStudyDates": ";".join(sorted(shifted_dates)),
            "ProcessingSeconds": round(time.time() - started, 2),
        })
        with MANIFEST.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
            writer.writeheader(); writer.writerows(sorted(manifest_rows, key=lambda r: r["ResearchID"]))
        with EXCLUDED.open("w", newline="", encoding="utf-8-sig") as f:
            fields = ["ResearchID", "Cohort", "Reason", "Modality", "SeriesDescription"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(excluded_rows)
        (out_case / "case_summary.json").write_text(json.dumps(manifest_rows[-1], ensure_ascii=False, indent=2), encoding="utf-8")
        completed.append(research_id)
        save_state(completed)

    if WORK.exists() and not any(WORK.iterdir()):
        WORK.rmdir()
    print(f"COMPLETE cases={len(completed)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

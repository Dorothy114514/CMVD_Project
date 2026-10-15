import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

import pydicom

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "deidentified"
MANIFEST = ROOT / "metadata" / "full_case_manifest.csv"
REPORT = ROOT / "audit" / "deidentification_validation.json"
RID = re.compile(r"^(POS|NEG)-\d{3}$")

rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
issues = []
checked = 0
started = time.time()
private_elements = 0
modalities = Counter()
patient_ids = Counter()

for path in OUT.rglob("*.dcm"):
    checked += 1
    if checked % 10000 == 0:
        elapsed = time.time() - started
        rate = checked / elapsed if elapsed else 0
        remaining = max(0, sum(int(r["DICOMWritten"]) for r in rows) - checked)
        eta_minutes = remaining / rate / 60 if rate else 0
        print(
            f"PROGRESS checked={checked} rate={rate:.1f}/s "
            f"elapsed_min={elapsed/60:.1f} eta_min={eta_minutes:.1f}",
            flush=True,
        )
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception as exc:
        issues.append(f"read:{path}:{exc}")
        continue
    rid = str(getattr(ds, "PatientID", ""))
    name = str(getattr(ds, "PatientName", ""))
    patient_ids[rid] += 1
    modalities[str(getattr(ds, "Modality", ""))] += 1
    if not RID.fullmatch(rid) or name != rid:
        issues.append(f"identity:{path}:{name}|{rid}")
    if str(getattr(ds, "PatientIdentityRemoved", "")) != "YES":
        issues.append(f"identity_removed:{path}")
    if str(getattr(ds, "Modality", "")).upper() in {"OT", "SC", "NM"}:
        issues.append(f"excluded_modality_present:{path}")
    for elem in ds.iterall():
        if elem.tag.is_private:
            private_elements += 1
            if private_elements <= 20:
                issues.append(f"private:{path}:{elem.tag}")

expected = sum(int(r["DICOMWritten"]) for r in rows)
report = {
    "manifest_cases": len(rows),
    "cohorts": dict(Counter(r["CohortLabel"] for r in rows)),
    "expected_dicom": expected,
    "checked_dicom": checked,
    "processing_errors": sum(int(r["ProcessingErrors"]) for r in rows),
    "excluded_pixel_phi_risk": sum(int(r["DICOMExcludedPixelPHIRisk"]) for r in rows),
    "non_dicom_files": sum(int(r["NonDICOMFiles"]) for r in rows),
    "modalities": dict(modalities),
    "research_ids": len(patient_ids),
    "private_elements": private_elements,
    "issue_count": len(issues),
    "issues_first_100": issues[:100],
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))

"""D5 local CLI. Reads explicit CSVs/snapshots only, never scans image folders."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from cmvd_demo.cohort import assemble_cohort, save_cohort, export_templates
from cmvd_demo.cohort.synthetic import create_synthetic_project
from cmvd_demo.io import DicomReadError
from cmvd_demo.perfusion import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--project", type=Path, help="Explicit local cohort_project.json")
    mode.add_argument("--synthetic-demo", action="store_true")
    mode.add_argument("--export-templates", action="store_true")
    parser.add_argument("--freeze", action="store_true", help="Explicitly freeze reviewed patient assignments in a new snapshot")
    args = parser.parse_args()
    if args.export_templates and args.freeze:
        parser.error("blank forms cannot freeze patient assignments")
    try:
        if args.export_templates:
            run = export_templates(ROOT, ROOT/"outputs/cohort_requests")
            print("Blank supplemental forms; real cases created: 0")
        else:
            path = create_synthetic_project(ROOT, ROOT/"outputs/cohort_fixtures") if args.synthetic_demo else (
                args.project or ROOT/"configs/cohort_project.template.json")
            cohort = assemble_cohort(ROOT, load_json(path))
            run = save_cohort(ROOT/"outputs/cohort", cohort, freeze=args.freeze)
            report = cohort["report"]
            print(f'mode={report["mode"]}; exams={report["manifest_case_count"]}; patients={report["associated_patient_count"]}; development_rows={report["development_included_count"]}')
            print(f'issues={len(report["issues"])}; patient_partitions_frozen={args.freeze}; model_trained=False')
            if report["mode"] == "synthetic_fixture":
                print("SYNTHETIC FIXTURE ONLY: real training inputs ready=False")
            for issue in report["issues"]:
                print(issue)
        print("output="+run.relative_to(ROOT).as_posix())
        return 0
    except (DicomReadError, OSError, ValueError) as error:
        print("Not completed: "+(error.code if isinstance(error, DicomReadError) else type(error).__name__))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

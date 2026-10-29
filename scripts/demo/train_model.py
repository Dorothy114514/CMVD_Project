"""D6 local training CLI. Real mode requires an explicitly frozen D5 snapshot."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from cmvd_demo.cohort import assemble_cohort, save_cohort
from cmvd_demo.cohort.synthetic import create_synthetic_project
from cmvd_demo.perfusion.storage import load_json
from cmvd_demo.io import DicomReadError
from cmvd_demo.training import train_snapshot, load_model_package


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cohort", type=Path, help="Complete frozen research D5 snapshot, never a raw data folder")
    modes.add_argument("--synthetic-demo", action="store_true")
    args = parser.parse_args()
    try:
        if args.synthetic_demo:
            path = create_synthetic_project(ROOT, ROOT/"outputs/training_fixtures", development_patients=80,
                protocol_version=2, reserve_independent=False, vary_features=True,
                progress=lambda stage, done, total: print(f"{stage} {done}/{total}", flush=True) if done % 10 == 0 or done == total else None)
            cohort = assemble_cohort(ROOT, load_json(path))
            snapshot = save_cohort(ROOT/"outputs/cohort", cohort, freeze=True)
        else:
            snapshot = args.cohort
        run = train_snapshot(ROOT, snapshot, ROOT/"outputs/training", allow_synthetic=args.synthetic_demo,
            progress=lambda stage, done, total: print(f"{stage} {done}/{total}", flush=True))
        package = load_model_package(run, allow_synthetic=args.synthetic_demo)
        report = package["report"]
        print(f'mode={report["mode"]} development_patients={report["development_patients"]} real_training={report["real_training_performed"]} independent_evaluation=False', flush=True)
        print("output="+run.relative_to(ROOT).as_posix(), flush=True)
        return 0
    except Exception as error:
        print("Not completed: "+(error.code if isinstance(error, DicomReadError) else type(error).__name__), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

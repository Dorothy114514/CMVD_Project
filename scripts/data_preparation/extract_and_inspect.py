"""Explicit preflight and non-overwriting batches; paths stay in restricted audit."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from cmvd_demo.preparation.archives import create_plan, execute_batch, load_json, summarize, write_json
from cmvd_demo.io import DicomReadError


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--output-base", type=Path, default=Path("E:/CMVD_Data/extracted"))
    parser.add_argument("--tool", type=Path, default=Path("E:/Conda/tools/7zip/26.04/7zr.exe"))
    parser.add_argument("--execute", type=Path, help="Explicit restricted plan created by --plan")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--finalize", type=Path)
    args = parser.parse_args()
    if sum((args.plan, args.execute is not None, args.finalize is not None)) != 1:
        parser.error("select one operation")
    def progress(stage, done, total):
        print(f"{stage} {done}/{total}", flush=True)
    try:
        if args.plan:
            path = create_plan(ROOT, args.output_base, args.tool, progress=progress)
            p = load_json(path)
            print("plan="+path.relative_to(ROOT).as_posix())
            print(f'archives={len(p["archives"])} space_ok={p["space_ok"]} preflight_ok={sum(r["preflight_ok"] for r in p["archives"])}')
            return 0 if p["space_ok"] and all(r["preflight_ok"] for r in p["archives"]) else 1
        if args.execute:
            report = execute_batch(args.execute, start=args.start, count=args.count, progress=progress)
        else:
            p = load_json(args.finalize)
            report = summarize(p)
            if not report["completed"]:
                raise DicomReadError("archive_final_report_requires_all_attempts")
            write_json(Path(p["report_root"])/"report.json", report)
        print(f'attempted={report["attempted"]}/{report["archive_count"]} extracted={report["extraction_passed"]} inspections_clean={report["inspection_passed"]} completed={report["completed"]}')
        return 0 if not report["archives_with_execution_failure"] else 1
    except (DicomReadError, OSError, ValueError) as error:
        print("not_completed="+(error.code if isinstance(error, DicomReadError) else type(error).__name__), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

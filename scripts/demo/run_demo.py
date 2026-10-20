"""Synthetic-only CLI. Run with the project's existing Conda interpreter."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cmvd_demo.framework import ContractError, predict_fixture


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-demo", action="store_true", required=True)
    args = parser.parse_args(argv)
    if not args.synthetic_demo:
        parser.error("Only --synthetic-demo is supported")

    inputs = {
        "case": ROOT / "examples/synthetic/case.json",
        "contract": ROOT / "configs/demo_contract.json",
        "model": ROOT / "examples/synthetic/model_fixture.json",
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = ROOT / "outputs/demo" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = predict_fixture(
            load_json(inputs["case"]), load_json(inputs["contract"]), load_json(inputs["model"])
        )
        result["run_id"] = run_id
        result["python_version"] = platform.python_version()
        result["input_sha256"] = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in inputs.items()
        }
        with (run_dir / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"SYNTHETIC ONLY; model is untrained. Result: {run_dir / 'result.json'}")
        return 0
    except (ContractError, ValueError, OSError, KeyError, TypeError) as error:
        # No raw input or exception values in reports: avoid accidental data disclosure.
        failure = {"status": "failed", "error_type": type(error).__name__, "run_id": run_id}
        with (run_dir / "failure.json").open("x", encoding="utf-8") as stream:
            json.dump(failure, stream, indent=2)
        print(f"Demo failed; see {run_dir / 'failure.json'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

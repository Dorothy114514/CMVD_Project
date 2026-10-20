"""D5 native Qt synthetic acceptance and read-only real-input availability audit."""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from PySide6.QtCore import QTimer, QEventLoop, QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
from cmvd_demo.cohort import assemble_cohort, save_cohort
from cmvd_demo.cohort.synthetic import create_synthetic_project
from cmvd_demo.cohort.storage import load_frozen
from cmvd_demo.perfusion.storage import load_json, new_run, write_json
from cmvd_demo.io import DicomReadError
from cmvd_demo.desktop.window import BrowserWindow


def wait_for(predicate, timeout=20):
    deadline = time.monotonic()+timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise DicomReadError("cohort_native_timeout")
        loop = QEventLoop()
        QTimer.singleShot(20, loop.quit)
        loop.exec()


def main():
    output = new_run(ROOT/"outputs/cohort_validation")
    report = {"schema_version": 1, "completed": False, "passed": False,
              "scope": "D5 synthetic native Qt and real CSV availability only; no image scan",
              "real_training_performed": False, "clinical_validated": False, "failures": []}
    write_json(output/"report.json", report)
    app = QApplication([sys.argv[0]])
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")
    window = BrowserWindow()
    window.show()
    window.open_cohort()
    dialog = window.cohort_dialog
    report["qt_platform"] = app.platformName()
    try:
        actual = assemble_cohort(ROOT, load_json(ROOT/"configs/cohort_project.template.json"))
        real_output = save_cohort(output/"availability_audit", actual)
        report["real_input_availability"] = {key: actual["report"][key] for key in
            ("issues", "manifest_case_count", "associated_patient_count", "development_included_count", "training_inputs_ready")}
        report["availability_audit"] = real_output.relative_to(ROOT).as_posix()
        path = create_synthetic_project(ROOT, output/"synthetic_forms")
        dialog.select_project(path)
        timer = QTimer(window)
        ticks = []
        timer.setInterval(5)
        timer.timeout.connect(lambda: ticks.append(True))
        timer.start()
        dialog.check_button.click()
        wait_for(lambda: not dialog.runner.busy)
        timer.stop()
        assert not dialog.last_error and len(ticks) > 0
        a = dialog.audit
        assert a["report"]["development_included_count"] == 6
        assert a["report"]["associated_patient_count"] == 7
        assert dialog.table.topLevelItemCount() == 8
        assert dialog.freeze_button.isEnabled()
        report["native_check"] = {"exams": 8, "patients": 7, "development_rows": 6, "ui_timer_ticks": len(ticks)}
        dialog.freeze_button.click()
        wait_for(lambda: not dialog.runner.busy)
        assert not dialog.last_error
        plan = load_frozen(dialog.last_output)
        independent = {p["patient_key"] for p in plan["patients"] if p["partition"] == "independent_test"}
        assert len(independent) == 1
        for fold in plan["folds"]["folds"]:
            train, valid = set(fold["train_patients"]), set(fold["validation_patients"])
            assert not train & valid and not (train | valid) & independent
        assert load_json(dialog.last_output/"audit.json")["training_inputs_ready"] is False
        report["synthetic_snapshot"] = dialog.last_output.relative_to(ROOT).as_posix()
        report["patient_isolation_checked"] = True
        report["synthetic_training_ready"] = False
        QCoreApplication.processEvents()
        assert dialog.grab().save(str(output/"synthetic-d5.png"))
    except Exception as error:
        report["failures"].append({"code": error.code if isinstance(error, DicomReadError) else "unexpected_"+type(error).__name__})
    finally:
        window.close()
        wait_for(lambda: not any(r.busy for r in window.background_runners()))
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    report.update(completed=True, passed=not report["failures"])
    # The initial report intentionally persists until a complete final report.
    import json
    (output/"report.json").write_text(json.dumps(report, allow_nan=False, indent=2), encoding="utf-8")
    print("Report: "+(output/"report.json").relative_to(ROOT).as_posix())
    print(f'passed={report["passed"]}; failures={report["failures"]}')
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

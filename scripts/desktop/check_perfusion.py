"""D4 native synthetic workflow; optionally reuse controlled D3 curve evidence.

Never scans patient images or supplies workstation MBF for real cases. Existing
D3 engineering polygons remain unreviewed; reused windows are numerical fixtures.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QMessageBox, QApplication

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from cmvd_demo.desktop.window import BrowserWindow
from cmvd_demo.io import DicomReadError
from cmvd_demo.roi import document_hash
from cmvd_demo.perfusion import mr_features, pet_request, load_json, save_request
from cmvd_demo.perfusion.storage import write_json
from check_desktop import wait_for


def native_synthetic(window, output):
    window.open_synthetic_perfusion()
    window.feature_panel.output_root = output/"synthetic_snapshots"
    f, p = window.feature_panel, window.roi_panel
    f.protocol.setText("SYNTHETIC-FIRST-PASS-V1")
    f.baseline.setValue(2)
    f.start.setValue(3)
    f.end.setValue(7)
    for condition, index in (("rest", 1), ("stress", 2)):
        window.select_choice("synthetic-perfusion-"+condition)
        wait_for(lambda: window.current_pixels is not None)
        for role in (0, 1):
            p.role.setCurrentIndex(role)
            p.add_reference()
        p.calculate()
        wait_for(lambda: p.result is not None or bool(p.last_error))
        if p.last_error:
            raise DicomReadError(p.last_error)
        f.condition.setCurrentIndex(index)
        f.calculate_mr()
        wait_for(lambda: f.candidate is not None or bool(f.last_error))
        if f.last_error:
            raise DicomReadError(f.last_error)
        assert "synthetic_signal_not_research_feature" in f.candidate["qc"]["warnings"]
        f.capture_mr()
    pair = f._mr_pair()
    np.testing.assert_allclose(pair["measurements"]["mr_mpri"]["value"], 2, atol=1e-12, rtol=1e-12)
    f.save()
    wait_for(lambda: f.saved_output is not None or bool(f.last_error))
    if f.last_error:
        raise DicomReadError(f.last_error)
    bundle = load_json(f.saved_output/"features.json")
    assert not bundle["qc"]["model_input_ready"] and not bundle["qc"]["perfusion_input_ready"]
    window.workspace_tabs.setCurrentIndex(1)
    window.time_slider.setValue(4)
    wait_for(lambda:window.displayed_indices==(4,0))
    window._set_levels(0,80)
    window.resize(1510, 1020)
    QApplication.processEvents()
    assert window.grab().save(str(output/"synthetic-d4.png"))
    return {"mr_pair_mpri": pair["measurements"]["mr_mpri"]["value"], "expected_mpri": 2,
            "snapshot": f.saved_output.relative_to(ROOT).as_posix(), "engineering_fixture_only": True,
            "model_input_ready": False, "clinical_validated": False}


def independent_curve_check(feature):
    """Scalar loop/trapezoids as an oracle for vectorized feature extraction."""
    p, observed = feature["parameters"], feature["measurements"]
    slopes = {}
    count = 0
    for curve in feature["input_curves"]["curves"]:
        role, t, y = curve["role"], curve["time_s"], curve["mean"]
        start, end, origin = p["window_start"], p["window_end"], p["window_origin_index"]
        b = p["baseline_start"]
        baseline = sum(y[b:b+p["baseline_frames"]])/p["baseline_frames"]
        peak = max(range(start, end+1), key=lambda i: y[i])
        slope = max((y[i+1]-y[i])/(t[i+1]-t[i]) for i in range(origin, peak))
        area = sum(((y[i]-baseline)+(y[i+1]-baseline))*(t[i+1]-t[i])/2 for i in range(origin, end))
        expected = {"baseline": baseline, "peak": y[peak], "peak_enhancement": y[peak]-baseline,
                    "relative_peak_enhancement": (y[peak]-baseline)/baseline, "peak_time": t[peak],
                    "ttp_from_window_origin": t[peak]-t[origin], "max_upslope": slope, "signed_enhancement_auc": area}
        for key, value in expected.items():
            np.testing.assert_allclose(observed[role+"."+key]["value"], value, rtol=1e-12, atol=1e-9)
            count += 1
        slopes[role] = slope
    np.testing.assert_allclose(observed["normalized_upslope"]["value"], slopes["myocardium"]/slopes["blood_pool"], rtol=1e-12)
    return count+1


def reuse_curves(path, output):
    previous = load_json(path)
    if previous.get("completed") is not True or previous.get("passed") is not True:
        raise DicomReadError("completed_passing_roi_report_required")
    rows, features = [], []
    for index, sample in enumerate(previous["samples"], 1):
        case = document_hash({"development_evidence_group": index, "report_sha256": document_hash(previous)})
        local, measurements, planes, blocked, attempted = {}, 0, 0, [], 0
        for entry in sample["roi"]:
            if entry["kind"] not in ("pet_dynamic", "mr_perfusion"):
                continue
            snapshot = (ROOT/entry["snapshot"]).resolve()
            if not snapshot.is_relative_to((ROOT/"outputs").resolve()):
                raise DicomReadError("controlled_roi_snapshot_required")
            doc, curves = load_json(snapshot/"roi.json"), load_json(snapshot/"curves.json")
            marker = load_json(snapshot/"bundle.json")
            assert marker["completed"] and doc["purpose"] == "engineering_fixture"
            assert marker["document_sha256"] == document_hash(doc) == curves["document_sha256"]
            condition = entry["condition_candidate"]
            if entry["kind"] == "pet_dynamic":
                local[condition] = doc, curves
            else:
                attempted += 1
                try:
                    f = mr_features(doc, curves, case, condition, "D3-REUSE-ENGINEERING-ONLY",
                                    baseline_frames=2, window_start=2,
                                    window_end=doc["binding"]["shape"][0]-1, window_reviewed=False)
                except DicomReadError as error:
                    if error.code != "feature_no_positive_enhancement":
                        raise
                    reasons = []
                    for curve in curves["curves"]:
                        y,t = curve["mean"],curve["time_s"]
                        baseline = (y[0]+y[1])/2
                        peak = max(range(2,len(y)),key=lambda k:y[k])
                        upslope = max((y[k+1]-y[k])/(t[k+1]-t[k]) for k in range(1,peak))
                        if y[peak] <= baseline or upslope <= 0:
                            reasons.append(curve["role"])
                    assert reasons
                    # D3 polygons are arbitrary engineering fixtures. A signal
                    # without enhancement must be rejected, not window-tuned to pass.
                    blocked.append({"source_signature":doc["binding"]["source_signature"],
                                    "condition_candidate":condition,"code":error.code,
                                    "independent_rejection_confirmed":True,"roles":reasons})
                    continue
                measurements += independent_curve_check(f)
                assert not f["qc"]["local_qc_passed"]
                planes += 1
                features.append(f)
        request = pet_request(case, local)
        assert all(request["branches"][c]["mbf"] is None for c in ("rest", "stress"))
        request_dir = save_request(output/("development-"+str(index)), request)
        rows.append({"development_group": index, "mr_planes_attempted":attempted,"mr_planes": planes,
                     "blocked_planes":blocked,"scalar_measurements_checked": measurements,
                     "pet_tac_branches": len(local), "pet_request": request_dir.relative_to(ROOT).as_posix(),
                     "workstation_mbf_available": False, "anatomical_pairing_reviewed": False,
                     "engineering_fixture_only": True, "source_images_rescanned": False})
        print(f"development group {index}: MR accepted={planes}/{attempted}, blocked={len(blocked)}, scalar checks={measurements}, PET templates only", flush=True)
    write_json(output/"reused_mr_plane_features.json", features)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-report", type=Path, help="completed D3 --samples --roi report; no image scan")
    args = parser.parse_args()
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    output = ROOT/"outputs"/"perfusion_validation"/run
    output.mkdir(parents=True, exist_ok=False)
    report = {"stage": "D4", "completed": False, "passed": False, "clinical_validated": False,
              "human_acceptance": False, "reference_workstation_comparison": False,
              "native_synthetic": None, "reused_development_curves": [], "failures": []}
    write_json(output/"report.json", report)
    app = QApplication([sys.argv[0]])
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")
    window = BrowserWindow(unsaved_prompt=lambda action: QMessageBox.Discard)
    window.show()
    report["qt_platform"] = app.platformName()
    try:
        report["native_synthetic"] = native_synthetic(window, output)
        if args.roi_report:
            report["reused_development_curves"] = reuse_curves(args.roi_report, output)
    except Exception as error:
        report["failures"].append({"code": error.code if isinstance(error, DicomReadError)
                                   else "unexpected_"+type(error).__name__})
    finally:
        window.close()
        wait_for(lambda: not any(r.busy for r in window.background_runners()))
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    report.update(completed=True, passed=not report["failures"])
    (output/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report: "+(output/"report.json").relative_to(ROOT).as_posix(), flush=True)
    print(f"passed={report['passed']} failures={report['failures']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

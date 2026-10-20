"""Automated native Qt regression: phantom and optionally two local samples.

Saves only a synthetic screenshot. No patient image, name or source path is
written into the report. This does not replace independent viewer/human checks.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import pydicom
from PySide6.QtCore import Qt, QTimer, QEventLoop,QPointF,QCoreApplication,QEvent
from PySide6.QtGui import QPainterPath,QPolygonF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cmvd_demo.desktop.window import BrowserWindow
from cmvd_demo.io import DicomReadError
from cmvd_demo.roi import document_hash
from cmvd_demo.roi.model import contour_at


def independent_roi_mask(contour,shape):
    """Independent Qt point-in-polygon oracle for non-boundary fixture vertices."""
    def path(vertices):
        p = QPainterPath()
        p.addPolygon(QPolygonF([QPointF(c,r) for r,c in vertices]))
        p.closeSubpath()
        return p
    area = path(contour["outer_rc"])
    if contour["inner_rc"] is not None:
        area = area.subtracted(path(contour["inner_rc"]))
    vertices = np.asarray(contour["outer_rc"])
    lo,hi = np.maximum(np.ceil(vertices.min(axis=0)).astype(int),0),np.minimum(np.floor(vertices.max(axis=0)).astype(int),np.array(shape)-1)
    mask = np.zeros(shape,dtype=bool)
    for r in range(lo[0],hi[0]+1):
        for c in range(lo[1],hi[1]+1):
            mask[r,c] = area.contains(QPointF(float(c),float(r)))
    return mask


def exercise_roi_choice(window,output_root):
    panel = window.roi_panel
    panel.output_root = output_root
    window.slice_slider.setValue(window.grid.shape[1]//2)
    window.time_slider.setValue(0)
    wait_for(lambda:window.displayed_indices==(0,window.grid.shape[1]//2))
    for index in (0,1):
        panel.role.setCurrentIndex(index)
        QTest.mouseClick(panel.add_button,Qt.LeftButton)
    panel.document["purpose"] = "engineering_fixture"
    window.time_slider.setValue(1)
    wait_for(lambda:window.displayed_indices==(1,window.grid.shape[1]//2))
    panel.scope.setCurrentIndex(1)
    item = panel.overlays["blood_pool"][0]
    item.setPos(item.pos()+QPointF(1,0))
    assert not panel.invalid_edit
    QTest.mouseClick(panel.calculate_button,Qt.LeftButton)
    wait_for(lambda:panel.result is not None or bool(panel.last_error))
    if panel.last_error:
        raise DicomReadError(panel.last_error)
    result = panel.result
    assert not result["qc"]["feature_input_ready"] and not result["qc"]["clinical_validated"]
    grid,doc = window.grid,panel.document
    z = result["slice_index"]
    masks = {}
    for t in range(grid.shape[0]):
        if hasattr(grid,"cells"):
            ds = pydicom.dcmread(grid.cells[t][z].path)
            pixels = (ds.pixel_array.astype(np.float64)*float(ds.get("RescaleSlope",1))+
                      float(ds.get("RescaleIntercept",0))).astype(np.float32)
        else:
            pixels = grid.read_plane(t,z)
        for track,curve in zip(doc["tracks"],result["curves"]):
            contour = contour_at(track,t)
            key = document_hash(contour)
            if key not in masks:
                masks[key] = independent_roi_mask(contour,grid.shape[-2:])
            values = pixels[masks[key]].astype(np.float64)
            np.testing.assert_allclose(curve["mean"][t],values.mean(),rtol=1e-12,atol=1e-10)
            assert curve["pixel_count"][t]==len(values)
            assert curve["area_mm2"][t]==len(values)*float(np.prod(grid.pixel_spacing_mm))
            assert curve["time_s"][t]==grid.times_s[z,t]
    QTest.mouseClick(panel.save_button,Qt.LeftButton)
    wait_for(lambda:panel.saved_output is not None or bool(panel.last_error))
    if panel.last_error:
        raise DicomReadError(panel.last_error)
    snapshot = panel.saved_output
    saved_hash = document_hash(doc)
    QTest.mouseClick(panel.remove_button,Qt.LeftButton)
    panel.load_roi(snapshot/"roi.json")
    wait_for(lambda:not panel.runner.busy)
    assert document_hash(panel.document)==saved_hash
    QTest.mouseClick(panel.calculate_button,Qt.LeftButton)
    wait_for(lambda:panel.result is not None or bool(panel.last_error))
    if panel.last_error:
        raise DicomReadError(panel.last_error)
    assert panel.result==result
    print(f"ROI {grid.kind}: frames={grid.shape[0]} roles=2, source means and reload match",flush=True)
    return {"kind":grid.kind,"condition_candidate":window.current_choice.condition,
            "slice_index":z,"timepoints":grid.shape[0],"roles":2,"independent_polygon_means_match":True,
            "manual_frame_override_tested":True,"reload_reproduces_curves":True,
            "engineering_fixture_only":True,"clinical_validated":False,
            "snapshot":snapshot.relative_to(ROOT).as_posix()}


def exercise_rois(window,choices,output_root):
    results = []
    for choice in choices:
        if choice.kind not in ("pet_dynamic","mr_perfusion","cine_stack"):
            continue
        print(f"ROI selection {len(results)+1}/11 {choice.kind}",flush=True)
        window.select_choice(choice.key)
        wait_for(lambda:window.current_pixels is not None or bool(window.last_error))
        if window.last_error:
            raise DicomReadError(window.last_error)
        results.append(exercise_roi_choice(window,output_root))
    assert len(results)==11
    return results


def wait_for(predicate, seconds=60):
    if predicate():
        return
    loop, timer, deadline = QEventLoop(), QTimer(), QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: loop.quit() if predicate() else None)
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    timer.start()
    deadline.start(int(seconds*1000))
    loop.exec()
    timer.stop()
    deadline.stop()
    if not predicate():
        raise DicomReadError("desktop_check_timeout")


def exercise_navigation(window, choices):
    """Paint acknowledgements and rapid controls; no patient screenshot."""
    results = []
    for kind in ("pet_dynamic", "mr_perfusion", "cine_stack"):
        choice = next(c for c in choices if c.kind == kind)
        window.select_choice(choice.key)
        wait_for(lambda: window.current_pixels is not None and window.image.painted_serial == window.image.serial)
        painted, blank = [], []
        def on_paint():
            painted.append((window.displayed_indices[0], time.monotonic()))
        def inspect():
            if window.current_pixels is None or window.image.image is None:
                blank.append(1)
        timer = QTimer()
        timer.setInterval(5)
        timer.timeout.connect(inspect)
        window.frame_painted.connect(on_paint)
        window.play_fps.setValue(25)
        timer.start()
        try:
            QTest.mouseClick(window.play_button, Qt.LeftButton)
            wait_for(lambda: len(painted) >= 8 or bool(window.last_error), seconds=15)
        finally:
            window._stop_play()
            timer.stop()
            window.frame_painted.disconnect(on_paint)
        if window.last_error:
            raise DicomReadError(window.last_error)
        assert [t for t, stamp in painted] == [i % window.grid.shape[0] for i in range(1, len(painted)+1)]
        intervals = np.diff([stamp for t, stamp in painted])
        assert len(intervals) and not blank
        # Keep a visible valid frame throughout continuous seeking, then verify
        # the last requested frame, not an intermediate completion, is shown.
        requests = iter([(i*7) % window.grid.shape[0] for i in range(50)]+[window.grid.shape[0]-1])
        drag = QTimer()
        drag.setInterval(8)
        def step():
            inspect()
            value = next(requests, None)
            if value is None:
                drag.stop()
                window.time_slider.sliderReleased.emit()
            else:
                window.time_slider.setValue(value)
        drag.timeout.connect(step)
        drag.start()
        try:
            wait_for(lambda: not drag.isActive() and window.displayed_indices == (window.grid.shape[0]-1, 0)
                     and window.image.painted_serial == window.image.serial)
        finally:
            drag.stop()
        assert not blank
        t, z = window.displayed_indices
        ds = pydicom.dcmread(window.grid.cells[t][z].path)
        expected = (ds.pixel_array.astype(np.float64)*float(ds.get("RescaleSlope", 1))+
                    float(ds.get("RescaleIntercept", 0))).astype(np.float32)
        np.testing.assert_array_equal(window.current_pixels, expected)
        assert window.cache.bytes_used <= window.cache.max_bytes
        item = {"kind": kind, "target_fps": 25, "painted_frames": len(painted),
                "measured_fps": round(1/float(np.mean(intervals)), 2),
                "paint_interval_p95_ms": round(float(np.percentile(intervals, 95))*1000, 2),
                "blank_observations": len(blank), "rapid_seek_requests": 51,
                "final_seek_source_pixels_match": True, "skipped_playback_frames": 0}
        results.append(item)
        print(f"navigation {kind}: fps={item['measured_fps']} blank={len(blank)}", flush=True)
    return results


def exercise_case(window, alias, path, navigation=False,roi_output=None):
    print(f"{alias}: indexing in desktop", flush=True)
    ticks = []
    timer = QTimer()
    timer.setInterval(40)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    window.open_case(path)
    assert window.current_pixels is None and window.cache.bytes_used == 0
    try:
        wait_for(lambda: window.current_pixels is not None or bool(window.last_error))
    finally:
        timer.stop()
    if window.last_error:
        raise DicomReadError(window.last_error)
    if window.index_summary["issues"]:
        raise DicomReadError("sample_index_has_issues")
    index_ticks = len(ticks)
    assert index_ticks > 2
    choices = [c for c in window.choices if c.kind in {"pet_dynamic", "pet_static", "pet_gated", "mr_perfusion", "cine_stack"}
               or (c.kind == "mr_cine" and "sax" not in c.source.tokens)]
    results = []
    for i, choice in enumerate(choices, 1):
        print(f"{alias}: choice {i}/{len(choices)} {choice.kind}", flush=True)
        window.select_choice(choice.key)
        wait_for(lambda: window.current_pixels is not None or bool(window.last_error))
        if window.last_error:
            raise DicomReadError(window.last_error)
        shape = window.grid.shape
        checks = []
        for t, z in ((0, 0), (shape[0]-1, shape[1]-1)):
            window.time_slider.setValue(t)
            window.slice_slider.setValue(z)
            wait_for(lambda: window.displayed_indices == (t, z) or bool(window.last_error))
            if window.last_error:
                raise DicomReadError(window.last_error)
            header = window.grid.cells[t][z]
            dataset = pydicom.dcmread(header.path)
            # Independent transfer/rescale arithmetic verifies the UI displays the
            # requested source frame. Spatial truth is separately tested by phantom.
            expected = (dataset.pixel_array.astype(np.float64)*float(dataset.get("RescaleSlope", 1))+
                        float(dataset.get("RescaleIntercept", 0))).astype(np.float32)
            np.testing.assert_array_equal(window.current_pixels, expected)
            before = window.current_pixels.copy()
            window.level_width.setValue(max(.001, window.level_width.value()*.8))
            np.testing.assert_array_equal(window.current_pixels, before)
            checks.append({"time_index": t, "slice_index": z, "source_pixels_match": True})
        if choice.kind == "cine_stack":
            assert "尚未同步" in window.qc_label.text()
            assert not np.allclose(window.grid.times_s, window.grid.times_s[0])
        assert window.cache.bytes_used <= window.cache.max_bytes
        results.append({"kind": choice.kind, "shape": shape, "frames": checks})
    assert len(results) == 18
    print(f"{alias}: passed {len(results)} choices, {index_ticks} index timer ticks", flush=True)
    navigation_results = exercise_navigation(window, choices) if navigation else []
    roi_results = exercise_rois(window,choices,roi_output) if roi_output is not None else []
    return {"alias": alias, "indexed_objects": window.index_summary["indexed_objects"],
            "index_issues": len(window.index_summary["issues"]), "index_gui_timer_ticks": index_ticks,
            "choices": results, "plane_cache_bytes": window.cache.bytes_used,
            "navigation": navigation_results,"roi":roi_results, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", action="store_true", help="also exercise two existing unpacked development samples")
    parser.add_argument("--navigation", action="store_true", help="with --samples, exercise playback paint timing and rapid seeking")
    parser.add_argument("--roi", action="store_true", help="check manual ROI means, motion override, snapshot/reload using engineering polygons")
    args = parser.parse_args()
    if args.navigation and not args.samples:
        parser.error("--navigation requires --samples")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    output = ROOT / "outputs" / "desktop_validation" / run_id
    output.mkdir(parents=True, exist_ok=False)
    app = QApplication([sys.argv[0]])
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")
    window = BrowserWindow()
    window.runner.progress.connect(lambda revision, stage, done, total: print(f"UI {stage} {done}/{total}", flush=True))
    window.show()
    report = {"stage": "D3" if args.roi else "D2", "qt_platform": app.platformName(), "automated_native_interaction": True,
              "human_acceptance": False, "independent_viewer_comparison": False,
              "clinical_validation": False, "samples": [], "failures": []}
    # Publish a started report so even native termination cannot be mistaken
    # for a completed pass. Only this new run's report is finalized below.
    report.update(completed=False,passed=False)
    (output / "report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    try:
        window.open_synthetic()
        wait_for(lambda: window.current_pixels is not None)
        if args.roi:
            report["synthetic_roi"] = exercise_roi_choice(window,output/"roi_artifacts"/"synthetic")
        QTest.qWait(100)
        assert window.grab().save(str(output / "synthetic-workstation.png"))
        report["synthetic_rendered"] = True
        if args.samples:
            for alias, folder in (("SAMPLE-A", "\u9633\u6027"), ("SAMPLE-B", "\u9634\u6027")):
                paths = [p for p in (ROOT / "data" / folder).iterdir() if p.is_dir() and not p.is_symlink()]
                if len(paths) != 1:
                    raise DicomReadError("exactly_one_unpacked_sample_required")
                report["samples"].append(exercise_case(window, alias, paths[0], args.navigation,
                                                      output/"roi_artifacts"/alias if args.roi else None))
    except Exception as error:
        report["failures"].append({"code": error.code if isinstance(error, DicomReadError)
                                   else "unexpected_"+type(error).__name__})
    finally:
        window.close()
        try:
            wait_for(lambda: not any(r.busy for r in window.background_runners()))
        except DicomReadError as error:
            report["failures"].append({"code": error.code})
        app.processEvents()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)
    report["completed"] = True
    report["passed"] = not report["failures"]
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Report: {output.relative_to(ROOT).as_posix()}/report.json", flush=True)
    print(f"passed={report['passed']} failures={report['failures']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

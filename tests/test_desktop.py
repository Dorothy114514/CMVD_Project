"""D2 tests exercise native Qt events, background work and known pixel truth.

No QT_QPA_PLATFORM override: Windows uses its native plugin. These are automated
interactions, not a claim of human acceptance or independent viewer comparison.
"""

from pathlib import Path
import sys
from threading import Event
import time
import unittest

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, QEventLoop, QCoreApplication, QEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QMessageBox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmvd_demo.desktop.data import PlaneCache, PhantomGrid, ReaderBackend, direction_label, auto_levels
from cmvd_demo.desktop.tasks import TaskRunner
from cmvd_demo.desktop.window import BrowserWindow
from cmvd_demo.desktop.release_check import check_product_language
from cmvd_demo.io import DicomReadError


APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)
APP.setStyle("Fusion")


def until(predicate, timeout=5000):
    if predicate():
        APP.processEvents()
        return
    loop, timer, deadline = QEventLoop(), QTimer(), QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: loop.quit() if predicate() else None)
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    timer.start()
    deadline.start(timeout)
    loop.exec()
    timer.stop()
    deadline.stop()
    if not predicate():
        raise AssertionError("qt_condition_timed_out")
    APP.processEvents()


def discard_unsaved(action):
    """Existing regression actions deliberately discard synthetic/development edits."""
    return QMessageBox.Discard


def dispose_window(window):
    """Drain workers, then destroy native graphs on their owning GUI thread.

    Leaving closed Qt graphs to Python cyclic GC can free connected wrappers in
    an unsafe order while a later test is running. close alone only hides them.
    """
    window.close()
    until(lambda: not any(r.busy for r in window.background_runners()))
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    APP.processEvents()


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.window = BrowserWindow(unsaved_prompt=discard_unsaved)
        self.window.show()
        QTest.qWait(25)

    def tearDown(self):
        dispose_window(self.window)

    def synthetic(self):
        self.window.open_synthetic()
        until(lambda: self.window.current_pixels is not None)
        until(lambda: not self.window.prefetch.busy and self.window.image.painted_serial == self.window.image.serial)
        return self.window

    def test_empty_workspace_has_no_model_or_image(self):
        self.assertEqual(self.window.model_label.text(), "模型未配置")
        self.assertIsNone(self.window.current_pixels)
        self.assertFalse(self.window.time_slider.isEnabled())

    def test_synthetic_demo_is_menu_only_and_opens_both_phantoms(self):
        w = self.window
        self.assertFalse(any("合成" in b.text() for b in w.centralWidget().findChildren(QPushButton)))
        self.assertEqual(w.series_label.text(), "打开病例目录以开始浏览")
        self.assertIn(w.synthetic_demo_action, w.demo_menu.actions())
        self.assertIn(w.perfusion_demo_action, w.demo_menu.actions())
        w.synthetic_demo_action.trigger()
        until(lambda: w.current_pixels is not None)
        self.assertEqual(w.grid.shape[:2], (12, 5))
        self.assertEqual(w.current_pixels[10, 15], 250)
        w.perfusion_demo_action.trigger()
        until(lambda: w.current_pixels is not None and w.current_choice.key.startswith("synthetic-perfusion-"))
        self.assertEqual(len(w.choices), 2)

    def test_product_empty_state_and_all_panel_language(self):
        w = self.window
        check_product_language(w)
        self.assertIs(w.viewer_stack.currentWidget(), w.empty_view)
        self.assertTrue(w.empty_open_button.isVisible())
        from unittest.mock import patch
        with patch("cmvd_demo.desktop.window.QFileDialog.getExistingDirectory", return_value="") as chooser:
            QTest.mouseClick(w.empty_open_button, Qt.LeftButton)
            chooser.assert_called_once()
        w.synthetic_demo_action.trigger()
        until(lambda: w.current_pixels is not None)
        self.assertIs(w.viewer_stack.currentWidget(), w.graphics)
        check_product_language(w)

    def test_known_asymmetric_pixels_physical_aspect_and_directions(self):
        w = self.synthetic()
        self.assertEqual(w.image.axisOrder, "row-major")
        self.assertEqual(w.current_pixels[10, 15], 250)
        point = w.image.mapToView(QPointF(15.5, 10.5))
        self.assertAlmostEqual(point.x(), 15)
        self.assertAlmostEqual(point.y(), 20)
        self.assertTrue(w.view.state["yInverted"])
        self.assertIn("上 A", w.direction.text())
        self.assertIn("左 R", w.direction.text())
        scene = w.image.mapToScene(QPointF(15.5, 10.5))
        w._mouse_moved(scene)
        self.assertIn("行 10 / 列 15", w.cursor_label.text())
        self.assertIn("(15.00, 20.00, 0.00)", w.cursor_label.text())

    def test_native_slider_keyboard_updates_true_time_and_slice(self):
        w = self.synthetic()
        w.time_slider.setFocus()
        QTest.keyClick(w.time_slider, Qt.Key_End)
        until(lambda: w.displayed_indices == (11, 0))
        self.assertIn("13.200 s", w.time_label.text())
        w.slice_slider.setFocus()
        QTest.keyClick(w.slice_slider, Qt.Key_End)
        until(lambda: w.displayed_indices == (11, 4))
        self.assertIn("5/5", w.slice_label.text())

    def test_window_changes_display_levels_without_changing_pixels(self):
        w = self.synthetic()
        before = w.current_pixels.copy()
        w.level_center.setValue(100)
        w.level_width.setValue(50)
        np.testing.assert_array_equal(before, w.current_pixels)
        np.testing.assert_allclose(w.image.getLevels(), [75, 125])
        QTest.mouseClick(w.auto_button, Qt.LeftButton)
        expected = auto_levels(before)
        np.testing.assert_allclose(w.image.getLevels(), expected, atol=.001)

    def test_manual_window_survives_frame_change(self):
        w = self.synthetic()
        w.level_center.setValue(80)
        w.level_width.setValue(20)
        w.time_slider.setValue(3)
        until(lambda: w.displayed_indices == (3, 0))
        np.testing.assert_allclose(w.image.getLevels(), [70, 90])

    def test_monochrome1_reverses_display_lut_only(self):
        w = self.window
        def mono_grid(choice, progress, cancel):
            grid = PhantomGrid()
            grid.photometric = "MONOCHROME1"
            return grid
        w.backend.grid = mono_grid
        w.open_synthetic()
        until(lambda: w.current_pixels is not None)
        self.assertEqual(w.image.lut[0, 0], 255)
        self.assertEqual(w.image.lut[-1, 0], 0)
        self.assertEqual(w.current_pixels[10, 15], 250)

    def test_enter_on_selected_series_retries_after_cancel(self):
        w = self.synthetic()
        w.cancel_current()
        self.assertIsNone(w.current_pixels)
        w.tree.setFocus()
        QTest.keyClick(w.tree, Qt.Key_Return)
        until(lambda: w.current_pixels is not None)

    def test_zoom_range_is_preserved_while_changing_frames(self):
        w = self.synthetic()
        w.view.setRange(xRange=(10, 40), yRange=(20, 50), padding=0)
        before = w.view.viewRange()
        w.time_slider.setValue(1)
        until(lambda: w.displayed_indices == (1, 0))
        np.testing.assert_allclose(w.view.viewRange(), before)

    def test_timer_playback_and_pause_use_ui_controls(self):
        w = self.synthetic()
        QTest.mouseClick(w.play_button, Qt.LeftButton)
        until(lambda: w.displayed_indices is not None and w.displayed_indices[0] > 0)
        QTest.mouseClick(w.play_button, Qt.LeftButton)
        self.assertFalse(w.play_timer.isActive())

    def test_debounced_rapid_frame_changes_publish_only_latest(self):
        w = self.synthetic()
        for t in range(1, 12):
            w.time_slider.setValue(t)
        self.assertIsNotNone(w.current_pixels)
        self.assertEqual(w.displayed_indices, (0, 0))
        self.assertIn("当前仍显示帧 1", w.status.text())
        until(lambda: w.displayed_indices == (11, 0))
        np.testing.assert_array_equal(w.current_pixels, PhantomGrid().read_plane(11, 0))

    def test_slow_load_retains_previous_image_and_matching_time_label(self):
        w = self.synthetic()
        started, release = Event(), Event()
        self.addCleanup(release.set)
        original = w.backend.plane
        def slow(*args):
            started.set()
            release.wait(3)
            return original(*args)
        w.backend.plane = slow
        before, label = w.current_pixels.copy(), w.time_label.text()
        w.time_slider.setValue(7)
        until(started.is_set)
        np.testing.assert_array_equal(w.current_pixels, before)
        self.assertEqual(w.displayed_indices, (0, 0))
        self.assertEqual(w.time_label.text(), label)
        self.assertIsNotNone(w.image.image)
        self.assertIn("目标帧 8", w.status.text())
        release.set()
        until(lambda: w.displayed_indices == (7, 0))
        self.assertIn("7.000 s", w.time_label.text())

    def test_playback_waits_for_slow_decode_then_paint_before_advancing(self):
        w = self.synthetic()
        started, release = Event(), Event()
        self.addCleanup(release.set)
        original = w.backend.plane
        def slow(grid, t, z, *args):
            if t == 1:
                started.set()
                release.wait(3)
            return original(grid, t, z, *args)
        w.backend.plane = slow
        presented = []
        w.frame_painted.connect(lambda: presented.append((w.displayed_indices, time.monotonic())))
        QTest.mouseClick(w.play_button, Qt.LeftButton)
        until(started.is_set)
        QTest.qWait(250)
        self.assertEqual(w.time_slider.value(), 1)
        self.assertEqual(w.displayed_indices, (0, 0))
        self.assertIsNotNone(w.image.image)
        release.set()
        until(lambda: any(indices == (2, 0) for indices, stamp in presented))
        w._stop_play()
        first, second = next(stamp for indices, stamp in presented if indices == (1, 0)), next(stamp for indices, stamp in presented if indices == (2, 0))
        self.assertGreaterEqual(second-first, w.play_timer.interval()/1000-.02)

    def test_playback_does_not_advance_unpainted_frame(self):
        w = self.synthetic()
        w.graphics.setUpdatesEnabled(False)
        try:
            QTest.mouseClick(w.play_button, Qt.LeftButton)
            until(lambda: w.displayed_indices == (1, 0))
            QTest.qWait(250)
            self.assertEqual(w.time_slider.value(), 1)
            self.assertNotEqual(w.image.painted_serial, w.image.serial)
        finally:
            w.graphics.setUpdatesEnabled(True)
            w.graphics.viewport().update()
        until(lambda: w.displayed_indices == (2, 0))
        w._stop_play()

    def test_drag_release_requests_latest_immediately_and_pauses_playback(self):
        w = self.synthetic()
        QTest.mouseClick(w.play_button, Qt.LeftButton)
        w.time_slider.sliderPressed.emit()
        w.time_slider.setValue(8)
        self.assertFalse(w._playing)
        self.assertTrue(w.debounce.isActive())
        w.time_slider.sliderReleased.emit()
        self.assertFalse(w.debounce.isActive())
        until(lambda: w.displayed_indices == (8, 0))

    def test_forward_prefetch_stays_bounded_and_next_frame_uses_cached_pixels(self):
        w = self.synthetic()
        calls = []
        original = w.backend.plane
        def recorded(grid, t, z, cached, progress, cancel):
            calls.append((t, cached is not None))
            return original(grid, t, z, cached, progress, cancel)
        w.backend.plane = recorded
        w.time_slider.setValue(1)
        until(lambda: w.displayed_indices == (1, 0))
        self.assertIn((1, True), calls)
        self.assertLessEqual(w.cache.bytes_used, w.cache.max_bytes)

    def test_late_prefetch_cannot_repopulate_old_case_cache(self):
        w = self.window
        started, release = Event(), Event()
        self.addCleanup(release.set)
        original = w.backend.plane
        def blocked(grid, t, z, *args):
            if t == 1:
                started.set()
                release.wait(3)
            return original(grid, t, z, *args)
        w.backend.plane = blocked
        w.open_synthetic()
        until(started.is_set)
        previous_case = w.case_revision
        w.open_synthetic()
        self.assertIsNone(w.current_pixels)
        self.assertEqual(w.cache.bytes_used, 0)
        release.set()
        until(lambda: w.current_pixels is not None and not w.prefetch.busy)
        self.assertGreater(w.case_revision, previous_case)
        self.assertTrue(all(key[0] == w.case_revision for key in w.cache._items))

    def test_continuous_scrub_renders_previews_without_blank_or_debounce_starvation(self):
        w = self.synthetic()
        frames = []
        w.frame_painted.connect(lambda: frames.append(w.displayed_indices))
        timer = QTimer()
        timer.setInterval(20)
        values = iter(range(1, 12))
        def drag():
            self.assertIsNotNone(w.current_pixels)
            value = next(values, None)
            if value is None:
                timer.stop()
                w.time_slider.sliderReleased.emit()
            else:
                w.time_slider.setValue(value)
        timer.timeout.connect(drag)
        timer.start()
        until(lambda: not timer.isActive() and w.displayed_indices == (11, 0))
        self.assertTrue(any(0 < t < 11 for t, z in frames))

    def test_browsing_fps_is_configurable(self):
        w = self.synthetic()
        w.play_fps.setValue(25)
        self.assertEqual(w.play_timer.interval(), 40)

    def test_opening_new_case_hides_previous_pixels_until_index_commits(self):
        w = self.synthetic()
        started, release = Event(), Event()
        self.addCleanup(release.set)
        def slow_index(path, progress, cancel):
            started.set()
            release.wait(3)
            return [], {"indexed_objects": 0, "series_count": 0, "issues": []}
        w.backend.index = slow_index
        import tempfile
        with tempfile.TemporaryDirectory() as source:
            w.open_case(source)
        self.assertIs(w.viewer_stack.currentWidget(),w.empty_view)
        self.assertIsNone(w.current_pixels)
        until(started.is_set)
        release.set()
        until(lambda: not w.runner.busy)
        self.assertIsNotNone(w.current_pixels)
        self.assertEqual(w.last_error,'case_no_indexed_objects')

    def test_loading_keeps_qt_timer_responsive_and_cancel_discards_late_result(self):
        w = self.synthetic()
        started, release = Event(), Event()
        self.addCleanup(release.set)
        original = w.backend.plane
        def slow_plane(*args):
            started.set()
            release.wait(3)
            return original(*args)
        w.backend.plane = slow_plane
        w.time_slider.setValue(2)
        until(started.is_set)
        ticks = []
        timer = QTimer()
        timer.setInterval(10)
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start()
        QTest.qWait(70)
        timer.stop()
        self.assertGreaterEqual(len(ticks), 2)
        QTest.mouseClick(w.cancel_button, Qt.LeftButton)
        release.set()
        until(lambda: not w.runner.busy)
        self.assertIsNone(w.current_pixels)
        self.assertIn("已取消", w.status.text())

    def test_queued_success_cancelled_before_event_delivery_is_ignored(self):
        w = self.synthetic()
        done = Event()
        def immediate(progress, cancel):
            done.set()
            return object()
        w._operation = ("grid", "unused")
        w.runner.submit(immediate)
        self.assertTrue(done.wait(2))
        w.cancel_current()
        until(lambda: not w.runner.busy)
        self.assertIsNone(w.current_pixels)

    def test_failure_clears_image_and_does_not_imply_negative_diagnosis(self):
        w = self.synthetic()
        def broken(*args):
            raise DicomReadError("test_decode_failure")
        w.backend.plane = broken
        w.time_slider.setValue(2)
        until(lambda: bool(w.last_error))
        self.assertIsNone(w.current_pixels)
        self.assertIn("错误不代表阴性", w.qc_label.text())
        self.assertIs(w.viewer_stack.currentWidget(), w.empty_view)
        self.assertTrue(w.empty_description.isVisible())
        self.assertNotIn("test_decode_failure", w.status.text())

    def test_close_during_load_is_nonblocking_and_reopen_has_no_old_result(self):
        w = self.synthetic()
        started, release = Event(), Event()
        self.addCleanup(release.set)
        original = w.backend.plane
        def slow(*args):
            started.set()
            release.wait(3)
            return original(*args)
        w.backend.plane = slow
        w.time_slider.setValue(1)
        until(started.is_set)
        w.close()
        self.assertFalse(w.isVisible())
        other = BrowserWindow(unsaved_prompt=discard_unsaved)
        try:
            other.show()
            self.assertIsNone(other.current_pixels)
            release.set()
            until(lambda: not w.runner.busy)
            self.assertIsNone(other.current_pixels)
        finally:
            other.close()


class CacheAndTaskTests(unittest.TestCase):
    def test_oversized_plane_rejected_before_decode(self):
        grid = PhantomGrid()
        grid.shape = (1, 1, 3000, 3000)
        called = []
        grid.read_plane = lambda *args: called.append(1)
        with self.assertRaisesRegex(DicomReadError, "desktop_plane_memory_budget_exceeded"):
            ReaderBackend().plane(grid, 0, 0, None, lambda *args: None, lambda: False)
        self.assertEqual(called, [])

    def test_lru_respects_bytes_and_recency(self):
        cache = PlaneCache(max_bytes=96)
        for key in ("a", "b"):
            cache.put(key, np.zeros((3, 4), dtype=np.float32))
        cache.get("a")
        cache.put("c", np.zeros((3, 4), dtype=np.float32))
        self.assertIsNone(cache.get("b"))
        self.assertIsNotNone(cache.get("a"))
        cache.put("large", np.zeros(100, dtype=np.float32))
        self.assertLessEqual(cache.bytes_used, 96)
        self.assertFalse(cache.get("a").flags.writeable)

    def test_task_queue_replaces_pending_work_and_ignores_stale_progress(self):
        runner = TaskRunner()
        started, release = Event(), Event()
        accepted, progress, executed = [], [], []
        runner.result.connect(lambda revision, payload: accepted.append(payload))
        runner.progress.connect(lambda *args: progress.append(args))
        def blocking(report, cancel):
            started.set()
            release.wait(3)
            report("old", 1, 1)
            return "stale"
        runner.submit(blocking)
        until(started.is_set)
        for i in range(10):
            def work(report, cancel, i=i):
                executed.append(i)
                return i
            runner.submit(work)
        release.set()
        until(lambda: not runner.busy)
        self.assertEqual(accepted, [9])
        self.assertEqual(executed, [9])
        self.assertEqual(progress, [])

    def test_oblique_labels_and_constant_window_are_finite(self):
        self.assertEqual(direction_label([.8, -.6, 0]), "LA")
        self.assertEqual(direction_label([0, 0, -1]), "I")
        low, high = auto_levels(np.ones((3, 4), dtype=np.float32))
        self.assertGreater(high, low)


if __name__ == "__main__":
    unittest.main()

"""Native Windows Qt D5 task lifecycle, stale results and cohort affordances."""

from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from test_desktop import APP, until, dispose_window
from PySide6.QtCore import QTimer
from cmvd_demo.desktop.window import BrowserWindow
from cmvd_demo.cohort.synthetic import create_synthetic_project


class CohortDesktopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.window = BrowserWindow()
        self.window.show()
        self.window.open_cohort()
        self.dialog = self.window.cohort_dialog
        self.addCleanup(lambda: dispose_window(self.window))
        # Keep test outputs out of the real workspace. Templates/config contract
        # are the only files copied; no metadata/ or data/ is inspected.
        import shutil
        shutil.copytree(ROOT/"configs", self.root/"configs", ignore=shutil.ignore_patterns("*.local.json"))
        shutil.copytree(ROOT/"examples/templates", self.root/"examples/templates")
        self.dialog.workspace = self.root
        self.dialog.select_project(self.root/"configs/cohort_project.template.json")

    def wait(self):
        until(lambda: not self.dialog.runner.busy, timeout=15000)
        self.assertEqual(self.dialog.last_error, "")

    def test_missing_real_tables_are_visible_no_cases_or_freeze(self):
        self.dialog.check_button.click()
        self.wait()
        self.assertEqual(self.dialog.table.topLevelItemCount(), 0)
        self.assertIn("旧研究编号", self.dialog.issues.toPlainText())
        self.assertFalse(self.dialog.freeze_button.isEnabled())
        self.assertEqual(self.dialog.audit["samples"], [])

    def test_synthetic_workflow_matrix_export_and_explicit_freeze(self):
        self.dialog.synthetic_button.click()
        self.wait()
        self.assertIn("完全合成", self.dialog.mode_label.text())
        self.assertEqual(self.dialog.table.topLevelItemCount(), 8)
        self.assertEqual(self.dialog.audit["report"]["development_included_count"], 6)
        self.assertTrue(self.dialog.freeze_button.isEnabled())
        self.assertTrue(self.dialog.output_button.isEnabled())
        self.assertFalse((self.dialog.last_output/"partitions.json").exists())
        self.dialog.freeze_button.click()
        self.wait()
        self.assertTrue((self.dialog.last_output/"partitions.json").is_file())
        self.assertIn("已冻结", self.dialog.status.text())
        self.assertFalse(self.dialog.audit["report"]["training_inputs_ready"])

    def test_blank_forms_selected_and_opening_another_project_clears_result(self):
        self.dialog.template_button.click()
        self.wait()
        self.assertTrue(self.dialog.project_path.is_file())
        self.assertTrue(self.dialog.output_button.isEnabled())
        self.dialog.select_project(self.root/"configs/cohort_project.template.json")
        self.assertFalse(self.dialog.output_button.isEnabled())
        self.assertIsNone(self.dialog.audit)
        self.assertFalse(self.dialog.freeze_button.isEnabled())

    def test_cancel_drops_result_and_main_window_still_processes_events(self):
        started, release, delivered = Event(), Event(), []
        self.dialog.runner.result.connect(lambda *args: delivered.append(args))
        def delayed(progress, cancel):
            started.set()
            release.wait(3)
            return {"must_not_publish": True}
        self.dialog.runner.submit(delayed)
        until(started.is_set)
        timer_ticks = []
        QTimer.singleShot(0, lambda: timer_ticks.append(True))
        until(lambda: bool(timer_ticks))
        self.dialog.cancel_button.click()  # Disabled in this direct task setup.
        self.dialog.cancel()
        release.set()
        until(lambda: not self.dialog.runner.busy)
        self.assertEqual(delivered, [])
        self.assertIsNone(self.dialog.audit)

    def test_switch_configuration_drops_inflight_result(self):
        started, release, delivered = Event(), Event(), []
        self.dialog.runner.result.connect(lambda *args: delivered.append(args))
        def delayed(progress, cancel):
            started.set()
            release.wait(3)
            return {"stale": True}
        self.dialog.runner.submit(delayed)
        until(started.is_set)
        self.dialog.select_project(self.root/"another.json")
        release.set()
        until(lambda: not self.dialog.runner.busy)
        self.assertEqual(delivered, [])

    def test_main_close_waits_for_fifth_task_runner_and_hides_dialog(self):
        started, release = Event(), Event()
        def delayed(progress, cancel):
            started.set()
            release.wait(3)
            return None
        self.dialog.runner.submit(delayed)
        until(started.is_set)
        self.window.close()
        self.assertTrue(self.dialog.runner.busy)
        self.assertFalse(self.dialog.isVisible())
        self.assertFalse(self.window.isVisible())
        release.set()
        until(lambda: not any(r.busy for r in self.window.background_runners()))

    def test_reopening_after_close_has_no_stale_results(self):
        self.dialog.synthetic_button.click()
        self.wait()
        self.dialog.close()
        self.assertIsNone(self.dialog.audit)
        self.window.open_cohort()
        self.assertTrue(self.dialog.isVisible())
        self.assertEqual(self.dialog.table.topLevelItemCount(), 0)

    def test_bad_project_shows_safe_error_and_can_retry(self):
        path = self.root/"bad.json"
        path.write_text('{"PRIVATE_SENTINEL": NaN}', encoding="utf-8")
        self.dialog.select_project(path)
        self.dialog.check_button.click()
        until(lambda: not self.dialog.runner.busy)
        self.assertTrue(self.dialog.last_error)
        self.assertNotIn("PRIVATE_SENTINEL", self.dialog.status.text())
        self.assertFalse(self.dialog.freeze_button.isEnabled())
        self.dialog.select_project(self.root/"configs/cohort_project.template.json")
        self.dialog.check_button.click()
        self.wait()


if __name__ == "__main__":
    unittest.main()

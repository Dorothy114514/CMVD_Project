"""Resource allowlist and source/frozen writable paths; no patient data."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from cmvd_demo import paths


class ReleasePathTests(unittest.TestCase):
    def tearDown(self):
        paths._workspace_override = None

    def test_source_paths_preserve_existing_workspace(self):
        self.assertEqual(paths.resource_root(), ROOT)
        self.assertEqual(paths.workspace_root(), ROOT)

    def test_frozen_default_uses_local_appdata_not_executable_or_cwd(self):
        with patch.object(sys, "frozen", True, create=True), patch.dict(os.environ, {"LOCALAPPDATA": "E:/SYNTHETIC-USER"}):
            self.assertEqual(paths.workspace_root(), Path("E:/SYNTHETIC-USER/CMVDResearchBrowser/workspace-v1"))

    def test_explicit_workspace_has_only_public_resources_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            work = paths.configure_workspace(Path(temp)/"中文 工作目录")
            actual = {p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file()}
            self.assertEqual(actual, set(paths.RESOURCE_FILES))
            selected = work/"configs/cohort_project.template.json"
            selected.write_text("SYNTHETIC USER EDIT", encoding="utf-8")
            paths.configure_workspace(work)
            self.assertEqual(selected.read_text(encoding="utf-8"), "SYNTHETIC USER EDIT")

    def test_frozen_resource_root_and_allowlist_exclude_controlled_material(self):
        with patch.object(sys, "frozen", True, create=True), patch.object(sys, "_MEIPASS", "E:/SYNTHETIC-BUNDLE/_internal", create=True):
            self.assertEqual(paths.resource_root(), Path("E:/SYNTHETIC-BUNDLE/_internal/resources"))
        self.assertTrue(all(name.startswith(("configs/", "examples/templates/")) or name=="docs/USAGE_CHECKLIST.md" for name in paths.RESOURCE_FILES))
        self.assertFalse(any("runtime.local" in name or "data/" in name or "outputs/" in name for name in paths.RESOURCE_FILES))


if __name__ == "__main__":
    unittest.main()

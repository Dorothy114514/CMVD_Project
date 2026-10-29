"""Known-value duplicate views; source files and strict reader remain intact."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import test_dicom_reader as fixtures
from cmvd_demo.io import DicomReadError, ReadCancelled, scan_case, build_grid
from cmvd_demo.preparation.duplicates import select_identical_sops
from cmvd_demo.preparation import duplicates
from cmvd_demo.preparation.archives import inspect_index
from cmvd_demo.preparation.qc import branch_inventory, review_status


class DuplicateSelectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ReaderTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.series = self.fixture.pet()
        self.truth = build_grid(self.series).materialize()
        self.original = self.series.objects[0].path

    def copy(self, source=None, name="copy.dcm"):
        target = self.root/name
        shutil.copyfile(source or self.original, target)
        return target

    def test_identical_group_preserves_exact_pixels_and_strict_default(self):
        self.copy()
        self.copy(name="copy-2.dcm")
        scan = scan_case(self.root)
        with self.assertRaisesRegex(DicomReadError, "duplicate_sop_instance"):
            build_grid(scan.series[0])
        selection = select_identical_sops(scan)
        self.assertEqual(selection.safe_summary()["excluded_from_transient_index"], 2)
        self.assertEqual(len(list(self.root.glob("*"))), 11)
        np.testing.assert_array_equal(build_grid(selection.scan.series[0]).materialize(), self.truth)
        with self.assertRaisesRegex(DicomReadError, "duplicate_sop_instance"):
            build_grid(scan_case(self.root).series[0])

    def test_one_conflicting_copy_blocks_entire_group_even_with_identical_subgroup(self):
        self.copy()
        changed = self.copy(name="different.dcm")
        ds = pydicom.dcmread(changed)
        ds.PixelData = (np.ones((3, 4), dtype=np.uint16)*1000).tobytes()
        ds.save_as(changed, enforce_file_format=True)
        selection = select_identical_sops(scan_case(self.root))
        self.assertEqual(selection.safe_summary()["conflicting_groups"], 1)
        self.assertEqual(selection.safe_summary()["excluded_from_transient_index"], 0)
        with self.assertRaisesRegex(DicomReadError, "duplicate_sop_instance"):
            build_grid(selection.scan.series[0])

    def test_header_only_difference_is_also_a_conflict(self):
        changed = self.copy()
        ds = pydicom.dcmread(changed)
        ds.RescaleSlope = 3
        ds.save_as(changed, enforce_file_format=True)
        selection = select_identical_sops(scan_case(self.root))
        self.assertEqual(selection.safe_summary()["conflicting_groups"], 1)

    def test_cancellation_and_source_change_reject_selection(self):
        copied = self.copy()
        scan = scan_case(self.root)
        with self.assertRaises(ReadCancelled):
            select_identical_sops(scan, cancel=lambda: True)
        copied.write_bytes(b"changed")
        with self.assertRaisesRegex(DicomReadError, "duplicate_source_changed"):
            select_identical_sops(scan)

    def test_source_mutation_during_hash_is_detected(self):
        self.copy()
        scan = scan_case(self.root)
        original_digest = duplicates._digest
        def changing_digest(header, cancel):
            value = original_digest(header, cancel)
            header.path.write_bytes(b"changed after hash")
            return value
        with patch.object(duplicates, "_digest", side_effect=changing_digest):
            with self.assertRaisesRegex(DicomReadError, "duplicate_source_changed"):
                select_identical_sops(scan)

    def test_selection_is_deterministic_and_safe_summary_has_no_paths_or_uid(self):
        self.copy()
        scan = scan_case(self.root)
        a = select_identical_sops(scan)
        reverse = replace(scan, series=tuple(replace(s, objects=tuple(reversed(s.objects))) for s in scan.series))
        b = select_identical_sops(reverse)
        self.assertEqual(a.groups, b.groups)
        safe = json.dumps(a.safe_summary())
        self.assertNotIn(str(self.root), safe)
        self.assertNotIn(self.series.objects[0].sop_uid, safe)
        self.assertEqual(a.groups[0]["kept_file_token"], min(m["file_token"] for m in a.groups[0]["members"]))

    def test_header_failures_remain_visible_and_inspection_uses_selected_index(self):
        self.copy()
        (self.root/"bad.dcm").write_bytes(b"\0"*128)
        selection = select_identical_sops(scan_case(self.root))
        restricted = self.root/"audit"
        restricted.mkdir()
        inspection = inspect_index(selection.scan, restricted, "SYNTHETIC")
        self.assertEqual(inspection["header_issue_count"], 1)
        self.assertEqual(len(inspection["grids"]), 1)
        self.assertFalse(inspection["inspection_passed"])

    def test_middle_frame_zero_slope_flagged_even_if_endpoints_pass(self):
        header = self.series.objects[4]
        ds = pydicom.dcmread(header.path)
        ds.RescaleSlope = 0
        ds.save_as(header.path, enforce_file_format=True)
        scan = scan_case(self.root)
        restricted = self.root/"audit"
        restricted.mkdir()
        inspection = inspect_index(scan, restricted, "SYNTHETIC")
        self.assertTrue(inspection["inspection_passed"])
        inventory = branch_inventory(scan, inspection)
        self.assertEqual(inventory["zero_slope_objects_all_indexed_series"], 1)
        self.assertTrue(inventory["series"][0]["quantitative_header_blocked"])

    def test_nonmainline_sop_conflict_prevents_whole_package_clear_status(self):
        status = review_status({"inspection_passed": True},
                               {"zero_slope_objects_all_indexed_series": 0}, {"conflicting_groups": 12})
        self.assertTrue(status["mainline_checks_passed_in_transient_view"])
        self.assertFalse(status["engineering_clear_in_transient_view"])
        self.assertFalse(status["clinical_or_training_ready"])


if __name__ == "__main__":
    unittest.main()

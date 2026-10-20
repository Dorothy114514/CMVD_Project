"""Known-value DICOM fixtures for D1; no patient data required."""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmvd_demo.io.dicom import (
    MR, PET, SC, DicomReadError, ReadCancelled, scan_case, build_grid,
    build_cine_stack, read_pixels,
)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study, self.reference = generate_uid(), generate_uid()

    def write(self, name, *, series, modality="PT", z=0, t=0, **changes):
        sop = PET if modality == "PT" else MR
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = changes.get("SOPClassUID", sop)
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = self.study, series
        ds.FrameOfReferenceUID = self.reference
        ds.PatientName, ds.PatientID = "SYNTHETIC^ONLY", "SYNTH-READER"
        ds.Modality = modality
        ds.SeriesNumber = 1
        ds.SeriesDescription = "rest dynamic" if modality == "PT" else "cine sax"
        ds.Rows, ds.Columns = 3, 4
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelSpacing = [2, 3]
        ds.SliceThickness, ds.SpacingBetweenSlices = 8, 10
        ds.ImageOrientationPatient = [0, 1, 0, -1, 0, 0]
        ds.ImagePositionPatient = [10, 20, 30+10*z]
        ds.InstanceNumber = 999-t
        ds.RescaleSlope, ds.RescaleIntercept = 2, 0
        if modality == "PT":
            ds.Units, ds.DecayCorrection = "BQML", "START"
            ds.FrameReferenceTime = [5000, 20000, 40000][t]
            ds.ActualFrameDuration = [10000, 20000, 20000][t]
            ds.NumberOfTimeSlices, ds.NumberOfSlices = 3, 3
        else:
            ds.TriggerTime, ds.CardiacNumberOfImages = t*100*(1+0.1*z), 3
        pixels = (np.arange(12).reshape(3, 4) + 100*t + 10*z).astype(np.uint16)
        for key, value in changes.items():
            setattr(ds, key, value)
        ds.PixelData = pixels.tobytes()
        if "PixelData" in changes:
            ds.PixelData = changes["PixelData"]
        path = self.root / name
        pydicom.dcmwrite(path, ds, enforce_file_format=True)
        return path

    def pet(self, *, omit=(), **changes):
        series = generate_uid()
        for t in range(3):
            for z in range(3):
                if (t, z) not in omit:
                    self.write(f"shuffled-{8-t*3-z}.ima", series=series, z=z, t=t, **changes)
        result = scan_case(self.root)
        self.assertEqual(result.issues, ())
        return result.series[0]

    def test_pet_grid_sorted_by_geometry_and_time_not_filename_or_instance(self):
        series = self.pet()
        self.assertEqual(series.kind, "pet_dynamic")
        grid = build_grid(series)
        self.assertEqual(grid.shape, (3, 3, 3, 4))
        np.testing.assert_allclose(grid.times_s, [[5, 20, 40]]*3)
        np.testing.assert_allclose(grid.frame_durations_s, [[10, 20, 20]]*3)
        np.testing.assert_allclose(grid.slice_center_distances_mm, [10, 10])
        array = grid.materialize()
        for t in range(3):
            for z in range(3):
                np.testing.assert_array_equal(array[t, z], 2*(np.arange(12).reshape(3, 4)+100*t+10*z))

    def test_asymmetric_pixel_spacing_and_orientation_round_trip(self):
        grid = build_grid(self.pet())
        np.testing.assert_allclose(grid.pixel_to_lps(1, row=2, column=3), [6, 29, 40])
        np.testing.assert_allclose(grid.lps_to_pixel(1, [6, 29, 40]), [2, 3, 0])

    def test_missing_cell_is_not_zero_filled(self):
        with self.assertRaisesRegex(DicomReadError, "missing_spatial_time_cell"):
            build_grid(self.pet(omit={(1, 1)}))

    def test_missing_whole_declared_frame_is_detected(self):
        with self.assertRaisesRegex(DicomReadError, "declared_frame_count_mismatch"):
            build_grid(self.pet(omit={(1, z) for z in range(3)}))

    def test_missing_whole_declared_slice_is_detected(self):
        with self.assertRaisesRegex(DicomReadError, "declared_slice_count_mismatch"):
            build_grid(self.pet(omit={(t, 1) for t in range(3)}))

    def test_spacing_detects_missing_plane_without_declared_count(self):
        with self.assertRaisesRegex(DicomReadError, "slice_spacing_mismatch_or_gap"):
            build_grid(self.pet(omit={(t, 1) for t in range(3)}, NumberOfSlices=None))

    def test_static_pet_has_spatial_axis_without_invented_time(self):
        uid = generate_uid()
        for z in range(3):
            self.write(f"{2-z}.ima", series=uid, z=z, FrameReferenceTime=None,
                       NumberOfTimeSlices=None, SeriesDescription="rest static")
        grid = build_grid(scan_case(self.root).series[0])
        self.assertEqual(grid.kind, "pet_static")
        self.assertEqual(grid.shape, (1, 3, 3, 4))
        self.assertIsNone(grid.times_s)
        self.assertEqual(grid.materialize()[0, 2, 0, 0], 40)

    def test_gated_pet_uses_trigger_axis_instead_of_frame_reference(self):
        uid = generate_uid()
        for t in range(3):
            for z in range(3):
                self.write(f"{2-t}-{2-z}.ima", series=uid, t=t, z=z,
                           SeriesDescription="stress gated", FrameReferenceTime=5000,
                           TriggerTime=100*t, NumberOfTimeSlots=3, NumberOfTimeSlices=None)
        grid = build_grid(scan_case(self.root).series[0])
        self.assertEqual(grid.kind, "pet_gated")
        np.testing.assert_allclose(grid.times_s, [[0, .1, .2]]*3)
        self.assertEqual(grid.materialize()[2, 1, 0, 0], 420)

    def test_duplicate_spatial_time_cell_is_rejected(self):
        series = self.pet()
        self.write("extra.ima", series=series.objects[0].series_uid, t=0, z=0)
        with self.assertRaisesRegex(DicomReadError, "duplicate_spatial_time_cell"):
            build_grid(scan_case(self.root).series[0])

    def test_duplicate_sop_instance_is_rejected(self):
        series = self.pet()
        (self.root / "duplicate.ima").write_bytes(series.objects[0].path.read_bytes())
        with self.assertRaisesRegex(DicomReadError, "duplicate_sop_instance"):
            build_grid(scan_case(self.root).series[0])

    def test_same_series_number_and_different_uid_are_not_merged(self):
        self.write("a.ima", series=generate_uid(), modality="MR", SeriesNumber=1)
        self.write("b.ima", series=generate_uid(), modality="MR", SeriesNumber=1)
        self.assertEqual(len(scan_case(self.root).series), 2)

    def test_study_uid_is_part_of_the_grouping_key(self):
        series = generate_uid()
        self.write("a.ima", series=series, modality="MR")
        self.write("b.ima", series=series, modality="MR", StudyInstanceUID=generate_uid())
        self.assertEqual(len(scan_case(self.root).series), 2)

    def test_valid_secondary_capture_with_empty_series_number_is_indexed(self):
        self.write("capture.ima", series=generate_uid(), modality="OT", SOPClassUID=SC, SeriesNumber=None)
        result = scan_case(self.root)
        self.assertEqual(result.issues, ())
        self.assertEqual(result.series[0].kind, "secondary_capture")
        self.assertIsNone(result.series[0].objects[0].series_number)
        with self.assertRaises(DicomReadError):
            build_grid(result.series[0])
        with self.assertRaises(DicomReadError):
            read_pixels(result.series[0].objects[0])

    def test_multiframe_is_indexed_but_not_silently_flattened(self):
        self.write("enhanced.ima", series=generate_uid(), NumberOfFrames=2)
        series = scan_case(self.root).series[0]
        self.assertEqual(series.kind, "unsupported_multiframe")
        with self.assertRaises(DicomReadError):
            build_grid(series)

    def test_failures_are_unclassified_and_archives_are_not_opened(self):
        (self.root / "private-name.ima").write_bytes(b"not a readable dataset")
        (self.root / "private-name.7z").write_bytes(b"not opened")
        result = scan_case(self.root)
        self.assertEqual(result.enumerated_files, 2)
        self.assertEqual(result.ignored_archives, 1)
        self.assertEqual(result.issues[0].code, "read_failed_unclassified")
        self.assertNotIn("private-name", repr(result))
        self.assertNotIn("private-name", json.dumps(result.safe_summary()))

    def test_safe_summary_contains_no_raw_identity_path_or_uid(self):
        self.pet()
        result = scan_case(self.root)
        summary = json.dumps(result.safe_summary()) + repr(result)
        for raw in (str(self.root), self.study, self.reference, "SYNTHETIC^ONLY", "SYNTH-READER"):
            self.assertNotIn(raw, summary)

    def test_pet_uses_each_object_rescale(self):
        uid = generate_uid()
        for t in range(3):
            for z in range(3):
                self.write(f"{t}-{z}.ima", series=uid, t=t, z=z, RescaleSlope=2+t)
        array = build_grid(scan_case(self.root).series[0]).materialize()
        self.assertEqual(array[2, 1, 0, 0], 4*210)

    def test_mr_monochrome1_does_not_invert_quantitative_pixels(self):
        self.write("mr.ima", series=generate_uid(), modality="MR", PhotometricInterpretation="MONOCHROME1", RescaleSlope=3, RescaleIntercept=-5)
        header = scan_case(self.root).series[0].objects[0]
        np.testing.assert_array_equal(read_pixels(header), np.arange(12).reshape(3, 4)*3-5)
        np.testing.assert_array_equal(read_pixels(header, scaled=False), np.arange(12).reshape(3, 4))

    def test_pet_without_units_or_rescale_cannot_produce_quantitative_pixels(self):
        series = self.pet(RescaleSlope=None)
        with self.assertRaisesRegex(DicomReadError, "pet_rescale_or_units_missing"):
            build_grid(series).read_plane(0, 0)

    def test_memory_guard_precedes_pixel_decode(self):
        grid = build_grid(self.pet())
        with patch("cmvd_demo.io.dicom.read_pixels") as read:
            with self.assertRaisesRegex(DicomReadError, "memory_budget_exceeded"):
                grid.materialize(max_bytes=1)
            read.assert_not_called()

    def test_cancellation_does_not_return_partial_scan_or_array(self):
        self.pet()
        with self.assertRaises(ReadCancelled):
            scan_case(self.root, cancel=lambda: True)
        grid = build_grid(scan_case(self.root).series[0])
        with self.assertRaises(ReadCancelled):
            grid.materialize(cancel=lambda: True)

    def test_enumeration_progress_can_cancel_before_headers_are_read(self):
        self.pet()
        counts = []
        with patch("cmvd_demo.io.dicom.pydicom.dcmread") as read:
            with self.assertRaises(ReadCancelled):
                scan_case(self.root, progress=lambda *args: counts.append(args), cancel=lambda: bool(counts))
            read.assert_not_called()
        self.assertEqual(counts[0][0], "enumerate")

    def test_source_changed_since_index_is_rejected(self):
        header = self.pet().objects[0]
        header.path.write_bytes(b"changed after indexing")
        with self.assertRaisesRegex(DicomReadError, "source_changed_since_index"):
            read_pixels(header)

    def test_cancellation_after_first_decoded_plane_discards_partial_array(self):
        grid = build_grid(self.pet())
        calls = []
        def progress(stage, done, total):
            calls.append(done)
        with self.assertRaises(ReadCancelled):
            grid.materialize(progress=progress, cancel=lambda: bool(calls))
        self.assertEqual(calls, [1])

    def test_corrupt_pixel_payload_has_a_safe_error(self):
        series = self.pet(PixelData=b"\0\0")
        with self.assertRaisesRegex(DicomReadError, "pixel_decode_failed_") as error:
            read_pixels(series.objects[0])
        self.assertNotIn(str(self.root), str(error.exception))

    def test_inconsistent_geometry_is_rejected(self):
        uid = generate_uid()
        for t in range(3):
            for z in range(3):
                self.write(f"{t}-{z}.ima", series=uid, t=t, z=z, PixelSpacing=[2, 4] if (t, z)==(1, 1) else [2, 3])
        with self.assertRaisesRegex(DicomReadError, "inconsistent_geometry"):
            build_grid(scan_case(self.root).series[0])

    def test_inconsistent_units_are_rejected(self):
        uid = generate_uid()
        for t in range(3):
            for z in range(3):
                self.write(f"{t}-{z}.ima", series=uid, t=t, z=z, Units="CNTS" if t==1 else "BQML")
        with self.assertRaisesRegex(DicomReadError, "inconsistent_quantitative_metadata"):
            build_grid(scan_case(self.root).series[0])

    def cine(self, mismatch_reference=False):
        for z in range(3):
            uid = generate_uid()
            reference = generate_uid() if mismatch_reference and z==1 else self.reference
            for t in range(3):
                self.write(f"{2-z}-{2-t}.ima", series=uid, modality="MR", z=z, t=t, FrameOfReferenceUID=reference)
        return scan_case(self.root).series

    def test_cross_series_cine_preserves_actual_per_slice_trigger_times(self):
        stack = build_cine_stack(list(reversed(self.cine())))
        self.assertEqual(stack.shape, (3, 3, 3, 4))
        self.assertIn("slice_trigger_grids_differ", stack.qc_warnings)
        np.testing.assert_allclose(stack.times_s[1], [0, .11, .22])
        np.testing.assert_allclose(stack.slice_center_distances_mm, [10, 10])
        self.assertEqual(stack.materialize()[2, 2, 0, 0], 440)
        self.assertFalse(stack.safe_summary()["phase_synchronization_performed"])

    def test_cross_reference_cine_cannot_be_merged(self):
        with self.assertRaisesRegex(DicomReadError, "cine_study_or_reference_mismatch"):
            build_cine_stack(self.cine(mismatch_reference=True))

    def perfusion(self, *, dated=True):
        uid = generate_uid()
        clocks = ["235959.100000", "235959.800000", "000001.200000"]
        for t, clock in enumerate(clocks):
            self.write(f"{2-t}.ima", series=uid, modality="MR", t=t,
                       SeriesDescription="rest dynamic tfl", TriggerTime=None, CardiacNumberOfImages=None,
                       AcquisitionTime=clock, AcquisitionDate=("20261019" if t < 2 else "20261020") if dated else None,
                       NumberOfTemporalPositions=3)
        return scan_case(self.root).series[0]

    def test_perfusion_uses_nonuniform_physical_times_across_midnight(self):
        grid = build_grid(self.perfusion())
        self.assertEqual(grid.shape, (3, 1, 3, 4))
        np.testing.assert_allclose(grid.times_s[0], [0, .7, 2.1], atol=1e-9)
        self.assertEqual(grid.materialize()[2, 0, 0, 0], 400)

    def test_midnight_without_a_date_is_not_guessed(self):
        with self.assertRaisesRegex(DicomReadError, "ambiguous_midnight_without_date"):
            build_grid(self.perfusion(dated=False))


if __name__ == "__main__":
    unittest.main()

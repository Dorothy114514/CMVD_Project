"""Synthetic archive safety, overwrite protection and actual 7zr roundtrip."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest
import zlib

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from cmvd_demo.preparation.archives import (ArchiveError, safe_member, parse_listing, verify_tree,
    create_plan, execute_batch, load_json, summarize, write_json, sha256_file)

TOOL = Path('E:/Conda/tools/7zip/26.04/7zr.exe')


def listing(records):
    return '7-Zip test\r\n----------\r\n'+''.join(
        '\r\n'.join(f'{k} = {v}' for k, v in row.items())+'\r\n\r\n' for row in records)


class ArchiveSafetyTests(unittest.TestCase):
    def test_paths_preserve_nested_unicode_names(self):
        self.assertEqual(safe_member('合成病例\\series\\001.dcm'), '合成病例/series/001.dcm')

    def test_escaping_absolute_ads_reserved_and_ambiguous_paths_rejected(self):
        for path in ('../x', 'a/../b', 'C:\\x', '\\root', '//host/share', 'a:stream', 'a./x',
                     'a /x', 'CON.txt', 'COM1.dcm', 'LPT¹/x', 'a//b', './x', 'a\nfile', 'a?file'):
            with self.subTest(path=path), self.assertRaises(ArchiveError):
                safe_member(path)

    def test_crlf_listing_multiple_records_folder_and_byte_counts(self):
        rows = parse_listing(listing([{'Path': 'root', 'Size': '0', 'Attributes': 'D drwxrwxrwx'},
            {'Path': 'root/file.dcm', 'Size': '123', 'Attributes': 'A -rw-rw-rw-', 'Encrypted': '-'}]))
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]['folder'])
        self.assertEqual(rows[1]['size'], 123)

    def test_empty_last_listing_field_is_legal(self):
        text = listing([{'Path': 'file.dcm', 'Size': '1', 'Block': ''}])
        self.assertEqual(len(parse_listing(text)), 1)

    def test_duplicate_member_contents_preserved_with_explicit_rename_mapping(self):
        records = [{'Path': 'file.dcm', 'Size': '3', 'CRC': f'{zlib.crc32(data):08X}'} for data in (b'abc', b'def')]
        members = parse_listing(listing(records), preserve_duplicate_files=True)
        self.assertTrue(all(e['duplicate_name'] for e in members))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'file.dcm').write_bytes(b'abc')
            (root/'file_1.dcm').write_bytes(b'def')
            mapping = []
            self.assertEqual(verify_tree(root, members, resolution=mapping), (2, 6))
            self.assertEqual(len(mapping), 2)
            (root/'file_1.dcm').write_bytes(b'abc')
            with self.assertRaisesRegex(ArchiveError, 'contents_not_preserved'):
                verify_tree(root, members)

    def test_link_encrypted_anti_duplicate_case_and_file_directory_conflicts_rejected(self):
        base = {'Path': 'root/a', 'Size': '1', 'Attributes': 'A -rw-rw-rw-'}
        for rows in ([{**base, 'Symbolic Link': '../outside'}], [{**base, 'Encrypted': '+'}],
                     [{**base, 'Anti': '+'}], [base, {**base, 'Path': 'ROOT/A'}],
                     [base, {**base, 'Path': 'root/a/child'}]):
            with self.assertRaises(ArchiveError):
                parse_listing(listing(rows))

    def test_tree_requires_exact_file_set_and_sizes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'x').write_bytes(b'abc')
            members = [{'path': 'x', 'folder': False, 'size': 3}]
            self.assertEqual(verify_tree(root, members), (1, 3))
            (root/'extra').write_bytes(b'')
            with self.assertRaisesRegex(ArchiveError, 'file_set_or_size'):
                verify_tree(root, members)

    def test_write_json_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'record.json'
            write_json(p, {'value': 1})
            with self.assertRaises(FileExistsError):
                write_json(p, {'value': 2})
            self.assertEqual(load_json(p), {'value': 1})


@unittest.skipUnless(TOOL.is_file(), 'local pinned 7zr unavailable')
class ArchiveRoundtripTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.workspace = self.base/'workspace'
        self.source = self.base/'synthetic_files'
        self.source.mkdir()
        destination = self.workspace/'data/SYNTHETIC'
        destination.mkdir(parents=True)
        study, series = generate_uid(), generate_uid()
        for t in range(3):
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0'*128)
            ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
            ds.Modality, ds.SeriesNumber, ds.SeriesDescription = 'MR', 1, 'rest perf'
            ds.Rows, ds.Columns, ds.SamplesPerPixel = 2, 3, 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
            ds.ImagePositionPatient, ds.ImageOrientationPatient, ds.PixelSpacing = [0, 0, 0], [1, 0, 0, 0, 1, 0], [1, 1]
            ds.AcquisitionDate, ds.AcquisitionTime = '20000101', f'12000{t}'
            ds.TemporalPositionIdentifier, ds.NumberOfTemporalPositions = t+1, 3
            ds.PixelData = (np.arange(6, dtype=np.uint16)+t).tobytes()
            pydicom.dcmwrite(self.source/f'SYNTH-SENTINEL-{t}.dcm', ds, enforce_file_format=True)
        self.archive = destination/'SYNTH-SENTINEL.7z'
        completed = subprocess.run([str(TOOL), 'a', '-t7z', '-mx=0', str(self.archive), str(self.source/'*')],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(completed.returncode, 0)
        self.plan_path = create_plan(self.workspace, self.base/'external', TOOL)

    def test_real_extractor_preflight_pixel_inspection_no_identity_output_and_no_overwrite(self):
        source_hash = sha256_file(self.archive)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report = execute_batch(self.plan_path, start=0, count=1)
        self.assertTrue(report['completed'])
        self.assertEqual(report['extraction_passed'], 1)
        self.assertEqual(report['inspection_passed'], 1)
        self.assertEqual(report['indexed_objects'], 3)
        self.assertEqual(report['sampled_pixel_planes'], 2)
        self.assertEqual(sha256_file(self.archive), source_hash)
        self.assertNotIn('SYNTH-SENTINEL', json.dumps(report)+output.getvalue())
        with self.assertRaisesRegex(ArchiveError, 'refuses_overwrite'):
            execute_batch(self.plan_path, start=0, count=1)

    def test_source_modified_after_plan_is_rejected_before_extraction(self):
        with self.archive.open('ab') as stream:
            stream.write(b'changed')
        report = execute_batch(self.plan_path, start=0, count=1)
        self.assertEqual(report['extraction_passed'], 0)
        self.assertEqual(len(report['archives_with_execution_failure']), 1)
        row = load_json(self.plan_path)['archives'][0]
        self.assertFalse(Path(row['output']).exists())

    def test_output_inside_project_refused(self):
        with self.assertRaisesRegex(ArchiveError, 'external_root'):
            create_plan(self.workspace, self.workspace/'outputs', TOOL)

    def test_real_7z_duplicate_names_retain_both_different_contents(self):
        # Create a valid small Copy-mode 7z, then rename one stored UTF-16 name
        # in its uncompressed header and recompute the two header CRCs. This
        # reproduces duplicate archive names without touching any real data.
        folder = self.base/'duplicate_fixture'
        folder.mkdir()
        (folder/'a.dcm').write_bytes(b'A'*32)
        (folder/'b.dcm').write_bytes(b'B'*32)
        archive = self.archive.with_name('duplicate.7z')
        result = subprocess.run([str(TOOL), 'a', '-t7z', '-mx=0', '-mhc=off', str(archive),
                                 str(folder/'a.dcm'), str(folder/'b.dcm')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0)
        raw = bytearray(archive.read_bytes())
        offset, size = struct.unpack_from('<QQ', raw, 12)
        header = bytes(raw[32+offset:32+offset+size])
        old, new = ('b.dcm\0').encode('utf-16-le'), ('a.dcm\0').encode('utf-16-le')
        self.assertEqual(header.count(old), 1)
        header = header.replace(old, new)
        raw[32+offset:32+offset+size] = header
        struct.pack_into('<I', raw, 28, zlib.crc32(header))
        struct.pack_into('<I', raw, 8, zlib.crc32(raw[12:32]))
        archive.write_bytes(raw)
        plan_path = create_plan(self.workspace, self.base/'external', TOOL)
        plan = load_json(plan_path)
        index = next(i for i, r in enumerate(plan['archives']) if r.get('duplicate_name_entries'))
        source_hash = sha256_file(archive)
        report = execute_batch(plan_path, start=index, count=1)
        self.assertEqual(report['extraction_passed'], 1)
        self.assertEqual(report['extracted_files'], 2)
        target = Path(plan['archives'][index]['output'])
        self.assertEqual({p.read_bytes() for p in target.rglob('*') if p.is_file()}, {b'A'*32, b'B'*32})
        token = plan['archives'][index]['archive_token']
        mapping = load_json(Path(plan['restricted_root'])/(token+'.duplicate_resolution.json'))
        self.assertEqual(len(mapping), 2)
        self.assertEqual(sha256_file(archive), source_hash)


if __name__ == '__main__':
    unittest.main()

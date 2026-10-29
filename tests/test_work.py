"""Known truth, source/serialization rejection and incomplete work records."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cmvd_demo.desktop.data import perfusion_synthetic_choices, ReaderBackend
from cmvd_demo.io import DicomReadError, ReadCancelled
from cmvd_demo.roi import new_document, set_track, extract_curves, document_hash
from cmvd_demo.perfusion import mr_features, case_token_for_choices, pair_mr
from cmvd_demo.work import KIND, save_record, load_record, restore_record, verify_record


def fixture(*, paired=False):
    choices = perfusion_synthetic_choices()
    case = case_token_for_choices(choices)
    entries, slots = {}, {}
    for choice in choices[:2 if paired else 1]:
        grid = choice.source
        doc = new_document(grid)
        doc['purpose'] = 'engineering_fixture'
        center = np.array([47.5,63.5])
        circle = np.column_stack([np.sin(np.arange(8)*np.pi/4),np.cos(np.arange(8)*np.pi/4)])
        for role,radius in [('myocardium',np.array([96,128])*.22),('blood_pool',np.array([96,128])*.07)]:
            contour = {'outer_rc':(center+circle*radius).tolist(),
                       'inner_rc':(center+circle*radius*.55).tolist() if role=='myocardium' else None}
            doc = set_track(doc,role,0,0,contour)
        curves = extract_curves(grid,doc,0)
        feature = mr_features(doc,curves,case,choice.condition,'SYNTH-TEST',
            baseline_frames=2,window_start=2,window_end=6,window_reviewed=False)
        entries[choice.key] = {'document':doc,'curves':{'0':curves}}
        slots[choice.condition] = {'key':choice.key,'document':doc,'curves':curves,'feature':feature}
    active = choices[1 if paired else 0]
    record = {'schema_version':1,'kind':KIND,'scope':'case_work' if paired else 'single_mr',
        'case_token':case,'active_key':active.key,'entries':entries,
        'view':{'time':4,'slice':0,'levels':[80.,160.],'view_range':[[0.,128.],[0.,192.]],
                'fps':15,'tab':1,'role':'blood_pool','scope':0,'tool':'points','scroll':[0,100]},
        'feature_state':{'settings':{'condition':active.condition,'protocol':'SYNTH-TEST',
            'baseline_start':1,'baseline_frames':2,'window_start':3,'window_end':7,
            'window_reviewed':False,'pairing_reviewed':False,'protocol_comparable':False},
            'candidate':{'key':active.key,'value':slots[active.condition]['feature']},
            'mr_slots':slots if paired else {},'pet_slots':{},'pet':None,'mr_reference':None}}
    return record,choices,ReaderBackend()


class WorkTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def save(self, **kwargs):
        record,choices,backend=fixture(**kwargs)
        return save_record(self.root,record,choices,backend),record,choices,backend

    def test_single_mr_complete_record_csv_and_no_model_readiness(self):
        run,record,choices,backend=self.save()
        loaded=load_record(run/'work.json')
        self.assertEqual(document_hash(loaded),document_hash(record))
        result=restore_record(run,choices,backend)
        self.assertEqual(result['record'],record)
        np.testing.assert_array_equal(result['pixels'],choices[0].source.read_plane(4,0))
        marker=json.loads((run/'bundle.json').read_text())
        self.assertFalse(marker['model_input_ready'])
        self.assertFalse(marker['clinical_validated'])
        self.assertTrue((run/'curves.json').is_file())
        self.assertIn('normalized_upslope',(run/'measurements.csv').read_text())
        self.assertFalse((run/'features.json').exists())
        self.assertNotIn('study_uid',(run/'work.json').read_text())

    def test_pair_all_documents_and_qc_survive_recomputation(self):
        run,record,choices,backend=self.save(paired=True)
        got=restore_record(run,choices,backend)['record']['feature_state']
        pair=pair_mr(got['mr_slots']['rest']['feature'],got['mr_slots']['stress']['feature'])
        self.assertAlmostEqual(pair['measurements']['mr_mpri']['value'],2.)
        self.assertEqual(got,record['feature_state'])
        self.assertFalse(pair['qc']['local_qc_passed'])

    def test_every_save_has_new_directory_and_identical_payload(self):
        run,record,choices,backend=self.save()
        another=save_record(self.root,record,choices,backend)
        self.assertNotEqual(run,another)
        self.assertEqual((run/'work.json').read_bytes(),(another/'work.json').read_bytes())

    def test_changed_file_bytes_rejected(self):
        run,*_=self.save()
        with (run/'work.json').open('a') as stream:stream.write(' ')
        with self.assertRaisesRegex(DicomReadError,'work_file_changed'):load_record(run)

    def test_forged_measurements_rejected_by_recalculation(self):
        record,choices,backend=fixture()
        record['feature_state']['candidate']['value']['measurements']['normalized_upslope']['value']+=1
        with self.assertRaisesRegex(DicomReadError,'work_recalculation_mismatch'):
            verify_record(record,choices,backend)

    def test_forged_curve_rejected_by_original_pixels(self):
        record,choices,backend=fixture()
        record['entries'][record['active_key']]['curves']['0']['curves'][0]['mean'][2]+=10
        with self.assertRaisesRegex(DicomReadError,'work_recalculation_mismatch'):
            verify_record(record,choices,backend)

    def test_changed_source_pixels_rejected_even_if_binding_unchanged(self):
        record,choices,backend=fixture()
        original=choices[0].source.read_plane
        choices[0].source.read_plane=lambda t,z:original(t,z)+1
        with self.assertRaisesRegex(DicomReadError,'work_recalculation_mismatch'):
            verify_record(record,choices,backend)

    def test_wrong_case_missing_series_and_geometry_rejected(self):
        for mode in ('case','series','geometry'):
            with self.subTest(mode=mode):
                record,choices,backend=fixture()
                if mode=='case':record['case_token']='0'*64
                elif mode=='series':record['active_key']='not-present'
                else:choices[0].source.pixel_spacing_mm=(3.,1.)
                with self.assertRaises(DicomReadError):verify_record(record,choices,backend)

    def test_invalid_view_and_analysis_parameter_mismatch_rejected(self):
        for field,value in [('time',70),('fps',100),('levels',[float('inf'),1]),('scope',True)]:
            with self.subTest(field=field):
                record,choices,backend=fixture()
                record['view'][field]=value
                with self.assertRaises(DicomReadError):verify_record(record,choices,backend)
        record,choices,backend=fixture()
        record['feature_state']['settings']['window_start']=4
        with self.assertRaisesRegex(DicomReadError,'work_invalid_settings'):verify_record(record,choices,backend)

    def test_partial_directory_and_arbitrary_marker_paths_rejected(self):
        run,*_=self.save()
        (run/'bundle.json').unlink()
        with self.assertRaisesRegex(DicomReadError,'work_completion_required'):load_record(run)
        run,*_=self.save()
        marker=json.loads((run/'bundle.json').read_text())
        marker['files_sha256']['../other.json']='0'*64
        (run/'bundle.json').write_text(json.dumps(marker))
        with self.assertRaisesRegex(DicomReadError,'work_file_set'):load_record(run)

    def test_csv_cannot_be_forged_by_updating_only_its_file_hash(self):
        run,*_=self.save()
        csv=run/'measurements.csv'
        csv.write_text(csv.read_text().replace('normalized_upslope','wrong_metric'))
        marker=json.loads((run/'bundle.json').read_text())
        marker['files_sha256']['measurements.csv']=hashlib.sha256(csv.read_bytes()).hexdigest()
        (run/'bundle.json').write_text(json.dumps(marker))
        with self.assertRaisesRegex(DicomReadError,'work_recalculation_mismatch'):load_record(run)

    def test_cancel_during_pixel_recompute_leaves_incomplete_evidence(self):
        record,choices,backend=fixture()
        original=backend.plane
        stopped=[]
        def plane(*args):
            value=original(*args)
            stopped.append(True)
            return value
        backend.plane=plane
        with self.assertRaises(ReadCancelled):
            save_record(self.root,record,choices,backend,cancel=lambda:bool(stopped))
        attempts=list(self.root.iterdir())
        self.assertEqual(len(attempts),1)
        self.assertTrue((attempts[0]/'attempt.json').is_file())
        self.assertFalse((attempts[0]/'bundle.json').exists())

    def test_write_failure_does_not_publish_completion(self):
        record,choices,backend=fixture()
        from cmvd_demo import work
        original=work.write_json
        def fail(path,value):
            if path.name=='work.json':raise OSError('INJECTED')
            original(path,value)
        with patch.object(work,'write_json',side_effect=fail):
            with self.assertRaises(OSError):save_record(self.root,record,choices,backend)
        self.assertFalse(any(p.name=='bundle.json' for p in self.root.rglob('*')))

    def test_curves_and_analysis_are_optional_for_unfinished_work(self):
        record,choices,backend=fixture()
        record['scope']='case_work'
        record['entries'][record['active_key']]['curves']={}
        record['feature_state']['candidate']=None
        record['feature_state']['settings']['protocol']=''
        run=save_record(self.root,record,choices,backend)
        self.assertFalse((run/'curves.json').exists())
        self.assertEqual(restore_record(run,choices,backend)['record'],record)

    def test_malformed_records_have_safe_error_codes(self):
        record,choices,backend=fixture()
        for bad in (None,{'kind':KIND},{**record,'feature_state':None}):
            with self.subTest(kind=type(bad).__name__):
                with self.assertRaises(DicomReadError):verify_record(bad,choices,backend)

    def test_record_changed_during_recompute_is_rejected_at_end(self):
        run,record,choices,backend=self.save()
        original=backend.plane
        changed=[]
        def plane(*args):
            if not changed:
                with (run/'work.json').open('a') as stream:stream.write(' ')
                changed.append(True)
            return original(*args)
        backend.plane=plane
        with self.assertRaisesRegex(DicomReadError,'work_file_changed'):restore_record(run,choices,backend)


if __name__=='__main__':unittest.main()

from test_desktop import discard_unsaved
"""Engineering states, advisory history, bounded evidence and native links."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cmvd_demo.io import DicomReadError,ReadCancelled
from cmvd_demo.sequence_review import SequenceReview,save_review,load_history
from cmvd_demo.desktop.data import Choice,PhantomGrid
from cmvd_demo.desktop.window import BrowserWindow
from test_desktop import APP,until,dispose_window


KEY='a'*16
def choices():return [SimpleNamespace(key=KEY,kind='mr_perfusion',count=7,condition='unknown')]


class SequenceModelTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.m=SequenceReview()
        self.m.populate(choices(),{'issues':[]},'b'*64)

    def report(self,**extra):
        row={'index':{'series':[{'series_token':KEY}],'issues':[]},'grids':[],'grid_failures':[]}
        row.update(extra)
        p=self.root/'ARCH-001.json';p.write_text(json.dumps(row))
        (self.root/'bundle.json').write_text(json.dumps({'completed':True,'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()}}))
        return p

    def test_other_frame_success_cannot_erase_a_failed_frame(self):
        self.m.grid_ok(KEY,(7,1,32,40));self.m.plane_ok(KEY,0,0)
        self.m.failure(KEY,'pixels','pixel_decode_failed_ValueError',[3,0])
        self.m.plane_ok(KEY,5,0)
        self.assertEqual(self.m.rows[KEY]['state'],'partial_failure')
        self.assertTrue(self.m.issues[0]['active'])
        self.m.plane_ok(KEY,3,0)
        self.assertEqual(self.m.rows[KEY]['state'],'previewed')
        self.assertFalse(self.m.issues[0]['active'])
        self.assertEqual(len(self.m.issues),1)

    def test_header_failure_stays_unassigned_and_arbitrary_text_not_copied(self):
        self.m.populate(choices(),{'issues':[{'stage':'header_read','code':'PATIENT_SENTINEL_PATH',
             'file_token':'c'*16,'path':'PATIENT_SENTINEL_PATH','error_type':'PATIENT_SENTINEL_NAME'}]},'b'*64)
        snapshot=self.m.snapshot()
        self.assertIsNone(snapshot['issues'][0]['sequence_key'])
        self.assertEqual(snapshot['issues'][0]['code'],'unknown_read_failure')
        self.assertNotIn('PATIENT_SENTINEL',json.dumps(snapshot))

    def test_cancel_and_rebuild_do_not_claim_all_pixels_or_training(self):
        self.m.begin(KEY,'grid');self.m.cancelled(KEY,'grid')
        self.assertEqual(self.m.rows[KEY]['state'],'cancelled')
        self.m.failure(KEY,'grid','missing_time_axis')
        self.assertEqual(self.m.rows[KEY]['state'],'failed')
        self.m.begin(KEY,'grid');self.m.grid_ok(KEY,(7,1,32,40));self.m.plane_ok(KEY,0,0)
        self.assertFalse(self.m.issues[0]['active'])
        self.assertFalse(self.m.snapshot()['all_pixels_checked'])
        self.assertFalse(self.m.snapshot()['training_ready'])

    def test_export_is_new_and_cancelled_attempt_not_completed(self):
        one=save_review(self.root,self.m.snapshot());two=save_review(self.root,self.m.snapshot())
        self.assertNotEqual(one,two)
        self.assertTrue((one/'issues.csv').is_file())
        self.assertEqual(load_history(one/'review.json',{KEY},'b'*64)['source_type'],'prior_session')
        with self.assertRaises(ReadCancelled):save_review(self.root,self.m.snapshot(),cancel=lambda:True)

    def test_write_failure_leaves_no_completion(self):
        from cmvd_demo import sequence_review
        original=sequence_review.write_json
        def fail(path,value):
            if path.name=='review.json':raise OSError('SYNTHETIC')
            return original(path,value)
        with patch.object(sequence_review,'write_json',side_effect=fail):
            with self.assertRaises(OSError):save_review(self.root,self.m.snapshot())
        self.assertFalse(any(self.root.rglob('bundle.json')))

    def test_cancellation_mid_write_keeps_incomplete_attempt(self):
        from cmvd_demo import sequence_review
        original=sequence_review.write_json;cancelled=Event()
        def write(path,value):
            original(path,value)
            if path.name=='review.json':cancelled.set()
        with patch.object(sequence_review,'write_json',side_effect=write):
            with self.assertRaises(ReadCancelled):save_review(self.root,self.m.snapshot(),cancel=cancelled.is_set)
        self.assertEqual(len(list(self.root.rglob('attempt.json'))),1)
        self.assertFalse(any(self.root.rglob('bundle.json')))

    def test_oversized_or_incomplete_history_cannot_be_silently_truncated(self):
        p=self.report(grid_failures=[{'grid_token':KEY,'code':'missing_time_axis'}]*2001)
        with self.assertRaisesRegex(DicomReadError,'sequence_report_too_large'):load_history(p,{KEY},'b'*64)
        marker=self.root/'bundle.json';row=json.loads(marker.read_text());row['completed']=False
        marker.write_text(json.dumps(row))
        with self.assertRaisesRegex(DicomReadError,'sequence_report_incomplete'):load_history(p,{KEY},'b'*64)

    def test_report_changed_while_loading_is_rejected(self):
        from cmvd_demo import sequence_review
        p=self.report();original=sequence_review.load_json
        def read(path):
            value=original(path)
            if path==p:p.write_text('{}')
            return value
        with patch.object(sequence_review,'load_json',side_effect=read):
            with self.assertRaisesRegex(DicomReadError,'sequence_report_changed'):load_history(p,{KEY},'b'*64)

    def test_prior_session_preserves_frame_and_combined_sequence_key_as_advice(self):
        key='cine-stack-1';self.m.rows[key]=dict(self.m.rows[KEY],key=key)
        self.m.issue(key,'pixels','missing_geometry',frame=[3,0])
        run=save_review(self.root,self.m.snapshot())
        row=load_history(run/'review.json',{KEY,key},'b'*64)['rows'][0]
        self.assertEqual(row['sequence_key'],key)
        self.assertEqual(row['frame'],[3,0])

    def test_history_rejects_wrong_inventory_hash_and_aggregate(self):
        p=self.report()
        with self.assertRaisesRegex(DicomReadError,'sequence_report_case_mismatch'):load_history(p,{'d'*16},'b'*64)
        p.write_text('{}')
        with self.assertRaisesRegex(DicomReadError,'sequence_report_incomplete'):load_history(p,{KEY},'b'*64)

    def test_historical_zero_slope_is_kept_without_granting_access(self):
        p=self.report(branch_inventory={'series':[{'series_token':KEY,'zero_rescale_slope_objects':1,
                         'unique_sop_count':3,'objects_in_selected_index':4}]},
                      patient_name='PATIENT_SENTINEL',duplicate_selection={'conflicting_groups':1})
        got=load_history(p,{KEY},'b'*64)
        self.assertEqual(got['source_type'],'transient_index')
        self.assertEqual({r['code'] for r in got['rows']},{'zero_rescale_slope','duplicate_sop_instance'})
        self.assertNotIn('PATIENT_SENTINEL',json.dumps(got))
        self.assertEqual(self.m.rows[KEY]['state'],'pending')

    def test_ledger_is_bounded_and_omission_explicit(self):
        for n in range(2100):self.m.issue(KEY,'pixels','unknown_read_failure',frame=[n,0])
        for n in range(300):self.m.event(KEY,'pixels','accepted',[n,0])
        snapshot=self.m.snapshot()
        self.assertEqual(len(snapshot['issues']),2000)
        self.assertEqual(snapshot['omitted_issues'],100)
        self.assertLessEqual(len(snapshot['events']),256)
        self.assertGreater(snapshot['omitted_events'],0)


class SequenceDesktopTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.w=BrowserWindow(unsaved_prompt=discard_unsaved);self.w.show()
        self.w.sequences.output_root=Path(self.temp.name)
        self.w.open_synthetic();until(lambda:self.w.current_pixels is not None)
        self.panel=self.w.sequences

    def tearDown(self):dispose_window(self.w);self.temp.cleanup()

    def test_status_filter_and_index_only_selection_do_not_decode(self):
        w=self.w;w._reset_case()
        grid=PhantomGrid()
        rows=[Choice('supported','S1','synthetic','unknown',60,grid),
              Choice('unsupported','S2','secondary_capture','unknown',1,None)]
        w._populate(rows,{'indexed_objects':61,'series_count':2,'issues':[]})
        until(lambda:w.current_pixels is not None)
        self.assertEqual(self.panel.model.rows['supported']['state'],'previewed')
        w.sequence_filter.setCurrentIndex(w.sequence_filter.findData('index_only'))
        self.assertTrue(w.tree.topLevelItem(0).isHidden())
        w.select_choice('unsupported')
        self.assertIsNone(w.current_pixels)
        self.assertIsNone(w.grid)
        self.assertIn('仅索引',w.empty_title.text())
        self.assertEqual(self.panel.model.rows['unsupported']['state'],'index_only')

    def test_grid_failure_is_visible_and_retry_keeps_history(self):
        key=self.w.current_choice.key;original=self.w.backend.grid
        def failed(*args):raise DicomReadError('missing_time_axis')
        self.w.backend.grid=failed;self.w.select_choice(key)
        until(lambda:bool(self.w.last_error))
        self.assertEqual(self.panel.model.rows[key]['state'],'partial_failure')
        self.panel.open();self.assertGreater(self.panel.table.rowCount(),0)
        self.w.backend.grid=original;self.panel.table.selectRow(0);self.panel.locate_button.click()
        until(lambda:self.w.current_pixels is not None)
        self.assertEqual(self.panel.model.rows[key]['state'],'previewed')
        self.assertFalse(self.panel.model.issues[0]['active'])
        self.panel.visibility.setCurrentIndex(1)
        self.assertGreater(self.panel.table.rowCount(),0)

    def test_disposition_export_and_own_report_import_are_advisory(self):
        key=self.w.current_choice.key
        self.panel.model.issue(key,'qc','condition_requires_review');self.panel.open()
        self.panel.table.selectRow(0)
        self.panel.disposition.setCurrentIndex(1);self.panel.mark_button.click()
        self.assertEqual(self.panel.model.issues[0]['disposition'],'hospital')
        self.panel.export_button.click();until(lambda:not self.panel.runner.busy)
        self.assertFalse(self.panel.last_error)
        run=self.panel.saved_output;self.assertTrue((run/'bundle.json').is_file())
        state=self.panel.model.rows[key]['state']
        self.panel.import_history(run/'review.json');until(lambda:not self.panel.runner.busy)
        self.assertFalse(self.panel.last_error)
        self.assertEqual(self.panel.model.rows[key]['state'],state)
        self.assertTrue(any(row['origin']=='history' for row in self.panel.model.issues))

    def test_case_switch_rejects_late_import_and_close_waits_for_seventh_channel(self):
        started,release=Event(),Event();self.addCleanup(release.set)
        def delayed(*args,**kwargs):
            started.set();release.wait(3)
            return {'rows':[],'source_sha256':'a'*64,'source_type':'prior_session'}
        with patch('cmvd_demo.desktop.sequence_panel.load_history',side_effect=delayed):
            self.panel.import_history('not-read');until(started.is_set)
            self.w.open_synthetic_perfusion();self.w.close()
            self.assertTrue(self.panel.runner.busy)
            release.set();until(lambda:not any(r.busy for r in self.w.background_runners()))
        self.assertFalse(any(row['origin']=='history' for row in self.panel.model.issues))

    def test_frame_failure_locates_exact_frame_and_recovery_does_not_rebuild_table_each_frame(self):
        w,p=self.w,self.panel;key=w.current_choice.key
        p.model.failure(key,'pixels','pixel_decode_failed',[3,2]);p.open()
        p.table.selectRow(0);p.locate_button.click()
        until(lambda:w.displayed_indices==(3,2))
        self.assertFalse(p.model.issues[0]['active'])
        self.assertEqual(p.model.rows[key]['state'],'previewed')
        p.visibility.setCurrentIndex(1);item=p.table.item(0,0)
        w.time_slider.setValue(4);until(lambda:w.displayed_indices==(4,2))
        self.assertIs(p.table.item(0,0),item)

    def test_switch_to_index_only_cancels_old_loading_status_and_late_result(self):
        w=self.w;original=w.backend.grid;started,release=Event(),Event();self.addCleanup(release.set)
        key=w.current_choice.key
        def delayed(*args):
            started.set();release.wait(3);return original(*args)
        w._reset_case();grid=PhantomGrid()
        w.backend.grid=delayed
        w._populate([Choice(key,'Synthetic','synthetic','unknown',60,grid),
                     Choice('index-only','Synthetic SC','secondary_capture','unknown',1,None)],
                    {'indexed_objects':61,'series_count':2,'issues':[]})
        until(started.is_set);w.select_choice('index-only')
        self.assertEqual(self.panel.model.rows[key]['state'],'cancelled')
        release.set();until(lambda:not w.runner.busy)
        self.assertIsNone(w.current_pixels)
        self.assertEqual(self.panel.model.rows['index-only']['state'],'index_only')


if __name__=='__main__':unittest.main()

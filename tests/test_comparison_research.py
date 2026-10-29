from test_desktop import discard_unsaved
"""Comparison timing, append-only exports and native usage/restore/error paths."""
import copy,csv,hashlib,json,tempfile,sys,unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cmvd_demo.io import DicomReadError,ReadCancelled
from cmvd_demo.roi import binding_for,document_hash
from cmvd_demo.comparison import relative_target,link_allowed
from cmvd_demo.sequence_review import SequenceReview
from cmvd_demo.research_record import save_research,load_research
from cmvd_demo.desktop.window import BrowserWindow
from cmvd_demo.desktop.data import Choice,PerfusionPhantomGrid
from test_work import fixture
from test_desktop import APP,until,dispose_window


def payload():
    work,choices,backend=fixture(paired=True)
    model=SequenceReview();model.populate(choices,{'issues':[]},work['case_token'])
    model.failure(choices[0].key,'pixels','pixel_decode_failed',[2,0]);model.plane_ok(choices[0].key,2,0)
    comparison={'linked_time':False,'views':{}}
    for c in choices:
        entry=work['entries'][c.key]
        comparison['views'][c.condition]={'key':c.key,'binding':binding_for(c.source),'time':4,'slice':0,'levels':[50.,100.],
            'roi_document_sha256':document_hash(entry['document']),'curves_sha256':document_hash(entry['curves']['0'])}
    return work,model.snapshot(),comparison,choices,backend


class CompareResearchCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def save(self):
        args=payload();run=save_research(self.root,*args,'0.7.0');return run,args

    def resign(self,run,name):
        marker=json.loads((run/'bundle.json').read_text());marker['files_sha256'][name]=hashlib.sha256((run/name).read_bytes()).hexdigest()
        (run/'bundle.json').write_text(json.dumps(marker))

    def test_relative_nonuniform_time_offsets_and_ties(self):
        self.assertEqual(relative_target([30,32,35],[100,101,103,110],1),1)
        self.assertEqual(relative_target([30,32,35],[100,101,103,110],2),2)
        with self.assertRaisesRegex(DicomReadError,'compare_time_outside'):relative_target([0,5,12],[0,2,7],2)

    def test_time_link_rejects_units_and_cardiac_phase(self):
        _,_,c,_,_=payload();left=c['views']['rest']['binding'];right=copy.deepcopy(left)
        self.assertTrue(link_allowed(left,right))
        right['units']=['BQML'];self.assertFalse(link_allowed(left,right))
        right=copy.deepcopy(left);right['timing_basis']='cardiac_trigger_time';self.assertFalse(link_allowed(left,right))

    def test_export_known_measurements_and_original_curves_reopen(self):
        run,args=self.save();got=load_research(run/'research.json')
        self.assertEqual(got['case_token'],args[0]['case_token'])
        with (run/'measurements.csv').open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertAlmostEqual(float(next(r['value'] for r in rows if r['measurement']=='mr_mpri')),2.)
        self.assertIn('frame_duration_s',(run/'curves.csv').read_text(encoding='utf-8-sig'))
        self.assertEqual(got['comparison'],args[2]);self.assertFalse(got['clinical_validated'])
        self.assertTrue((run/got['work_relative']).is_file())
        self.assertIn('不代表', (run/'summary.html').read_text(encoding='utf-8'))

    def test_each_export_is_new_and_no_patient_source_paths(self):
        one,args=self.save();two=save_research(self.root,*args,'0.7.0')
        self.assertNotEqual(one,two)
        for file in (one/'research.json',one/'summary.html'):
            self.assertNotIn('study_uid',file.read_text(encoding='utf-8'))
            self.assertNotIn('source_path',file.read_text(encoding='utf-8'))

    def test_wrong_case_and_stale_roi_comparison_are_rejected(self):
        work,review,comparison,choices,backend=payload();review['case_token']='f'*64
        with self.assertRaisesRegex(DicomReadError,'research_case_mismatch'):save_research(self.root,work,review,comparison,choices,backend,'0.7.0')
        review['case_token']=work['case_token'];comparison['views']['rest']['roi_document_sha256']='f'*64
        with self.assertRaisesRegex(DicomReadError,'research_stale_result'):save_research(self.root,work,review,comparison,choices,backend,'0.7.0')
        self.assertFalse(any(p.name=='bundle.json' for p in self.root.glob('*/bundle.json')))

    def test_source_binding_mismatch_is_not_hidden_by_unannotated_view(self):
        work,review,comparison,choices,backend=payload();view=comparison['views']['rest']
        view['roi_document_sha256']=view['curves_sha256']=None;view['binding']['orientation_lps'][0]=-1
        with self.assertRaisesRegex(DicomReadError,'research_stale_result'):save_research(self.root,work,review,comparison,choices,backend,'0.7.0')
        self.assertFalse(any(self.root.glob('*/bundle.json')))

    def test_cancel_mid_write_keeps_partial_without_outer_marker(self):
        from cmvd_demo import research_record as module
        cancelled=Event();original=module.write_json
        def write(path,value):
            original(path,value)
            if path.name=='research.json':cancelled.set()
        with patch.object(module,'write_json',side_effect=write):
            with self.assertRaises(ReadCancelled):save_research(self.root,*payload(),'0.7.0',cancel=cancelled.is_set)
        self.assertTrue(any(self.root.glob('*/attempt.json')))
        self.assertFalse(any(self.root.glob('*/bundle.json')))

    def test_export_write_error_leaves_incomplete_attempt(self):
        from cmvd_demo import research_record as module
        original=module.write_json
        def fail(path,value):
            if path.name=='research.json':raise OSError('SYNTHETIC')
            original(path,value)
        with patch.object(module,'write_json',side_effect=fail):
            with self.assertRaises(OSError):save_research(self.root,*payload(),'0.7.0')
        self.assertFalse(any(self.root.glob('*/bundle.json')))

    def test_missing_marker_and_changed_csv_are_rejected(self):
        run,_=self.save();(run/'curves.csv').write_text('changed')
        with self.assertRaisesRegex(DicomReadError,'research_file_changed'):load_research(run)
        (run/'bundle.json').unlink()
        with self.assertRaisesRegex(DicomReadError,'research_incomplete'):load_research(run)

    def test_resigned_inconsistent_csv_or_html_is_rejected(self):
        run,_=self.save();(run/'measurements.csv').write_text('fabricated');self.resign(run,'measurements.csv')
        with self.assertRaisesRegex(DicomReadError,'research_file_changed'):load_research(run)
        other,_=self.save();(other/'summary.html').write_text('<script>not trusted</script>');self.resign(other,'summary.html')
        with self.assertRaisesRegex(DicomReadError,'research_file_changed'):load_research(other)

    def test_manifest_path_escape_rejected_before_file_access(self):
        run,_=self.save();marker=json.loads((run/'bundle.json').read_text())
        value=marker['files_sha256'].pop('curves.csv');marker['files_sha256']['../outside.csv']=value
        (run/'bundle.json').write_text(json.dumps(marker))
        with self.assertRaisesRegex(DicomReadError,'research_invalid_record'):load_research(run)

    def test_current_work_sources_rechecked_after_export_writes(self):
        from cmvd_demo import research_record as module
        original=module.check_sources
        with patch.object(module,'check_sources',side_effect=DicomReadError('source_changed_since_index')):
            with self.assertRaisesRegex(DicomReadError,'source_changed_since_index'):save_research(self.root,*payload(),'0.7.0')
        self.assertTrue(any(self.root.glob('*/summary.html')))
        self.assertFalse(any(self.root.glob('*/bundle.json')))


class CompareResearchDesktopTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.windows=[]
        self.w=self.window();self.c=self.w.comparison;self.r=self.w.research

    def window(self):
        w=BrowserWindow(unsaved_prompt=discard_unsaved);self.windows.append(w);w.research.output_root=self.root/'research';w.work.output_root=self.root/'work'
        w.show();w.open_synthetic_perfusion();until(lambda:w.current_pixels is not None);return w

    def tearDown(self):
        for w in self.windows:dispose_window(w)
        self.temp.cleanup()

    def compare(self):
        self.c.open();until(lambda:all(p.displayed is not None for p in self.c.panes.values()))

    def test_open_two_sides_does_not_change_main_selection_or_values(self):
        main=self.w.current_choice.key;pixels=self.w.current_pixels.copy();self.compare()
        for s,p in self.c.panes.items():
            self.assertEqual(p.key,'synthetic-perfusion-'+s)
            np.testing.assert_array_equal(p.pixels,p.grid.read_plane(0,0))
        self.assertEqual(self.w.current_choice.key,main);np.testing.assert_array_equal(self.w.current_pixels,pixels)
        self.assertFalse(self.c.link.isChecked())

    def test_independent_navigation_then_optional_relative_link(self):
        self.compare();rest,stress=(self.c.panes[s] for s in ('rest','stress'))
        rest.time.setValue(4);until(lambda:rest.displayed==(4,0))
        self.assertEqual(stress.displayed,(0,0))
        self.c.link.setChecked(True);rest.time.setValue(5)
        until(lambda:rest.displayed==(5,0) and stress.displayed==(5,0))

    def test_link_out_of_range_keeps_other_image_and_final_drag_position(self):
        self.compare();rest,stress=(self.c.panes[s] for s in ('rest','stress'))
        self.c.link.setChecked(True)
        stress.grid.times_s=np.array([[0.,.1,.2,.3,.4,.5,.6]])
        rest.time.setValue(6);until(lambda:rest.displayed==(6,0))
        self.assertEqual(stress.displayed,(0,0));self.assertIn('范围外',self.c.status.text())
        self.c.link.setChecked(False)
        for t in (1,4,2,5):rest.time.setValue(t)
        until(lambda:rest.displayed==(5,0));np.testing.assert_array_equal(rest.pixels,rest.grid.read_plane(5,0))

    def test_same_grid_wait_keeps_old_image_and_metadata_then_exact_final_frame(self):
        self.compare();p=self.c.panes['rest'];started,release=Event(),Event();self.addCleanup(release.set)
        original=self.w.backend.plane
        def slow(grid,t,z,cached,progress,cancel):
            if t==3:started.set();release.wait(3)
            return original(grid,t,z,cached,progress,cancel)
        self.w.backend.plane=slow;p.time.setValue(3);until(started.is_set)
        old=p.pixels.copy();self.assertEqual(p.displayed,(0,0));self.assertIn('已显示帧 1/',p.label.text())
        np.testing.assert_array_equal(p.pixels,old)
        p.time.setValue(5);release.set();until(lambda:p.displayed==(5,0))

    def test_case_switch_and_close_cancel_all_ten_channels_without_late_publication(self):
        self.compare();started,release=Event(),Event();self.addCleanup(release.set);original=self.w.backend.plane
        def slow(*args):started.set();release.wait(3);return original(*args)
        self.w.backend.plane=slow;self.c.panes['rest'].time.setValue(2);until(started.is_set)
        self.w.open_synthetic();self.w.close();release.set()
        until(lambda:not any(r.busy for r in self.w.background_runners()))
        self.assertEqual(len(self.w.background_runners()),10)
        self.assertIsNone(self.c.panes['rest'].pixels)

    def test_export_reopen_verify_and_restore_from_research_dialog(self):
        self.compare();self.r.open();self.r.export_button.click();until(lambda:not self.r.busy,timeout=15000)
        self.assertFalse(self.r.last_error,self.r.last_error);run=self.r.last_output
        self.assertIsNotNone(run);self.assertTrue((run/'summary.html').is_file())
        self.r.verify(run/'research.json');until(lambda:not self.r.busy)
        self.assertFalse(self.r.last_error,self.r.last_error)
        self.w.level_center.setValue(123.)
        self.r.restore_button.click();until(lambda:not self.w.work.busy)
        self.assertFalse(self.w.work.last_error,self.w.work.last_error)
        self.assertNotEqual(self.w.level_center.value(),123.)

    def test_retained_two_state_measurements_and_readonly_overlay_survive_navigation(self):
        from test_work_desktop import WorkDesktopTests
        for state in ('rest','stress'):
            WorkDesktopTests.calculate(self,self.w,state);self.w.feature_panel.capture_mr()
        self.compare();self.assertEqual(self.c.table.rowCount(),17)
        self.assertIn('心肌/血池',self.c.table.item(0,0).text())
        p=self.c.panes['rest'];ids=[id(v) for v in p.overlays.values()]
        for t in (4,2,6):
            p.time.setValue(t);until(lambda:p.displayed==(t,0))
            self.assertEqual([id(v) for v in p.overlays.values()],ids)
            self.assertTrue(all(v.isVisible() for v in p.overlays.values()))
        self.r.export();until(lambda:not self.r.busy,timeout=15000)
        self.assertFalse(self.r.last_error,self.r.last_error)
        text=(self.r.last_output/'summary.html').read_text(encoding='utf-8')
        self.assertIn('MR灌注储备指数',text)
        self.assertEqual(self.w.feature_panel._mr_pair()['qc']['clinical_validated'],False)

    def test_comparison_failure_keeps_other_side_and_can_export_problem(self):
        self.compare();left=self.c.panes['rest'];right=self.c.panes['stress'];original=self.w.backend.grid
        def failed(choice,*args):
            if choice.condition=='rest':raise DicomReadError('missing_time_axis')
            return original(choice,*args)
        self.w.backend.grid=failed;left.retry.click();until(lambda:bool(left.last_error))
        self.assertIsNone(left.pixels);self.assertIsNotNone(right.pixels)
        self.assertTrue(any(i['code']=='missing_time_axis' for i in self.w.sequences.model.issues))
        self.r.export();until(lambda:not self.r.busy,timeout=15000)
        # The main work source is the failed series; strict reassembly must reject it.
        self.assertIsNone(self.r.last_output);self.assertEqual(self.r.last_error,'missing_time_axis')
        self.w.backend.grid=original
        self.r.export();until(lambda:not self.r.busy,timeout=15000)
        self.assertFalse(self.r.last_error,self.r.last_error)
        self.assertIsNone(self.r.data['comparison']['views']['rest'])

    def test_display_levels_are_independent_and_leave_raw_pixels_unchanged(self):
        self.compare();left,right=(self.c.panes[s] for s in ('rest','stress'))
        before=left.pixels.copy();other=right.center.value();left.center.setValue(123.);left.width.setValue(77.)
        np.testing.assert_array_equal(left.pixels,before);self.assertEqual(right.center.value(),other)

    def test_each_slice_uses_own_times_and_no_layer_link(self):
        class TwoSlice(PerfusionPhantomGrid):
            shape=(7,2,96,128)
            def __init__(self,state):
                super().__init__(state);self.synthetic_id+='-2SLICE'
                self.times_s=np.array([[0,1,2,4,7,8,10],[0,.5,1,2,3,5,8]],dtype=float)
        self.w._reset_case()
        choices=[Choice('two-'+s,'Synthetic '+s,'mr_perfusion',s,14,TwoSlice(s)) for s in ('rest','stress')]
        self.w._populate(choices,{'indexed_objects':28,'series_count':2,'issues':[]})
        until(lambda:self.w.current_pixels is not None);self.compare()
        left,right=(self.c.panes[s] for s in ('rest','stress'))
        left.slice.setValue(1);until(lambda:left.displayed==(0,1));self.assertEqual(right.slice.value(),0)
        self.c.link.setChecked(True);left.time.setValue(4)
        until(lambda:left.displayed==(4,1) and right.displayed==(2,0))

    def test_menu_lifetimes_and_close_comparison_then_reopen(self):
        self.compare();left=self.c.panes['rest']
        self.assertIs(left.plot.getPlotItem().ctrlMenu.parent(),left.plot)
        self.c.dialog.close();until(lambda:not self.c.busy)
        self.assertIsNone(left.pixels)
        self.compare();self.assertIsNotNone(left.pixels)

    def test_generated_pet_exports_real_units_durations_and_imported_mfr(self):
        from test_perfusion_desktop import PerfusionDesktopTests
        self.f=self.w.feature_panel;self.f.output_root=self.root/'features'
        path=PerfusionDesktopTests.prepare_pet(self);self.f.load_pet(path)
        until(lambda:self.f.pet is not None or bool(self.f.last_error));self.assertFalse(self.f.last_error)
        self.compare();self.r.export();until(lambda:not self.r.busy,timeout=15000)
        self.assertFalse(self.r.last_error,self.r.last_error)
        run=self.r.last_output
        with (run/'curves.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
        self.assertTrue(all(r['unit']=='BQML' for r in rows));self.assertTrue(all(float(r['frame_duration_s'])>0 for r in rows))
        with (run/'measurements.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
        self.assertAlmostEqual(float(next(r['value'] for r in rows if r['measurement']=='pet_mfr')),3.)
        self.assertTrue(all(r['clinical_validated']=='False' for r in rows))

    def test_source_changed_after_nested_save_is_rejected_before_outer_completion(self):
        import os
        from test_perfusion_desktop import PerfusionDesktopTests
        from cmvd_demo import research_record as module
        self.f=self.w.feature_panel;self.f.output_root=self.root/'features'
        PerfusionDesktopTests.prepare_pet(self)
        header=self.f.pet_slots['rest']['grid'].cells[0][0];original=module.tables
        def change(*args,**kwargs):
            tables=original(*args,**kwargs);stat=header.path.stat()
            os.utime(header.path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000));return tables
        with patch.object(module,'tables',side_effect=change):
            self.r.export();until(lambda:not self.r.busy,timeout=15000)
        self.assertEqual(self.r.last_error,'source_changed_since_index');self.assertIsNone(self.r.last_output)
        self.assertFalse(any(self.r.output_root.glob('*/bundle.json')))

    def test_edited_roi_clears_compare_curves_and_exported_clinical_flags_stay_false(self):
        from test_work_desktop import WorkDesktopTests
        WorkDesktopTests.calculate(self,self.w,'rest');self.w.feature_panel.capture_mr();self.compare()
        p=self.c.panes['rest'];doc,result=p.content();self.assertIsNotNone(result)
        self.w.roi_panel.undo();self.assertIsNone(p.content()[1])
        values=p.curve_lines['myocardium'].xData
        self.assertTrue(values is None or len(values)==0)

    def test_cancelled_export_and_changed_case_reject_queued_success(self):
        self.compare();started,release=Event(),Event();self.addCleanup(release.set)
        def slow(*args,**kwargs):started.set();release.wait(3);return self.root,{}
        with patch('cmvd_demo.desktop.research_controller.save_research',side_effect=slow):
            self.r.export();until(started.is_set);self.r.cancel();self.w.open_synthetic();release.set()
            until(lambda:not self.r.busy)
        self.assertIsNone(self.r.last_output);self.assertFalse(self.r.last_error)

    def test_close_research_dialog_requests_cancel_and_unlocks_editor(self):
        self.r.open();started,release=Event(),Event();self.addCleanup(release.set)
        def slow(*args,**kwargs):started.set();release.wait(3);raise ReadCancelled()
        with patch('cmvd_demo.desktop.research_controller.save_research',side_effect=slow):
            self.r.export();until(started.is_set);self.assertFalse(self.w.centralWidget().isEnabled())
            self.r.dialog.close();self.assertTrue(self.w.centralWidget().isEnabled())
            release.set();until(lambda:not self.r.busy)
        self.assertIsNone(self.r.last_output);self.assertFalse(self.r.last_error)


if __name__=='__main__':unittest.main()

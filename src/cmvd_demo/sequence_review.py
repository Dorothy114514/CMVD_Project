"""Bounded engineering observations; no patient labels or reading-policy overrides."""
import copy
import csv
import hashlib
from pathlib import Path
import re

from cmvd_demo.io import DicomReadError, ReadCancelled
from cmvd_demo.perfusion.storage import load_json,write_json,new_run

STATES={'pending':'未预览','loading':'读取中','previewed':'已预览','failed':'读取失败',
        'partial_failure':'部分读取失败','cancelled':'已取消','index_only':'仅索引'}
DISPOSITIONS={'pending':'待核对','hospital':'需医院核源/补导出','deferred':'暂缓处理'}
SUPPORTED={'pet_dynamic','pet_gated','pet_static','mr_perfusion','mr_cine','cine_stack','synthetic'}
KINDS=SUPPORTED|{'mr_other','secondary_capture','unsupported_sop_class','unsupported_multiframe'}
CODES=set('''nonfinite_metadata invalid_geometry invalid_acquisition_time missing_identity_uid case_directory_missing
missing_geometry invalid_orientation inconsistent_geometry unsupported_pixel_layout inconsistent_slice_origins
series_not_supported_for_assembly duplicate_sop_instance inconsistent_frame_of_reference inconsistent_quantitative_metadata
inconsistent_acquisition_reference ambiguous_midnight_without_date missing_time_axis declared_frame_count_mismatch
declared_slice_count_mismatch duplicate_spatial_time_cell missing_spatial_time_cell slice_spacing_mismatch_or_gap
irregular_slice_spacing invalid_frame_durations cine_selection_required cine_study_or_reference_mismatch
cine_phase_count_or_plane_mismatch duplicate_cine_plane irregular_cine_slice_spacing cine_slice_spacing_mismatch_or_gap
source_changed_since_index source_changed_during_read pixel_shape_mismatch pet_rescale_or_units_missing
zero_rescale_slope nonfinite_scaled_pixels memory_budget_exceeded desktop_plane_memory_budget_exceeded
read_failed_unclassified metadata_parse_failed indexed_only historical_unverified_report qc_pending
unknown_read_failure pixel_decode_failed cancelled archive_plane_memory_budget archive_nonfinite_sample_pixels condition_requires_review phase_synchronization_not_performed'''.split())
STAGES={'header_read','metadata_parse','index','grid','pixels','capability','history','qc'}


def safe_code(code):
    if isinstance(code,str) and code in CODES:return code
    if isinstance(code,str) and code.startswith('pixel_decode_failed_'):return 'pixel_decode_failed'
    return 'unknown_read_failure'


def safe_key(key):
    if not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',key):
        raise DicomReadError('sequence_invalid_record')
    return key


def cancel_check(cancel):
    if cancel and cancel():raise ReadCancelled()


class SequenceReview:
    def __init__(self):
        self.rows={};self.issues=[];self.events=[];self.omitted_issues=0;self.omitted_events=0
        self.case_token=None;self.revision=0;self.index_complete=False

    def event(self,key,stage,result,frame=None):
        self.revision+=1
        self.events.append({'sequence_key':key,'stage':stage,'result':result,'frame':frame})
        if len(self.events)>256:self.events.pop(0);self.omitted_events+=1

    def issue(self,key,stage,code,*,frame=None,file_token=None,origin='session',source_sha256=None):
        stage=stage if stage in STAGES else 'history'
        code=safe_code(code)
        token=file_token if isinstance(file_token,str) and re.fullmatch('[0-9a-f]{16}',file_token) else None
        row={'sequence_key':key,'stage':stage,'code':code,'frame':frame,'file_token':token,
             'origin':origin,'source_sha256':source_sha256,'active':True,'disposition':'pending'}
        previous=next((r for r in self.issues if all(r[k]==row[k] for k in ('sequence_key','stage','code','frame','file_token','origin','source_sha256'))),None)
        if previous:previous['active']=True
        elif len(self.issues)<2000:self.issues.append(row)
        else:self.omitted_issues+=1

    def populate(self,choices,summary,case_token):
        self.__init__();self.case_token=case_token;self.index_complete=True
        for choice in choices:
            key=safe_key(choice.key)
            kind=choice.kind if choice.kind in KINDS else 'unsupported_sop_class'
            supported=kind in SUPPORTED
            self.rows[key]={'key':key,'kind':kind,'objects':int(choice.count),'state':'pending' if supported else 'index_only',
                'condition':choice.condition if choice.condition in ('rest','stress','unknown') else 'unknown',
                'frames':[],'frame_record_truncated':False,'shape':None}
            if not supported:self.issue(key,'capability','indexed_only')
        for row in summary.get('issues',[]):
            self.issue(None,row.get('stage'),row.get('code'),file_token=row.get('file_token'))
        self.event(None,'index','completed')

    def begin(self,key,stage,frame=None):
        if key not in self.rows:return
        for row in self.rows.values():
            if row['state']=='loading':row['state']='cancelled'
        self.rows[key]['state']='loading'
        self.event(key,stage,'started',frame)

    def grid_ok(self,key,shape,warnings=()):
        if key not in self.rows:return
        self.rows[key]['shape']=list(shape)
        self.recover(key,'grid')
        for warning in warnings:
            if warning!='synthetic_only':self.issue(key,'qc',warning if warning in CODES else 'qc_pending')
        self.event(key,'grid','accepted')

    def recover(self,key,stage,frame=None):
        for row in self.issues:
            if row['origin']=='session' and row['sequence_key']==key and row['stage']==stage and row['frame']==frame:
                row['active']=False

    def plane_ok(self,key,t,z):
        if key not in self.rows:return
        row=self.rows[key];frame=[int(t),int(z)]
        if frame not in row['frames']:
            if len(row['frames'])<2048:row['frames'].append(frame)
            else:row['frame_record_truncated']=True
        self.recover(key,'pixels',frame)
        row['state']='partial_failure' if any(i['active'] and i['origin']=='session' and i['sequence_key']==key
            and i['stage'] in ('grid','pixels') for i in self.issues) else 'previewed'
        self.event(key,'pixels','accepted',frame)

    def failure(self,key,stage,code,frame=None):
        if key in self.rows:self.rows[key]['state']='partial_failure' if self.rows[key]['frames'] else 'failed'
        self.issue(key if key in self.rows else None,stage,code,frame=frame)
        self.event(key,stage,'failed',frame)

    def cancelled(self,key,stage):
        if key in self.rows:self.rows[key]['state']='cancelled'
        self.event(key,stage,'cancelled')

    def snapshot(self):
        return copy.deepcopy({'schema_version':1,'kind':'sequence_review','case_token':self.case_token,
            'index_complete':self.index_complete,'revision':self.revision,'sequences':[self.rows[k] for k in sorted(self.rows)],
            'issues':self.issues,'events':self.events,'omitted_issues':self.omitted_issues,'omitted_events':self.omitted_events,
            'all_pixels_checked':False,'clinical_validated':False,'training_ready':False})


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_review(root,snapshot,*,cancel=None):
    cancel_check(cancel);run=new_run(root)
    write_json(run/'attempt.json',{'completed':False,'kind':'sequence_review'})
    write_json(run/'review.json',snapshot)
    with (run/'issues.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        keys=['sequence_key','stage','code','frame','file_token','origin','active','disposition','source_sha256']
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in snapshot['issues']:
            cancel_check(cancel);writer.writerow({k:row[k] for k in keys})
    cancel_check(cancel)
    write_json(run/'bundle.json',{'completed':True,'kind':'sequence_review','schema_version':1,
        'files_sha256':{p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file()},
        'clinical_validated':False,'all_pixels_checked':False})
    return run


def load_history(path,series_keys,case_token,*,cancel=None):
    """Whitelist evidence fields; never use a historical success to grant access."""
    try:
        path=Path(path)
        if path.stat().st_size>8*1024**2:raise DicomReadError('sequence_report_too_large')
        marker=load_json(path.parent/'bundle.json')
        expected=marker.get('files_sha256',{}).get(path.name)
        if marker.get('completed') is not True or not expected or path.is_symlink() or sha(path)!=expected:
            raise DicomReadError('sequence_report_incomplete')
        cancel_check(cancel);value=load_json(path)
        rows=[]
        def items(value):
            if not isinstance(value,list):raise DicomReadError('sequence_invalid_report')
            if len(value)>2000:raise DicomReadError('sequence_report_too_large')
            if not all(isinstance(item,dict) for item in value):raise DicomReadError('sequence_invalid_report')
            return value
        def add(key,stage,code,file_token=None,frame=None):
            if key is not None and key not in series_keys:key=None
            token=file_token if isinstance(file_token,str) and re.fullmatch('[0-9a-f]{16}',file_token) else None
            if not (isinstance(frame,list) and len(frame)==2 and all(type(v) is int and 0<=v<1000000 for v in frame)):frame=None
            rows.append({'sequence_key':key,'stage':stage,'code':safe_code(code),'file_token':token,'frame':frame})
        if value.get('kind')=='sequence_review':
            if not case_token or value.get('case_token')!=case_token:raise DicomReadError('sequence_report_case_mismatch')
            for item in items(value['issues']):
                add(item.get('sequence_key'),'history',item.get('code'),item.get('file_token'),item.get('frame'))
            source_type='prior_session'
        else:
            inspection=value.get('inspection',value)
            index=inspection.get('index',{})
            imported={safe_key(s['series_token']) for s in items(index.get('series',[]))}
            raw_keys={k for k in series_keys if re.fullmatch('[0-9a-f]{16}',k)}
            if not imported or imported!=raw_keys:raise DicomReadError('sequence_report_case_mismatch')
            for item in items(index.get('issues',[])):add(None,'history',item.get('code'),item.get('file_token'))
            for item in items(inspection.get('grid_failures',[])):add(item.get('grid_token'),'history',item.get('code'))
            for item in items(value.get('branch_inventory',{}).get('series',[])):
                if item.get('zero_rescale_slope_objects',0)>0:add(item.get('series_token'),'history','zero_rescale_slope')
                if item.get('unique_sop_count',0)<item.get('objects_in_selected_index',0):
                    add(item.get('series_token'),'history','duplicate_sop_instance')
            source_type='transient_index' if 'duplicate_selection' in value else 'archive_inspection'
        if len(rows)>2000:raise DicomReadError('sequence_report_too_large')
        cancel_check(cancel)
        if sha(path)!=expected or load_json(path.parent/'bundle.json')!=marker:
            raise DicomReadError('sequence_report_changed')
        return {'rows':rows,'source_sha256':expected,'source_type':source_type}
    except DicomReadError:raise
    except (OSError,KeyError,TypeError,ValueError):raise DicomReadError('sequence_invalid_report') from None

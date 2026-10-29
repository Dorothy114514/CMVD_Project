"""Append-only research exports; reuse verified image work records and reader."""
import copy,csv,hashlib,io,re
from html import escape
from pathlib import Path,PurePosixPath
import numpy as np
from cmvd_demo.io import DicomReadError
from cmvd_demo.roi import document_hash,binding_for,validate_document
from cmvd_demo.perfusion import pair_mr
from cmvd_demo.perfusion.labels import measurement_label,unit_label
from cmvd_demo.perfusion.storage import new_run,write_json,load_json
from cmvd_demo.work import save_record,load_record,check_sources,check_cancel,require
from cmvd_demo.sequence_review import safe_key,STATES,DISPOSITIONS,CODES,STAGES,KINDS

KIND='research_record'
MAX_BYTES=8*1024**2


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
def position(value):return isinstance(value,list) and len(value)==2 and all(type(v) is int and 0<=v<1000000 for v in value)


def validate_review(v,case):
    required={'schema_version','kind','case_token','index_complete','revision','sequences','issues','events',
        'omitted_issues','omitted_events','all_pixels_checked','clinical_validated','training_ready'}
    require(isinstance(v,dict) and set(v)==required,'research_invalid_record')
    require(v['schema_version']==1 and v['kind']=='sequence_review' and v['case_token']==case,'research_case_mismatch')
    require(all(v[k] is False for k in ('all_pixels_checked','clinical_validated','training_ready')),'research_invalid_record')
    require(type(v['index_complete']) is bool and all(type(v[k]) is int and v[k]>=0 for k in ('revision','omitted_issues','omitted_events')),'research_invalid_record')
    require(isinstance(v['sequences'],list) and len(v['sequences'])<=10000,'research_invalid_record')
    keys=set()
    for row in v['sequences']:
        require(set(row)=={'key','kind','objects','state','condition','frames','frame_record_truncated','shape'},'research_invalid_record')
        key=safe_key(row['key']);require(key not in keys,'research_invalid_record');keys.add(key)
        require(row['kind'] in KINDS and row['state'] in STATES and row['condition'] in ('rest','stress','unknown'),'research_invalid_record')
        require(type(row['objects']) is int and row['objects']>=0 and type(row['frame_record_truncated']) is bool,'research_invalid_record')
        require(isinstance(row['frames'],list) and len(row['frames'])<=2048 and all(position(p) for p in row['frames']),'research_invalid_record')
        require(row['shape'] is None or isinstance(row['shape'],list) and len(row['shape'])==4 and all(type(x) is int and x>0 for x in row['shape']),'research_invalid_record')
    require(isinstance(v['issues'],list) and len(v['issues'])<=2000 and isinstance(v['events'],list) and len(v['events'])<=256,'research_invalid_record')
    for row in v['issues']:
        require(set(row)=={'sequence_key','stage','code','frame','file_token','origin','source_sha256','active','disposition'},'research_invalid_record')
        require(row['sequence_key'] is None or row['sequence_key'] in keys,'research_invalid_record')
        require(row['stage'] in STAGES and row['code'] in CODES and row['origin'] in ('session','history') and type(row['active']) is bool and row['disposition'] in DISPOSITIONS,'research_invalid_record')
        require(row['file_token'] is None or re.fullmatch('[0-9a-f]{16}',row['file_token']) is not None,'research_invalid_record')
        require(row['source_sha256'] is None or digest(row['source_sha256']),'research_invalid_record')
        require(row['frame'] is None or position(row['frame']),'research_invalid_record')
    for event in v['events']:
        require(set(event)=={'sequence_key','stage','result','frame'} and (event['sequence_key'] is None or event['sequence_key'] in keys),'research_invalid_record')
        require(event['stage'] in STAGES and event['result'] in ('started','accepted','completed','cancelled','failed','imported') and (event['frame'] is None or position(event['frame'])),'research_invalid_record')


def validate_comparison(v,record):
    require(isinstance(v,dict) and set(v)=={'linked_time','views'} and type(v['linked_time']) is bool,'research_invalid_record')
    require(isinstance(v['views'],dict) and set(v['views'])=={'rest','stress'},'research_invalid_record')
    for view in v['views'].values():
        if view is None:continue
        require(set(view)=={'key','binding','time','slice','levels','roi_document_sha256','curves_sha256'},'research_invalid_record')
        safe_key(view['key']);b=view['binding'];t,z=view['time'],view['slice']
        validate_document({'schema_version':1,'purpose':'research_draft','binding':b,'tracks':[],
            'reviews':{'orientation_and_coverage':False,'motion_and_contours':False}})
        require(type(t) is int and type(z) is int and 0<=t<b['shape'][0] and 0<=z<b['shape'][1],'research_invalid_record')
        levels=np.asarray(view['levels'],dtype=float)
        require(levels.shape==(2,) and np.isfinite(levels).all() and abs(levels[0])<=1e12 and .001<=levels[1]<=1e12,'research_invalid_record')
        require(digest(b['source_signature']),'research_invalid_record')
        for key in ('roi_document_sha256','curves_sha256'):
            require(view[key] is None or digest(view[key]),'research_invalid_record')
        entry=record['entries'].get(view['key'])
        if view['roi_document_sha256'] is not None:
            require(entry and document_hash(entry['document'])==view['roi_document_sha256'] and entry['document']['binding']==b,'research_stale_result')
        if view['curves_sha256'] is not None:
            require(entry and str(z) in entry['curves'] and document_hash(entry['curves'][str(z)])==view['curves_sha256'],'research_stale_result')


def csv_text(headers,rows,cancel=None):
    stream=io.StringIO(newline='');writer=csv.writer(stream);writer.writerow(headers)
    for row in rows:
        check_cancel(cancel)
        writer.writerow([("'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v) for v in row])
        require(stream.tell()<=MAX_BYTES,'research_file_too_large')
    return stream.getvalue()


def measurement_rows(work):
    state=work['feature_state'];measurements=[];analyses={}
    if state['candidate']:analyses[document_hash(state['candidate']['value'])]=state['candidate']['value']
    for slot in state['mr_slots'].values():analyses[document_hash(slot['feature'])]=slot['feature']
    for _,value in sorted(analyses.items()):
        for name,item in sorted(value['measurements'].items()):
            measurements.append(['MR',value['condition'],name,item['value'],item['unit'],value['qc']['local_qc_passed'],False])
    if len(state['mr_slots'])==2:
        try:
            pair=pair_mr(state['mr_slots']['rest']['feature'],state['mr_slots']['stress']['feature'],
                pairing_reviewed=state['settings']['pairing_reviewed'],protocol_comparable=state['settings']['protocol_comparable'])
            for name,item in sorted(pair['measurements'].items()):measurements.append(['MR','pair',name,item['value'],item['unit'],pair['qc']['local_qc_passed'],False])
        except DicomReadError:pass  # Individual drafts remain exportable; no fabricated pair.
    if state['pet']:
        for name,item in sorted(state['pet']['measurements'].items()):measurements.append(['PET','workstation',name,item['value'],item['unit'],state['pet']['qc']['local_qc_passed'],False])
    return measurements


def tables(work,review,cancel=None,synthetic_data=False):
    measurements=measurement_rows(work)
    curves=[]
    for key,entry in sorted(work['entries'].items()):
        for z,result in sorted(entry['curves'].items()):
            for curve in result['curves']:
                for i,time in enumerate(curve['time_s']):
                    curves.append([key,int(z),curve['role'],i,time,None if curve['frame_duration_s'] is None else curve['frame_duration_s'][i],
                        curve['mean'][i],curve['std'][i],curve['pixel_count'][i],curve['area_mm2'][i],curve['units'],False])
    series=[[r['key'],r['kind'],r['condition'],r['objects'],STATES[r['state']],len(r['frames']),r['frame_record_truncated'],False] for r in review['sequences']]
    issues=[[r['sequence_key'],r['stage'],r['code'],r['frame'],r['file_token'],r['origin'],r['active'],DISPOSITIONS[r['disposition']],r['source_sha256']] for r in review['issues']]
    def output(headers,rows):return csv_text([*headers,'synthetic_data'],([*r,synthetic_data] for r in rows),cancel)
    return {'measurements.csv':output(['branch','state','measurement','value','unit','local_qc_passed','clinical_validated'],measurements),
        'curves.csv':output(['sequence_key','slice_zero_based','role','frame_zero_based','time_s','frame_duration_s','mean','std','pixel_count','area_mm2','unit','clinical_validated'],curves),
        'sequences.csv':output(['sequence_key','kind','condition_candidate','objects','state','observed_positions','positions_truncated','clinical_validated'],series),
        'issues.csv':output(['sequence_key','stage','code','frame_zero_based','file_token','origin','active','disposition','source_sha256'],issues)}


def summary_html(work,review,comparison,version,synthetic_data=False):
    def row(values):return '<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in values)+'</tr>'
    state=work['feature_state']
    summary=[('软件版本',version),('数据性质','完全合成演示：全部数值不是患者结果。' if synthetic_data else '受控研究资料，未临床验证。'),('病例安全摘要',work['case_token']),('已标注/活动序列条目',len(work['entries'])),
        ('保存的曲线层数',sum(len(e['curves']) for e in work['entries'].values())),('保留MR状态',len(state['mr_slots'])),
        ('PET工作站结果','已导入' if state['pet'] else '未导入'),('序列问题/提示',len(review['issues'])),
        ('省略的问题 / 事件',str(review['omitted_issues'])+' / '+str(review['omitted_events']))]
    body=''.join(row(r) for r in summary)
    views=''.join(row([s,'未选择' if v is None else v['key'],'—' if v is None else v['time']+1,'—' if v is None else v['slice']+1]) for s,v in comparison['views'].items())
    measures=''.join(row([branch,{'rest':'静息','stress':'负荷','pair':'配对','workstation':'工作站'}.get(state,state),measurement_label(name),f'{value:.6g}',unit_label(unit),'本地标记已记录' if qc else '待核对','未临床验证'])
        for branch,state,name,value,unit,qc,clinical in measurement_rows(work))
    if not measures:measures=row(['尚无已计算测量；不会补造缺失值。'])
    tables_html='<h2>已计算或导入的测量</h2><table><tr><th>分支</th><th>状态</th><th>测量</th><th>值</th><th>单位</th><th>本地质控</th><th>研究边界</th></tr>'+measures+'</table>'
    analyses=[state['candidate']['value']] if state['candidate'] else []
    analyses.extend(slot['feature'] for slot in state['mr_slots'].values())
    unique={document_hash(a):a for a in analyses}
    windows=''
    for _,a in sorted(unique.items()):
        params=a['parameters']
        windows+=row([a['condition'],params['baseline_start']+1,params['baseline_start']+params['baseline_frames'],
            params['window_start']+1,params['window_end']+1,'本地已勾选' if params['window_reviewed'] else '待核对'])
    tables_html+='<h2>MR分析窗口（帧号从1计）</h2><table><tr><th>状态</th><th>基线起帧</th><th>基线末帧</th><th>首过起帧</th><th>首过末帧</th><th>窗口复核</th></tr>'+windows+'</table>'
    return '<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>研究记录摘要</title><style>body{font:15px/1.7 "Microsoft YaHei",sans-serif;max-width:1080px;margin:32px auto;color:#16354c}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #c8d8df;padding:8px;text-align:left;overflow-wrap:anywhere}th{background:#e7f2f2}.notice{background:#fff4dc;padding:16px}</style><h1>心灌影像工作站 · 研究记录摘要</h1><div class="notice">研究草稿。无诊断输出，未临床验证；已预览不代表全帧通过，历史提示不代表当前像素已审核。文件哈希核对不等于匿名化或定量有效性证明。</div><h2>记录范围</h2><table>'+body+'</table>'+tables_html+'<h2>点击导出时的只读对照位置</h2><p>rest=静息侧、stress=负荷侧，仅表示手动选择；不自动证明采集状态、解剖同层或协议可比。帧/层面从1计，CSV中从0计。相对时间联动：'+('开启' if comparison['linked_time'] else '关闭')+'</p><table><tr><th>侧别</th><th>序列标识</th><th>帧</th><th>层面</th></tr>'+views+'</table><h2>打开与继续工作</h2><ol><li>使用CSV查看测量、原始曲线、序列状态及问题；未保留或未计算的结果不会补造。</li><li>在程序的“研究记录导出”中核对完整目录。此操作只核对文件完整性，不重读原影像或执行临床审核。</li><li>先打开原病例或对应合成演示，再从核对窗口“从此记录恢复工作”；仍按原工作记录规则核对来源并复算。</li></ol><p>不包含影像像素、患者源路径或新增患者编号。工作记录保留ROI与分析来源，仍是受控研究资料。问题及对照位置是本次导出快照；恢复只应用嵌套影像工作记录，不自动恢复处置结论或对照窗口。</p><p>PET血流数值来自工作站；MR为探索性半定量。缺临床/医生标签/审核信息时不可据此放行训练。请保留完整目录，不修改完成工件。</p></html>'


def save_research(root,work,review,comparison,choices,backend,version,*,progress=None,cancel=None):
    check_cancel(cancel);run=new_run(root);write_json(run/'attempt.json',{'completed':False,'kind':KIND})
    require(isinstance(version,str) and re.fullmatch(r'\d+\.\d+\.\d+',version) is not None,'research_invalid_record')
    validate_review(review,work['case_token']);validate_comparison(comparison,work)
    nested=save_record(run/'work',work,choices,backend,progress=progress,cancel=cancel)
    require(document_hash(load_record(nested))==document_hash(work),'research_stale_result')
    grids={};available={c.key:c for c in choices}
    for state,view in comparison['views'].items():
        check_cancel(cancel)
        if view is None:continue
        require(view['key'] in available,'research_stale_result')
        grid=backend.grid(available[view['key']],progress or (lambda *a:None),cancel or (lambda:False))
        require(binding_for(grid)==view['binding'],'research_stale_result');grids[state]=grid
        backend.plane(grid,view['time'],view['slice'],None,progress or (lambda *a:None),cancel or (lambda:False))
    data={'schema_version':1,'kind':KIND,'case_token':work['case_token'],'product_version':version,
        'work_relative':(nested/'work.json').relative_to(run).as_posix(),'work_sha256':document_hash(work),
        'sequence_review':copy.deepcopy(review),'comparison':copy.deepcopy(comparison),
        'synthetic_data':all(bool(getattr(c.source,'synthetic_id',None)) for c in choices),
        'clinical_validated':False,'model_input_ready':False,'diagnosis_available':False}
    write_json(run/'research.json',data)
    for name,content in tables(work,review,cancel,data['synthetic_data']).items():
        check_cancel(cancel)
        with (run/name).open('x',encoding='utf-8-sig',newline='') as stream:stream.write(content)
    with (run/'summary.html').open('x',encoding='utf-8') as stream:stream.write(summary_html(work,review,comparison,version,data['synthetic_data']))
    # Recheck every work source, including annotated sequences not open in comparison.
    source_grids={view['key']:grids[state] for state,view in comparison['views'].items() if view is not None}
    for key,entry in work['entries'].items():
        check_cancel(cancel)
        if key not in source_grids:source_grids[key]=backend.grid(available[key],progress or (lambda *a:None),cancel or (lambda:False))
        require(binding_for(source_grids[key])==entry['document']['binding'],'research_stale_result')
    check_sources(source_grids,cancel);check_cancel(cancel)
    write_json(run/'bundle.json',{'completed':True,'schema_version':1,'kind':KIND,'clinical_validated':False,
        'files_sha256':{p.relative_to(run).as_posix():sha(p) for p in sorted(run.rglob('*')) if p.is_file()}})
    return run


def _load_research(path,cancel=None):
    path=Path(path);root=path if path.is_dir() else path.parent
    require(path.is_dir() or path.name in ('research.json','bundle.json'),'research_invalid_record')
    marker=load_json(root/'bundle.json');require(marker.get('completed') is True and marker.get('kind')==KIND and marker.get('schema_version')==1 and marker.get('clinical_validated') is False,'research_incomplete')
    hashes=marker.get('files_sha256',{});require(isinstance(hashes,dict) and 12<=len(hashes)<=14,'research_invalid_record')
    before={}
    for name,h in sorted(hashes.items()):
        check_cancel(cancel);relative=PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts and re.fullmatch(r'[A-Za-z0-9_./-]+',name) is not None and digest(h),'research_invalid_record')
        file=root/relative
        require(all(not p.is_symlink() for p in [file,*file.parents] if p!=root.parent),'research_invalid_record')
        require(file.stat().st_size<=MAX_BYTES and sha(file)==h,'research_file_changed');before[name]=h
    data=load_json(root/'research.json')
    require(set(data)=={'schema_version','kind','case_token','product_version','work_relative','work_sha256','sequence_review','comparison','synthetic_data','clinical_validated','model_input_ready','diagnosis_available'},'research_invalid_record')
    require(type(data['synthetic_data']) is bool,'research_invalid_record')
    require(data['schema_version']==1 and data['kind']==KIND and all(data[k] is False for k in ('clinical_validated','model_input_ready','diagnosis_available')),'research_invalid_record')
    require(re.fullmatch(r'\d+\.\d+\.\d+',data['product_version']) is not None,'research_invalid_record')
    relative=PurePosixPath(data['work_relative'])
    require(len(relative.parts)==3 and relative.parts[0]=='work' and re.fullmatch(r'[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}',relative.parts[1]) and relative.parts[2]=='work.json','research_invalid_record')
    work=load_record(root/relative);require(document_hash(work)==data['work_sha256'] and work['case_token']==data['case_token'],'research_file_changed')
    validate_review(data['sequence_review'],data['case_token']);validate_comparison(data['comparison'],work)
    expected={'attempt.json','research.json','summary.html','measurements.csv','curves.csv','sequences.csv','issues.csv'}
    nested=load_json(root/relative.parent/'bundle.json')
    expected|={str(relative.parent/p) for p in ['bundle.json',*nested['files_sha256']]}
    require(set(hashes)==expected,'research_invalid_record')
    for name,content in tables(work,data['sequence_review'],cancel,data['synthetic_data']).items():
        with (root/name).open(encoding='utf-8-sig',newline='') as stream:require(stream.read()==content,'research_file_changed')
    require((root/'summary.html').read_text(encoding='utf-8')==summary_html(work,data['sequence_review'],data['comparison'],data['product_version'],data['synthetic_data']),'research_file_changed')
    check_cancel(cancel)
    require(load_json(root/'bundle.json')==marker and all(sha(root/n)==h for n,h in before.items()),'research_file_changed')
    return data


def load_research(path,*,cancel=None):
    try:return _load_research(path,cancel)
    except DicomReadError:raise
    except OSError:raise DicomReadError('research_incomplete') from None
    except (KeyError,TypeError,ValueError,AttributeError,IndexError,OverflowError,RecursionError):
        raise DicomReadError('research_invalid_record') from None

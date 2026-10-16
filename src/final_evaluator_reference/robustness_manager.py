"""One predeclared development stress queue; no training or automated tuning."""
import ast,hashlib,json,math,os,shutil,sys,tempfile,time,traceback,uuid,zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

def audit_queue_rows(root,protocol):
    digest=canonical(protocol);rows=[]
    for job in schedule(protocol['options']):
        row=verified_episode(root,digest,job)
        if row is not None:
            verify_initial_pair(root,row,rows);rows.append(row)
    return rows

def verify_initial_pair(root,row,previous):
    name=next(n for n in row['artifacts'] if Path(n).name=='initial_state.npy')
    initial=np.load(safe_child(root,name),allow_pickle=False)
    assert initial.shape==(32,) and np.isfinite(initial).all()
    for other in previous:
        if other['environment_seed']==row['environment_seed']:
            path=safe_child(root,next(n for n in other['artifacts'] if Path(n).name=='initial_state.npy'))
            np.testing.assert_allclose(initial,np.load(path,allow_pickle=False),rtol=0,atol=1e-9,
                err_msg='Reset differs across condition/modality; stress pairing blocked.')

def report_statistics(rows,options):
    groups=[];paired=[];changes=[]
    for training in options['training_seeds']:
        seed_rows=[r for r in rows if r['training_seed']==training]
        for c in options['conditions']:
            chosen=[r for r in seed_rows if r['condition_id']==c['id']]
            for m in ['rgb','events']:
                selected=[r for r in chosen if r['modality']==m];n=len(selected)
                groups.append({'condition_id':c['id'],'modality':m,'training_seed':training,'completed':n,'planned':50,
                    'any_success':sum(r['any_success'] for r in selected),'sustained_success':sum(r['sustained_success'] for r in selected),
                    'any_rate':sum(r['any_success'] for r in selected)/n if n else None,
                    'sustained_rate':sum(r['sustained_success'] for r in selected)/n if n else None,
                    'conditional_descriptive_wilson_95pct':wilson(sum(r['sustained_success'] for r in selected),n)})
            paired.append({'condition_id':c['id'],'training_seed':training,
                **paired_counts(chosen,'rgb_best','events_best','sustained_success')})
        for m in ['rgb','events']:
            baseline={r['environment_seed']:r for r in seed_rows if r['condition_id']=='normal' and r['modality']==m}
            stressed={r['environment_seed']:r for r in seed_rows if r['condition_id']=='dark_gain025' and r['modality']==m}
            common=sorted(set(baseline)&set(stressed))
            delta=[int(stressed[e]['sustained_success'])-int(baseline[e]['sustained_success']) for e in common]
            changes.append({'condition_id':'dark_gain025','training_seed':training,'modality':m,'paired_resets':len(common),
                'stress_minus_normal_percentage_points':float(np.mean(delta)*100) if common else None,
                'lost_successes':sum(d==-1 for d in delta),'gained_successes':sum(d==1 for d in delta)})
    return groups,paired,changes

def make_stress_bundle(root,session,rows,include_videos):
    target=root/('final_evaluation_full_bundle.zip' if include_videos else 'final_evaluation_core_evidence.zip')
    temporary=target.with_name(target.name+'.tmp-'+uuid.uuid4().hex);files={}
    for p in root.iterdir():
        if p.is_file() and p.suffix in ['.json','.csv','.md','.png','.pdf']:files[p.name]=p
    for row in rows:
        pointer=root/'episodes'/(row['key']+'.json');files[str(pointer.relative_to(root))]=pointer
        for name,digest in row['artifacts'].items():
            p=safe_child(root,name)
            if not include_videos and p.suffix in ['.mp4','.npz']:continue
            files[name]=p
    for p in session.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.csv','.md','.py','.log','.png','.xml'] and '.tmp-' not in p.name:
            files['session_evidence/'+str(p.relative_to(session))]=p
    manifest=[]
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
            for name,p in sorted(files.items()):
                z.write(p,name);manifest.append({'path':name,'sha256':sha_file(p),'bytes':p.stat().st_size})
            z.writestr('bundle_manifest.json',json.dumps({'files':manifest,'weights_included':False,'videos_included':include_videos}))
        os.replace(temporary,target)
    finally:temporary.unlink(missing_ok=True)
    return target

def publish_stress(root,protocol,rows,status,session,final=False,error=None,execution=None):
    if final:rows=audit_queue_rows(root,protocol)
    groups,paired,changes=report_statistics(rows,protocol['options'])
    if status=='COMPLETED':assert len(rows)==600
    report={'status':status,'queue_id':root.name,'protocol_hash':canonical(protocol),'study_hash':STUDY_HASH,
        'updated_utc':utc(),'planned_episodes':600,'completed_episodes':len(rows),'training_performed':False,
        'historical_imports':0,'role':'locked final evaluation; three fixed trained pairs and fifty held-out resets',
        'primary_endpoint':'Lift height predicate for10 consecutive20Hz steps','groups':groups,'paired':paired,
        'paired_changes_from_normal':changes,'episodes':rows,'error':error,
        'verification_scope':'fresh all-artifact audit' if final else 'just-verified durable rows',
        'limitations':['Synthetic ideal noiseless V2E; gain is not calibrated low-light/HDR sensor simulation.',
            'One fixed TRAIN scene, three fixed trained pairs and50 held-out resets; the dark condition was selected on separate development data.',
            'Both predeclared conditions and all training seeds reported; no condition retuning or retraining.',
            'Darkness is the unchanged gain0.25 from reset; source, control and planner horizons are unchanged.',
            'Reports retain all assigned valid trials, including task failures; infrastructure failures block completion.',
            'Shared bounded x0 sampler unchanged. Stress advantage for physical Event cameras is not established.',
            'Slow playback is presentation only; scored times use20Hz simulated time.',
            'Both modalities retain matched source work; wall time does not compare native sensor efficiency.']}
    atomic_json(root/'final_evaluation_summary.json',report);write_csv(root/'condition_summary.csv',groups)
    write_csv(root/'paired_condition_changes.csv',changes);write_csv(root/'episode_summary.csv',[
        {k:r[k] for k in ['key','training_seed','condition_id','modality','environment_seed','policy_seed','steps','any_success','sustained_success','wall_seconds']} for r in rows])
    statistics=replication_statistics(rows,protocol['options'])
    report['replication_statistics']=statistics
    report['development_reference']=DEVELOPMENT_REFERENCE
    report['analysis_plan']=ANALYSIS_PLAN
    atomic_json(root/'final_evaluation_summary.json',report)
    atomic_json(root/'replication_statistics.json',statistics)
    write_csv(root/'paired_modality_counts.csv',[{**{k:v for k,v in p.items() if k not in ['counts','paired_seeds']},**p['counts']} for p in paired])
    write_csv(root/'training_seed_variability.csv',statistics['training_seed_variability'])
    reset_rows=reset_population_rows(root,rows)
    write_csv(root/'reset_population.csv',reset_rows)
    lines=['# Notebook08 — locked normal/dark final evaluation','',
        f'Status: {status}; completed {len(rows)}/600. No training.',
        'Training seeds42/43/44; final resets1000-1049; independent policy seeds4000-4049.',
        'The condition was selected on disclosed development evidence; this NEW locked final cohort is not used for further tuning.','',
        '| Condition | Training seed | Modality | Completed | Any Lift | Sustained Lift |',
        '|---|---:|---|---:|---:|---:|']
    lines += [f"| {g['condition_id']} | {g['training_seed']} | {g['modality']} | {g['completed']}/50 | {g['any_success']} | {g['sustained_success']} |" for g in groups]
    lines += ['','## Scope']+['- '+x for x in report['limitations']]+[
        '- Reset-bootstrap intervals are conditional on six fixed trained models; they do not quantify training-population uncertainty.',
        '- Report per-training-seed rates and sample standard deviation; do not select the best training seed.',
        '- Reset population CSV and per-episode reset_contract.json document actual positions, yaw, joint settings and sampler metadata.',
        '', '## Upload first',str(root/'final_evaluation_core_evidence.zip')]
    (root/'final_evaluation_summary.md').write_text('\n'.join(lines)+'\n')
    if rows:replication_plots(root,groups,changes,protocol['options'])
    if final:
        if execution is not None:
            execution['scored_episodes']=len(rows)
            atomic_json(session/'execution_summary.json',execution)
            atomic_json(root/'execution_summary.json',execution)
        publish_delivery(root,session,rows,status)
        if EXPORT_FULL_BUNDLE and status=='COMPLETED':make_stress_bundle(root,session,rows,True)
    return report

def study_identity():
    return {'version':VERSION,'build_sha256':BUILD_SHA,'dataset_id':DATASET_ID,'data_fingerprint':DATA_HASH,
        'normalization_hash':NORM_HASH,'sources':SOURCE_CATALOG,'options':OPTIONS,
        'role':'locked final normal/dark evaluation; no tuning on these outcomes',
        'training_performed':False,'selection_rule':'all six fixed SMALL FP32 EMA4500 models; selected on development only',
        'development_reference':DEVELOPMENT_REFERENCE,'analysis_plan':ANALYSIS_PLAN}

def run_stress_pilot():
    global GPU_ENV,GPU_INDEX,GPU_LIBRARY,GRAPHICS,CACHED,PROTOCOL_HASH,OUTPUT,PROTOCOL,STUDY_HASH
    for stale in ['PROTOCOL','PROTOCOL_HASH']:globals().pop(stale,None)
    select_training_seed(42)
    validate_budget(OPTIONS);identity=study_identity();STUDY_HASH=canonical(identity)
    OUTPUT=PROJECT_ROOT/'results/final_evaluation'/('d200_tail_final_normal_dark_v1_'+STUDY_HASH[:12])
    OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/'episodes').mkdir(exist_ok=True)
    freeze_final_protocol(OUTPUT)
    print('Final evaluation directory:',OUTPUT,'\n600 declared trials; no training. Both normal and fixed gain0.25; all three trained pairs.',flush=True)
    failure=None;rows=[];summary={'status':'RUNNING','started_utc':utc(),'training_performed':False};start=time.perf_counter()
    try:
        if (OUTPUT/'protocol.json').is_file():
            PROTOCOL=load_json(OUTPUT/'protocol.json');assert PROTOCOL['study_identity']==identity
            PROTOCOL_HASH=canonical(PROTOCOL)
            rows=audit_queue_rows(OUTPUT,PROTOCOL)
            if len(rows)==600:
                print('All600 completed trials verified; no GPU/source/model work is needed.',flush=True)
                summary.update(status='COMPLETED',recovered_completed=True,scored_episodes=600)
                return summary
        verify_new_seed_roles(PROJECT_ROOT,OUTPUT,OPTIONS,SESSION)
        runtime=prepare_dependencies();atomic_json(SESSION/'runtime.json',runtime)
        source_manifest=audit_inputs()
        GPU_ENV,GPU_INDEX,GPU_LIBRARY,probe=hardware_selection();GRAPHICS=probe['graphics']
        request=source_request(OPTIONS['scene_demo'],'train',SCENE_SOURCE_PROTOCOL_HASH,GPU_INDEX,GPU_LIBRARY)
        gate=run_process(PRODUCTION_WORKER_PATH,{**request,'id':'fresh_complete_source_gate','task':'online',
            'offline_directory':str(SOURCE_FOLDER)},GPU_ENV,SESSION/'fresh_complete_source_gate',900)
        assert gate['parity']['status']==gate['physics_noninterference']['status']==gate['physics']['status']=='PASS'
        assert gate['parity']['event_max_abs']<=1e-5
        load_selected_models()
        candidate={'version':VERSION,'study_identity':identity,'options':OPTIONS,'runtime':runtime,'graphics':GRAPHICS,
            'scene_xml_sha256':source_manifest['provenance']['raw_scene_xml_sha256'],
            'source_protocol':SOURCE_PROTOCOL,'tail_export_protocol':TAIL_EXPORT_PROTOCOL,
            'action_bounds':{'raw_lo':RAW_LO,'raw_hi':RAW_HI,'scope':'TRAIN_ONLY'},
            'sampling':'unchanged100-step FP32 EMA eps DDPM; shared TRAIN-range x0 projection',
            'initial_history':'duplicate RGB reset / zero Event startup; persistent causal200Hz source',
            'source_guard':'three native draws, unchanged native amplitude checks, independent transformed primary/shadow V2E',
            'reset_contract':'seeded fresh reset; sampler XY[-0.03,0.03]m, random yaw Z, recorded robot initialization noise; paired at1e-9',
            'development_selection':DEVELOPMENT_REFERENCE,'analysis_plan':ANALYSIS_PLAN,
            'pre_scoring_freeze_sha256':canonical(final_freeze_payload()),
            'scope':'one fixed TRAIN scene; locked held-out final cohort; synthetic gain; no tuning or training'}
        if (OUTPUT/'protocol.json').is_file():
            assert canonical(candidate)==PROTOCOL_HASH, 'Runtime/source/protocol changed. Preserve results; do not mix sessions.'
        else:atomic_json(OUTPUT/'protocol.json',candidate)
        PROTOCOL=candidate;PROTOCOL_HASH=canonical(PROTOCOL);atomic_json(SESSION/'protocol.json',PROTOCOL)
        gates=[]
        for training in OPTIONS['training_seeds']:
            activate_parent_seed(training)
            for m in ['rgb','events']:
                result=run_process(LIVE_WORKER_PATH,live_request('preflight',m,'best'),GPU_ENV,
                    SESSION/('normal_preflight_'+str(training)+'_'+m),900)
                assert result['status']=='PASS';gates.append(result)
            result=run_process(LIVE_WORKER_PATH,live_request('stress_preflight','events','best',condition_id='dark_gain025'),
                GPU_ENV,SESSION/('dark_preflight_'+str(training)+'_events'),900)
            assert result['status']=='PASS';gates.append(result)
        atomic_json(SESSION/'preflight.json',{'status':'PASS','source_gate':gate,'model_and_condition_gates':gates,'scored_episodes':0})
        rows=[]
        for job in schedule(OPTIONS):
            activate_parent_seed(job['training_seed'])
            row=verified_episode(OUTPUT,PROTOCOL_HASH,job)
            if row is None:
                refs=[r for r in rows if r['environment_seed']==job['environment_seed']]
                reference=str(safe_child(OUTPUT,next(n for n in refs[0]['artifacts'] if Path(n).name=='initial_state.npy'))) if refs else None
                assert_final_protocol_lock(OUTPUT)
                mark_final_scoring_start(OUTPUT,job)
                print('Running:',job['key'],flush=True)
                result=run_process(LIVE_WORKER_PATH,live_request('episode',job['modality'],'best',job=job,initial_reference=reference),
                    GPU_ENV,SESSION/('worker_'+job['key']),1200)
                row=verified_episode(OUTPUT,PROTOCOL_HASH,job);assert row is not None and result['status']=='COMPLETED'
            else:print('Reusing verified completed episode:',job['key'],flush=True)
            verify_initial_pair(OUTPUT,row,rows);rows.append(row)
            print(len(rows),'/600 |',job['key'],'any=',row['any_success'],'sustained=',row['sustained_success'],'steps=',row['steps'],flush=True)
            atomic_json(OUTPUT/'progress.json',{'status':'RUNNING','completed':len(rows),'planned':600,'updated_utc':utc()})
            if len(rows)%10==0:publish_stress(OUTPUT,PROTOCOL,rows,'RUNNING',SESSION)
        summary.update(status='COMPLETED',scored_episodes=600)
    except BaseException as exc:
        failure=exc;summary.update(status='ABORTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=str(exc),traceback=traceback.format_exc())
        print('Stopped safely:',summary['status'],'Completed episode pointers remain resumable.',flush=True)
    finally:
        summary.update(ended_utc=utc(),wall_seconds=time.perf_counter()-start)
        if 'PROTOCOL' in globals():
            try:
                report=publish_stress(OUTPUT,PROTOCOL,rows,summary['status'],SESSION,final=True,error=summary.get('error'),execution=summary)
                summary['scored_episodes']=report['completed_episodes']
            except BaseException as exc:
                if failure is None:failure=exc
                summary.update(status='FAILED',error='Evidence publication failed: '+str(exc),traceback=traceback.format_exc())
                atomic_json(SESSION/'publication_failure.json',summary)
        atomic_json(SESSION/'execution_summary.json',summary);atomic_json(OUTPUT/'execution_summary.json',summary)
        make_bundle(SESSION,SESSION/'final_evaluation_session_evidence.zip',False)
        if (OUTPUT/'final_evaluation_core_evidence.zip').is_file():print('UPLOAD FIRST:',OUTPUT/'final_evaluation_core_evidence.zip',flush=True)
        print('SELECTED VIDEOS:',OUTPUT/'final_evaluation_selected_videos.zip',flush=True)
        print('DETAILED AUDIT PARTS:',OUTPUT/'audit_parts',flush=True)
        print('ON FAILURE ALSO UPLOAD:',SESSION/'final_evaluation_session_evidence.zip',flush=True)
    if failure is not None:raise failure
    return summary

def disconnect_on_complete(execution):
    if not AUTO_DISCONNECT_ON_COMPLETE or execution.get('status')!='COMPLETED':return {'status':'NOT_REQUESTED'}
    report=load_json(OUTPUT/'final_evaluation_summary.json')
    assert report['status']=='COMPLETED' and report['completed_episodes']==600
    assert_final_protocol_lock(OUTPUT)
    validate_export_archives(OUTPUT)
    try:
        from google.colab import drive,runtime
        assert callable(getattr(drive,'flush_and_unmount',None)) and callable(getattr(runtime,'unassign',None))
    except Exception as exc:
        print('All results saved; disconnect/delete the runtime manually:',str(exc),flush=True)
        return {'status':'UNAVAILABLE','error':str(exc)}
    atomic_json(OUTPUT/'disconnect_status.json',{'status':'REQUESTED_AFTER_VERIFIED_SAVE','utc':utc()})
    print('All600 final trials and delivery bundles verified. Flushing Drive, then requesting disconnect.',flush=True)
    try:
        drive.flush_and_unmount();runtime.unassign();return {'status':'REQUESTED'}
    except Exception as exc:
        print('Disconnect request did not complete; disconnect/delete runtime manually:',str(exc),flush=True)
        return {'status':'REQUEST_FAILED','error':str(exc)}


def activate_parent_seed(seed):
    global CACHED,RUN_RECORDS
    select_training_seed(seed)
    CACHED=CACHED_BY_SEED[seed];RUN_RECORDS=RECORDS_BY_SEED[seed]

def load_selected_models():
    global CACHED_BY_SEED,RECORDS_BY_SEED
    CACHED_BY_SEED={};RECORDS_BY_SEED={}
    for seed in OPTIONS['training_seeds']:
        select_training_seed(seed)
        records={}
        for modality,identity in SOURCES.items():
            directory=PROJECT_ROOT/'artifacts/checkpoints'/modality/identity['run_id']
            record=load_json(directory/'run_config.json');validate_run_record(record,modality)
            assert canonical(load_json(directory/'normalization.json'))==NORM_HASH
            status=load_json(directory/'status.json')
            assert status['run_id']==identity['run_id'] and status['status']=='COMPLETED' and status['durable_step']==4500
            assert status['stop_reason']=='budget_exhausted'
            records[modality]=record
        a=records['rgb']['config'].copy();b=records['events']['config'].copy()
        for key in ['modality','visual_normalization']:a.pop(key,None);b.pop(key,None)
        assert a==b,'Paired training recipe differs for training seed'+str(seed)
        RECORDS_BY_SEED[seed]=records
        CACHED_BY_SEED[seed]=cache_checkpoints()
    activate_parent_seed(42)
    atomic_json(SESSION/'six_checkpoint_identities.json',{'status':'PASS','sources':SOURCE_CATALOG,'data_hash':DATA_HASH,
        'normalization_hash':NORM_HASH,'training_performed':False})

def verify_new_seed_roles(project,own_output,options,session):
    requested=set(options['environment_seeds'])|set(options['policy_seeds']);checked=[]
    results=Path(project)/'results'
    for path in sorted(results.rglob('protocol.json')):
        if path.resolve().is_relative_to(Path(own_output).resolve()):continue
        record=load_json(path)
        if record.get('study_identity')==study_identity():continue
        used=set()
        def visit(value):
            if isinstance(value,dict):
                for key,item in value.items():
                    if key in ['environment_seeds','policy_seeds']:
                        assert isinstance(item,list) and all(type(x) is int for x in item)
                        used.update(item)
                    visit(item)
            elif isinstance(value,list):
                for item in value:visit(item)
        visit(record)
        assert not requested&used,'Protected final/policy seed IDs were already used in another result protocol: '+str(path)
        checked.append(str(path.relative_to(project)))
    atomic_json(Path(session)/'new_seed_role_audit.json',{'status':'PASS','environment_seeds':options['environment_seeds'],
        'policy_seeds':options['policy_seeds'],'prior_protocol_files_checked':checked,
        'scope':'protected final reset IDs1000-1049 and dedicated policy IDs4000-4049 checked against previous result protocols'})

def reset_population_rows(root,rows):
    result=[];seen=set()
    for r in rows:
        seed=r['environment_seed']
        if seed in seen:continue
        seen.add(seed)
        name=next(n for n in r['artifacts'] if Path(n).name=='initial_state.npy')
        state=np.load(safe_child(root,name),allow_pickle=False);assert state.shape==(32,)
        w,x,y,z=state[13:17]
        yaw=math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))%360
        result.append({'environment_seed':seed,'cube_x_world_m':float(state[10]),'cube_y_world_m':float(state[11]),
            'cube_z_world_m':float(state[12]),'cube_yaw_deg':yaw,
            **{'arm_joint'+str(i+1)+'_rad':float(state[1+i]) for i in range(7)},
            'gripper_joint1_m':float(state[8]),'gripper_joint2_m':float(state[9]),
            'verified_scope':'paired initial state; all matching rows checked separately at1e-9'})
    return result

def replication_statistics(rows,options):
    variability=[]
    for condition in ['normal','dark_gain025']:
        for modality in ['rgb','events']:
            rates=[]
            for training in options['training_seeds']:
                selected=[r for r in rows if r['condition_id']==condition and r['modality']==modality and r['training_seed']==training]
                if len(selected)==50:rates.append(sum(r['sustained_success'] for r in selected)/50)
            variability.append({'condition_id':condition,'modality':modality,'completed_training_seeds':len(rates),
                'mean_seed_success_rate':float(np.mean(rates)) if rates else None,
                'sample_std_across_training_seeds':float(np.std(rates,ddof=1)) if len(rates)>1 else None,
                'minimum_seed_rate':min(rates) if rates else None,'maximum_seed_rate':max(rates) if rates else None})
    result={'status':'COMPLETED' if len(rows)==600 else 'PENDING','training_seed_variability':variability,
        'conditional_paired_reset_bootstrap':{'status':'PENDING'},
        'scope':'locked final evaluation of six fixed trained models; three training-pair variability remains descriptive'}
    if len(rows)!=600:return result
    by={(r['training_seed'],r['condition_id'],r['modality'],r['environment_seed']):int(r['sustained_success']) for r in rows}
    assert len(by)==600
    rng=np.random.default_rng(98439);index=rng.integers(0,50,size=(4000,50))
    differences={}
    for condition in ['normal','dark_gain025']:
        values=np.asarray([[by[t,condition,'events',e]-by[t,condition,'rgb',e] for e in options['environment_seeds']]
            for t in options['training_seeds']],dtype=float)
        differences[condition]=values.mean(axis=0)
    contrasts={'events_minus_rgb_normal':differences['normal'],'events_minus_rgb_dark':differences['dark_gain025'],
        'events_minus_rgb_change_dark_vs_normal':differences['dark_gain025']-differences['normal']}
    result['conditional_paired_reset_bootstrap']={'status':'COMPLETED','draws':4000,'summary_seed':98439,
        'sampling_unit':'the same reset ID resampled jointly across all six models and both conditions',
        'scope':'conditional on these six fixed trained models; excludes training-population, selection, scene and sensor uncertainty',
        'contrasts':{k:{'mean_rate_difference':float(v.mean()),
            'descriptive_reset_bootstrap_95pct':np.quantile(v[index].mean(axis=1),[.025,.975]).tolist()} for k,v in contrasts.items()}}
    return result

def replication_plots(root,groups,changes,options):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for m,offset in [('rgb',-.12),('events',.12)]:
        for cond,marker in [('normal','o'),('dark_gain025','s')]:
            selected=[g for g in groups if g['modality']==m and g['condition_id']==cond]
            axes[0].plot(np.arange(3)+offset,[g['sustained_rate'] if g['completed']==50 else np.nan for g in selected],
                marker=marker,label=m.upper()+' '+cond)
        cs=[c for c in changes if c['modality']==m]
        axes[1].plot(np.arange(3)+offset,[c['stress_minus_normal_percentage_points'] if c['paired_resets']==50 else np.nan for c in cs],
            marker='o',label=m.upper())
    for ax in axes:ax.set_xticks(range(3),[str(x) for x in options['training_seeds']]);ax.set_xlabel('Training seed');ax.legend(fontsize=8)
    axes[0].set_ylim(0,1.05);axes[0].set_ylabel('Sustained Lift fraction');axes[0].set_title('All three fixed pairs; complete50-reset groups only')
    axes[1].axhline(0,color='black',linewidth=.7);axes[1].set_ylabel('Dark minus normal, percentage points');axes[1].set_title('Same-reset paired degradation')
    fig.tight_layout();fig.savefig(Path(root)/'stress_results.png',dpi=160);fig.savefig(Path(root)/'stress_results.pdf');plt.close(fig)


def final_freeze_payload():
    return {'study_identity':study_identity(),'analysis_plan':ANALYSIS_PLAN,
        'endpoint':'cube height above table>0.04m for10 consecutive20Hz steps',
        'policy_noise_role':'4000-4049, separate from environment reset1000-1049',
        'queue_stopping':'all600 valid completed trials; no outcome-dependent queue stopping',
        'frozen_source':'exact six EMA4500 models, data, normalizer, sampler, source and conditions',
        'video_selection':'training42,reset1000/1001,both modalities,both conditions; selected before outcomes'}

def freeze_final_protocol(root):
    expected=final_freeze_payload();path=Path(root)/'protocol_freeze.json'
    if path.exists():
        saved=load_json(path);assert saved['payload']==expected and saved['freeze_sha256']==canonical(expected)
    else:
        atomic_json(path,{'status':'FROZEN_BEFORE_SCORING','created_utc':utc(),'freeze_sha256':canonical(expected),'payload':expected})
    assert_final_protocol_lock(root)
    return path

def assert_final_protocol_lock(root):
    saved=load_json(Path(root)/'protocol_freeze.json')
    assert saved['status']=='FROZEN_BEFORE_SCORING'
    assert saved['payload']==final_freeze_payload() and saved['freeze_sha256']==canonical(saved['payload']), 'Frozen final protocol changed; do not mix or retune final results.'
    return True

def mark_final_scoring_start(root,job):
    assert_final_protocol_lock(root);path=Path(root)/'scoring_started.json'
    if path.exists():
        saved=load_json(path);assert saved['study_hash']==STUDY_HASH and saved['freeze_sha256']==canonical(final_freeze_payload())
    else:atomic_json(path,{'started_utc':utc(),'study_hash':STUDY_HASH,'freeze_sha256':canonical(final_freeze_payload()),'first_job_key':job['key']})

CORE_EPISODE_FILES={'initial_state.npy','steps.csv','reset_contract.json','acquisition_guard_summary.json',
    'render_stability_summary.json','stress_audit.json','presentation_audit.json','presentation_frames.csv','event_stats.csv'}

def write_delivery_zip(target,files,metadata):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.tmp-'+uuid.uuid4().hex)
    manifest=[]
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for name,p in sorted(files.items()):
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
                z.write(p,name);manifest.append({'path':name,'bytes':p.stat().st_size,'sha256':sha_file(p)})
            z.writestr('bundle_manifest.json',json.dumps({'files':manifest,'weights_included':False,**metadata}))
        os.replace(temporary,target)
    finally:temporary.unlink(missing_ok=True)
    return {'path':str(target),'bytes':target.stat().st_size,'sha256':sha_file(target)}

def publish_delivery(root,session,rows,status):
    root=Path(root);previous=load_json(root/'bundle_index.json') if (root/'bundle_index.json').is_file() else {}
    old={r['relative_path']:r for r in previous.get('audit_parts',[])};parts=[]
    for number,(training,condition,modality) in enumerate((t,c,m) for t in [42,43,44] for c in ['normal','dark_gain025'] for m in ['rgb','events']):
        chosen=[r for r in rows if r['training_seed']==training and r['condition_id']==condition and r['modality']==modality]
        if not chosen:continue
        name='audit_parts/final_evaluation_audit_part_'+str(number+1).zfill(2)+'_of_12.zip'
        signature=canonical({'protocol_hash':rows[0]['protocol_hash'],'rows':chosen,'package':'all nonvideo/nonNPZ artifacts'})
        target=root/name;prior=old.get(name)
        if prior and prior['content_signature']==signature and target.is_file() and sha_file(target)==prior['sha256']:
            part=prior
        else:
            files={}
            for r in chosen:
                pointer=root/'episodes'/(r['key']+'.json');files[str(pointer.relative_to(root))]=pointer
                for path,digest in r['artifacts'].items():
                    if Path(path).suffix in ['.mp4','.npz']:continue
                    files[path]=safe_child(root,path)
            result=write_delivery_zip(target,files,{'videos_included':False,'trajectory_query_npz_included':False,
                'scope':'detailed per-frame acquisition/source/Event CSV and PNG evidence; weights and videos excluded'})
            part={'relative_path':name,'sha256':result['sha256'],'bytes':result['bytes'],'content_signature':signature,
                'training_seed':training,'condition_id':condition,'modality':modality,'completed_episodes':len(chosen),'planned_episodes':50}
        parts.append(part)
    chosen=[r for r in rows if r['training_seed']==42 and r['environment_seed'] in [1000,1001]]
    video_files={}
    for r in chosen:
        for name,digest in r['artifacts'].items():
            if Path(name).name in ['rollout.mp4','rollout_slow025.mp4']:video_files[name]=safe_child(root,name)
        pointer=root/'episodes'/(r['key']+'.json');video_files[str(pointer.relative_to(root))]=pointer
    video=None
    if video_files:
        video=write_delivery_zip(root/'final_evaluation_selected_videos.zip',video_files,{'videos_included':True,
            'selection':'FIXED BEFORE OUTCOMES:training42,resets1000/1001,bothconditions,bothmodalities',
            'completed_selected_episodes':len(chosen),'planned_selected_episodes':8,'scope':'presentation subset,not evidence of aggregate success'})
        video={'relative_path':'final_evaluation_selected_videos.zip','sha256':video['sha256'],'bytes':video['bytes'],
            'completed_selected_episodes':len(chosen),'planned_selected_episodes':8}
    index={'status':status,'completed_episodes':len(rows),'planned_episodes':600,'audit_parts':parts,'selected_videos':video,
        'core_scope':'all episode pointers,initial states,step outcomes,reset/guard summaries andcurrent-session gate/code evidence; dense audits in12parts; NPZ/video bytes remain on Drive',
        'all_videos_preserved_in_episode_artifacts':True,'optional_full_bundle_export':bool(EXPORT_FULL_BUNDLE)}
    atomic_json(root/'bundle_index.json',index)
    files={}
    for p in root.iterdir():
        if p.is_file() and p.suffix in ['.json','.csv','.md','.png','.pdf'] and p.name!='disconnect_status.json':files[p.name]=p
    for r in rows:
        pointer=root/'episodes'/(r['key']+'.json');files[str(pointer.relative_to(root))]=pointer
        for name,digest in r['artifacts'].items():
            if Path(name).name in CORE_EPISODE_FILES:files[name]=safe_child(root,name)
    for p in Path(session).rglob('*'):
        if p.is_file() and p.suffix in ['.json','.py','.log','.xml'] and '.tmp-' not in p.name:
            files['session_evidence/'+str(p.relative_to(session))]=p
    write_delivery_zip(root/'final_evaluation_core_evidence.zip',files,{'videos_included':False,'dense_frame_audits_included':False,
        'scope':index['core_scope']})
    return index

def validate_export_archives(root):
    root=Path(root);index=load_json(root/'bundle_index.json')
    assert index['status']=='COMPLETED' and index['completed_episodes']==600 and len(index['audit_parts'])==12
    assert all(p['completed_episodes']==p['planned_episodes']==50 for p in index['audit_parts'])
    assert index['selected_videos'] and index['selected_videos']['completed_selected_episodes']==8
    archives=[root/'final_evaluation_core_evidence.zip']
    for r in index['audit_parts']+[index['selected_videos']]:
        p=safe_child(root,r['relative_path']);assert p.stat().st_size==r['bytes'] and sha_file(p)==r['sha256'];archives.append(p)
    for p in archives:
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            manifest=json.loads(z.read('bundle_manifest.json'))
            for r in manifest['files']:
                digest=hashlib.sha256();size=0
                with z.open(r['path']) as f:
                    for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block);size+=len(block)
                assert size==r['bytes'] and digest.hexdigest()==r['sha256']
    if EXPORT_FULL_BUNDLE:
        with zipfile.ZipFile(root/'final_evaluation_full_bundle.zip') as z:assert z.testzip() is None
    return True

"""D200 evaluation-only orchestration; exact-protocol completion reuse only."""
import ast, importlib, importlib.util, importlib.metadata, signal, subprocess, sys, tempfile, traceback

def load_json(path): return json.loads(Path(path).read_text())

def audit_inputs():
    from tail_data_contract import verify_paired_identity,verify_tail_shared,COMMON_KEYS,ARRAY_KEYS
    import h5py
    global DATASET_ROOT,DATASET,SOURCE_PROTOCOL,NORMALIZATION,SOURCE_FOLDER,EXPORT_FOLDER,RUN_RECORDS,RAW_LO,RAW_HI,SCENE_SOURCE_PROTOCOL_HASH,SCENE_SOURCE_RECORD,TAIL_EXPORT_PROTOCOL
    raw_path=PROJECT_ROOT/'data/raw/lift_ph_raw.hdf5'
    assert sha_file(raw_path)==RAW_SHA, 'Raw dataset changed.'
    split_path=PROJECT_ROOT/'configs/training/d200_rgb_event_train_val.json'
    assert sha_file(split_path)==D200_SPLIT_SHA, 'D200 split changed.'
    index=load_json(PROJECT_ROOT/'configs/processed_sources/paired_gpu_d200_tail_training_ready.json')
    DATASET_ROOT=safe_child(PROJECT_ROOT,index['relative_root'])
    assert sha_file(DATASET_ROOT/'paired_dataset.json')==index['dataset_manifest_sha256']
    DATASET=load_json(DATASET_ROOT/'paired_dataset.json');TAIL_EXPORT_PROTOCOL=load_json(DATASET_ROOT/'protocol.json')
    norm_path=safe_child(DATASET_ROOT,DATASET['normalization_file'])
    assert sha_file(norm_path)==DATASET['normalization_sha256'];NORMALIZATION=load_json(norm_path)
    split=verify_paired_identity(index,DATASET,TAIL_EXPORT_PROTOCOL,NORMALIZATION)
    assert split==load_json(split_path) and TAIL_EXPORT_PROTOCOL['split_sha256']==D200_SPLIT_SHA
    train,val=split['train_demos'],split['validation_demos'];assert OPTIONS['scene_demo'] in train
    for key,shape in [('action_mean',(7,)),('action_std',(7,)),('proprio_mean',(36,)),('proprio_std',(36,))]:
        value=np.asarray(NORMALIZATION[key]);assert value.shape==shape and np.isfinite(value).all()
        if key.endswith('std'):assert (value>0).all()
    # Tail exports preserve the exact source; source resolution uses the parent protocol.
    parent=safe_child(PROJECT_ROOT,'data/processed/paired_gpu_source/'+PARENT_DATASET_ID)
    parent_data=load_json(parent/'paired_dataset.json');SOURCE_PROTOCOL=load_json(parent/'protocol.json')
    assert canonical(SOURCE_PROTOCOL)==parent_data['source_protocol_sha256']==D200_SOURCE_PROTOCOL_HASH
    assert SOURCE_PROTOCOL['version']=='paired-gpu-source-d200-v3' and SOURCE_PROTOCOL['build_sha256']==D200_PREPARATION_BUILD
    assert SOURCE_PROTOCOL['raw_sha256']==RAW_SHA and SOURCE_PROTOCOL['split_sha256']==D200_SPLIT_SHA
    assert SOURCE_PROTOCOL['split']==parent_data['split']==split and parent_data['dataset_id']==PARENT_DATASET_ID
    assert parent_data['status']=='COMPLETED' and parent_data['schema']==2
    fingerprint=canonical({'source_protocol':canonical(SOURCE_PROTOCOL),'export_schema':2,
        'demos':[{k:v for k,v in r.items() if k not in ['absolute_directory','relative_directory']} for r in parent_data['demos']],
        'normalization':PARENT_NORM_HASH,'split':split})
    assert fingerprint==parent_data['paired_data_fingerprint']==PARENT_DATA_HASH
    parent_rows={r['demo_id']:r for r in parent_data['demos']}
    assert len(parent_rows)==200 and set(parent_rows)==set(train+val)
    assert SOURCE_PROTOCOL['source_contract']==CONTRACT
    assert SOURCE_PROTOCOL['verified_source_sha256']==SOURCE_SHA and SOURCE_PROTOCOL['frozen_helpers_sha256']==FROZEN_SHA
    require_pinned_v2e(SOURCE_PROTOCOL['pinned_v2e_settings'])
    inventory=load_json(parent/'source_inventory.json')
    assert inventory['status']=='COMPLETED' and inventory['dataset_id']==PARENT_DATASET_ID
    assert inventory['source_protocol_hash']==D200_SOURCE_PROTOCOL_HASH
    source_records={r['demo_id']:r for r in inventory['demos']}
    assert len(source_records)==len(inventory['demos'])==200 and set(source_records)==set(train+val)
    lows=[];highs=[];audit=[]
    with h5py.File(raw_path,'r') as raw:
        for i,row in enumerate(DATASET['demos']):
            demo=row['demo_id'];folder=safe_child(DATASET_ROOT,row['relative_directory'])
            assert sha_file(folder/'manifest.json')==row['manifest_sha256']
            manifest=load_json(folder/'manifest.json');role='train' if demo in train else 'validation'
            assert manifest['status']=='COMPLETED' and manifest['schema']==3 and manifest['dataset_id']==DATASET_ID
            assert manifest['demo_id']==demo and manifest['dataset_role']==row['dataset_role']==role
            assert manifest['export_code_sha256']==EXPORT_BUILD and manifest['export_protocol_hash']==EXPORT_PROTOCOL_HASH
            assert manifest['tail_target_policy']==TAIL_POLICY
            assert manifest['visual_validation']['all_source_window_reconstruction']=='PASS'
            assert manifest['visual_validation']['historical_prefix_parity']=='PASS'
            assert manifest['visual_validation']['independent_event_max_abs']<=1e-5
            assert manifest['source_candidate_fingerprint']==row['source_candidate_fingerprint']==parent_rows[demo]['source_candidate_fingerprint']
            rec=source_records[demo]
            assert rec['status']=='COMPLETED' and rec['dataset_role']==role
            assert rec['manifest_sha256']==manifest['source_manifest_sha256']
            assert rec['candidate_fingerprint']==row['source_candidate_fingerprint']
            binding=load_json(parent/'demos'/demo/'complete.json')
            assert binding['kind']=='d200_source_binding' and binding['dataset_id']==PARENT_DATASET_ID
            assert binding['status']=='COMPLETED' and binding['demo_id']==demo
            for field in ['dataset_role','source_origin','manifest_sha256','candidate_fingerprint']:assert binding[field]==rec[field]
            pointer=load_json(DATASET_ROOT/'paired_exports'/demo/'complete.json')
            assert pointer['status']=='COMPLETED' and pointer['kind']=='paired_tail_export' and pointer['schema']==3
            assert pointer['dataset_id']==DATASET_ID and pointer['demo_id']==demo and pointer['manifest_sha256']==row['manifest_sha256']
            assert safe_child(DATASET_ROOT,pointer['relative_directory'])==folder
            shared={}
            for modality,key in [('rgb','rgb_sha256'),('events','events_sha256')]:
                name=modality+'_policy_samples.npz';assert sha_file(folder/name)==row[key]==manifest['files'][name]
                with np.load(folder/name,allow_pickle=False) as z:
                    assert set(z.files)==ARRAY_KEYS;shared[modality]={k:z[k] for k in COMMON_KEYS}
            group=raw['data/'+demo]
            n=verify_tail_shared(shared['rgb'],shared['events'],group['states'][:],group['actions'][:]);assert n==row['samples']
            if role=='train':
                assert NORMALIZATION['train_export_manifest_sha256'][demo]==row['manifest_sha256']
                actions=shared['rgb']['action_targets'];lows.append(actions.min((0,1)));highs.append(actions.max((0,1)))
            if demo==OPTIONS['scene_demo']:EXPORT_FOLDER=folder
            audit.append({'demo_id':demo,'role':role,'samples':n,'endpoint_repeat_slots':120,'manifest_sha256':row['manifest_sha256'],'source_origin':rec['source_origin']['kind']})
            if i%20==0 or i==199:print('Verified tail exports:',i+1,'/200',flush=True)
    RAW_LO=np.min(lows,axis=0).astype(np.float32);RAW_HI=np.max(highs,axis=0).astype(np.float32)
    assert (RAW_LO>=-1-1e-6).all() and (RAW_HI<=1+1e-6).all() and (RAW_HI>RAW_LO).all()
    SCENE_SOURCE_RECORD=source_records[OPTIONS['scene_demo']]
    SOURCE_FOLDER,source_manifest=resolve_demo_source(parent,SCENE_SOURCE_RECORD,SOURCE_PROTOCOL)
    assert SCENE_SOURCE_RECORD['source_origin']['kind']=='reviewed_d50_reference'
    SCENE_SOURCE_PROTOCOL_HASH=SCENE_SOURCE_RECORD['source_origin']['source_protocol_hash'];assert SCENE_SOURCE_PROTOCOL_HASH==D50_PROTOCOL_HASH
    RUN_RECORDS={}
    for modality,identity in SOURCES.items():
        run=PROJECT_ROOT/'artifacts/checkpoints'/modality/identity['run_id']
        record=load_json(run/'run_config.json');validate_run_record(record,modality)
        assert canonical(load_json(run/'normalization.json'))==NORM_HASH
        status=load_json(run/'status.json')
        assert status['run_id']==identity['run_id'] and status['status']=='COMPLETED' and status['durable_step']==4500
        assert status['stop_reason']=='budget_exhausted';RUN_RECORDS[modality]=record
    a=RUN_RECORDS['rgb']['config'].copy();b=RUN_RECORDS['events']['config'].copy()
    for key in ['modality','visual_normalization']:a.pop(key,None);b.pop(key,None)
    assert a==b, 'Paired training recipe mismatch.'
    write_csv(SESSION/'paired_data_audit.csv',audit)
    atomic_json(SESSION/'input_audit.json',{'status':'PASS','dataset_id':DATASET_ID,'data_fingerprint':DATA_HASH,
        'normalization_hash':NORM_HASH,'export_protocol_hash':EXPORT_PROTOCOL_HASH,'parent_dataset_id':PARENT_DATASET_ID,
        'raw_train_min':RAW_LO,'raw_train_max':RAW_HI,'demos':200,'train_samples':7460,'validation_samples':1806,
        'scene_source_record':SCENE_SOURCE_RECORD,'runs':RUN_RECORDS})
    return source_manifest



def prepare_dependencies():
    modules=split_frozen(FROZEN_PATH.read_text())
    installer=next(n for n in ast.parse(modules['01_setup.py']).body if isinstance(n,ast.FunctionDef) and n.name=='ensure_dependencies')
    exec(compile(ast.Module(body=[installer],type_ignores=[]),'verified-dependency-installer','exec'),globals())
    ensure_dependencies()
    requirements={'cv2':'opencv-python-headless','argcomplete':'argcomplete','engineering_notation':'engineering-notation',
      'screeninfo':'screeninfo','easygui':'easygui','plyer':'plyer','dv_processing':'dv-processing','scipy':'scipy','tqdm':'tqdm'}
    missing=[package for module,package in requirements.items() if importlib.util.find_spec(module) is None]
    if missing: subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
    importlib.invalidate_caches()
    import torch, torchvision, cv2, h5py
    assert torch.cuda.is_available(), 'Use a fresh Colab T4 GPU runtime.'
    return {'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,
        'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'opencv':cv2.__version__,
        'h5py':h5py.__version__,'mujoco':importlib.metadata.version('mujoco'),'robosuite':importlib.metadata.version('robosuite')}

def run_process(worker,request,env,output,timeout=600):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    request={**request,'output':str(output)};atomic_json(output/'request_parent.json',request)
    started=time.perf_counter();last=started
    with open(output/'stdout.log','w') as log:
        process=subprocess.Popen([sys.executable,str(worker),str(output/'request_parent.json')],
            env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                try:process.wait(timeout=min(5,max(.01,timeout-(time.perf_counter()-started))))
                except subprocess.TimeoutExpired:pass
                elapsed=time.perf_counter()-started
                if process.poll() is None and elapsed>=timeout: raise TimeoutError('Worker budget exceeded: '+request['id'])
                if process.poll() is None and time.perf_counter()-last>=30:
                    progress=load_json(output/'progress.json') if (output/'progress.json').is_file() else None
                    print(request['id'],'running',round(elapsed),'s',progress or '',flush=True);last=time.perf_counter()
        except BaseException as exc:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                deadline=time.perf_counter()+45
                while process.poll() is None and time.perf_counter()<deadline:
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:print('Finalizing interrupted worker evidence...',flush=True)
                if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            preserve_worker_failure(output,exc)
            raise
    result=load_json(output/'worker_result.json') if (output/'worker_result.json').is_file() else {}
    if process.returncode!=0 or result.get('status') not in ['PASS','LOADABLE','COMPLETED']:
        preserve_worker_failure(output,result.get('error','Missing worker result'))
        print('Worker failure stage:',result.get('failure_stage','unknown'),flush=True)
        if result.get('traceback'):print(result['traceback'],flush=True)
        raise AssertionError('Worker failed: '+request['id']+' | '+str(result.get('error') or result.get('stop_reason') or 'No error message')+' | stage='+str(result.get('failure_stage','unknown'))+' | inspect '+str(output/'stdout.log'))
    result.update(output_directory=str(output),exit_code=process.returncode)
    atomic_json(output/'parent_result.json',result)
    return result

def split_frozen(source):
    import re
    starts=list(re.finditer(r'^# MODULE ([A-Za-z0-9_]+\.py)\n',source,re.MULTILINE))
    return {m.group(1):source[m.end():starts[i+1].start() if i+1<len(starts) else len(source)] for i,m in enumerate(starts)}

def cache_checkpoints():
    records={}
    for modality,identity in SOURCES.items():
        for stage in ['best']:
            selected=identity[stage]
            source=safe_child(PROJECT_ROOT/'artifacts/checkpoints'/modality/identity['run_id'],selected['relative'])
            assert source.is_file(), 'Exact selected checkpoint unavailable: '+str(source)
            assert sha_file(source)==selected['sha256'], 'Selected checkpoint checksum changed.'
            cached=WORKSPACE/(str(TRAINING_SEED)+'_'+modality+'_'+stage+'.ckpt');shutil.copyfile(source,cached)
            assert sha_file(cached)==selected['sha256']
            records[modality+'_'+stage]={'local':str(cached),'source':str(source),**selected}
            print('Verified exact checkpoint:',modality,stage,selected['step'],flush=True)
    return records

def live_request(task,modality,stage,**extra):
    return {'id':task+'_'+modality+'_'+stage,'task':task,'training_seed':TRAINING_SEED,'project_root':str(PROJECT_ROOT),
        'payload_root':str(WORKSPACE),'source_folder':str(SOURCE_FOLDER),'export_folder':str(EXPORT_FOLDER),
        'record':RUN_RECORDS[modality],'normalization':NORMALIZATION,'raw_lo':RAW_LO,'raw_hi':RAW_HI,
        'modality':modality,'stage':stage,'checkpoint':CACHED[modality+'_'+stage]['local'],
        'options':OPTIONS,'device_id':GPU_INDEX,'protocol_hash':PROTOCOL_HASH,'output_root':str(OUTPUT),
        'expected_graphics':GRAPHICS,'selected_library':GPU_LIBRARY['resolved_path'],
        'selected_library_sha256':GPU_LIBRARY['sha256'],**extra}

def verify_paired_reset(job,row,previous_rows):
    name=next(n for n in row['artifacts'] if Path(n).name=='initial_state.npy')
    initial=np.load(safe_child(OUTPUT,name),allow_pickle=False)
    assert initial.shape==(32,) and np.isfinite(initial).all()
    for other in previous_rows:
        if other['status']!='COMPLETED' or other['environment_seed']!=job['environment_seed']: continue
        ref=next(n for n in other['artifacts'] if Path(n).name=='initial_state.npy')
        np.testing.assert_allclose(initial,np.load(safe_child(OUTPUT,ref),allow_pickle=False),atol=1e-9,rtol=0,
            err_msg='Paired reset differs. Comparison blocked.')

def run_all():
    global GPU_ENV, GPU_INDEX, GPU_LIBRARY, GRAPHICS, CACHED, PROTOCOL_HASH, OUTPUT, PROTOCOL
    for stale in ['OUTPUT','PROTOCOL','PROTOCOL_HASH']: globals().pop(stale,None)
    summary={'status':'RUNNING','code_version':VERSION,'build_sha256':BUILD_SHA,'started_utc':utc(),
        'training_performed':False,'scored_episodes':0,'scope':'development only; not locked final evaluation'}
    failure=None;rows=[];started=time.perf_counter()
    try:
        validate_budget(OPTIONS)
        runtime=prepare_dependencies();atomic_json(SESSION/'runtime.json',runtime)
        source_manifest=audit_inputs()
        # Reuse the exercised NVIDIA selection code, with no source production call.
        GPU_ENV,GPU_INDEX,GPU_LIBRARY,probe=hardware_selection();GRAPHICS=probe['graphics']
        assert GRAPHICS['hardware_nvidia_verified'] is True
        request=source_request(OPTIONS['scene_demo'],'train',SCENE_SOURCE_PROTOCOL_HASH,GPU_INDEX,GPU_LIBRARY)
        source_gate=run_process(PRODUCTION_WORKER_PATH,{**request,'id':'fresh_complete_source_gate','task':'online',
            'offline_directory':str(SOURCE_FOLDER)},GPU_ENV,SESSION/'fresh_complete_source_gate',900)
        assert source_gate['parity']['status']==source_gate['physics_noninterference']['status']==source_gate['physics']['status']=='PASS'
        assert source_gate['parity']['event_max_abs']<=1e-5
        CACHED=cache_checkpoints()
        PROTOCOL={'version':VERSION,'build_sha256':BUILD_SHA,'dataset_id':DATASET_ID,'data_fingerprint':DATA_HASH,
            'normalization_hash':NORM_HASH,'sources':SOURCES,'options':OPTIONS,'source_protocol':SOURCE_PROTOCOL,'tail_export_protocol':TAIL_EXPORT_PROTOCOL,
            'scene_xml_sha256':source_manifest['provenance']['raw_scene_xml_sha256'],'runtime':runtime,'graphics':GRAPHICS,
            'action_bounds':{'raw_lo':RAW_LO,'raw_hi':RAW_HI,'scope':'160 D200 training demonstration targets only'},
            'sampling':'100-step DDPM epsilon FP32 EMA; empirical training-range x0 posterior projection in z-score coordinates',
            'initial_history':'duplicate RGB reset; zero previous/current Events at t=0; then causal consecutive 20Hz observations',
            'source_lifecycle':'BOTH modalities use three native draws, primary draw3 and independent shadow draw2 V2E; RGB uses 20Hz endpoints',
            'rollout_acquisition':ROLLOUT_ACQUISITION,
            'completion_reuse':{'rule':'Only verified completed episodes in this exact D200 protocol; no historical pilot imports'},
            'scene_source_origin':SCENE_SOURCE_RECORD,
            'action_alignment':'actions[t:t+16]; execute plan[0:8]',
            'scope':'one fixed training scene geometry, new seeded resets; one training seed; 10 pairs; selected EMA4500 is both validation best and fixed-budget final; development evaluation only'}
        PROTOCOL_HASH=canonical(PROTOCOL)
        OUTPUT=PROJECT_ROOT/'results/closed_loop_development'/('paired_gpu_d200_tail4500_v1_'+PROTOCOL_HASH[:12])
        OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/'episodes').mkdir(exist_ok=True)
        if (OUTPUT/'protocol.json').exists(): assert canonical(load_json(OUTPUT/'protocol.json'))==PROTOCOL_HASH
        else: atomic_json(OUTPUT/'protocol.json',PROTOCOL)
        atomic_json(SESSION/'protocol.json',PROTOCOL)
        print('Pilot directory:',OUTPUT,'\nD200 tail budget: 20 selected4500 episodes (10 pairs); exact-protocol resume only; no training.',flush=True)
        preflights=[]
        for stage in ['best']:
            for modality in ['rgb','events']:
                result=run_process(LIVE_WORKER_PATH,live_request('preflight',modality,stage),GPU_ENV,
                    SESSION/('preflight_'+modality+'_'+stage),900)
                assert result['status']=='PASS';preflights.append(result)
        atomic_json(SESSION/'preflight.json',{'status':'PASS','source_gate':source_gate,'model_gates':preflights})
        jobs=schedule(OPTIONS)
        for job in jobs:
            row=verified_episode(OUTPUT,PROTOCOL_HASH,job)
            if row is None:
                initial_reference=None
                for other in rows:
                    if other['environment_seed']==job['environment_seed']:
                        initial_reference=str(safe_child(OUTPUT,next(n for n in other['artifacts'] if Path(n).name=='initial_state.npy')));break
                print('Running:',job['key'],'policy seed:',job['policy_seed'],flush=True)
                result=run_process(LIVE_WORKER_PATH,live_request('episode',job['modality'],job['stage'],
                    job=job,initial_reference=initial_reference),GPU_ENV,SESSION/('worker_'+job['key']),OPTIONS['episode_timeout_seconds'])
                row=verified_episode(OUTPUT,PROTOCOL_HASH,job)
                assert row is not None and result['status']=='COMPLETED', 'Episode completion not durable.'
            else: print('Reusing verified completed episode:',job['key'],flush=True)
            verify_paired_reset(job,row,rows);rows.append(row)
            print(len(rows),'/',len(jobs),'|',job['key'],'COMPLETED | any=',row['any_success'],
                '| sustained=',row['sustained_success'],'| steps=',row['steps'],'| wall min=',round(row['wall_seconds']/60,2),flush=True)
            summary['scored_episodes']=len(rows);summary['episodes']=rows
            atomic_json(OUTPUT/'pilot_progress.json',summary)
            summarize(OUTPUT,PROTOCOL,rows,len(jobs),final=False)
        summary.update(status='COMPLETED',episodes=rows,scored_episodes=len(rows))
    except BaseException as exc:
        failure=exc;summary.update(status='ABORTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=str(exc),traceback=traceback.format_exc(),episodes=rows)
        print('Stopped safely:',summary['status'],'Completed episode pointers remain resumable.',flush=True)
    finally:
        summary.update(ended_utc=utc(),wall_seconds=time.perf_counter()-started)
        atomic_json(SESSION/'execution_summary.json',summary)
        if 'OUTPUT' in globals():
            atomic_json(OUTPUT/'pilot_progress.json',summary)
            try: summarize(OUTPUT,PROTOCOL,rows,len(schedule(OPTIONS)),final=summary['status']=='COMPLETED',make_bundles=True)
            except Exception as exc:
                summary.update(status='FAILED',error='Evidence publication failed: '+str(exc),traceback=traceback.format_exc())
                if failure is None:failure=exc
                atomic_json(SESSION/'summary_failure.json',{'error':str(exc),'traceback':traceback.format_exc()})
        atomic_json(SESSION/'execution_summary.json',summary)
        if 'OUTPUT' in globals():atomic_json(OUTPUT/'pilot_progress.json',summary)
        bundle=make_bundle(SESSION,SESSION/'pilot_session_evidence.zip',include_videos=False)
        print('Session evidence (also upload on failure):',bundle,flush=True)
        if 'OUTPUT' in globals(): print('Main evidence:',OUTPUT/'pilot_evidence.zip','\nWith videos:',OUTPUT/'pilot_full_bundle.zip',flush=True)
    if failure is not None: raise failure
    return summary


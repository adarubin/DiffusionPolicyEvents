"""Run All source generation, durable per-demo commits and one fresh-runtime preflight."""
import importlib.metadata, signal, subprocess, sys, tempfile, traceback

def preserve_worker_failure(output, message):
    output=Path(output)
    target=SESSION/'failed_worker_evidence'/(output.name+'_'+uuid.uuid4().hex[:8])
    target.mkdir(parents=True,exist_ok=False)
    atomic_json(target/'failure_location.json',{'source_directory':str(output),'error':str(message)})
    for p in output.glob('*'):
        if p.is_file() and (p.suffix in ['.json','.csv','.png','.log']):shutil.copyfile(p,target/p.name)

def run_process(worker, request, env, output, timeout=600):
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
                    print(request['id'],'running',round(elapsed),'s',flush=True);last=time.perf_counter()
        except BaseException as exc:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            preserve_worker_failure(output,exc)
            raise
    result=json.loads((output/'worker_result.json').read_text()) if (output/'worker_result.json').is_file() else {}
    if process.returncode!=0 or result.get('status') not in ['PASS','LOADABLE','COMPLETED']:
        message='Worker failed; inspect '+str(output/'stdout.log')+' | '+str(result.get('error'))
        preserve_worker_failure(output,message)
        raise AssertionError(message)
    result.update(output_directory=str(output),exit_code=process.returncode)
    atomic_json(output/'parent_result.json',result)
    return result

def hardware_selection():
    library=Path(OPTIONS['nvidia_library']);assert library.is_file(), 'NVIDIA library unavailable. Select a fresh T4 runtime.'
    env=os.environ.copy();env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0')
    for k in ['LIBGL_ALWAYS_SOFTWARE','MESA_LOADER_DRIVER_OVERRIDE','GALLIUM_DRIVER']:env.pop(k,None)
    env['LD_LIBRARY_PATH']=':'.join(dict.fromkeys([str(library.parent),str(library.resolve().parent)]+[p for p in env.get('LD_LIBRARY_PATH','').split(':') if p]))
    load=run_process(LIBRARY_WORKER_PATH,{'id':'load','task':'load','library':str(library)},env,SESSION/'nvidia_library',20)
    vendor=SESSION/'private_nvidia_vendor.json';atomic_json(vendor,{'file_format_version':'1.0.0','ICD':{'library_path':load['resolved_path']}})
    env['__EGL_VENDOR_LIBRARY_FILENAMES']=str(vendor);env.pop('__EGL_VENDOR_LIBRARY_DIRS',None)
    base={'frozen_source':str(FROZEN_PATH),'frozen_sha256':FROZEN_SHA,
        'selected_library':load['resolved_path'],'selected_library_sha256':load['sha256']}
    inventory=run_process(RENDERER_WORKER_PATH,{**base,'id':'inventory','task':'inventory'},env,SESSION/'nvidia_inventory',90)
    for d in inventory.get('devices',[])[:2]:
        idx=int(d['index']);trial=env.copy();trial['MUJOCO_EGL_DEVICE_ID']=str(idx)
        try:
            probe=run_process(RENDERER_WORKER_PATH,{**base,'device_id':idx,'id':'probe','task':'probe'},trial,SESSION/('nvidia_probe_'+str(idx)),90)
        except AssertionError:
            continue
        if probe.get('graphics',{}).get('hardware_nvidia_verified') is True:
            return trial,idx,load,probe
    raise AssertionError('No NVIDIA hardware context. CPU fallback is blocked.')

def source_request(demo, role, protocol_hash, device, library):
    return {'frozen_source':str(FROZEN_PATH),'frozen_sha256':FROZEN_SHA,
        'renderer_worker':str(RENDERER_WORKER_PATH),'renderer_worker_sha256':sha_file(RENDERER_WORKER_PATH),
        'verified_source_worker':str(VERIFIED_SOURCE_WORKER_PATH),'verified_source_sha256':SOURCE_SHA,
        'source_protocol_hash':protocol_hash,'split_sha256':SPLIT_SHA,'demo':demo,'dataset_role':role,
        'max_frames':OPTIONS['max_frames_per_camera'],'device_id':device,
        'selected_library':library['resolved_path'],'selected_library_sha256':library['sha256']}

def source_generation(root, demo, role, request, env):
    pointer=root/'demos'/demo/'complete.json'
    if pointer.is_file():
        folder,rec=checked_pointer(root,pointer,DATASET_ID,'source')
        manifest=verify_source(folder,demo,request['source_protocol_hash']);check_source_draw_rows(folder,manifest)
        assert rec['dataset_role']==role and rec['candidate_fingerprint']==manifest['candidate_fingerprint']
        print('Reusing verified source:',demo,flush=True)
        return folder,manifest,True
    # A failed/partial generation is never considered complete; new attempts have new folders.
    generation=root/'demos'/demo/'generations'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    print('Generating',demo,role,flush=True)
    run_process(PRODUCTION_WORKER_PATH,{**request,'id':demo,'task':'offline'},env,generation,OPTIONS['source_timeout_s'])
    try:
        manifest=verify_source(generation,demo,request['source_protocol_hash']);check_source_draw_rows(generation,manifest)
    except BaseException as exc:preserve_worker_failure(generation,exc);raise
    assert manifest['provenance']['graphics']['nvidia_driver']==CURRENT_GRAPHICS['nvidia_driver']
    # Verify exact semantic arrays before publishing the pointer. Full Event reconstruction is Notebook03's independent check.
    import h5py
    with h5py.File(PROJECT_ROOT/'data/raw/lift_ph_raw.hdf5','r') as f:
        group=f['data/'+demo]
        validate_paired(generation,group['states'][:],group['actions'][:],group.attrs['model_file'],full_events=False)
    rec={'status':'COMPLETED','kind':'source','dataset_id':DATASET_ID,'demo_id':demo,'dataset_role':role,
        'relative_directory':str(generation.relative_to(root)),'manifest_sha256':sha_file(generation/'manifest.json'),
        'candidate_fingerprint':manifest['candidate_fingerprint'],'created_utc':utc()}
    atomic_json(pointer,rec)
    return generation,manifest,False

def fresh_preflight(folder, request, env):
    online=run_process(PRODUCTION_WORKER_PATH,{**request,'id':'runtime_causal_preflight','task':'online',
        'offline_directory':str(folder)},env,SESSION/'runtime_causal_preflight',OPTIONS['source_timeout_s'])
    assert online['parity']['status']=='PASS' and online['physics_noninterference']['status']=='PASS'
    assert online['parity']['render_state_exact_frames']==online['parity']['frames_per_camera']
    cpu_env=os.environ.copy();cpu_env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',
        LIBGL_ALWAYS_SOFTWARE='1',GALLIUM_DRIVER='llvmpipe')
    vendor=SESSION/'private_mesa_vendor.json';atomic_json(vendor,{'file_format_version':'1.0.0','ICD':{'library_path':'libEGL_mesa.so.0'}})
    cpu_env['__EGL_VENDOR_LIBRARY_FILENAMES']=str(vendor);cpu_env.pop('__EGL_VENDOR_LIBRARY_DIRS',None)
    cpu_request={k:v for k,v in request.items() if k not in ['selected_library','selected_library_sha256']}
    cpu=run_process(PRODUCTION_WORKER_PATH,{**cpu_request,'device_id':0,'id':'runtime_cpu_physics','task':'physics_cpu'},
        cpu_env,SESSION/'runtime_cpu_physics',180)
    a=np.load(SESSION/'runtime_cpu_physics/fixed_physics_trace.npy',allow_pickle=False)
    b=np.load(SESSION/'runtime_causal_preflight/fixed_physics_trace.npy',allow_pickle=False)
    assert a.shape==b.shape==(4,32) and np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_allclose(a,b,atol=1e-9,rtol=0)
    for key in ['compiled_scene','raw_scene_xml_sha256','rewritten_scene_xml_sha256','asset_files','state_array','action_array']:
        left=json.loads((SESSION/'runtime_cpu_physics/provenance.json').read_text())
        right=json.loads((SESSION/'runtime_causal_preflight/provenance.json').read_text())
        assert left[key]==right[key], 'Current runtime CPU/GPU scene mismatch: '+key
    report={'status':'PASS','training_demo':'demo_10','full_trajectory_parity':online['parity'],
        'render_emulation_noninterference':online['physics_noninterference'],
        'cpu_gpu_fixed_physics_max_abs':float(np.max(np.abs(a-b))),
        'scope':'fresh-runtime source and three fixed control actions; no policy training/rollout'}
    atomic_json(SESSION/'preflight.json',report);print('Fresh-runtime source/physics preflight PASS',flush=True)
    return report

def generate_d50():
    global DATASET_ID,DATASET_ROOT,CURRENT_GRAPHICS
    summary={'status':'RUNNING','started_utc':utc(),'training_performed':False,'policy_loaded':False,
        'scored_episodes':0,'expected_demos':50,'completed_demos':[],'code_version':CODE_VERSION,'code_sha256':BUILD_SHA}
    records=[];started=time.perf_counter();failure=None
    try:
        verify_validation_bundle(VALIDATION_BUNDLE)
        assert sha_file(PROJECT_ROOT/'data/raw/lift_ph_raw.hdf5')==RAW_SHA
        split=verify_split(PROJECT_ROOT/'configs/training/d50_rgb_event_train_val.json')
        env,idx,library,probe=hardware_selection();CURRENT_GRAPHICS=probe['graphics']
        stack={p:importlib.metadata.version(p) for p in ['numpy','torch','torchvision','mujoco','robosuite']}
        import cv2
        stack['opencv']=cv2.__version__
        protocol={'version':CODE_VERSION,'build_sha256':BUILD_SHA,'verified_source_sha256':SOURCE_SHA,
            'frozen_helpers_sha256':FROZEN_SHA,'validation_bundle_sha256':VALIDATION_SHA,'raw_sha256':RAW_SHA,
            'split_sha256':SPLIT_SHA,'split':split,'source_contract':CONTRACT,'options':OPTIONS,
            'runtime_stack':stack,'python_major_minor':list(sys.version_info[:2]),
            'graphics':{k:CURRENT_GRAPHICS[k] for k in ['vendor','renderer','version','nvidia_driver']},
            'nvidia_library_sha256':library['sha256'],'schema':1,
            'policy_contract':{'proprio_per_observation':18,'observation_horizon':2,'prediction_horizon':16,
                'execution_horizon':8,'control_hz':20,'source_hz':200,'first_control_step':2},
            'adapter_scope':'explicit train/validation membership and removal of historical checkpoint-directory existence check; verified render/physics/emulation definitions unchanged',
            'normalization':'not fitted here; Notebook03 fits NEW training-only statistics'}
        source_hash=canonical(protocol);DATASET_ID='d50_paired_gpu_v1_'+source_hash[:12]
        DATASET_ROOT=PROJECT_ROOT/'data/processed/paired_gpu_source'/DATASET_ID
        DATASET_ROOT.mkdir(parents=True,exist_ok=True)
        if (DATASET_ROOT/'protocol.json').is_file():assert canonical(json.loads((DATASET_ROOT/'protocol.json').read_text()))==source_hash
        else:atomic_json(DATASET_ROOT/'protocol.json',protocol)
        summary.update(dataset_id=DATASET_ID,dataset_root=str(DATASET_ROOT),source_protocol_hash=source_hash,graphics=CURRENT_GRAPHICS)
        first='demo_10';request=source_request(first,'train',source_hash,idx,library)
        folder,manifest,reused=source_generation(DATASET_ROOT,first,'train',request,env)
        summary['preflight']=fresh_preflight(folder,request,env)
        all_demos=[first]+[d for d in sorted(split['train_demos']+split['validation_demos']) if d!=first]
        for demo in all_demos:
            role='train' if demo in split['train_demos'] else 'validation'
            if demo!=first:folder,manifest,reused=source_generation(DATASET_ROOT,demo,role,source_request(demo,role,source_hash,idx,library),env)
            rec={'demo_id':demo,'dataset_role':role,'status':'COMPLETED','reused':reused,
                'frames_per_camera':manifest['frames_per_camera'],'event_windows':manifest['event_windows'],
                'samples':manifest['paired_policy_samples'],'candidate_fingerprint':manifest['candidate_fingerprint'],
                'manifest_sha256':sha_file(folder/'manifest.json'),'generation_relative':str(folder.relative_to(DATASET_ROOT)),
                'raw_scene_xml_sha256':manifest['provenance']['raw_scene_xml_sha256'],
                'render_seconds':manifest['render_seconds'],'local_grayscale_failures':0}
            records.append(rec);summary['completed_demos']=[r['demo_id'] for r in records]
            atomic_json(SESSION/'dataset_build_summary.json',summary);write_csv(SESSION/'source_generation_progress.csv',records)
            print(len(records),'/50 durable |',demo,'samples',rec['samples'],flush=True)
        assert len(records)==50 and set(summary['completed_demos'])==set(split['train_demos']+split['validation_demos'])
        summary.update(status='COMPLETED',source_demos=records,train_samples=sum(r['samples'] for r in records if r['dataset_role']=='train'),
            validation_samples=sum(r['samples'] for r in records if r['dataset_role']=='validation'),
            frames_both_cameras=2*sum(r['frames_per_camera'] for r in records))
        atomic_json(DATASET_ROOT/'source_inventory.json',{'status':'COMPLETED','dataset_id':DATASET_ID,'source_protocol_hash':source_hash,'demos':records})
        # This index selects a completed SOURCE, not a training-ready dataset. Notebook03 has a separate final index.
        atomic_json(PROJECT_ROOT/'configs/processed_sources/paired_gpu_d50_latest.json',{'status':'COMPLETED','dataset_id':DATASET_ID,
            'relative_root':str(DATASET_ROOT.relative_to(PROJECT_ROOT)),'protocol_sha256':sha_file(DATASET_ROOT/'protocol.json'),
            'inventory_sha256':sha_file(DATASET_ROOT/'source_inventory.json'),'schema':1})
    except BaseException as exc:
        failure=exc;summary.update(status='ABORTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=str(exc),traceback=traceback.format_exc())
    finally:
        summary.update(ended_utc=utc(),elapsed_seconds=time.perf_counter()-started)
        atomic_json(SESSION/'dataset_build_summary.json',summary);write_csv(SESSION/'source_generation_progress.csv',records)
        (SESSION/'dataset_build_summary.md').write_text('# Notebook02 paired GPU source\n\n'+json.dumps({k:v for k,v in summary.items() if k not in ['preflight','source_demos','traceback']},indent=2))
        bundle=evidence_bundle(SESSION,'paired_source_build_bundle.zip')
        print('Status:',summary['status'],'\nUpload:',bundle,'\nCompleted demos:',len(records),'/50',flush=True)
    if failure is not None:raise failure
    print('Next: Notebook03 Run All on CPU. No model has been trained.',flush=True)
    return summary

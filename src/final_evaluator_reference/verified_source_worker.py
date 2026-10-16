"""One-demo source version candidate; no policy model or optimizer is constructed."""
import hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def write_json(path,value):
    def default(v):
        if hasattr(v,'tolist'):return v.tolist()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v).__name__)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=default));os.replace(tmp,path)

def timeline(states):
    assert states.ndim==2 and states.shape[1]==32 and len(states)>=18
    assert np.isfinite(states).all()
    np.testing.assert_allclose(states[:,0],np.arange(len(states))/20.,atol=1e-7,rtol=0)
    return np.arange((len(states)-1)*10+1,dtype=np.float64)/200.,np.arange(2,len(states),dtype=np.float64)/20.

def array_digest(value):
    value=np.ascontiguousarray(value)
    return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(value.tobytes()).hexdigest()}

def preserve_endpoints(interpolator):
    """Keep recorded endpoint states byte-for-byte; SLERP only reconstructs intermediate states."""
    def interpolate(a,b,alpha):
        assert 0<=alpha<=1
        if alpha==0:return a.copy()
        if alpha==1:return b.copy()
        return interpolator(a,b,alpha)
    return interpolate

def validate_events(events,final_time):
    assert events.ndim==2 and events.shape[1]==4 and np.isfinite(events).all()
    if len(events):
        assert np.all(np.diff(events[:,0])>=-1e-7)
        assert events[:,0].min()>=-1e-6 and events[:,0].max()<=final_time+1e-6
        assert np.isin(events[:,3],[-1,1]).all()
        xy=events[:,1:3];assert ((xy>=0)&(xy<84)).all() and np.array_equal(xy,xy.astype(np.int64))

def parity_row(reference,current,camera,kind,index):
    assert reference.shape==current.shape and np.isfinite(current).all()
    delta=current.astype(np.float64)-reference.astype(np.float64)
    return {'camera':camera,'kind':kind,'index':int(index),'max_abs':float(np.max(np.abs(delta))),
      'mae':float(np.mean(np.abs(delta))),'rms':float(np.sqrt(np.mean(delta**2)))}

RENDER_LIFECYCLE_CONTRACT={
    'version':'interleaved-render-emulate-v1',
    'order':'ensure V2E imports; construct renderer and warm up; create persistent per-camera emulators; restore/forward/render each frame; emulate both cameras before next frame',
    'offline':'retain complete raw packets; build all window tensors after the complete trajectory',
    'causal':'retain/prune pending packets; build windows at each completed control endpoint',
    'scope':'shared runtime lifecycle, independent processes and tensor construction; no cached-image substitution or changed graphics flags'}

RENDER_STABILITY_CONTRACT={
    'version':'fixed-three-draws-gray-exact-rgb-bounded-v2',
    'draws_per_observation':3,
    'selection':'always return draw3; audit draw2/3 grayscale exactly; bound native RGB by one level in at most two color values per camera per observation',
    'rgb_max_abs_uint8':1,'rgb_max_changed_color_values_per_camera':2,
    'event_input':'native cv2.cvtColor RGB2GRAY, unchanged; exact grayscale acceptance for BOTH cameras',
    'state_rule':'flattened state and recorded derived body/geometry/camera/light arrays must be unchanged across all three draws',
    'feed_rule':'one observation/event-emulator feed per source timestamp, not three',
    'scope':'explicit NEW acceptance protocol: bounded native RGB; exact grayscale and existing Event tensor tolerance. Historical exact-RGB failures remain failures. No image transformation, reference-dependent selection or adaptive retries.'}

def input_fidelity(ctx,reference,current):
    assert reference.shape==current.shape==(84,84,3)
    assert reference.dtype==current.dtype==np.uint8
    delta=current.astype(np.int16)-reference.astype(np.int16)
    a=ctx['cv2'].cvtColor(reference,ctx['cv2'].COLOR_RGB2GRAY)
    b=ctx['cv2'].cvtColor(current,ctx['cv2'].COLOR_RGB2GRAY)
    assert a.shape==b.shape==(84,84) and a.dtype==b.dtype==np.uint8
    maximum=int(np.max(np.abs(delta)));changed=int(np.count_nonzero(delta))
    return {'rgb_max_abs_uint8':maximum,'rgb_changed_color_values':changed,
      'rgb_changed_fraction':changed/reference.size,
      'rgb_within_bound':maximum<=1 and changed<=2,
      'grayscale_max_abs_uint8':int(np.max(np.abs(b.astype(np.int16)-a.astype(np.int16)))),
      'grayscale_changed_pixels':int(np.count_nonzero(a!=b)),
      'grayscale_exact':bool(np.array_equal(a,b))}

def save_native_pair_evidence(ctx,folder,label,reference,current,record):
    folder=Path(folder);errors=[]
    write_json(folder/(label+'.json'),record)
    delta=current.astype(np.int16)-reference.astype(np.int16)
    gray_a=ctx['cv2'].cvtColor(reference,ctx['cv2'].COLOR_RGB2GRAY)
    gray_b=ctx['cv2'].cvtColor(current,ctx['cv2'].COLOR_RGB2GRAY)
    arrays={'reference_rgb':reference,'current_rgb':current,
      'rgb_difference_amplified':np.clip(np.abs(delta)*128,0,255).astype(np.uint8),
      'reference_gray':gray_a,'current_gray':gray_b}
    for name,array in arrays.items():
        try:
            value=array[:,:,::-1] if array.ndim==3 else array
            if not ctx['cv2'].imwrite(str(folder/(label+'_'+name+'.png')),value):errors.append(name)
        except Exception as exc:errors.append(name+': '+str(exc))
    write_json(folder/(label+'.json'),{**record,'image_export_errors':errors})

def render_state_snapshot(env):
    result={'flattened_state':array_digest(env.sim.get_state().flatten()),'derived':{}}
    for name in ['xpos','xmat','geom_xpos','geom_xmat','cam_xpos','cam_xmat','light_xpos','light_xdir']:
        try:result['derived'][name]=array_digest(np.asarray(getattr(env.sim.data,name)))
        except AttributeError:result['derived'][name]={'available':False}
    return result

def install_render_stability(ctx,folder):
    folder=Path(folder)
    if ctx.get('_render_stability_installed'):
        if ctx['_render_stability_folder']!=folder:
            ctx['_render_stability_folder']=folder;ctx['_render_stability_audit'].clear()
        return
    raw_observation=ctx['get_observation'];audit=[]
    def stable_observation(env):
        before=render_state_snapshot(env);before_time=float(env.sim.get_state().flatten()[0])
        draws=[];states_unchanged=[];native_seconds=0.
        for _ in range(3):
            started=time.perf_counter();value=raw_observation(env);native_seconds+=time.perf_counter()-started
            # Preserve each native uint8 read independently; no averaging or value conversion.
            draws.append({k:np.asarray(v).copy() for k,v in value.items()})
            states_unchanged.append(render_state_snapshot(env)==before)
        pairs=[]
        for a,b in [(draws[0],draws[2]),(draws[1],draws[2])]:
            deltas=[b[c].astype(np.int16)-a[c].astype(np.int16) for c in ctx['CAMERAS']]
            pairs.append((max(int(np.max(np.abs(d))) for d in deltas),sum(int(np.count_nonzero(d)) for d in deltas)))
        prop_exact=bool(np.array_equal(draws[1]['proprio'],draws[2]['proprio']))
        fidelity={cam:input_fidelity(ctx,draws[1][cam],draws[2][cam]) for cam in ctx['CAMERAS']}
        row={'logical_read':len(audit),'sim_timestamp_s':before_time,'native_draws':3,
          'state_unchanged_each_draw':bool(all(states_unchanged)),
          'first_to_third_max_abs':pairs[0][0],'first_to_third_changed_color_values':pairs[0][1],
          'second_to_third_max_abs':pairs[1][0],'second_to_third_changed_color_values':pairs[1][1],
          'last_two_proprio_exact':prop_exact,'native_draw_seconds':native_seconds,
          'last_two_grayscale_exact':all(v['grayscale_exact'] for v in fidelity.values()),
          'last_two_rgb_within_bound':all(v['rgb_within_bound'] for v in fidelity.values())}
        audit.append(row)
        for cam,v in fidelity.items():
            marker=ctx['_render_stability_folder']/('same_state_first_difference_'+cam+'.json')
            if v['rgb_max_abs_uint8'] and not marker.exists():
                save_native_pair_evidence(ctx,marker.parent,marker.stem,draws[1][cam],draws[2][cam],
                  {'row':row,'camera':cam,'fidelity':v,'contract':RENDER_STABILITY_CONTRACT})
        if not all(states_unchanged) or not row['last_two_rgb_within_bound'] or not prop_exact:
            write_json(ctx['_render_stability_folder']/'render_stability_failure.json',{'contract':RENDER_STABILITY_CONTRACT,'row':row,
              'scope':'local fixed-state rendering failed; no Event feed or reference-dependent retry follows'})
            raise AssertionError('Same-state acquisition violates RGB bound, state or proprioception contract.')
        # A grayscale mismatch is audited through the full bounded trajectory, but NEVER accepted.
        # One unmodified third image is fed to V2E; final parity rejects any such mismatch.
        return draws[2]
    ctx.update(get_observation=stable_observation,_render_stability_installed=True,_render_stability_audit=audit,
               _render_stability_folder=folder)

def save_render_stability_audit(ctx,folder):
    rows=ctx.get('_render_stability_audit',[])
    ctx['write_csv'](Path(folder)/'render_stability.csv',rows)
    result={'contract':RENDER_STABILITY_CONTRACT,'logical_observation_reads':len(rows),
      'native_two_camera_draws':3*len(rows),
      'reads_with_first_to_third_difference':sum(r['first_to_third_max_abs']>0 for r in rows),
      'reads_with_unstable_last_two':sum(r['second_to_third_max_abs']>0 for r in rows),
      'reads_with_grayscale_mismatch':sum(not r['last_two_grayscale_exact'] for r in rows),
      'reads_outside_rgb_bound':sum(not r['last_two_rgb_within_bound'] for r in rows),
      'reads_with_state_change':sum(not r['state_unchanged_each_draw'] for r in rows),
      'reads_with_proprio_change':sum(not r['last_two_proprio_exact'] for r in rows),
      'scope':'all logical reads in this worker, including warmup/physics reads; stream feeds remain one per timestamp'}
    write_json(Path(folder)/'render_stability_summary.json',result)
    return result

def rgb_failure_evidence(ctx,folder,camera,index,reference,current,expected,env,offline_state):
    delta=current.astype(np.int16)-reference.astype(np.int16);changed=np.argwhere(delta!=0)
    ref_gray=ctx['cv2'].cvtColor(reference,ctx['cv2'].COLOR_RGB2GRAY)
    cur_gray=ctx['cv2'].cvtColor(current,ctx['cv2'].COLOR_RGB2GRAY)
    actual=env.sim.get_state().flatten();live_state=render_state_snapshot(env)
    record={'camera':camera,'frame':int(index),'timestamp_s':index/200.,
      'max_abs_uint8':int(np.max(np.abs(delta))),'changed_color_values':len(changed),
      'changed_pixels':int(np.count_nonzero(np.any(delta,axis=2))),
      'grayscale_max_abs':int(np.max(np.abs(cur_gray.astype(np.int16)-ref_gray.astype(np.int16)))),
      'grayscale_changed_pixels':int(np.count_nonzero(cur_gray!=ref_gray)),
      'first_32_color_differences':[{'y':int(y),'x':int(x),'channel':int(k),'offline':int(reference[y,x,k]),'causal':int(current[y,x,k])} for y,x,k in changed[:32]],
      'restored_state_exact':bool(np.array_equal(actual,expected)),
      'restored_state_max_abs':float(np.max(np.abs(actual-expected))),
      'offline_render_state':offline_state,'causal_render_state':live_state,
      'render_state_exact':offline_state==live_state,
      'acceptance':'FAILED: exact RGB gate unchanged; grayscale equality alone is not acceptance'}
    # Two fresh reads of the same current state: diagnose instability without replacing input bytes.
    repeats=[];errors=[]
    for _ in range(2):
        try:
            again=ctx['get_observation_original'](env)[camera]
            repeats.append({'max_abs_vs_first_causal':int(np.max(np.abs(again.astype(np.int16)-current.astype(np.int16)))),
                            'max_abs_vs_offline':int(np.max(np.abs(again.astype(np.int16)-reference.astype(np.int16)))),
                            'sha256':array_digest(again)['sha256']})
        except Exception as exc:errors.append('Repeat render: '+str(exc))
    record['same_state_repeat_renders']=repeats
    write_json(folder/'rgb_parity_failure.json',record)
    # Lossless portable images belong in the small evidence bundle.
    for name,array in [('rgb_reference.png',reference),('rgb_causal.png',current),
                       ('rgb_difference_amplified.png',np.clip(np.abs(delta)*128,0,255).astype(np.uint8))]:
        try:
            ok=ctx['cv2'].imwrite(str(folder/name),array[:,:,::-1])
            if not ok:errors.append('Cannot write '+name)
        except Exception as exc:errors.append(name+': '+str(exc))
    record['diagnostic_errors']=errors
    write_json(folder/'rgb_parity_failure.json',record)

EVENT_TIME_CONTRACT={
    'version':'timestamp-dtype-half-open-v1',
    'stored_timestamp_dtype':'float32',
    'boundary_rule':'cast both half-open bounds explicitly to the stored timestamp dtype before comparison',
    'surface_rule':'decay from the same representable upper bound; latest timestamps accumulated in float64',
    'pruning_rule':'retain timestamps >= the same representable previous-window lower bound',
    'scope':'quantized V2E timestamps; not sub-float32 timing accuracy; no tolerance-based reassignment'}

def event_bounds(events,start,end):
    assert events.ndim==2 and events.shape[1]==4 and events.dtype in [np.dtype('float32'),np.dtype('float64')]
    assert np.isfinite(start) and np.isfinite(end) and start<end
    lo,hi=events.dtype.type(start),events.dtype.type(end)
    assert lo<hi, 'Window collapsed at stored timestamp precision.'
    return lo,hi

def event_window_mask(events,start,end):
    lo,hi=event_bounds(events,start,end)
    return (events[:,0]>=lo)&(events[:,0]<hi)

def install_event_time_contract(ctx):
    """New candidate adapter only; never modify the frozen historical evaluator."""
    original_tensor=ctx['build_event_tensor'];original_stream=ctx['EventStream']
    def tensor(events,start,end):
        lo,hi=event_bounds(events,start,end)
        return original_tensor(events,lo,hi)
    class CanonicalEventStream(original_stream):
        def windows(self,end):
            assert self.last_frame_time is not None and abs(self.last_frame_time-float(end))<1e-8
            for cam in ctx['CAMERAS']:
                events=self.pending[cam]
                self.previous[cam]=tensor(events,end-.10,end-.05)
                self.current[cam]=tensor(events,end-.05,end)
                self.last_counts[cam]=int(np.count_nonzero(event_window_mask(events,end-.05,end)))
                lo,_=event_bounds(events,end-.10,end-.05)
                # Preserve exact-boundary events for the next half-open window.
                self.pending[cam]=events[events[:,0]>=lo]
            return self.previous,self.current
    ctx.update(build_event_tensor=tensor,EventStream=CanonicalEventStream)

def load_stream_contract(modules,ctx,worker):
    # The historical module also executes model-normalization/audit gates at top level.
    # This candidate needs its helper definitions only; initialize external handles explicitly.
    ctx.update(EventEmulator=None,cv2=None,EVENT_DEVICE=None)
    worker.execute_definitions(modules['04a_event_stream.py'],ctx)
    install_event_time_contract(ctx)
    ctx['interpolated_state']=preserve_endpoints(ctx['interpolated_state'])
    worker.execute_definitions(modules['04b_event_gates.py'],ctx)

def bind_runtime(request,folder):
    helper=Path(request['renderer_worker']);assert digest(helper)==request['renderer_worker_sha256']
    spec=importlib.util.spec_from_file_location('verified_renderer_helpers',helper)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    source=Path(request['frozen_source']).read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==request['frozen_sha256']
    if request.get('selected_library'):
        assert digest(request['selected_library'])==request['selected_library_sha256'], 'NVIDIA library changed after probe.'
    modules=worker.split_modules(source)
    ctx=worker.load_context(modules,folder,request.get('device_id'),False)
    root=ctx['PROJECT_ROOT'];raw=root/'data/raw/lift_ph_raw.hdf5'
    assert digest(raw)==ctx['EXPECTED_RAW_SHA256']
    meta=ctx['ENV_META']
    assert meta['env_name']=='Lift' and meta['env_version']=='1.5.0'
    assert meta['env_kwargs']['robots']==['Panda'] and meta['env_kwargs']['control_freq']==20
    assert meta['env_kwargs']['lite_physics'] is False
    controller=meta['env_kwargs']['controller_configs']
    assert controller['type']=='BASIC' and set(controller['body_parts'])=={'right'}
    arm=controller['body_parts']['right']
    assert arm['type']=='OSC_POSE' and arm['control_delta'] is True
    assert arm['input_ref_frame']=='world' and arm['impedance_mode']=='fixed'
    assert arm['input_min']==-1 and arm['input_max']==1 and arm['gripper']['type']=='GRIP'
    np.testing.assert_allclose(arm['output_max'],[.05,.05,.05,.5,.5,.5])
    split_path=root/'configs/training/d50_rgb_event_train_val.json'
    assert digest(split_path)==request['split_sha256'], 'Development split changed.'
    split=json.loads(split_path.read_text());train=split['train_demos'];val=split['validation_demos']
    assert len(train)==40 and len(val)==10 and set(train).isdisjoint(val)
    demo=request['demo'];assert demo in train, 'Source pilot is restricted to one training demo.'
    with ctx['h5py'].File(raw,'r') as f:
        group=f['data/'+demo];states=np.asarray(group['states'],dtype=np.float64);actions=np.asarray(group['actions'],dtype=np.float32)
        original=group.attrs['model_file'];original=original.decode() if isinstance(original,bytes) else original
        assert int(group.attrs['num_samples'])==len(states)==len(actions)
    assert actions.shape==(len(states),7) and np.isfinite(actions).all()
    times,ends=timeline(states)
    np.testing.assert_allclose(np.linalg.norm(states[:,13:17],axis=1),1.,atol=1e-6,rtol=0)
    assert len(times)<=request['max_frames'], 'One-demo frame budget exceeded; no truncation silently applied.'
    ctx['SCENE_DEMO']=demo;ctx['ORIGINAL_XML']=original;ctx['SCENE_XML']=ctx['repair_xml'](original)
    ctx['RAW_LO']=actions.min(0);ctx['RAW_HI']=actions.max(0)
    ctx['SCENE_TEMPLATE_DEMO']=demo
    # Import definitions from the frozen stream only. The old model/artifact-loading modules never execute.
    load_stream_contract(modules,ctx,worker)
    settings_path=root/'data/processed/events'/demo/'manifest.json'
    settings_record=json.loads(settings_path.read_text())
    assert settings_record['demo_id']==demo and settings_record['v2e']['device']=='cpu', 'Keep pinned CPU V2E, distinct from GPU image rendering.'
    settings=settings_record['v2e']
    assert settings['resolved_commit']==ctx['V2E_COMMIT'] and settings['emulator_seed']==42
    assert settings['pos_thres']==settings['neg_thres']==.2
    assert all(settings[k]==0 for k in ['sigma_thres','cutoff_hz','leak_rate_hz','shot_noise_rate_hz','refractory_period_s'])
    env=ctx['repair_make_environment'](ctx['SCENE_XML'],'pilot')
    try:
        ctx['verify_environment'](env);ctx['get_observation'](env)
        gl=getattr(getattr(env.sim,'_render_context_offscreen',None),'gl_ctx',None)
        if gl is not None and callable(getattr(gl,'make_current',None)):gl.make_current()
        graphics=ctx['graphics_identity']()
        graphics.update(hardware_nvidia_verified=worker.hardware_verified(graphics),context_class=type(gl).__name__,
            context_module=type(gl).__module__,render_gpu_device_id=getattr(env,'render_gpu_device_id',None))
        if request['task']=='physics_cpu':
            assert graphics.get('vendor')=='Mesa' and 'llvmpipe' in str(graphics.get('renderer','')).lower()
        else:assert worker.hardware_verified(graphics), 'NVIDIA identity required; CPU fallback is blocked.'
        scene=ctx['repair_model_snapshot'](env)
        xml_root=ctx['ET'].fromstring(ctx['SCENE_XML']);assets={}
        for node in xml_root.iter():
            name=node.get('file')
            if name:assets[str(Path(name).resolve())]=digest(name)
        provenance={'demo_id':demo,'raw_sha256':digest(raw),'split_sha256':digest(split_path),
          'raw_scene_xml_sha256':hashlib.sha256(original.encode()).hexdigest(),
          'rewritten_scene_xml_sha256':hashlib.sha256(ctx['SCENE_XML'].encode()).hexdigest(),
          'asset_files':assets,'compiled_scene':scene,'environment_metadata':ctx['ENV_META'],
          'runtime':ctx['RUNTIME'],'graphics':graphics,'state_array':array_digest(states),'action_array':array_digest(actions),
          'source_hz':200,'control_hz':20,'camera_names':ctx['CAMERAS'],'image_shape':[84,84,3],
          'rgb_orientation':'top-origin, wrapper render vertically flipped','interpolation':'linear with cube quaternion SLERP at intermediate times; exact recorded endpoints, not action replay',
          'scene_contract':'each demonstration uses its own recorded XML; this pilot validates one train demo only',
          'event_settings_manifest_sha256':digest(settings_path),'v2e_settings':settings,
          'event_time_contract':EVENT_TIME_CONTRACT,
          'render_lifecycle_contract':RENDER_LIFECYCLE_CONTRACT,
          'render_stability_contract':RENDER_STABILITY_CONTRACT,
          'normalization':'none learned in this one-demo pilot; future model statistics must use training demos only'}
        write_json(folder/'provenance.json',provenance)
        (folder/'recorded_scene.xml').write_text(original);(folder/'runtime_scene.xml').write_text(ctx['SCENE_XML'])
    finally:env.close()
    install_render_stability(ctx,folder)
    return ctx,states,actions,times,ends,provenance

def fixed_physics(ctx,folder):
    env=None
    try:
        env=ctx['make_env'](1995);ctx['verify_environment'](env)
        values=[env.sim.get_state().flatten().copy()]
        for action in [[.1,0,0,0,0,0,-.2],[0,.1,0,0,0,0,.2],[0,0,.1,0,0,0,0]]:
            before=float(env.sim.data.time);env.step(np.asarray(action,dtype=np.float32))
            np.testing.assert_allclose(float(env.sim.data.time)-before,.05,atol=1e-7,rtol=0)
            values.append(env.sim.get_state().flatten().copy())
        trace=np.asarray(values);assert np.isfinite(trace).all()
        np.save(folder/'fixed_physics_trace.npy',trace)
        return {'status':'PASS','steps':3,'seed':1995,'scope':'disposable fixed actions; no learned policy or Lift score'}
    finally:
        if env is not None:env.close()

def offline_source(ctx,states,actions,times,ends,folder,provenance):
    install_render_stability(ctx,folder)
    env=None;cameras=ctx['CAMERAS'];render_times=[];render_audit=[];emulators={}
    blocks={c:[] for c in cameras};emulation_seconds={c:0. for c in cameras}
    rgb={c:np.empty((len(times),84,84,3),dtype=np.uint8) for c in cameras}
    prop=np.empty((len(states),18),dtype=np.float32)
    # Match causal initialization and render/emulation ordering without reusing causal tensors.
    runtime=ctx['ensure_v2e']();write_json(folder/'v2e_runtime.json',runtime)
    try:
        env=ctx['repair_make_environment'](ctx['SCENE_XML'],'pilot');ctx['verify_environment'](env)
        ctx['get_observation'](env)  # Disposable warmup outside render timings.
        for cam in cameras:emulators[cam]=ctx['new_emulator']()
        for frame,stamp in enumerate(times):
            started=time.perf_counter();state=ctx['repair_state_at_frame'](states,frame,times,'notebook02_index')
            env.sim.set_state_from_flattened(state);env.sim.forward()
            np.testing.assert_allclose(env.sim.get_state().flatten(),state,atol=1e-9,rtol=0)
            obs=ctx['get_observation'](env);render_times.append(time.perf_counter()-started)
            render_audit.append(render_state_snapshot(env))
            for cam in cameras:rgb[cam][frame]=obs[cam]
            if frame%10==0:prop[frame//10]=obs['proprio']
            for cam in cameras:
                started=time.perf_counter()
                events=emulators[cam].generate_events(ctx['cv2'].cvtColor(obs[cam],ctx['cv2'].COLOR_RGB2GRAY),float(stamp))
                if events is not None and len(events):blocks[cam].append(np.asarray(events,dtype=np.float32).copy())
                emulation_seconds[cam]+=time.perf_counter()-started
            if frame%100==0:print('Offline NVIDIA source frame:',frame,'/',len(times),flush=True)
    finally:
        for emulator in emulators.values():
            cleanup=getattr(emulator,'cleanup',None)
            if callable(cleanup):cleanup()
        if env is not None:env.close()
        write_json(folder/'render_state_audit.json',render_audit)
    np.save(folder/'source_timestamps.npy',times)
    np.savez_compressed(folder/'control_state_action.npz',states=states,actions=actions,robot_proprio=prop,control_timestamps=states[:,0])
    ctx['write_csv'](folder/'render_frame_timings.csv',[{'frame':i,'render_pair_seconds':float(v)} for i,v in enumerate(render_times)])
    totals={};counts=[]
    for cam in cameras:
        np.savez_compressed(folder/(cam+'_rgb.npz'),frames=rgb[cam])
        started=time.perf_counter()
        raw=np.concatenate(blocks[cam]) if blocks[cam] else np.empty((0,4),dtype=np.float32)
        validate_events(raw,float(times[-1]))
        previous=np.stack([ctx['build_event_tensor'](raw,end-.10,end-.05) for end in ends])
        current=np.stack([ctx['build_event_tensor'](raw,end-.05,end) for end in ends])
        np.savez_compressed(folder/(cam+'_events.npz'),events=raw,previous=previous,current=current)
        totals[cam]={'raw_events':len(raw),'v2e_seconds':emulation_seconds[cam],
          'tensor_pack_seconds':time.perf_counter()-started,'v2e_timing_scope':'sum of per-frame grayscale conversion, emulator call and raw packet copy; excludes tensor packing and initialization'}
        for index,end in enumerate(ends):
            counts.append({'camera':cam,'window':index,'timestamp_s':float(end),
              'current_events':int(np.count_nonzero(event_window_mask(raw,end-.05,end))),
              'positive_active_pixels':int(np.count_nonzero(current[index,0])),
              'negative_active_pixels':int(np.count_nonzero(current[index,1]))})
    np.save(folder/'window_timestamps.npy',ends);ctx['write_csv'](folder/'event_activity.csv',counts)
    # Common policy windows: previous/current observations at t-1,t, targets[t:t+16].
    indices=np.arange(2,len(states)-15,dtype=np.int64)
    assert len(indices)>0
    targets=np.stack([actions[t:t+16] for t in indices]);assert targets.shape==(len(indices),16,7)
    packed={'control_steps':indices,'control_timestamps':indices/20.,'action_targets':targets,
       'proprio_previous':prop[indices-1],'proprio_current':prop[indices]}
    for cam in cameras:
        packed[cam+'_rgb_previous']=rgb[cam][(indices-1)*10];packed[cam+'_rgb_current']=rgb[cam][indices*10]
        with np.load(folder/(cam+'_events.npz'),allow_pickle=False) as z:
            packed[cam+'_events_previous']=z['previous'][indices-2];packed[cam+'_events_current']=z['current'][indices-2]
    np.savez_compressed(folder/'paired_policy_samples.npz',**packed)
    save_render_stability_audit(ctx,folder)
    generated={p.name:digest(p) for p in folder.glob('*') if p.is_file() and p.suffix in ['.npy','.npz','.xml']}
    generated['render_state_audit.json']=digest(folder/'render_state_audit.json')
    for name in ['render_stability.csv','render_stability_summary.json']:generated[name]=digest(folder/name)
    manifest={'status':'COMPLETED_CANDIDATE','demo_id':provenance['demo_id'],'source_protocol_sha256':request_identity(ctx),
      'provenance':provenance,'frames_per_camera':len(times),'control_states':len(states),'event_windows':len(ends),
      'paired_policy_samples':len(indices),'robot_state_per_observation':18,'proprio_horizon_total':36,
      'action_alignment':'targets[t:t+16]; execute indices 0:8','half_open_windows':True,
      'raw_event_counts':totals,'render_seconds':sum(render_times),'median_render_pair_seconds':float(np.median(render_times)),
      'render_timing_scope':'restore/forward plus three native two-camera reads and stability/state checks; excludes emulation and outer audit-file writes; different scope from single-draw historical timings',
      'files':generated,'production_adopted':False,'training_performed':False}
    manifest['candidate_fingerprint']=canonical({'source_protocol_sha256':manifest['source_protocol_sha256'],
      'provenance':provenance,'files':generated})
    write_json(folder/'manifest.json',manifest)
    return manifest

def request_identity(ctx):return ctx['SOURCE_REQUEST_PROTOCOL_HASH']

def verify_offline(folder,protocol_hash):
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['status']=='COMPLETED_CANDIDATE' and manifest['source_protocol_sha256']==protocol_hash
    assert manifest['candidate_fingerprint']==canonical({'source_protocol_sha256':manifest['source_protocol_sha256'],
      'provenance':manifest['provenance'],'files':manifest['files']}), 'Candidate manifest fingerprint changed.'
    for name,sha in manifest['files'].items():
        p=folder/name;assert p.resolve().is_relative_to(folder.resolve()) and digest(p)==sha, 'Candidate data hash changed.'
    return manifest

def online_source(ctx,states,actions,times,ends,folder,offline,provenance):
    install_render_stability(ctx,folder)
    manifest=verify_offline(offline,request_identity(ctx))
    for key in ['demo_id','raw_sha256','split_sha256','raw_scene_xml_sha256','rewritten_scene_xml_sha256',
                'asset_files','compiled_scene','environment_metadata','state_array','action_array','v2e_settings','event_settings_manifest_sha256','event_time_contract','render_lifecycle_contract','render_stability_contract']:
        assert manifest['provenance'][key]==provenance[key], 'Offline/online source identity differs: '+key
    write_json(folder/'v2e_runtime.json',ctx['ensure_v2e']())
    rgb={};event_refs={}
    for cam in ctx['CAMERAS']:
        with np.load(offline/(cam+'_rgb.npz'),allow_pickle=False) as z:rgb[cam]=z['frames'].copy()
        with np.load(offline/(cam+'_events.npz'),allow_pickle=False) as z:event_refs[cam]=(z['previous'].copy(),z['current'].copy())
    offline_render_audit=json.loads((offline/'render_state_audit.json').read_text())
    assert len(offline_render_audit)==len(times)
    rows=[];state_rows=[];env=None;stream=None;online={c:([],[]) for c in ctx['CAMERAS']};frame=[0];event_failure_saved=[False]
    original=ctx['get_observation'];parity_atol=1e-5
    def compare_images(render_env):
        obs=original(render_env);index=frame[0]
        expected=ctx['repair_state_at_frame'](states,index,times,'notebook02_index')
        np.testing.assert_allclose(render_env.sim.get_state().flatten(),expected,atol=1e-9,rtol=0)
        current_audit=render_state_snapshot(render_env)
        state_rows.append({'frame':index,'render_state_exact':current_audit==offline_render_audit[index],
          'flattened_state_exact':current_audit['flattened_state']==offline_render_audit[index]['flattened_state'],
          'derived_state_exact':current_audit['derived']==offline_render_audit[index]['derived']})
        for cam in ctx['CAMERAS']:
            row=parity_row(rgb[cam][index],obs[cam],cam,'rgb',index);rows.append(row)
            fidelity=input_fidelity(ctx,rgb[cam][index],obs[cam]);row.update(fidelity)
            if row['max_abs']!=0. and not (folder/('cross_process_first_difference_'+cam+'.json')).exists():
                save_native_pair_evidence(ctx,folder,'cross_process_first_difference_'+cam,rgb[cam][index],obs[cam],
                  {'row':row,'frame':index,'timestamp_s':index/200.,'offline_render_state':offline_render_audit[index],
                   'causal_render_state':current_audit,'contract':RENDER_STABILITY_CONTRACT})
            assert fidelity['rgb_within_bound'], 'Independent offline/causal RGB exceeds bounded acquisition contract.'
        frame[0]+=1
        return obs
    try:
        env=ctx['repair_make_environment'](ctx['SCENE_XML'],'pilot');ctx['verify_environment'](env)
        ctx['get_observation'](env)  # Disposable context warmup, not stream input.
        env.sim.set_state_from_flattened(states[0]);env.sim.forward()
        ctx['get_observation']=compare_images;stream=ctx['EventStream']()
        stream.feed(compare_images(env),0.)
        for step in range(1,len(states)):
            # This uses only the completed pair of control endpoints; never a future state.
            ctx['advance_event_stream'](env,stream,states[step-1],states[step],step)
            if step>=2:
                index=step-2
                for cam in ctx['CAMERAS']:
                    for kind,array,target in [('previous',stream.previous[cam],event_refs[cam][0]),('current',stream.current[cam],event_refs[cam][1])]:
                        row=parity_row(target[index],array,cam,kind,index)
                        # One consistent CSV schema, including both RGB and Event rows.
                        row.update({k:None for k in ['rgb_max_abs_uint8','rgb_changed_color_values','rgb_changed_fraction',
                          'rgb_within_bound','grayscale_max_abs_uint8','grayscale_changed_pixels','grayscale_exact']});rows.append(row)
                        if row['max_abs']>parity_atol and not event_failure_saved[0]:
                            # Small, uploadable evidence if another mismatch remains; no weakened gate.
                            end=float(ends[index]);lo,hi=(end-.10,end-.05) if kind=='previous' else (end-.05,end)
                            changed=np.argwhere(np.abs(target[index].astype(np.float64)-array)>parity_atol)
                            pixels=set((int(y),int(x)) for _,y,x in changed[:32])
                            with np.load(offline/(cam+'_events.npz'),allow_pickle=False) as z:raw=z['events'].copy()
                            def excerpts(events):
                                lower,upper=event_bounds(events,lo,hi)
                                near=(events[:,0]>=np.nextafter(lower,-np.inf,dtype=events.dtype))&(events[:,0]<=np.nextafter(upper,np.inf,dtype=events.dtype))
                                at_pixel=np.array([(int(y),int(x)) in pixels for _,x,y,_ in events],dtype=bool)
                                selected=events[near&at_pixel]
                                return {'matching_rows':len(selected),'first_48':selected[:48].tolist(),
                                  'window_count':int(np.count_nonzero(event_window_mask(events,lo,hi))),'dtype':str(events.dtype)}
                            write_json(folder/'event_parity_failure.json',{'row':row,'window_start':lo,'window_end':hi,
                              'representable_bounds':[float(v) for v in event_bounds(raw,lo,hi)],'contract':EVENT_TIME_CONTRACT,
                              'changed_values_count':len(changed),'first_32_changed_values':[
                                {'channel':int(k),'y':int(y),'x':int(x),'offline':float(target[index,k,y,x]),'causal':float(array[k,y,x])}
                                for k,y,x in changed[:32]],'offline_events':excerpts(raw),'causal_events':excerpts(stream.pending[cam])})
                            event_failure_saved[0]=True
                        # Collect the whole bounded source before declaring failure; never accept a mismatch.
                    online[cam][0].append(stream.previous[cam].copy());online[cam][1].append(stream.current[cam].copy())
            if step%10==0:print('Causal NVIDIA source control step:',step,'/',len(states)-1,flush=True)
        assert frame[0]==len(times) and stream.frame_count==len(times)
        for cam,(p,c) in online.items():
            np.savez_compressed(folder/(cam+'_online_events.npz'),previous=np.stack(p),current=np.stack(c))
        # Check every packed window's action/proprio alignment independently against raw endpoints.
        with np.load(offline/'paired_policy_samples.npz',allow_pickle=False) as z:
            for i,t in enumerate(z['control_steps']):
                t=int(t);np.testing.assert_array_equal(z['action_targets'][i],actions[t:t+16])
                for which,k in [('previous',t-1),('current',t)]:
                    expected=np.r_[states[k,1:10],states[k,17:26]].astype(np.float32)
                    np.testing.assert_allclose(z['proprio_'+which][i],expected,atol=1e-6,rtol=1e-6)
                for cam in ctx['CAMERAS']:
                    # Full tensor comparisons above cover these same temporal rows.
                    assert z[cam+'_events_previous'][i].shape==online[cam][0][t-2].shape
                    assert z[cam+'_events_current'][i].shape==online[cam][1][t-2].shape
        rgb_rows=[r for r in rows if r['kind']=='rgb'];event_rows=[r for r in rows if r['kind']!='rgb']
        gray_failures=sum(not r['grayscale_exact'] for r in rgb_rows)
        local_gray_failures=sum(not r['last_two_grayscale_exact'] for r in ctx['_render_stability_audit'])
        offline_gray_failures=json.loads((offline/'render_stability_summary.json').read_text())['reads_with_grayscale_mismatch']
        event_failures=sum(r['max_abs']>parity_atol for r in event_rows)
        status='PASS' if gray_failures==local_gray_failures==offline_gray_failures==event_failures==0 else 'FAILED'
        report={'status':status,'frames_per_camera':len(times),'windows_per_camera':len(ends),
          'render_state_exact_frames':sum(r['render_state_exact'] for r in state_rows),
          'render_state_checked_frames':len(state_rows),'render_state_scope':'diagnostic hashes; mandatory restored-state tolerance remains1e-9',
          'rgb_max_abs':max(r['max_abs'] for r in rows if r['kind']=='rgb'),
          'event_max_abs':max(r['max_abs'] for r in event_rows),'rgb_exact_required':False,'event_atol':parity_atol,
          'rgb_nonexact_rows':sum(r['max_abs']>0 for r in rgb_rows),'rgb_checked_rows':len(rgb_rows),
          'rgb_max_changed_color_values':max(r['rgb_changed_color_values'] for r in rgb_rows),
          'rgb_bound_max_abs_uint8':1,'rgb_bound_max_changed_color_values_per_camera':2,
          'cross_process_grayscale_failed_rows':gray_failures,'local_grayscale_failed_reads':local_gray_failures,
          'offline_local_grayscale_failed_reads':offline_gray_failures,
          'event_failed_rows':event_failures,'event_checked_rows':len(event_rows),
          'acceptance_contract':RENDER_STABILITY_CONTRACT,
          'rgb_fidelity':'BYTE_EXACT' if all(r['max_abs']==0 for r in rgb_rows) else 'BOUNDED_NOT_BYTE_EXACT',
          'event_input_fidelity':'BYTE_EXACT' if gray_failures==local_gray_failures==offline_gray_failures==0 else 'FAILED',
          'scope':'independent processes, same runtime/driver, one recorded training trajectory; no learned policy or rollout-equivalence claim',
          'paired_window_action_proprio_alignment':'PASS'}
        write_json(folder/'source_fidelity_summary.json',report)
        return report
    finally:
        ctx['get_observation']=original;ctx['write_csv'](folder/'offline_online_parity.csv',rows)
        ctx.pop('get_observation_original',None)
        ctx['write_csv'](folder/'render_state_comparison.csv',state_rows)
        if stream is not None:stream.close()
        if env is not None:env.close()

def main(request):
    folder=Path(request['output']);folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    result={'id':request['id'],'task':request['task'],'status':'RUNNING','policy_loaded':False,'training_performed':False,'scored_episodes':0}
    ctx=None;phase='RUNTIME_BINDING'
    try:
        ctx,states,actions,times,ends,provenance=bind_runtime(request,folder)
        ctx['SOURCE_REQUEST_PROTOCOL_HASH']=request['source_protocol_hash']
        result['graphics']=provenance['graphics'];print('Actual source renderer:',result['graphics'],flush=True)
        if request['task']=='offline':
            phase='OFFLINE_SOURCE';result['source_manifest']=offline_source(ctx,states,actions,times,ends,folder,provenance)
        elif request['task']=='online':
            phase='CAUSAL_SOURCE_PARITY'
            result['parity']=online_source(ctx,states,actions,times,ends,folder,Path(request['offline_directory']),provenance)
            assert result['parity']['status']=='PASS', 'Full source fidelity failed: grayscale or Event mismatch; see full-trajectory evidence.'
            # This check is independent of equivalence to old CPU-rendered Event data.
            phase='RENDER_EMULATION_NONINTERFERENCE'
            result['physics_noninterference']=ctx['validate_physics_parity']()
            phase='GPU_FIXED_ACTION_PHYSICS'
            result['physics']=fixed_physics(ctx,folder)
            assert all(r['last_two_grayscale_exact'] for r in ctx['_render_stability_audit']), 'Physics helper acquisition has a grayscale mismatch.'
        elif request['task']=='physics_cpu':
            phase='CPU_FIXED_ACTION_PHYSICS';result['physics']=fixed_physics(ctx,folder)
        else:raise ValueError('Unsupported source-pilot task.')
        result['status']='PASS'
    except BaseException as exc:
        gate={'Same-state RGB rendering did not stabilize without state changes.':'LOCAL_RENDER_STABILITY',
              'Same-state acquisition violates RGB bound, state or proprioception contract.':'LOCAL_ACQUISITION_BOUND_OR_STATE',
              'Independent offline/causal RGB exceeds bounded acquisition contract.':'CROSS_PROCESS_RGB_BOUND',
              'Full source fidelity failed: grayscale or Event mismatch; see full-trajectory evidence.':'FULL_TRAJECTORY_GRAY_OR_EVENT_FIDELITY',
              'Independent offline/causal RGB frames differ.':'CROSS_PROCESS_RGB_EXACT',
              'Offline/causal Event tensors differ.':'CROSS_PROCESS_EVENT_TENSOR'}.get(str(exc),phase)
        result.update(status='ABORTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',failed_gate=gate,
                      error=str(exc),traceback=traceback.format_exc())
    finally:
        if ctx is not None and ctx.get('_render_stability_installed'):
            result['render_stability']=save_render_stability_audit(ctx,folder)
        result['elapsed_seconds']=time.perf_counter()-started;write_json(folder/'worker_result.json',result)
        print('Source worker ended:',result['id'],result['status'],flush=True)
    return result

if __name__=='__main__':main(json.loads(Path(sys.argv[1]).read_text()))

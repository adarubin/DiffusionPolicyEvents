"""Isolated renderer experiment. The parent supplies verified historical helper source."""
import ast,copy,hashlib,importlib.metadata,inspect,json,os,re,sys,time,traceback
from pathlib import Path

def split_modules(source):
    starts=list(re.finditer(r'^# MODULE ([A-Za-z0-9_]+\.py)\n',source,re.MULTILINE))
    return {m.group(1):source[m.end():starts[i+1].start() if i+1<len(starts) else len(source)] for i,m in enumerate(starts)}

def setup_without_mount(source):
    tree=ast.parse(source);body=[]
    for n in tree.body:
        if isinstance(n,ast.Assert) and 'SOURCE_RUN.is_dir()' in ast.unparse(n.test):
            n.test=ast.parse('RAW_PATH.is_file()',mode='eval').body
        if isinstance(n,ast.ImportFrom) and n.module=='google.colab':continue
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='drive' and n.value.func.attr=='mount':continue
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODE' for t in n.targets):
            n.value=ast.Constant('renderer_repair')
        body.append(n)
    return compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'verified-setup-no-remount','exec')

def execute_definitions(source,ctx):
    body=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    exec(compile(ast.Module(body=body,type_ignores=[]),'verified-helper-definitions','exec'),ctx)

def repair_source_with_timing_evidence(source):
    marker="        write_csv(folder/'pixels.csv',pixels);write_csv(folder/'event_channels.csv',event_rows)"
    assert source.count(marker)==1, 'Historical timing evidence insertion point changed.'
    extra="\n        write_csv(folder/'render_frame_timings.csv',[{'frame':int(i),'render_pair_seconds':float(v)} for i,v in enumerate(frame_times)])"
    # Save the already measured times after the loop. No rendering, thresholds or timing boundaries change.
    return source.replace(marker,marker+extra)

def hardware_verified(graphics):
    vendor=str(graphics.get('vendor','')).lower();renderer=str(graphics.get('renderer','')).lower()
    return 'nvidia' in vendor and bool(renderer.strip()) and renderer!='none' and not any(s in renderer for s in ['llvmpipe','softpipe','software','swrast'])

def write_json(path,value):
    def default(v):
        if hasattr(v,'tolist'):return v.tolist()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v).__name__)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=default));os.replace(tmp,path)

def inventory():
    from mujoco.egl import egl_ext as EGL
    devices=list(EGL.eglQueryDevicesEXT());rows=[]
    for i,d in enumerate(devices):
        row={'index':i}
        query=getattr(EGL,'eglQueryDeviceStringEXT',None)
        if query:
            try:
                value=query(d,0x3055)  # EGL_EXTENSIONS; diagnostic only, not GPU certification.
                row['extensions']=value.decode(errors='replace') if isinstance(value,bytes) else str(value)
            except Exception as exc:row['query_error']=str(exc)
        rows.append(row)
    return {'status':'COMPLETED','devices':rows,'scope':'EGL enumeration; actual NVIDIA GL context still required'}

def load_context(modules,folder,device_id,full):
    ctx={'__name__':'renderer_audit_worker','EVAL_MODALITY':'events'}
    exec(setup_without_mount(modules['01_setup.py']),ctx)
    ctx['OUTPUT']=folder;ctx['SESSION']=folder
    if full:
        # Keep all historical config/data/normalization gates, without deserializing policy weights.
        text=modules['02_artifacts.py']
        prefix=text.split('checkpoint_path=SOURCE_RUN/CHECKPOINT_RELATIVE')[0]
        suffix='SPLIT=json.loads'+text.split('SPLIT=json.loads',1)[1]
        suffix=suffix.split("PROTOCOL={'mode':",1)[0]
        exec(compile(prefix,'verified-artifact-identities','exec'),ctx)
        selected=ctx['SOURCE_RUN']/ctx['CHECKPOINT_RELATIVE']
        assert ctx['file_hash'](selected)==ctx['EXPECTED_CHECKPOINT_SHA256'], 'Historical checkpoint SHA changed.'
        exec(compile(suffix,'verified-data-scene-contract','exec'),ctx)
    else:
        import h5py,xml.etree.ElementTree as ET
        ctx['ET']=ET
        ctx['asset_root']=Path(ctx['robosuite'].__file__).resolve().parent/'models/assets'
        with h5py.File(ctx['RAW_PATH'],'r') as f:
            ctx['ENV_META']=json.loads(f['data'].attrs['env_args'])
            raw=f['data/demo_154'].attrs['model_file']
        ctx['ORIGINAL_XML']=raw.decode() if isinstance(raw,bytes) else raw
    ctx['CAMERAS']=['agentview','robot0_eye_in_hand']
    ctx['ROBOT_JOINTS']=[f'robot0_joint{i}' for i in range(1,8)]+['gripper0_right_finger_joint1','gripper0_right_finger_joint2']
    execute_definitions(modules['04_policy_and_environment.py'],ctx)
    if full:
        exec(compile(modules['04a_event_stream.py'],'verified-V2E-and-source-audit','exec'),ctx)
        execute_definitions(modules['04b_event_gates.py'],ctx)
    execute_definitions(repair_source_with_timing_evidence(modules['04c_renderer_repair.py']),ctx)
    ctx['SCENE_XML']=ctx['repair_xml'](ctx['ORIGINAL_XML'])
    original_make=ctx['robosuite'].make
    if device_id is not None:
        def selected_make(*args,**kwargs):
            # The only constructor change is the offscreen EGL device selection.
            kwargs['render_gpu_device_id']=int(device_id)
            return original_make(*args,**kwargs)
        ctx['robosuite'].make=selected_make
    for mod in ['robosuite.utils.binding_utils','robosuite.renderers.context.egl_context']:
        m=sys.modules.get(mod)
        if m and getattr(m,'__file__',None):
            p=Path(m.__file__);(folder/(mod.replace('.','_')+'.py')).write_bytes(p.read_bytes())
    return ctx

def graphics_probe(ctx):
    env=None
    try:
        env=ctx['repair_make_environment'](ctx['SCENE_XML'],'pilot')
        ctx['get_observation'](env)
        context=getattr(env.sim,'_render_context_offscreen',None)
        gl=getattr(context,'gl_ctx',None)
        if gl is not None and callable(getattr(gl,'make_current',None)):gl.make_current()
        identity=ctx['graphics_identity']()
        identity.update(context_class=type(gl).__name__ if gl is not None else None,
            context_module=type(gl).__module__ if gl is not None else None,
            render_gpu_device_id=getattr(env,'render_gpu_device_id',None),
            hardware_nvidia_verified=hardware_verified(identity))
        binding=sys.modules.get('robosuite.utils.binding_utils')
        identity['binding_mujoco_gl']=getattr(binding,'_MUJOCO_GL',None)
        identity['gpu_rendering_macro']=getattr(ctx['robosuite'].macros,'MUJOCO_GPU_RENDERING',None) if hasattr(ctx['robosuite'],'macros') else None
        return identity
    finally:
        if env is not None:env.close()

def full_case(ctx,request,graphics):
    if request['require_nvidia']:assert hardware_verified(graphics), 'Candidate did not create an NVIDIA hardware context; no CPU fallback accepted.'
    if request.get('require_llvmpipe'):
        assert graphics.get('vendor')=='Mesa' and 'llvmpipe' in str(graphics.get('renderer','')).lower(), 'Historical CPU baseline did not use Mesa llvmpipe.'
    records=[]
    for demo in request['demos']:
        assert demo in ctx['train_ids'], 'Only training trajectories may be used.'
        case=ctx['repair_load_training_case'](demo)
        assert case['max_frame']==70 and len(case['ends'])==6, 'Fixed benchmark prefix geometry changed.'
        ctx['repair_check_source'](case)
        candidate={'id':request['id'],'scene':'template','construction':'pilot','interpolation':'notebook02_index','backend':'wrapper'}
        record=ctx['repair_candidate'](case,candidate,ctx['SCENE_XML']);records.append(record)
    physics=None
    if all(r['status']=='PASS' for r in records):
        physics=ctx['validate_physics_parity']()
        env=None
        try:
            env=ctx['make_env'](1994);ctx['verify_environment'](env)
            states=[env.sim.get_state().flatten().copy()]
            import numpy as np
            for action in [[.1,0,0,0,0,0,-.2],[0,.1,0,0,0,0,.2],[0,0,.1,0,0,0,0]]:
                env.step(np.asarray(action,dtype=np.float32));states.append(env.sim.get_state().flatten().copy())
            np.save(ctx['SESSION']/'fixed_physics_trace.npy',np.asarray(states))
        finally:
            if env is not None:env.close()
    return {'status':'PASS' if all(r['status']=='PASS' for r in records) and physics and physics['status']=='PASS' else 'INCOMPATIBLE',
        'graphics':graphics,'records':records,'physics':physics,'policy_loaded':False,'scored_episodes':0,
        'scope':'three training prefixes and disposable physics probes; not rollout equivalence or real-time performance'}

def main(request):
    folder=Path(request['output']);folder.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();result={'id':request['id'],'task':request['task'],'status':'RUNNING'}
    write_json(folder/'request.json',request)
    try:
        source=Path(request['frozen_source']).read_text()
        assert hashlib.sha256(source.encode()).hexdigest()==request['frozen_sha256']
        if request.get('selected_library'):
            selected=Path(request['selected_library'])
            assert hashlib.sha256(selected.read_bytes()).hexdigest()==request['selected_library_sha256'], 'Selected library changed after its load probe.'
        if request['task']=='inventory':result.update(inventory())
        else:
            ctx=load_context(split_modules(source),folder,request.get('device_id'),request['task']=='full')
            graphics=graphics_probe(ctx);write_json(folder/'graphics_identity.json',graphics)
            print('Actual renderer:',graphics,flush=True)
            if request['task']=='probe':result.update(status='COMPLETED',graphics=graphics)
            else:result.update(full_case(ctx,request,graphics))
    except BaseException as exc:
        result.update(status='ABORTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=str(exc),traceback=traceback.format_exc())
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        write_json(folder/'worker_result.json',result)
        print('Worker ended:',result['id'],result['status'],flush=True)
    return result

if __name__=='__main__':main(json.loads(Path(sys.argv[1]).read_text()))

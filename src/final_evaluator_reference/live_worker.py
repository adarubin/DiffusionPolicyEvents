"""One isolated immutable-model preflight or episode; no optimizer is constructed."""
import ast, copy, gc, hashlib, importlib.util, inspect, json, logging, math, os, random
import shutil, signal, subprocess, sys, tempfile, time, traceback, uuid
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from contracts import *
from acquisition_guard import install_guard, save_guard
from stress_conditions import install_stress,condition_by_id,apply_impulse,save_stress,policy_panels,presentation_replay,stress_preflight,save_reset_contract

def definitions(source,ctx):
    body=[node for node in ast.parse(source).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))]
    exec(compile(ast.Module(body=body,type_ignores=[]),'verified-definition-adapter','exec'),ctx)

def initialize(request,folder):
    import torch, torchvision, h5py, mujoco, robosuite, cv2, imageio.v2 as imageio
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import resnet18
    from PIL import Image, ImageDraw
    import xml.etree.ElementTree as ET
    assert torch.cuda.is_available() and mujoco.__version__=='3.3.0' and robosuite.__version__=='1.5.1'
    assert sha_file(request['selected_library'])==request['selected_library_sha256']
    assert sha_file(Path(request['payload_root'])/'verified_source_worker.py')==SOURCE_SHA
    assert sha_file(Path(request['payload_root'])/'verified_verified_paired_v4.py')==FROZEN_SHA
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(request['project_root']);source=Path(request['source_folder']);export=Path(request['export_folder'])
    provenance=json.loads((source/'manifest.json').read_text())['provenance']
    original=(source/'recorded_scene.xml').read_text()
    assert hashlib.sha256(original.encode()).hexdigest()==provenance['raw_scene_xml_sha256']
    assert provenance['demo_id']==request['options']['scene_demo'] and provenance['raw_sha256']==RAW_SHA
    runtime={'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,
        'mujoco':mujoco.__version__,'robosuite':robosuite.__version__,'opencv':cv2.__version__,
        'gpu':torch.cuda.get_device_name(0),'nvidia_driver':subprocess.check_output(
            ['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()}
    modules=__import__('re').split(r'^# MODULE ',(Path(request['payload_root'])/'verified_verified_paired_v4.py').read_text(),flags=__import__('re').MULTILINE)[1:]
    modules=dict(piece.split('\n',1) for piece in modules)
    ctx=dict(globals());ctx.update(torch=torch,nn=nn,F=F,resnet18=resnet18,robosuite=robosuite,mujoco=mujoco,
        h5py=h5py,cv2=cv2,Image=Image,ImageDraw=ImageDraw,imageio=imageio,ET=ET,
        asset_root=Path(robosuite.__file__).resolve().parent/'models/assets',DEVICE=torch.device('cuda'),
        RUNTIME=runtime,PILOT=request['options'],ENV_META=provenance['environment_metadata'],
        ORIGINAL_XML=original,RAW_LO=np.asarray(request['raw_lo'],np.float32),RAW_HI=np.asarray(request['raw_hi'],np.float32),
        EVENT_OPTIONS={'tau_s':.05},CAMERAS=CAMERAS,
        ROBOT_JOINTS=[f'robot0_joint{i}' for i in range(1,8)]+['gripper0_right_finger_joint1','gripper0_right_finger_joint2'])
    definitions(modules['04_policy_and_environment.py'],ctx)
    definitions(modules['04c_renderer_repair.py'],ctx)
    ctx['SCENE_XML']=ctx['repair_xml'](original)
    original_make=robosuite.make
    def selected_make(*args,**kwargs):
        kwargs['render_gpu_device_id']=int(request['device_id'])
        return original_make(*args,**kwargs)
    robosuite.make=selected_make
    # Only source definitions are imported. Historical top-level artifact/model/setup code never executes.
    helper_path=Path(request['payload_root'])/'verified_source_worker.py'
    spec=importlib.util.spec_from_file_location('verified_source_contract',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    definitions(modules['04a_event_stream.py'],ctx)
    helper.install_event_time_contract(ctx)
    ctx['interpolated_state']=helper.preserve_endpoints(ctx['interpolated_state'])
    settings=provenance['v2e_settings']
    commit='cac638444557a224ba6ee5c29260916f152c00bd'
    assert settings['device']=='cpu' and settings['resolved_commit']==commit and settings['emulator_seed']==42
    assert settings['pos_thres']==settings['neg_thres']==.2
    assert all(settings[k]==0 for k in ['sigma_thres','cutoff_hz','leak_rate_hz','shot_noise_rate_hz','refractory_period_s'])
    repo=Path('/content/v2e-notebook07-'+commit[:12])
    # The mandatory full source gate already cloned/verified this exact repository.
    assert repo.is_dir() and subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==commit
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
    sys.path.insert(0,str(repo))
    from v2ecore.emulator import EventEmulator
    assert Path(inspect.getfile(EventEmulator)).resolve().is_relative_to(repo.resolve())
    ctx.update(EventEmulator=EventEmulator,EVENT_DEVICE='cpu',cv2=cv2)
    # Expected empty frames remain counted in event_stats; suppress only known repetitive display/no-signal messages.
    class QuietExpectedMessages(logging.Filter):
        def filter(self,record):
            message=record.getMessage()
            return not (message.startswith('no signal events generated for frame') or message.startswith('cannot get screen size for window placement'))
    logging.getLogger('v2ecore.emulator').addFilter(QuietExpectedMessages())
    ctx['helper']=helper
    install_guard(ctx,folder)
    record=request['record'];cfg=validate_run_record(record,request['modality']);norm=request['normalization']
    assert canonical(norm)==NORM_HASH
    selected=SOURCES[request['modality']][request['stage']]
    assert sha_file(request['checkpoint'])==selected['sha256']
    saved=torch.load(request['checkpoint'],map_location='cpu',weights_only=False)
    validate_checkpoint(saved,record,request['modality'],request['stage'])
    weights=saved['ema'];del saved;gc.collect()
    assert all(not v.is_floating_point() or (v.dtype==torch.float32 and bool(torch.isfinite(v).all())) for v in weights.values())
    model_ctx=dict(ctx);model_ctx['CFG']=cfg
    model_path=Path(request['payload_root'])/'model_inference.py'
    exec(compile(model_path.read_text(),str(model_path),'exec'),model_ctx)
    model=model_ctx['VisualConditionedDiffusionPolicy']()
    model.load_state_dict(weights,strict=True);del weights;gc.collect()
    assert sum(p.numel() for p in model.parameters())==33894535
    model=model.to(ctx['DEVICE']).eval().requires_grad_(False)
    assert not any(m.training for m in model.modules())
    ctx.update(MODEL=model,CFG=cfg,NORMALIZATION=norm,helper=helper,request=request,
        ACT_MEAN=np.asarray(norm['action_mean'],np.float32),ACT_STD=np.asarray(norm['action_std'],np.float32),
        PROP_MEAN=np.asarray(norm['proprio_mean'],np.float32),PROP_STD=np.asarray(norm['proprio_std'],np.float32))
    assert ctx['PROP_STD'].shape==(36,) and (ctx['PROP_STD']>0).all() and (ctx['ACT_STD']>0).all()
    def cosine(t): return math.cos((t+.008)/1.008*math.pi/2)**2
    betas=torch.tensor([min(1-cosine((i+1)/100)/cosine(i/100),.999) for i in range(100)],device=ctx['DEVICE'],dtype=torch.float32)
    alphas=1-betas;abar=alphas.cumprod(0);prev=torch.cat([torch.ones(1,device=ctx['DEVICE']),abar[:-1]])
    ctx.update(betas=betas,alphas=alphas,abar=abar,prev=prev,
        variance=(betas*(1-prev)/(1-abar)).clamp_min(1e-20),
        Z_LO=torch.tensor((ctx['RAW_LO']-ctx['ACT_MEAN'])/ctx['ACT_STD'],device=ctx['DEVICE']),
        Z_HI=torch.tensor((ctx['RAW_HI']-ctx['ACT_MEAN'])/ctx['ACT_STD'],device=ctx['DEVICE']))
    assert float(abar[-1])<1e-6
    atomic_json(folder/'runtime.json',runtime)
    return ctx

def fingerprint(ctx):
    h=hashlib.sha256()
    for name,tensor in ctx['MODEL'].state_dict().items():
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def preprocess(ctx,previous,current):
    torch=ctx['torch'];norm=ctx['NORMALIZATION'];modality=ctx['request']['modality'];batch={}
    for cam in CAMERAS:
        a=np.stack([previous[cam],current[cam]])
        if modality=='rgb':
            assert a.shape==(2,84,84,3) and a.dtype==np.uint8
            a=np.moveaxis(a.astype(np.float32)/255.,-1,1)
            mean=np.asarray(norm['rgb_image_mean'],np.float32);std=np.asarray(norm['rgb_image_std'],np.float32)
        else:
            assert a.shape==(2,3,84,84) and a.dtype==np.float32 and (a>=0).all()
            mean=np.asarray(norm['event_pixel_zscore'][cam]['mean'],np.float32);std=np.asarray(norm['event_pixel_zscore'][cam]['std'],np.float32)
        a=(a[:,:,4:80,4:80]-mean[None,:,None,None])/std[None,:,None,None]
        assert np.isfinite(a).all();batch[cam]=torch.from_numpy(np.ascontiguousarray(a[None])).to(ctx['DEVICE'])
    prop=np.r_[previous['proprio'],current['proprio']].astype(np.float32)
    assert prop.shape==(36,)
    prop=(prop-ctx['PROP_MEAN'])/ctx['PROP_STD'];assert np.isfinite(prop).all()
    batch['proprio']=torch.from_numpy(prop[None]).to(ctx['DEVICE'])
    return batch

def predict(ctx,previous,current,generator):
    torch=ctx['torch'];model=ctx['MODEL']
    with torch.inference_mode():
        batch=preprocess(ctx,previous,current)
        condition=model.encode_observation(batch).float()
        x=torch.randn((1,16,7),device=ctx['DEVICE'],dtype=torch.float32,generator=generator)
        traces=[]
        for t in range(99,-1,-1):
            eps=model.unet(x,condition,torch.full((1,),t,device=ctx['DEVICE'],dtype=torch.long)).float()
            x0=(x-(1-ctx['abar'][t]).sqrt()*eps)/ctx['abar'][t].sqrt()
            clipped=torch.minimum(torch.maximum(x0,ctx['Z_LO']),ctx['Z_HI'])
            mean=ctx['betas'][t]*ctx['prev'][t].sqrt()/(1-ctx['abar'][t])*clipped
            mean=mean+(1-ctx['prev'][t])*ctx['alphas'][t].sqrt()/(1-ctx['abar'][t])*x
            x=mean+ctx['variance'][t].sqrt()*torch.randn(x.shape,device=x.device,generator=generator) if t>0 else clipped
            traces.append(((x0<ctx['Z_LO'])|(x0>ctx['Z_HI'])).float().mean())
            assert bool(torch.isfinite(x).all()), 'Non-finite sampled trajectory.'
        result=(x[0]*torch.tensor(ctx['ACT_STD'],device=x.device)+torch.tensor(ctx['ACT_MEAN'],device=x.device)).cpu().numpy()
        projection=torch.stack(traces).cpu().numpy()
    assert result.shape==(16,7) and np.isfinite(result).all()
    assert np.all(result>=ctx['RAW_LO']-1e-5) and np.all(result<=ctx['RAW_HI']+1e-5)
    return result,projection

def warm_renderer(ctx,env):
    # No warmup timestamp is fed. This is the verified acquisition lifecycle.
    ctx['get_observation'](env)
    context=getattr(env.sim,'_render_context_offscreen',None);gl=getattr(context,'gl_ctx',None)
    if gl is not None and callable(getattr(gl,'make_current',None)): gl.make_current()
    graphics=ctx['graphics_identity']()
    assert 'NVIDIA' in graphics.get('vendor','') and not any(s in graphics.get('renderer','').lower() for s in ['llvmpipe','software','softpipe'])
    expected=ctx['request']['expected_graphics']
    assert graphics['vendor']==expected['vendor'] and graphics['renderer']==expected['renderer'] and graphics['nvidia_driver']==expected['nvidia_driver']
    return graphics

def raw_proprio(env): return np.r_[env.sim.data.qpos[:9],env.sim.data.qvel[:9]].astype(np.float32)

def policy_observation(ctx,stream,prop):
    source=ctx['_last_images'] if ctx['request']['modality']=='rgb' else stream.current
    return {**{cam:source[cam].copy() for cam in CAMERAS},'proprio':np.asarray(prop,np.float32).copy()}

def frame(ctx,seed,step,success,stream):
    return policy_panels(ctx,seed,step,success,stream)

def preflight(ctx,folder):
    env=None;stream=None;torch=ctx['torch'];before=fingerprint(ctx)
    try:
        env=ctx['make_env'](1998);mapping=ctx['verify_environment'](env);initial=env.sim.get_state().flatten().copy()
        save_reset_contract(ctx,env,folder)
        assert not env._check_success();env.close();env=ctx['make_env'](1998)
        np.testing.assert_allclose(initial,env.sim.get_state().flatten(),atol=1e-9,rtol=0)
        env.close();env=ctx['make_env'](1999)
        assert not np.allclose(initial,env.sim.get_state().flatten());env.close()
        env=ctx['repair_make_environment'](ctx['SCENE_XML'],'pilot');mapping=ctx['verify_environment'](env)
        graphics=warm_renderer(ctx,env)
        source=Path(ctx['request']['source_folder']);export=Path(ctx['request']['export_folder'])
        with np.load(source/'control_state_action.npz',allow_pickle=False) as z: states=z['states'];actions=z['actions']
        stream=ctx['EventStream']()
        env.sim.set_state_from_flattened(states[0]);env.sim.forward()
        stream.feed(ctx['get_observation'](env),0.)
        observations=[]
        for step in [1,2]:
            ctx['advance_event_stream'](env,stream,states[step-1],states[step],step)
            observations.append(policy_observation(ctx,stream,raw_proprio(env)))
        # At t=.1 the first packed sample has previous t=.05 and current t=.1.
        parity=[]
        for modality in ['rgb','events']:
            with np.load(export/(modality+'_policy_samples.npz'),allow_pickle=False) as z:
                assert int(z['control_steps'][0])==2
                np.testing.assert_array_equal(z['action_targets'][0],actions[2:18])
                for suffix,step,obs in [('previous',1,observations[0]),('current',2,observations[1])]:
                    np.testing.assert_allclose(z['proprio_'+suffix][0],np.r_[states[step,1:10],states[step,17:26]],atol=1e-6,rtol=1e-6)
                    for cam in CAMERAS:
                        if modality=='events':
                            cur=stream.previous[cam] if suffix=='previous' else stream.current[cam]
                            error=float(np.abs(cur-z[cam+'_'+suffix][0]).max());assert error<=1e-5
                        elif ctx['request']['modality']=='rgb':
                            fidelity=ctx['source_input_fidelity'](ctx,z[cam+'_'+suffix][0],obs[cam])
                            assert fidelity['rgb_within_bound'] and fidelity['grayscale_exact'];error=fidelity['rgb_max_abs_uint8']
                        else: continue
                        parity.append({'modality':modality,'camera':cam,'observation':suffix,'max_abs':error})
        # Event observations at consecutive times must agree with canonical previous/current tensors.
        if ctx['request']['modality']=='events':
            for cam in CAMERAS: np.testing.assert_array_equal(observations[0][cam],stream.previous[cam])
        with np.load(export/(ctx['request']['modality']+'_policy_samples.npz'),allow_pickle=False) as z:
            packed=[{**{cam:z[cam+'_'+suffix][0] for cam in CAMERAS},'proprio':z['proprio_'+suffix][0]} for suffix in ['previous','current']]
            live_batch=preprocess(ctx,*observations);packed_batch=preprocess(ctx,*packed)
            for key in ['proprio']: np.testing.assert_allclose(live_batch[key].cpu(),packed_batch[key].cpu(),atol=1e-5,rtol=1e-5)
            if ctx['request']['modality']=='events':
                for cam in CAMERAS: np.testing.assert_allclose(live_batch[cam].cpu(),packed_batch[cam].cpu(),atol=1e-4,rtol=1e-5)
        generator=torch.Generator(device=ctx['DEVICE']).manual_seed(90415)
        plan,projection=predict(ctx,*observations,generator)
        np.savez_compressed(folder/'preflight_plan.npz',actions=plan,projection=projection)
        writer=ctx['imageio'].get_writer(str(folder/'video_probe.mp4'),fps=10,codec='libx264',macro_block_size=1)
        try:
            for _ in range(2): writer.append_data(frame(ctx,1999,2,False,stream))
        finally: writer.close()
        assert (folder/'video_probe.mp4').stat().st_size>0
        reader=ctx['imageio'].get_reader(str(folder/'video_probe.mp4'))
        try: assert reader.get_data(0).shape==(512,512,3), 'Encoded video cannot be decoded.'
        finally: reader.close()
        stream.close();stream=None;env.close();env=None
        # Live-environment stepping plus renderer/emulator acquisition must not modify live state.
        env=ctx['make_env'](1997);ctx['verify_environment'](env)
        before_time=float(env.sim.data.time);env.step(np.zeros(7,np.float32))
        np.testing.assert_allclose(float(env.sim.data.time)-before_time,.05,atol=1e-7,rtol=0)
        assert np.isfinite(env.sim.get_state().flatten()).all()
        assert fingerprint(ctx)==before
        return {'status':'PASS','controller_mapping':mapping,'graphics':graphics,'input_parity':parity,
            'reset_reproducibility':'PASS','different_seeds':'PASS','normalization':'PASS','temporal_alignment':'plan[0:8]',
            'FP32_EMA_inference':'PASS','video_encoding':'PASS','weights_unchanged':True,'scored_episodes':0}
    finally:
        if stream is not None: stream.close()
        if env is not None: env.close()

def run_episode(ctx,folder):
    request=ctx['request'];job=request['job'];options=request['options'];torch=ctx['torch']
    root=Path(request['output_root']);attempt=root/'attempts'/job['key']/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True,exist_ok=False);local=Path(tempfile.mkdtemp(prefix='nb07-episode-',dir='/content'))
    row={**job,'group':job['modality']+'_'+job['stage'],'protocol_hash':request['protocol_hash'],
        'source_run_id':request['record']['run_id'],'checkpoint_step':SOURCES[job['modality']][job['stage']]['step'],
        'checkpoint_sha256':SOURCES[job['modality']][job['stage']]['sha256'],'status':'RUNNING','started_utc':utc()}
    atomic_json(attempt/'status.json',row)
    env=None;renderer=None;stream=None;writer=None;step=0;rows=[];states=[];plans=[];projections=[];times=[];event_stats=[]
    any_success=False;consecutive=0;max_consecutive=0;first_success=None;stop='horizon';wall=time.perf_counter()
    before_weights=fingerprint(ctx)
    condition=condition_by_id(job['condition_id']);stress=install_stress(ctx,condition,folder)
    frame_steps=[];input_archive=[];query_steps=[]
    try:
        env=ctx['make_env'](job['environment_seed']);ctx['verify_environment'](env)
        initial=env.sim.get_state().flatten().copy();assert not env._check_success()
        row['reset_contract']=save_reset_contract(ctx,env,local)
        if request.get('initial_reference'):
            np.testing.assert_allclose(initial,np.load(request['initial_reference'],allow_pickle=False),atol=1e-9,rtol=0)
        np.save(local/'initial_state.npy',initial);states.append(initial)
        renderer=ctx['repair_make_environment'](ctx['SCENE_XML'],'pilot');ctx['verify_environment'](renderer)
        warm_renderer(ctx,renderer)
        stream=ctx['EventStream']()
        renderer.sim.set_state_from_flattened(initial);renderer.sim.forward()
        stream.feed(ctx['get_observation'](renderer),0.)
        np.testing.assert_array_equal(env.sim.get_state().flatten(),initial)
        current=policy_observation(ctx,stream,raw_proprio(env));previous={k:v.copy() for k,v in current.items()}
        generator=torch.Generator(device=ctx['DEVICE']).manual_seed(job['policy_seed'])
        writer=ctx['imageio'].get_writer(str(local/'policy_panels.mp4'),fps=10,codec='libx264',quality=7,macro_block_size=1)
        writer.append_data(frame(ctx,job['environment_seed'],0,False,stream));frame_steps.append(0)
        ctx['Image'].fromarray(frame(ctx,job['environment_seed'],0,False,stream)).save(local/'initial.png')
        peak=-np.inf;low,high=map(np.asarray,env.action_spec)
        while step<options['max_control_steps']:
            torch.cuda.synchronize();start=time.perf_counter()
            query_steps.append(step)
            input_archive.append({cam:np.stack([previous[cam],current[cam]]).copy() for cam in CAMERAS})
            plan,projection=predict(ctx,previous,current,generator)
            torch.cuda.synchronize();inference=time.perf_counter()-start
            plans.append(plan);projections.append(projection);times.append(inference)
            # This project's packed targets begin at t. Never apply the official code's different +1 offset.
            for action_index,requested in enumerate(plan[:options['execution_horizon']]):
                if step>=options['max_control_steps']: break
                assert np.isfinite(requested).all() and (requested>=low-1e-5).all() and (requested<=high+1e-5).all()
                action=np.clip(requested,low,high);previous=current
                apply_impulse(env,condition,step+1,stress)
                a=env.sim.get_state().flatten().copy();_,reward,done,_=env.step(action);step+=1
                b=env.sim.get_state().flatten().copy();assert np.isfinite(b).all();states.append(b)
                np.testing.assert_allclose(b[0]-a[0],.05,atol=1e-7,rtol=0)
                start=time.perf_counter();counts=ctx['advance_event_stream'](renderer,stream,a,b,step)
                source_seconds=time.perf_counter()-start
                np.testing.assert_array_equal(env.sim.get_state().flatten(),b)
                current=policy_observation(ctx,stream,raw_proprio(env))
                if job['modality']=='events':
                    previous={**{cam:stream.previous[cam].copy() for cam in CAMERAS},'proprio':previous['proprio']}
                success=bool(env._check_success());any_success|=success
                if success and first_success is None: first_success=step
                consecutive=consecutive+1 if success else 0;max_consecutive=max(max_consecutive,consecutive)
                height=float(env.sim.data.body_xpos[env.cube_body_id][2]);table=float(env.model.mujoco_arena.table_offset[2])
                rows.append({'step':step,'simulation_elapsed_s':step/20,'plan':len(plans)-1,'plan_action_index':action_index,
                    'reward':float(reward),'success':success,'consecutive_success':consecutive,'cube_above_table':height-table,
                    **{f'cube_{axis}_world_m':float(env.sim.data.body_xpos[env.cube_body_id][i]) for i,axis in enumerate('xyz')},
                    'inference_seconds':inference,'source_pipeline_seconds':source_seconds,
                    'render_seconds':stream.last_render_seconds,'v2e_seconds':stream.last_v2e_seconds,
                    **{f'action_{i}':float(v) for i,v in enumerate(action)}})
                for cam in CAMERAS:
                    event_stats.append({'step':step,'camera':cam,'events':int(counts[cam]),
                        'active_fraction':float(np.mean(stream.current[cam][:2].sum(0)>0)),
                        'total_generated':int(stream.generated[cam])})
                if height>peak:
                    peak=height;ctx['Image'].fromarray(frame(ctx,job['environment_seed'],step,success,stream)).save(local/'peak.png')
                if step%options['video_stride']==0:
                    writer.append_data(frame(ctx,job['environment_seed'],step,success,stream));frame_steps.append(step)
                if step%20==0:
                    progress={'job':job['key'],'step':step,'max_steps':300,'any_success':any_success,
                        'consecutive_success':consecutive,'wall_minutes':round((time.perf_counter()-wall)/60,2)}
                    atomic_json(folder/'progress.json',progress);atomic_json(attempt/'progress.json',progress)
                    print('Progress:',progress,flush=True)
                if consecutive>=options['success_hold_steps']: stop='sustained_success';break
                if done: stop='environment_done';break
            if stop!='horizon': break
        if step%2:
            writer.append_data(frame(ctx,job['environment_seed'],step,success,stream));frame_steps.append(step)
        row.update(status='COMPLETED',stop_reason=stop)
    except BaseException as exc:
        row.update(status='ABORTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',stop_reason=type(exc).__name__,
            error=str(exc),traceback=traceback.format_exc())
    finally:
        for resource in [writer,stream,renderer,env]:
            if resource is not None:
                try: resource.close()
                except Exception as exc: row.update(status='FAILED',close_error=str(exc))
        write_csv(local/'steps.csv',rows);write_csv(local/'event_stats.csv',event_stats)
        np.savez_compressed(local/'trajectory.npz',states=np.asarray(states),plans=np.asarray(plans),
            x0_projection_fraction=np.asarray(projections),inference_seconds=np.asarray(times))
        # Save the exact local acquisition checks before a completion pointer can be published.
        stability=ctx['helper'].save_render_stability_audit(ctx,local)
        guard=save_guard(ctx,local)
        passed=all(stability[k]==0 for k in ['reads_outside_rgb_bound','reads_with_state_change','reads_with_proprio_change']) and guard['status']=='PASS' and guard['control_endpoints_checked']==step
        unchanged=fingerprint(ctx)==before_weights
        np.savez_compressed(local/'policy_inputs.npz',query_steps=np.asarray(query_steps,dtype=np.int64),
            **{cam:np.asarray([q[cam] for q in input_archive]) for cam in CAMERAS})
        row['stress_audit']=save_stress(ctx,local)
        if row['status']=='COMPLETED':
            try:row['presentation']=presentation_replay(ctx,local,np.asarray(states),frame_steps,job,rows)
            except BaseException as exc:
                row.update(status='ABORTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',
                    stop_reason='presentation_failed',error=str(exc),traceback=traceback.format_exc(),
                    behavior_completed=True,presentation={'status':'FAILED','error':str(exc)})
        if row['status']=='COMPLETED':
            try:
                reader=ctx['imageio'].get_reader(str(local/'rollout.mp4'))
                try: assert reader.get_data(0).shape==(512,1024,3)
                finally: reader.close()
            except Exception as exc: row.update(status='FAILED',stop_reason='video_decode',error=str(exc))
        if not passed or not unchanged: row.update(status='FAILED',stop_reason='acquisition_or_weight_mutation')
        row.update(ended_utc=utc(),steps=step,simulation_seconds=step/20,wall_seconds=time.perf_counter()-wall,
            any_success=bool(any_success),sustained_success=max_consecutive>=10,max_consecutive_success_steps=max_consecutive,
            first_success_step=first_success,weights_unchanged=unchanged,render_gate='PASS' if passed else 'FAILED',
            acquisition_guard=guard,
            generated_events={c:int(stream.generated[c]) for c in CAMERAS} if stream is not None else {},
            source_frames_per_camera=stream.frame_count if stream is not None else 0,
            mean_inference_seconds=float(np.mean(times)) if times else None,
            source_pipeline_seconds=sum(r['source_pipeline_seconds'] for r in rows),
            render_seconds=sum(r['render_seconds'] for r in rows),v2e_seconds=sum(r['v2e_seconds'] for r in rows),
            max_cube_above_table=max((r['cube_above_table'] for r in rows),default=None))
        publish_episode(root,attempt,local,row);shutil.rmtree(local)
        atomic_json(attempt/'status.json',row)
    return row

def replay_incident(ctx,folder):
    """Re-render the complete saved incident history, with no live action/policy execution."""
    incident=ctx['request']['incident'];path=Path(incident['path'])
    assert sha_file(path)==incident['sha256']
    with np.load(path,allow_pickle=False) as z: states=np.asarray(z['states']).copy()
    steps=int(incident['control_steps'])
    assert steps in [270,272] and states.shape==(steps+1,32) and np.isfinite(states).all()
    np.testing.assert_allclose(np.diff(states[:,0]),.05,atol=1e-7,rtol=0)
    env=None;stream=None;before=fingerprint(ctx)
    try:
        env=ctx['repair_make_environment'](ctx['SCENE_XML'],'pilot');ctx['verify_environment'](env)
        warm_renderer(ctx,env);stream=ctx['EventStream']()
        env.sim.set_state_from_flattened(states[0]);env.sim.forward()
        stream.feed(ctx['get_observation'](env),0.)
        for step in range(1,len(states)):
            ctx['advance_event_stream'](env,stream,states[step-1],states[step],step)
            if step%50==0:print('Incident history replay:',step,'/',steps,'; no policy execution',flush=True)
        assert fingerprint(ctx)==before
        guard=save_guard(ctx,folder)
        assert guard['status']=='PASS' and guard['control_endpoints_checked']==steps
        return {'status':'PASS','control_steps':steps,'historical_trajectory_sha256':incident['sha256'],
            'guard':guard,'scored_episodes':0,'weights_unchanged':True,
            'scope':'Replay of saved completed endpoint states; not proof of identical driver failure recurrence or a task success.'}
    finally:
        if stream is not None:stream.close()
        if env is not None:env.close()

def main(request):
    folder=Path(request['output']);folder.mkdir(parents=True,exist_ok=True);ctx=None
    result={'status':'RUNNING','task':request['task'],'training_performed':False,'started_utc':utc()}
    def interrupted(signum,stack): raise KeyboardInterrupt('Parent stopped this isolated worker.')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        select_training_seed(request['training_seed'])
        assert request['record']['config']['seed']==request['training_seed']
        if request['task']=='episode':assert request['job']['training_seed']==request['training_seed']
        ctx=initialize(request,folder)
        if request['task']=='preflight': result.update(preflight(ctx,folder))
        elif request['task']=='stress_preflight': result.update(stress_preflight(ctx,folder))
        elif request['task']=='episode': result.update(run_episode(ctx,folder))
        elif request['task']=='incident_replay': result.update(replay_incident(ctx,folder))
        else: raise ValueError('Only preflight/episode/incident_replay tasks are supported.')
    except BaseException as exc:
        result.update(status='ABORTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=str(exc),traceback=traceback.format_exc())
    finally:
        if ctx is not None:
            result['render_stability']=ctx['helper'].save_render_stability_audit(ctx,folder)
            result['acquisition_guard']=save_guard(ctx,folder)
            if any(result['render_stability'][k] for k in ['reads_outside_rgb_bound','reads_with_state_change','reads_with_proprio_change']) or result['acquisition_guard']['status']!='PASS': result['status']='FAILED'
        result['ended_utc']=utc();atomic_json(folder/'worker_result.json',result)
    print('Worker ended:',request['id'],result['status'],flush=True)
    return result

if __name__=='__main__': main(json.loads(Path(sys.argv[1]).read_text()))

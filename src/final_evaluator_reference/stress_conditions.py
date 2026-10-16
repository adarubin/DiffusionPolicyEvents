"""Declared development interventions and post-rollout observer videos.

Native acquisition checks precede source-gain transforms. Primary and shadow V2E
receive the same declared transform and retain independent persistent histories.
"""
import json, math, time
from pathlib import Path
import numpy as np
from contracts import CAMERAS, atomic_json, canonical, write_csv

CONDITIONS = [{'id':'normal','kind':'none'},{'id':'dark_gain025','kind':'source_gain','gain':0.25}]
PRESENTATION = {'layout':'left external frontview; right agentview above wrist',
    'frame_shape':[512,1024,3],'external_camera':'frontview','robot_display_crop':[4,80],
    'native_fps':10,'slow_playback_multiplier':0.25,'event_scales':{'positive':3.0,'negative':3.0,'time_surface':1.0},
    'observer_method':'post-rollout saved-state rendering; no policy, V2E or env.step',
    'right_views':'cached current observations; center76 crop before zscore, enlarged for display',
    'plan_input_archive':'exact previous/current84 observations for every policy query; separate from lossy MP4'}

def condition_by_id(name):
    matches=[c for c in CONDITIONS if c['id']==name]
    assert len(matches)==1, 'Unknown stress condition.'
    return dict(matches[0])

def gain_at(condition, timestamp_s):
    if condition['kind']=='source_gain': return condition['gain']
    if condition['kind']=='source_gain_step':
        return condition['gain_after'] if timestamp_s+1e-9>=condition['change_time_s'] else condition['gain_before']
    return 1.0

def transform_rgb(value, condition, timestamp_s):
    assert value.dtype==np.uint8 and value.ndim==3 and value.shape[-1]==3
    gain=gain_at(condition,timestamp_s)
    if gain==1.0: return value.copy()
    return np.clip(np.rint(value.astype(np.float64)*gain),0,255).astype(np.uint8)

def visual_material(env,condition):
    if condition['kind']!='cube_color': return None
    model=env.sim.model;body=int(env.cube_body_id)
    ids=np.flatnonzero(np.asarray(model.geom_bodyid)==body)
    assert len(ids)>0, 'Cube geometry not found.'
    before=env.sim.get_state().flatten().copy()
    physical={k:np.asarray(getattr(model,k)).copy() for k in ['body_mass','body_inertia','geom_friction','geom_size','geom_pos','geom_quat']}
    originals={'rgba':np.asarray(model.geom_rgba)[ids].copy(),'matid':np.asarray(model.geom_matid)[ids].copy()}
    model.geom_matid[ids]=-1
    model.geom_rgba[ids]=np.asarray(condition['rgba'],dtype=float)
    env.sim.forward()
    np.testing.assert_array_equal(before,env.sim.get_state().flatten())
    for k,v in physical.items():np.testing.assert_array_equal(v,getattr(model,k))
    return {'geom_ids':ids.tolist(),'original_rgba':originals['rgba'].tolist(),'original_matid':originals['matid'].tolist(),
        'new_rgba':condition['rgba'],'physics_unchanged':True,'material_override':'solid color; cube texture disabled'}

def install_stress(ctx,condition,folder):
    assert condition==condition_by_id(condition['id'])
    raw=ctx['get_observation'];state={'condition':condition,'reads':[],'material_changes':{},'impulses':[]}
    ctx['_stress']=state
    def acquire(env):
        if condition['kind']=='cube_color' and id(env) not in state['material_changes']:
            state['material_changes'][id(env)]=visual_material(env,condition)
        before=env.sim.get_state().flatten().copy()
        images=raw(env)  # Three native draws, physics checks and native amplitude bound.
        stamp=float(env.sim.data.time);gain=gain_at(condition,stamp)
        originals={cam:images[cam].copy() for cam in CAMERAS}
        changed={**images,**{cam:transform_rgb(images[cam],condition,stamp) for cam in CAMERAS}}
        ctx['_last_draw2']={cam:transform_rgb(ctx['_last_draw2'][cam],condition,stamp) for cam in CAMERAS}
        ctx['_last_images']={k:v.copy() for k,v in changed.items()}
        for cam in CAMERAS:
            unclipped=originals[cam].astype(np.float64)*gain
            state['reads'].append({'timestamp_s':stamp,'camera':cam,'gain':gain,
                'changed_fraction':float(np.mean(changed[cam]!=originals[cam])),
                'saturated_fraction':float(np.mean(unclipped>255)),
                'zero_fraction':float(np.mean(changed[cam]==0)),'mean_uint8':float(changed[cam].mean())})
        np.testing.assert_array_equal(before,env.sim.get_state().flatten())
        return changed
    ctx['get_observation']=acquire
    atomic_json(Path(folder)/'declared_condition.json',{'condition':condition,'condition_hash':canonical(condition),
        'source_gain_scope':'digital uint8 source gain before both RGB input and persistent V2E; not calibrated lux/HDR',
        'native_acquisition_contract':'unchanged; transformed primary/shadow Event windows remain guarded'})
    return state

def apply_impulse(env,condition,control_step,stress):
    if condition['kind']!='velocity_impulse' or control_step!=condition['at_control_step']:return False
    assert not stress['impulses'], 'Impulse must occur once.'
    model=env.sim.model;before=env.sim.get_state().flatten().copy();qpos=np.asarray(env.sim.data.qpos).copy()
    joints=np.flatnonzero(np.asarray(model.jnt_bodyid)==int(env.cube_body_id))
    assert len(joints)==1 and int(model.jnt_type[int(joints[0])])==0, 'Expected a single free cube joint.'
    dof=int(model.jnt_dofadr[int(joints[0])]);delta=np.asarray(condition['delta_velocity_world_m_s'],float)
    old=np.asarray(env.sim.data.qvel).copy()
    env.sim.data.qvel[dof:dof+3]+=delta
    env.sim.forward()
    np.testing.assert_array_equal(qpos,env.sim.data.qpos)
    expected=old.copy();expected[dof:dof+3]+=delta
    np.testing.assert_allclose(env.sim.data.qvel,expected,rtol=0,atol=1e-12)
    after=env.sim.get_state().flatten().copy();assert before[0]==after[0]
    stress['impulses'].append({'applied_before_control_step':control_step,'simulation_time_s':float(env.sim.data.time),
        'delta_velocity_world_m_s':delta.tolist(),'cube_mass_kg':float(model.body_mass[int(env.cube_body_id)]),
        'impulse_world_N_s':(delta*float(model.body_mass[int(env.cube_body_id)])).tolist(),
        'position_unchanged_at_impulse':True,'before_state':before.tolist(),'after_state':after.tolist()})
    return True

def save_stress(ctx,folder):
    state=ctx.get('_stress');folder=Path(folder)
    if state is None:return None
    write_csv(folder/'source_transform_audit.csv',state['reads'])
    result={'condition':state['condition'],'condition_hash':canonical(state['condition']),
        'material_changes':list(state['material_changes'].values()),'impulses':state['impulses'],
        'reads':len(state['reads']),'history_reset_on_light_change':False,
        'max_saturated_fraction':max((r['saturated_fraction'] for r in state['reads']),default=0.0)}
    atomic_json(folder/'stress_audit.json',result)
    return result

def policy_panels(ctx,seed,step,success,stream):
    Image=ctx['Image'];ImageDraw=ctx['ImageDraw'];modality=ctx['request']['modality']
    canvas=Image.new('RGB',(512,512),'black');draw=ImageDraw.Draw(canvas)
    source=ctx['_last_images'] if modality=='rgb' else stream.current
    condition=ctx.get('_stress',{}).get('condition',{'id':'preflight'})
    for i,cam in enumerate(CAMERAS):
        if modality=='rgb': pixels=source[cam][4:80,4:80]
        else:
            a=source[cam][:,4:80,4:80]
            pixels=(255*np.stack([np.clip(a[0]/3,0,1),np.clip(a[2],0,1),np.clip(a[1]/3,0,1)],-1)).astype(np.uint8)
        picture=Image.fromarray(pixels).resize((224,224),resample=Image.Resampling.NEAREST)
        canvas.paste(picture,(144,i*256+26))
        draw.text((8,i*256+6),f'{modality.upper()} | {cam} | t={step/20:.2f}s',fill='white')
    draw.text((5,490),'Events: red ON / blue OFF / green time; fixed scales' if modality=='events' else 'RGB: current sensor view; center76 crop',fill='white')
    return np.asarray(canvas)

def compose_frame(ctx,external,panels,job,step,success,slow=False):
    assert external.shape==(512,512,3) and panels.shape==(512,512,3)
    Image=ctx['Image'];ImageDraw=ctx['ImageDraw']
    out=Image.fromarray(np.concatenate([external,panels],axis=1));draw=ImageDraw.Draw(out)
    draw.rectangle((0,0,511,46),fill='black')
    speed='0.25x slow playback' if slow else '1x simulated-time playback'
    draw.text((8,5),f"EXTERNAL MUJOCO | {job['condition_id']} | {job['modality'].upper()}",fill='white')
    draw.text((8,22),f"seed {job['environment_seed']} | t={step/20:.2f}s | Lift={bool(success)} | {speed}",fill='white')
    draw.text((8,478),'Recorded policy trajectory; observer replay only',fill='white')
    return np.asarray(out)

def presentation_replay(ctx,local,states,frame_steps,job,rows):
    """Render saved physical states AFTER scoring; never step or modify the policy stream."""
    local=Path(local);condition=condition_by_id(job['condition_id']);env=None;normal=None;slow=None;reader=None
    expected={0:False,**{int(r['step']):bool(r['success']) for r in rows}}
    started=time.perf_counter();audit=[]
    try:
        env=ctx['repair_make_environment'](ctx['SCENE_XML'],'pilot')
        ctx['verify_environment'](env)
        camera_id=env.sim.model.camera_name2id(PRESENTATION['external_camera']);assert camera_id>=0
        material=visual_material(env,condition)
        reader=ctx['imageio'].get_reader(str(local/'policy_panels.mp4'))
        normal=ctx['imageio'].get_writer(str(local/'rollout.mp4'),fps=10,codec='libx264',quality=7,macro_block_size=1)
        slow=ctx['imageio'].get_writer(str(local/'rollout_slow025.mp4'),fps=2.5,codec='libx264',quality=7,macro_block_size=1)
        count=0
        for index,panels in enumerate(reader):
            assert index<len(frame_steps), 'Extra recorded sensor frame.'
            step=int(frame_steps[index]);env.sim.set_state_from_flattened(states[step]);env.sim.forward()
            before=env.sim.get_state().flatten().copy()
            np.testing.assert_allclose(before,states[step],rtol=0,atol=1e-9)
            pixels=np.ascontiguousarray(env.sim.render(height=512,width=512,camera_name=PRESENTATION['external_camera'])[::-1])
            pixels=transform_rgb(pixels,condition,step/20.)
            np.testing.assert_array_equal(before,env.sim.get_state().flatten())
            image=compose_frame(ctx,pixels,panels,job,step,expected[step]);normal.append_data(image)
            slow.append_data(compose_frame(ctx,pixels,panels,job,step,expected[step],slow=True))
            if index==0:ctx['Image'].fromarray(image).save(local/'initial_presentation.png')
            audit.append({'video_frame':index,'control_step':step,'simulation_time_s':step/20.,'observer_state_max_abs':float(np.max(np.abs(before-states[step]))),
                'source_gain':gain_at(condition,step/20.),'observer_noninterference':True})
            count+=1
        assert count==len(frame_steps)>0
    finally:
        for obj in [normal,slow,reader,env]:
            if obj is not None:obj.close()
    for name in ['rollout.mp4','rollout_slow025.mp4']:
        r=ctx['imageio'].get_reader(str(local/name))
        try:assert r.get_data(0).shape==(512,1024,3)
        finally:r.close()
    write_csv(local/'presentation_frames.csv',audit)
    result={'status':'PASS','frames':count,'presentation':PRESENTATION,'material':material,'wall_seconds':time.perf_counter()-started,
        'scored_episodes':0,'env_steps':0,'maximum_state_restore_error':max(r['observer_state_max_abs'] for r in audit),
        'scope':'Separate saved-state observer rendering after scored policy execution; sensor panels are cached, lossy encoded display.'}
    atomic_json(local/'presentation_audit.json',result)
    return result

def stress_preflight(ctx,folder):
    from live_worker import fingerprint,raw_proprio,policy_observation,predict,warm_renderer
    env=None;renderer=None;stream=None;before=fingerprint(ctx)
    condition=condition_by_id(ctx['request']['condition_id']);stress=install_stress(ctx,condition,folder)
    initial_views=None
    try:
        env=ctx['make_env'](1997);ctx['verify_environment'](env);visual_material(env,condition)
        renderer=ctx['repair_make_environment'](ctx['SCENE_XML'],'pilot');ctx['verify_environment'](renderer)
        warm_renderer(ctx,renderer)
        initial=env.sim.get_state().flatten().copy();renderer.sim.set_state_from_flattened(initial);renderer.sim.forward()
        stream=ctx['EventStream']();images=ctx['get_observation'](renderer);initial_views={c:images[c].copy() for c in CAMERAS}
        stream.feed(images,0.);current=policy_observation(ctx,stream,raw_proprio(env));previous=current
        initial_cube=np.asarray(env.sim.data.body_xpos[env.cube_body_id]).copy()
        states=[initial];steps=[];before_impulse_cube=None
        for step in range(1,25):
            if condition['kind']=='velocity_impulse' and step==condition['at_control_step']:
                before_impulse_cube=np.asarray(env.sim.data.body_xpos[env.cube_body_id]).copy()
            apply_impulse(env,condition,step,stress)
            a=env.sim.get_state().flatten().copy();env.step(np.zeros(7,np.float32));b=env.sim.get_state().flatten().copy();states.append(b)
            ctx['advance_event_stream'](renderer,stream,a,b,step)
            np.testing.assert_array_equal(b,env.sim.get_state().flatten())
            previous=current;current=policy_observation(ctx,stream,raw_proprio(env))
            if step in [19,20,21]:steps.append({'step':step,'gain':gain_at(condition,step/20.),'current_event_mass':float(sum(stream.current[c][:2].sum() for c in CAMERAS))})
        if condition['kind']=='source_gain_step':
            assert {r['gain'] for r in stress['reads']}=={1.0,0.25}
        if condition['kind']=='velocity_impulse':assert len(stress['impulses'])==1
        if condition['kind']=='cube_color':assert stress['material_changes']
        gen=ctx['torch'].Generator(device=ctx['DEVICE']).manual_seed(90418)
        plan,projection=predict(ctx,previous,current,gen);assert plan.shape==(16,7)
        np.savez_compressed(Path(folder)/'stress_preflight_plan.npz',actions=plan,projection=projection)
        ctx['Image'].fromarray(policy_panels(ctx,1997,24,False,stream)).save(Path(folder)/'stress_sensor_preview.png')
        write_csv(Path(folder)/'transition_preview.csv',steps)
        # Observer probe is deliberately after the last policy call, in a separate simulator.
        body=ctx['repair_make_environment'](ctx['SCENE_XML'],'pilot')
        try:
            visual_material(body,condition);body.sim.set_state_from_flattened(states[-1]);body.sim.forward();saved=body.sim.get_state().flatten().copy()
            external=np.ascontiguousarray(body.sim.render(height=512,width=512,camera_name='frontview')[::-1])
            np.testing.assert_array_equal(saved,body.sim.get_state().flatten())
            image=compose_frame(ctx,transform_rgb(external,condition,1.2),policy_panels(ctx,1997,24,False,stream),
                {'condition_id':condition['id'],'modality':ctx['request']['modality'],'environment_seed':1997},24,False)
            ctx['Image'].fromarray(image).save(Path(folder)/'presentation_preview.png')
        finally:body.close()
        assert fingerprint(ctx)==before
        save_stress(ctx,folder)
        return {'status':'PASS','condition':condition,'scored_episodes':0,'weights_unchanged':True,
            'control_endpoints':24,'persistent_history':'PASS','FP32_EMA_inference':'PASS','observer_view':'PASS',
            'probe_cube_xy_displacement_m':float(np.linalg.norm(np.asarray(env.sim.data.body_xpos[env.cube_body_id])[:2]-initial_cube[:2])),
            'probe_cube_displacement_after_impulse_m':float(np.linalg.norm(np.asarray(env.sim.data.body_xpos[env.cube_body_id])[:2]-before_impulse_cube[:2])) if before_impulse_cube is not None else None,
            'scope':'Disposable zero-action probe; not policy rollout success or physical sensor calibration.'}
    finally:
        for obj in [stream,renderer,env]:
            if obj is not None:obj.close()


def save_reset_contract(ctx,env,folder):
    """Record actual sampler/noise settings, not inferred bounds from finite reset samples."""
    sampler=env.placement_initializer
    np.testing.assert_allclose(sampler.x_range,[-.03,.03],atol=1e-12,rtol=0)
    np.testing.assert_allclose(sampler.y_range,[-.03,.03],atol=1e-12,rtol=0)
    assert sampler.rotation is None and sampler.rotation_axis=='z'
    np.testing.assert_allclose(sampler.reference_pos,[0,0,.8],atol=1e-12,rtol=0)
    assert [getattr(r,'initialization_noise',None) for r in env.robots]==[{'magnitude':.02,'type':'gaussian'}], 'Robot reset noise differs from the frozen development contract.'
    state=env.sim.get_state().flatten().copy()
    assert state.shape==(32,) and np.isfinite(state).all()
    report={'status':'PASS','environment_seed':ctx['request'].get('job',{}).get('environment_seed'),
        'placement_sampler_class':type(sampler).__name__,'x_range_m':list(sampler.x_range),
        'y_range_m':list(sampler.y_range),'rotation':sampler.rotation,'rotation_axis':sampler.rotation_axis,
        'reference_pos_m':np.asarray(sampler.reference_pos).tolist(),
        'metadata_initialization_noise':ctx['ENV_META']['env_kwargs'].get('initialization_noise','default'),
        'actual_robot_initialization_noise':[getattr(r,'initialization_noise',None) for r in env.robots],
        'actual_arm_qpos_rad':state[1:8].tolist(),'actual_gripper_qpos_m':state[8:10].tolist(),
        'actual_cube_qpos_xyz_wxyz':state[10:17].tolist(),
        'reset_protocol':'fixed TRAIN XML; deterministic_reset=False; seeded fresh task reset',
        'scope':'actual reset values and sampler contract; not proof of out-of-distribution coverage'}
    atomic_json(Path(folder)/'reset_contract.json',report)
    return report

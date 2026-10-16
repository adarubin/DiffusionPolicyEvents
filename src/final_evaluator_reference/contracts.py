"""Evaluation-only persistence, exact source identities and paired development budget."""
import csv, hashlib, json, os, shutil, time, uuid, zipfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

VERSION = 'notebook08-d200-tail-final-normal-dark-seeds42-43-44-v1'
DATASET_ID = 'd200_paired_tail_v1_88bac96f5257'
DATA_HASH = '7e86659ebc890c6ce6aa7ec6907b3bfa63d0045ed30a580aac475b89384e15c7'
NORM_HASH = '8cfb1b1d216f5c0c9752628ef745fa601c7ee1667c445e5c0d7b5a6dfc1355a5'
TRAIN_CODE = 'notebook04-05-paired-gpu-d200-tail-v1'
TRAIN_CODE_SHA = '62c010b9cf90328a4534614c1fb1ad5ab914d61bd48b5ce429084c99460d7aba'
RAW_SHA = '68f2a8cfe2502de1224f0708fb82e46da07abce108c63e721d72fc1618272f37'
SPLIT_SHA = '3aaecbdeb837980170aa5a4253dfcd9729d6dae5f38b94f4376a9fa41b213e44'
SOURCE_SHA = 'ee77cb475a9589ab21634c16b7316dfdec5abe95625fef29213335a0c28c0079'
FROZEN_SHA = '48d0045520329ab21c8e7cc70ec38cbc26b8a1de892c3c0ea62a085fc6e875e4'
CAMERAS = ['agentview', 'robot0_eye_in_hand']
SOURCE_CATALOG = {'42': {'rgb': {'run_id': 'd200_rgb_paired_gpu_tail_fixed4500_v1_seed42_f4221f88a689', 'config_hash': 'f4221f88a6892d2a3f10ec530f01c10e5ade75611d652a7129a5867a778b666f', 'best': {'step': 4500, 'relative': 'best/step_004500_01791714935441234706_0f7dcd1a.ckpt', 'sha256': '09ca161ee0a231146faaf7a5dc0c482a73bfbd30f81b286dd550fcc0cac02a13'}}, 'events': {'run_id': 'd200_events_paired_gpu_tail_fixed4500_v1_seed42_78e67b2b54a1', 'config_hash': '78e67b2b54a106c045f69e96ab7a38a490ddc6798b56682155fde12485341288', 'best': {'step': 4500, 'relative': 'best/step_004500_01791718502057605491_64e73a6e.ckpt', 'sha256': 'c2383ef36146abfd9e872e16dbd1105af81a47e1bd319e98a7155621cce27194'}}}, '43': {'rgb': {'run_id': 'd200_rgb_paired_gpu_tail_fixed4500_v1_seed43_85f45fcb540b', 'config_hash': '85f45fcb540b5c621165cb9bbb66b2dfec07daa0f6fc3c1f6aa74d519ff82f3b', 'best': {'step': 4500, 'relative': 'best/step_004500_01791744910726681940_1de9a22d.ckpt', 'sha256': 'cd8da3e1b127737b9e5a1adecdd5c2d796007a355f560ea1832cb085579fba95'}}, 'events': {'run_id': 'd200_events_paired_gpu_tail_fixed4500_v1_seed43_c9e81662cb8d', 'config_hash': 'c9e81662cb8d907c0a01ddc27f19d960b15752968db6aa9604863435af90a0d3', 'best': {'step': 4500, 'relative': 'best/step_004500_01791748635011251479_7b72ef6c.ckpt', 'sha256': '995c6035fbe499f6d529a59a33c831e545bf71245a759f2d12c72b1d6bc957b1'}}}, '44': {'rgb': {'run_id': 'd200_rgb_paired_gpu_tail_fixed4500_v1_seed44_a0fe3c3db462', 'config_hash': 'a0fe3c3db46289ad830eba075ce11f09f46671924216e96dadafb54e619205a2', 'best': {'step': 4500, 'relative': 'best/step_004500_01791746143897191881_f2fac252.ckpt', 'sha256': '4cff99b364f5420142b1a9b40c3d81b685d40179dd92f15d41d378ea5745da11'}}, 'events': {'run_id': 'd200_events_paired_gpu_tail_fixed4500_v1_seed44_ab47122ba734', 'config_hash': 'ab47122ba7344f15b45134ecaf76d923abb366f395764892a3c7084dffa4d5fa', 'best': {'step': 4500, 'relative': 'best/step_004500_01791749891742660507_88a50998.ckpt', 'sha256': '7744db85d1afa3166d45294f6551e33c86a1d8eb7080a93e17d6c888d00c81bd'}}}}
TRAINING_SEED = 42
SOURCES = dict(SOURCE_CATALOG["42"])

def select_training_seed(seed):
    global TRAINING_SEED
    assert type(seed) is int and seed in [42,43,44]
    TRAINING_SEED=seed
    SOURCES.clear();SOURCES.update(SOURCE_CATALOG[str(seed)])
    return SOURCES

def utc(): return datetime.now(timezone.utc).isoformat()
def sha_file(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
def json_default(value):
    if isinstance(value,Path): return str(value)
    if isinstance(value,(np.ndarray,np.generic)): return value.tolist()
    raise TypeError(type(value).__name__)
def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        allow_nan=False,default=json_default).encode()).hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value,indent=2,allow_nan=False,default=json_default),encoding='utf8')
        os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)
def safe_child(root,relative):
    root=Path(root).resolve();relative=Path(relative)
    assert not relative.is_absolute() and '..' not in relative.parts, 'Unsafe artifact path.'
    path=(root/relative).resolve();assert path!=root and path.is_relative_to(root)
    return path
def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with open(temporary,'w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
        os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)
def validate_budget(options):
    from stress_conditions import CONDITIONS,PRESENTATION
    assert options['max_control_steps']==300 and options['execution_horizon']==8
    assert options['success_hold_steps']==10 and options['video_stride']==2 and options['scene_demo']=='demo_10'
    assert options['environment_seeds']==list(range(1000,1050)) and options['policy_seeds']==list(range(4000,4050))
    assert options['training_seeds']==[42,43,44] and options['planned_episodes']==600
    assert options['evaluation_role']=='locked_final_normal_dark' and options['no_adaptive_stopping'] is True
    assert options['source_policy']=='strict' and options['sampler']=='train_range_x0'
    assert options['conditions']==CONDITIONS and options['presentation']==PRESENTATION
    assert options['episode_timeout_seconds']==1200
    historical=set(range(2000,2010))|set(range(2100,2150))|set(range(2200,2210))|set(range(2300,2320))
    historical|=set(range(3000,3010))|set(range(3100,3150))|set(range(3200,3210))|set(range(3300,3320))|{42,43,44,20260927}
    assert not (set(options['environment_seeds'])|set(options['policy_seeds']))&historical
    return True
def episode_key(modality,stage,seed):
    assert modality in SOURCES and stage == 'best'
    return f'{modality}_{stage}_seed_{int(seed)}'
def schedule(options):
    validate_budget(options)
    from stress_conditions import CONDITIONS
    return [{'key':'train'+str(training)+'__'+c['id']+'__'+episode_key(m,'best',seed),
        'training_seed':training,'modality':m,'stage':'best','environment_seed':seed,'policy_seed':policy,
        'condition_id':c['id'],'condition_hash':canonical(c)}
        for training in options['training_seeds'] for c in CONDITIONS
        for seed,policy in zip(options['environment_seeds'],options['policy_seeds']) for m in ['rgb','events']]
def validate_run_record(record,modality):
    identity=SOURCES[modality]
    assert record['run_id']==identity['run_id'] and record['config_hash']==identity['config_hash']
    assert record['modality']==modality and record['data_fingerprint']==DATA_HASH
    assert record['paired_sample_fingerprint']==DATA_HASH and record['normalization_hash']==NORM_HASH
    assert record['dataset_id']==DATASET_ID
    assert record['code_version']==TRAIN_CODE and record['code_sha256']==TRAIN_CODE_SHA
    assert canonical({'code':record['code_version'],'code_sha256':record['code_sha256'],
        'config':record['config'],'data':DATA_HASH,'normalization':NORM_HASH})==identity['config_hash'], 'Training config identity changed.'
    cfg=record['config']
    assert cfg['checkpoint_schema']==6 and cfg['unet_dims']==[128,256,512]
    assert cfg['state_per_observation']==18 and cfg['observation_horizon']==2
    assert cfg['prediction_horizon']==16 and cfg['action_dim']==7 and cfg['crop_size']==76
    assert cfg['diffusion_steps']==100 and cfg['max_beta']==.999 and cfg['seed']==TRAINING_SEED
    assert cfg['max_steps']==4500 and cfg['lr_schedule_steps']==10000 and cfg['early_stop_enabled'] is False
    assert cfg['export_schema']==3 and cfg['dataset_id']==DATASET_ID
    assert cfg['execution_horizon']==8 and cfg['split_seed']==20260927
    assert cfg['tail_target_policy']==TAIL_POLICY
    return cfg
def validate_checkpoint(saved,record,modality,stage):
    cfg=validate_run_record(record,modality);identity=SOURCES[modality][stage]
    assert saved['run_id']==record['run_id'] and saved['config_hash']==record['config_hash']
    assert saved['global_step']==identity['step'] and saved['schema']==6
    assert saved['modality']==modality and saved['dataset_id']==DATASET_ID
    assert saved['data_fingerprint']==saved['paired_sample_fingerprint']==DATA_HASH
    assert saved['paired_normalization_hash']==NORM_HASH
    assert saved['code_version']==TRAIN_CODE and saved['code_sha256']==TRAIN_CODE_SHA
    assert canonical(saved['config'])==canonical(cfg) and canonical(saved['normalization'])==NORM_HASH
    assert stage=='best' and saved['kind']=='best_raw_ema_fp32'
    assert 'ema' in saved
    return True
def validate_episode_files(root,protocol_hash,job,row):
    assert row['protocol_hash']==protocol_hash
    from stress_conditions import condition_by_id
    assert row['condition_id']==job['condition_id']
    assert row['condition_hash']==job['condition_hash']==canonical(condition_by_id(job['condition_id']))
    assert row['presentation']['status']=='PASS' and row['presentation']['env_steps']==0
    assert row['stress_audit']['condition_hash']==row['condition_hash']
    for key in ['key','modality','stage','environment_seed','policy_seed']: assert row[key]==job[key]
    assert row['status']=='COMPLETED'
    assert row['training_seed']==job['training_seed'] and job['training_seed'] in [42,43,44]
    identity=SOURCE_CATALOG[str(job['training_seed'])][job['modality']]
    assert row['source_run_id']==identity['run_id']
    assert row['group']==job['modality']+'_best' and 1<=row['steps']<=300
    assert type(row['any_success']) is bool and type(row['sustained_success']) is bool
    assert 0<=row['max_consecutive_success_steps']<=row['steps']
    assert row['sustained_success']==(row['max_consecutive_success_steps']>=10)
    assert row['any_success']==(row['first_success_step'] is not None)
    if row['first_success_step'] is not None:assert 1<=row['first_success_step']<=row['steps']
    assert row['checkpoint_sha256']==identity[job['stage']]['sha256']
    assert row['checkpoint_step']==identity[job['stage']]['step']
    assert row['artifacts'] and row['render_gate']=='PASS' and row['weights_unchanged'] is True
    guard=row['acquisition_guard']
    assert guard['status']=='PASS' and guard['control_endpoints_checked']==row['steps']
    assert guard['tensor_checks']==4*row['steps'] and guard['max_abs_tensor_difference']<=1e-5
    basenames={Path(n).name for n in row['artifacts']}
    assert {'initial_state.npy','trajectory.npz','steps.csv','rollout.mp4','render_stability_summary.json','render_stability.csv','acquisition_guard_summary.json','event_equivalence.csv','acquisition_read_audit.csv','rollout_slow025.mp4','policy_panels.mp4','policy_inputs.npz','stress_audit.json','presentation_audit.json','presentation_frames.csv','reset_contract.json'}<=basenames
    for name,digest in row['artifacts'].items():
        path=safe_child(root,name)
        assert path.is_file() and path.stat().st_size>0 and sha_file(path)==digest
    return True

def verified_episode(root,protocol_hash,job):
    pointer=Path(root)/'episodes'/(job['key']+'.json')
    if not pointer.exists(): return None
    try:
        row=json.loads(pointer.read_text())
        if row['status']!='COMPLETED': return None
        validate_episode_files(root,protocol_hash,job,row)
        return row
    except (OSError,ValueError,AssertionError,KeyError,TypeError):
        print('Incomplete/corrupt attempt will be retained and rerun:',job['key'],flush=True)
        return None
def publish_episode(root,attempt,local,row):
    assert row['status'] in ['COMPLETED','FAILED','ABORTED']
    row['artifacts']={}
    for path in sorted(Path(local).iterdir()):
        if not path.is_file(): continue
        target=Path(attempt)/path.name;shutil.copyfile(path,target)
        digest=sha_file(path);assert sha_file(target)==digest, 'Evidence transfer failed.'
        row['artifacts'][str(target.relative_to(root))]=digest
    atomic_json(Path(attempt)/'result.json',row)
    # Only complete, checked evidence can become the resumable completion pointer.
    if row['status']=='COMPLETED':
        temporary_job={k:row[k] for k in ['key','training_seed','modality','stage','environment_seed','policy_seed','condition_id','condition_hash']}
        validate_episode_files(root,row['protocol_hash'],temporary_job,row)
        candidate=Path(root)/'episodes'/(row['key']+'.json')
        atomic_json(candidate,row)
        assert verified_episode(root,row['protocol_hash'],temporary_job) is not None
    return row
def paired_counts(rows,first,second,outcome):
    a={r['environment_seed']:r for r in rows if r['status']=='COMPLETED' and r['group']==first}
    b={r['environment_seed']:r for r in rows if r['status']=='COMPLETED' and r['group']==second}
    common=sorted(set(a)&set(b));counts=dict(both_success=0,first_only=0,second_only=0,both_fail=0)
    for seed in common:
        av,bv=bool(a[seed][outcome]),bool(b[seed][outcome])
        counts['both_success' if av and bv else 'first_only' if av else 'second_only' if bv else 'both_fail']+=1
    return {'first':first,'second':second,'outcome':outcome,'paired_seeds':common,'n':len(common),'counts':counts}

D200_SPLIT_SHA = '3428033261f30bca390c104b7978bcd387133a7fedbcf5ba64eae6a56059688d'
D200_SOURCE_PROTOCOL_HASH = '93f4f6bf54a9469be3ae4e41e2d98dda71850f41a4ca863e865f152a052e3ab3'
D200_PREPARATION_BUILD = '0dba021949ead3a91c4c40bf9ffe20e79b067061ae7c202c5e3188d09c28ae2d'

TAIL_POLICY = {'first_input': 2, 'last_input': 'T-1', 'observation_horizon': 2, 'prediction_horizon': 16, 'execution_horizon': 8, 'action_alignment': 'actions[t:t+16]; execute indices0:8', 'tail_padding': 'repeat_last_recorded_action_within_demo', 'observation_padding': 'none', 'loss_policy': 'unmasked_epsilon_MSE_all_16_targets_including_endpoint_repeats', 'action_is_recorded_scope': 'provenance indicator; NOT a training loss mask', 'normalization_scope': 'TRAIN_ONLY_OVERLAPPING_WINDOWS_INCLUDING_ENDPOINT_REPEATS', 'statistic_estimator': 'same overlapping-window mean/std method as historical export', 'terminal_semantics': 'Endpoint continuation convention; repeated command is not a measured post-demo action or guaranteed zero-motion hold.'}
EXPORT_PROTOCOL_HASH = '88bac96f525758fdee4b2f059324bc90cce81f1255af30352fb8d6ecea49820e'
EXPORT_BUILD = '91f30c3f7c12b3b7ba79a2b4667f8613167ef6c1d0e1a8bd55ea2107b9757302'
PARENT_DATA_HASH = '79257cd4b409d4125f682ee8517006d65ca3ff54dae9803aae06b480e2e3623b'
PARENT_NORM_HASH = '65a31662ca4501984c1f40d7226a70c5d5e5b9752718f91c2e1f27e6ca34d193'
PARENT_DATASET_ID = 'd200_paired_gpu_v2_93f4f6bf54a9'

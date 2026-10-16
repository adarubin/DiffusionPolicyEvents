"""Versioned paired-source persistence and numerical contracts. No policy training."""
import csv, hashlib, io, json, os, shutil, time, uuid, zipfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

RAW_SHA = '68f2a8cfe2502de1224f0708fb82e46da07abce108c63e721d72fc1618272f37'
SPLIT_SHA = '3aaecbdeb837980170aa5a4253dfcd9729d6dae5f38b94f4376a9fa41b213e44'
VALIDATION_SHA = '8836068c3468ace9b3f3b57e9f4f27875f9fb003214d0144d65835b65144376a'
SOURCE_SHA = 'ee77cb475a9589ab21634c16b7316dfdec5abe95625fef29213335a0c28c0079'
FROZEN_SHA = '48d0045520329ab21c8e7cc70ec38cbc26b8a1de892c3c0ea62a085fc6e875e4'
CONTRACT = 'fixed-three-draws-gray-exact-rgb-bounded-v2'
CAMERAS = ['agentview', 'robot0_eye_in_hand']
TRAIN_GATE_DEMOS = ['demo_10', 'demo_198', 'demo_87']

def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def json_default(v):
    if isinstance(v, Path): return str(v)
    if hasattr(v, 'tolist'): return v.tolist()
    raise TypeError(type(v).__name__)

def canonical(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'),
        allow_nan=False, default=json_default).encode()).hexdigest()

def utc(): return datetime.now(timezone.utc).isoformat()

def safe_child(root, relative):
    root = Path(root).resolve(); p = (root/relative).resolve()
    assert p != root and p.is_relative_to(root), 'Artifact path escapes its dataset.'
    return p

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=json_default), encoding='utf8')
        os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)

def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with open(temp, 'w', newline='', encoding='utf8') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        os.replace(temp, path)
    finally: temp.unlink(missing_ok=True)

def verify_split(path):
    assert sha_file(path) == SPLIT_SHA, 'Locked D50 train/validation split changed.'
    s = json.loads(Path(path).read_text()); train = s['train_demos']; val = s['validation_demos']
    assert len(train) == len(set(train)) == 40 and len(val) == len(set(val)) == 10
    assert set(train).isdisjoint(val) and all(d in train for d in TRAIN_GATE_DEMOS)
    return s

def verify_validation_bundle(path):
    assert sha_file(path) == VALIDATION_SHA, 'Expected reviewed three-demo evidence bundle.'
    with zipfile.ZipFile(path) as z:
        s = json.loads(z.read('gpu_source_validation_summary.json'))
        assert s['status'] == 'SOURCE_VALIDATION_PASS'
        assert s['required_demos'] == s['successful_demos'] == s['completed_demos'] == TRAIN_GATE_DEMOS
        assert s['source_worker_sha256'] == SOURCE_SHA and s['acceptance_contract'] == CONTRACT
        assert hashlib.sha256(z.read('source_worker.py')).hexdigest() == SOURCE_SHA
        assert hashlib.sha256(z.read('verified_paired_v4.py')).hexdigest() == FROZEN_SHA
        for rec in s['demo_records']:
            assert rec['status'] == 'SOURCE_PILOT_PASS'
            assert rec['offline_online_parity']['status'] == 'PASS'
            assert rec['physics_noninterference']['status'] == rec['cpu_gpu_physics']['status'] == 'PASS'
    return s

def verify_source(folder, expected_demo=None, protocol_hash=None, split_sha256=None):
    folder = Path(folder); m = json.loads((folder/'manifest.json').read_text())
    assert m['status'] == 'COMPLETED_CANDIDATE'
    if expected_demo is not None: assert m['demo_id'] == expected_demo
    if protocol_hash is not None: assert m['source_protocol_sha256'] == protocol_hash
    assert m['candidate_fingerprint'] == canonical({'source_protocol_sha256':m['source_protocol_sha256'],
        'provenance':m['provenance'], 'files':m['files']})
    required = {'recorded_scene.xml','runtime_scene.xml','source_timestamps.npy','window_timestamps.npy',
        'control_state_action.npz','paired_policy_samples.npz','render_state_audit.json',
        'render_stability.csv','render_stability_summary.json'}
    required |= {cam+suffix for cam in CAMERAS for suffix in ['_rgb.npz','_events.npz']}
    assert required <= set(m['files']), 'Incomplete source manifest.'
    for name, digest in m['files'].items(): assert sha_file(safe_child(folder, name)) == digest, 'Source hash changed: '+name
    p = m['provenance']; assert p['demo_id'] == m['demo_id'] and p['raw_sha256'] == RAW_SHA and p['split_sha256'] == (SPLIT_SHA if split_sha256 is None else split_sha256)
    assert p['render_stability_contract']['version'] == CONTRACT
    assert p['event_time_contract']['version'] == 'timestamp-dtype-half-open-v1'
    assert p['render_lifecycle_contract']['version'] == 'interleaved-render-emulate-v1'
    g = p['graphics']; assert g['hardware_nvidia_verified'] is True and 'NVIDIA' in g['vendor']
    assert not any(k in g['renderer'].lower() for k in ['llvmpipe','softpipe','software'])
    assert m['robot_state_per_observation'] == 18 and m['proprio_horizon_total'] == 36
    v = p['v2e_settings']; assert v['device'] == 'cpu' and v['emulator_seed'] == 42
    assert v['resolved_commit'] == 'cac638444557a224ba6ee5c29260916f152c00bd'
    assert v['pos_thres'] == v['neg_thres'] == .2
    assert all(v[k] == 0 for k in ['sigma_thres','cutoff_hz','leak_rate_hz','shot_noise_rate_hz','refractory_period_s'])
    a = json.loads((folder/'render_stability_summary.json').read_text())
    assert a['contract']['version'] == CONTRACT
    for key in ['reads_with_grayscale_mismatch','reads_outside_rgb_bound','reads_with_state_change','reads_with_proprio_change']:
        assert a[key] == 0, 'Source acquisition failed: '+key
    assert a['logical_observation_reads'] == m['frames_per_camera']+1
    assert a['native_two_camera_draws'] == 3*a['logical_observation_reads']
    return m

def check_source_draw_rows(folder,manifest):
    with open(Path(folder)/'render_stability.csv',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==manifest['frames_per_camera']+1
    for row in rows:
        assert int(row['native_draws'])==3
        assert all(row[k]=='True' for k in ['state_unchanged_each_draw','last_two_proprio_exact','last_two_grayscale_exact','last_two_rgb_within_bound'])
    return True

def checked_pointer(root, pointer_path, dataset_id, kind):
    rec = json.loads(Path(pointer_path).read_text())
    assert rec['dataset_id'] == dataset_id and rec['kind'] == kind and rec['status'] == 'COMPLETED'
    folder = safe_child(root, rec['relative_directory'])
    assert sha_file(folder/'manifest.json') == rec['manifest_sha256'], 'Generation manifest hash changed.'
    return folder, rec

def event_tensor(events, start, end):
    assert events.ndim == 2 and events.shape[1] == 4 and events.dtype == np.float32
    lo, hi = events.dtype.type(start), events.dtype.type(end)
    assert lo < hi
    chosen = events[(events[:,0] >= lo) & (events[:,0] < hi)]
    tensor = np.zeros((3,84,84), dtype=np.float32)
    if not len(chosen): return tensor
    x, y = chosen[:,1].astype(np.int64), chosen[:,2].astype(np.int64)
    assert np.isfinite(chosen).all() and ((x >= 0)&(x < 84)&(y >= 0)&(y < 84)).all()
    assert np.array_equal(chosen[:,1:3], chosen[:,1:3].astype(np.int64)) and np.isin(chosen[:,3], [-1,1]).all()
    for ch, mask in [(0,chosen[:,3]>0),(1,chosen[:,3]<0)]:
        counts = np.zeros((84,84),dtype=np.float32); np.add.at(counts,(y[mask],x[mask]),1.)
        tensor[ch] = np.log1p(counts)
    latest = np.full((84,84),-np.inf,dtype=np.float64); np.maximum.at(latest,(y,x),chosen[:,0])
    valid = np.isfinite(latest); tensor[2,valid] = np.exp(-(hi-latest[valid])/.05)
    return tensor

def event_tensor_sorted(events, start, end):
    """Select an exact float32 half-open interval without scanning the full stream."""
    lo, hi = np.float32(start), np.float32(end)
    assert lo < hi
    first = int(np.searchsorted(events[:,0], lo, side='left'))
    last = int(np.searchsorted(events[:,0], hi, side='left'))
    return event_tensor(events[first:last], start, end)


def validate_paired(folder, raw_states=None, raw_actions=None, raw_xml=None, full_events=True, split_sha256=None):
    folder = Path(folder); m = verify_source(folder, split_sha256=split_sha256)
    with np.load(folder/'control_state_action.npz',allow_pickle=False) as z:
        states, actions, props = z['states'], z['actions'], z['robot_proprio']
        times_control = z['control_timestamps']
    n = len(states); assert states.shape == (n,32) and actions.shape == (n,7) and props.shape == (n,18)
    assert np.isfinite(states).all() and np.isfinite(actions).all() and np.isfinite(props).all()
    np.testing.assert_allclose(states[:,0],np.arange(n)/20,atol=1e-7,rtol=0)
    np.testing.assert_array_equal(times_control,states[:,0])
    np.testing.assert_allclose(props,np.c_[states[:,1:10],states[:,17:26]],atol=1e-6,rtol=1e-6)
    if raw_states is not None: np.testing.assert_array_equal(states,raw_states)
    if raw_actions is not None: np.testing.assert_array_equal(actions,np.asarray(raw_actions,dtype=np.float32))
    if raw_xml is not None:
        if isinstance(raw_xml,bytes): raw_xml=raw_xml.decode()
        assert hashlib.sha256(raw_xml.encode()).hexdigest() == m['provenance']['raw_scene_xml_sha256']
        assert (folder/'recorded_scene.xml').read_text() == raw_xml
    high = np.load(folder/'source_timestamps.npy',allow_pickle=False)
    ends = np.load(folder/'window_timestamps.npy',allow_pickle=False)
    np.testing.assert_array_equal(high,np.arange((n-1)*10+1,dtype=np.float64)/200.)
    np.testing.assert_array_equal(ends,np.arange(2,n,dtype=np.float64)/20.)
    assert m['frames_per_camera'] == len(high) and m['event_windows'] == n-2 and m['paired_policy_samples'] == n-17
    with np.load(folder/'paired_policy_samples.npz',allow_pickle=False) as z: packed={k:z[k] for k in z.files}
    steps = np.arange(2,n-15,dtype=np.int64)
    keys={'control_steps','control_timestamps','action_targets','proprio_previous','proprio_current'}
    keys |= {cam+'_'+mod+'_'+suffix for cam in CAMERAS for mod in ['rgb','events'] for suffix in ['previous','current']}
    assert set(packed) == keys
    np.testing.assert_array_equal(packed['control_steps'],steps)
    np.testing.assert_array_equal(packed['control_timestamps'],steps/20.)
    np.testing.assert_array_equal(packed['action_targets'],np.stack([actions[t:t+16] for t in steps]))
    np.testing.assert_array_equal(packed['proprio_previous'],props[steps-1]); np.testing.assert_array_equal(packed['proprio_current'],props[steps])
    maximum = 0.; events_total=0
    for cam in CAMERAS:
        with np.load(folder/(cam+'_rgb.npz'),allow_pickle=False) as z: frames=z['frames']
        assert frames.shape == (len(high),84,84,3) and frames.dtype == np.uint8
        for suffix, ix in [('previous',(steps-1)*10),('current',steps*10)]:
            np.testing.assert_array_equal(packed[cam+'_rgb_'+suffix],frames[ix])
        del frames
        with np.load(folder/(cam+'_events.npz'),allow_pickle=False) as z: ev=z['events']; previous=z['previous']; current=z['current']
        assert ev.dtype == previous.dtype == current.dtype == np.float32
        assert ev.shape[1:] == (4,) and np.isfinite(ev).all() and np.isfinite(previous).all() and np.isfinite(current).all()
        assert previous.shape == current.shape == (len(ends),3,84,84)
        assert (previous>=0).all() and (current>=0).all() and (previous[:,2]<=1).all() and (current[:,2]<=1).all()
        if len(ev):
            assert np.all(np.diff(ev[:,0])>=-1e-7) and ev[:,0].min()>=-1e-6 and ev[:,0].max()<=high[-1]+1e-6
        events_total+=len(ev)
        if full_events:
            # Minor accepted timestamp-order noise is handled in this audit view only.
            # Stored packets and source/emulation output remain immutable.
            ordered = ev if np.all(np.diff(ev[:,0])>=0) else ev[np.argsort(ev[:,0],kind='stable')]
            for i,end in enumerate(ends):
                for ref,start,finish in [(previous[i],end-.10,end-.05),(current[i],end-.05,end)]:
                    rebuilt=event_tensor_sorted(ordered,start,finish); error=float(np.max(np.abs(ref-rebuilt)))
                    maximum=max(maximum,error); assert error<=1e-5, 'Independent Event reconstruction failed.'
        for suffix,value in [('previous',previous),('current',current)]:
            np.testing.assert_array_equal(packed[cam+'_events_'+suffix],value[steps-2])
    return m,packed,{'demo_id':m['demo_id'],'samples':len(steps),'event_windows':len(ends),
        'frames_per_camera':len(high),'raw_events_both_cameras':events_total,'independent_event_max_abs':maximum,
        'robot_state_per_observation':18,'actions_proprio_rgb_events_alignment':'PASS'}

def evidence_bundle(session, name):
    session=Path(session); dest=session/name
    temp=dest.with_name(dest.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(session.rglob('*')):
                if p.is_file() and p != temp and p.suffix not in ['.zip','.npz','.npy']:
                    z.write(p,str(p.relative_to(session)))
        os.replace(temp,dest)
    finally:temp.unlink(missing_ok=True)
    return dest

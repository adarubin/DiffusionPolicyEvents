# MODULE 01_setup.py
# Run All on a fresh Colab T4. No training; historical RGB/Event runs are read-only.
PILOT = {
    'tag': 'rgb_d50_best1500_bounded_v1',
    'environment_seeds': list(range(2000, 2010)),
    'policy_seeds': list(range(3000, 3010)),
    'max_control_steps': 300,
    'execution_horizon': 8,
    'success_hold_steps': 10,
    'video_stride': 2, 'video_resolution': 256,
    'render_match_mae_limit': 2.0,  # uint8 levels; hard gate, not a learned threshold
}
CODE_VERSION = 'notebook07-paired-template154-v4'
CODE_SHA256 = '98032fe3c2a64fde6481fe50b51c5cab30a4fc16286ca3f9c13f21c4ff35eeb7'
# One active Notebook07: paired evaluation is the default; repair remains an explicit diagnostic mode.
MODE = 'paired_pilot'  # Run All evaluates RGB first, then Events. No training.
MODALITY = globals().get('EVAL_MODALITY','rgb')
PAIRED_RGB_CONTRACT = globals().get('PAIRED_RGB_CONTRACT')
SCENE_TEMPLATE_DEMO = 'demo_154'
REPAIR_OPTIONS = {'historical_template_demo':'demo_154','calibration_windows':6,'verification_demos':2}
assert MODE in ['renderer_repair','pilot','paired_pilot']
assert MODE!='renderer_repair' or MODALITY=='events'
# strict: stop on renderer drift; exploratory: record drift and run development trials only.
EVENT_RENDER_POLICY = 'strict'
assert MODE!='paired_pilot' or EVENT_RENDER_POLICY=='strict', 'Paired evaluation requires strict renderer compatibility.'
assert MODALITY in ['rgb','events']
SOURCES = {
 'rgb': {'run_id':'d50_rgb_run4_smallunet_seed42_1ad98fd703af',
  'checkpoint':'best/step_001500_01791057690723302567_5352279c.ckpt',
  'sha256':'3aa345e63be74339489a154d303015c5d3775218cb4e181d414d3aeb9a6e1232',
  'config':'1ad98fd703af7f9e2ee19119a7f2c4a76dbe85761f43ff0af812fdaca1589cfd',
  'data':'d108c1f91be38b3fe7fd79e956d843f9e3b16768f7953b52ddd7059eed6c91d8'},
 'events': {'run_id':'d50_events_matched_run4_seed42_c9eae534117c',
  'checkpoint':'best/step_001500_01791140000689365899_4bed7e55.ckpt',
  'sha256':'63b2675fdf4138f68782944c61c054a03cdc43c4ee1e341c13df6f1b1248cc2a',
  'config':'c9eae534117c8803ae5a1b18ed57dce867b0df711d695a47688fba78ed6e7b34',
  'data':'a30196d2dd0471a250d3aa71a0fee6c31a1162d231e0f114554e7e58eab282f2'}
}
SOURCE_RUN_ID=SOURCES[MODALITY]['run_id']
CHECKPOINT_RELATIVE=SOURCES[MODALITY]['checkpoint']
EXPECTED_CHECKPOINT_SHA256=SOURCES[MODALITY]['sha256']
EXPECTED_CONFIG_HASH=SOURCES[MODALITY]['config']
EXPECTED_DATA_FINGERPRINT=SOURCES[MODALITY]['data']
PAIRED_SAMPLE_FINGERPRINT='d108c1f91be38b3fe7fd79e956d843f9e3b16768f7953b52ddd7059eed6c91d8'
V2E_COMMIT='cac638444557a224ba6ee5c29260916f152c00bd'
EVENT_OPTIONS={'source_hz':200,'control_hz':20,'window_s':.05,'tau_s':.05,
 'device':'match_source_manifest','seed':42,'threshold':.2,'source_tensor_atol':1e-5,
 'render_mae_limit':2.,'calibration_windows':6,'physics_parity_atol':1e-9}
PILOT['tag']=f'{MODALITY}_d50_best1500_template154_v4' if MODE in ['pilot','paired_pilot'] else 'events_renderer_repair_v3'
EXPECTED_RAW_SHA256 = '68f2a8cfe2502de1224f0708fb82e46da07abce108c63e721d72fc1618272f37'
import os, sys, subprocess, importlib, importlib.util, importlib.metadata
import json, math, random, hashlib, copy, time, uuid, csv, io, shutil, traceback, inspect, zipfile, gc
from pathlib import Path
from datetime import datetime, timezone
START_PERF = time.perf_counter()
# Set graphics backend BEFORE importing MuJoCo/OpenGL/robosuite.
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from google.colab import drive
drive.mount('/content/drive')
PROJECT_ROOT = Path('/content/drive/MyDrive/DiffusionPolicyEvents')
SOURCE_RUN = PROJECT_ROOT/'artifacts/checkpoints'/MODALITY/SOURCE_RUN_ID
RAW_PATH = PROJECT_ROOT/'data/raw/lift_ph_raw.hdf5'
PACKED_ROOT = PROJECT_ROOT/'data/processed/rgb_policy/d50'
assert SOURCE_RUN.is_dir() and RAW_PATH.is_file(), 'Existing selected training artifacts and raw dataset are required.'
assert len(PILOT['environment_seeds']) == len(PILOT['policy_seeds']) == 10
assert len(set(PILOT['environment_seeds']))==10
assert set(PILOT['environment_seeds']).isdisjoint(range(1000,1050))
assert set(PILOT['policy_seeds']).isdisjoint(range(1000,1050))
assert PILOT['execution_horizon']==8 and PILOT['video_stride']==2
assert PILOT['max_control_steps']>0 and PILOT['success_hold_steps']>0
assert all(s not in [42,43,44,20260927] for s in PILOT['environment_seeds']+PILOT['policy_seeds'])

def json_default(v):
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.bool_): return bool(v)
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, Path): return str(v)
    raise TypeError(type(v).__name__)
def canonical_hash(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),default=json_default).encode()).hexdigest()
def file_hash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def atomic_json(p,v):
    p=Path(p)
    assert not p.resolve().is_relative_to(SOURCE_RUN.resolve()), 'Historical source training run is read-only.'
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
    try:
        tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=json_default),encoding='utf8')
        os.replace(tmp,p)
    finally: tmp.unlink(missing_ok=True)
def write_csv(p,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(p,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def ensure_dependencies():
    # Match the rendering stack that generated Notebook 02's verified RGB inputs.
    if 'mujoco' in sys.modules or 'robosuite' in sys.modules:
        import mujoco, robosuite
        assert mujoco.__version__=='3.3.0' and robosuite.__version__=='1.5.1', 'Use a fresh runtime; simulator already imported with another version.'
    specs=[]
    for package,version in [('mujoco','3.3.0'),('robosuite','1.5.1')]:
        try: installed=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: installed=None
        if installed!=version:
            specs.append('mujoco==3.3.0' if package=='mujoco' else 'git+https://github.com/ARISE-Initiative/robosuite.git@v1.5.1')
    for module,package in [('h5py','h5py'),('matplotlib','matplotlib'),('imageio','imageio'),('imageio_ffmpeg','imageio-ffmpeg'),('PIL','pillow')]:
        if importlib.util.find_spec(module) is None: specs.append(package)
    if specs: subprocess.check_call([sys.executable,'-m','pip','install','-q',*specs])
    importlib.invalidate_caches()
ensure_dependencies()
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import h5py, mujoco, robosuite
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image, ImageDraw
assert torch.cuda.is_available(), 'Select T4 GPU and Run All.'
assert mujoco.__version__=='3.3.0' and robosuite.__version__=='1.5.1'
DEVICE=torch.device('cuda')
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
RUNTIME={'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,
    'mujoco':mujoco.__version__,'robosuite':robosuite.__version__,'gpu':torch.cuda.get_device_name(0),
    'cuda':torch.version.cuda,'numpy':np.__version__}
try:
    RUNTIME['nvidia_driver']=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()
except Exception:RUNTIME['nvidia_driver']='unavailable'
print('Mode:',MODE,'| no training | Runtime:',RUNTIME)

assert EVENT_RENDER_POLICY in ['strict','exploratory']


# MODULE 02_artifacts.py
assert file_hash(RAW_PATH)==EXPECTED_RAW_SHA256, 'Raw data hash changed.'
source_record=json.loads((SOURCE_RUN/'run_config.json').read_text())
assert source_record['run_id']==SOURCE_RUN_ID
assert source_record['config_hash']==EXPECTED_CONFIG_HASH
assert source_record['data_fingerprint']==EXPECTED_DATA_FINGERPRINT
CFG=source_record['config']
NORMALIZATION=json.loads((SOURCE_RUN/'normalization.json').read_text())
EXPECTED_SCHEMA={'rgb':4,'events':5}[MODALITY]
config_identity={'code':source_record['code_version'],'code_sha256':source_record['code_sha256'],
    'config':CFG,'data':EXPECTED_DATA_FINGERPRINT}
if MODALITY=='events':
    assert source_record['modality']=='events', 'Selected source is not an Events training run.'
    assert source_record['paired_sample_fingerprint']==PAIRED_SAMPLE_FINGERPRINT, 'Paired sample identity changed.'
    normalization_hash=canonical_hash(NORMALIZATION)
    assert source_record['normalization_hash']==normalization_hash, 'Events normalization hash changed.'
    config_identity['normalization']=normalization_hash
assert canonical_hash(config_identity)==EXPECTED_CONFIG_HASH, 'Training config identity does not match the selected modality protocol.'
assert CFG['unet_dims']==[128,256,512] and CFG['checkpoint_schema']==EXPECTED_SCHEMA
assert CFG['state_per_observation']==18 and CFG['prediction_horizon']==16 and CFG['observation_horizon']==2
assert CFG['action_dim']==7 and CFG['crop_size']==76 and CFG['diffusion_steps']==100 and CFG['max_beta']==.999
if MODALITY=='events' and MODE=='paired_pilot':
    assert isinstance(PAIRED_RGB_CONTRACT,dict), 'Missing RGB training contract.'
    matched_keys=['seed','split_seed','policy_lr','encoder_lr','weight_decay','adam_betas','adam_eps',
        'micro_batch','accumulation','warmup_steps','lr_schedule_steps','min_lr_ratio','diffusion_steps',
        'max_beta','unet_dims','time_embed_dim','unet_kernel','group_norm_groups','crop_size',
        'prediction_horizon','observation_horizon','execution_horizon','action_dim','state_per_observation',
        'state_contract','packed_state_indices','bn_policy','ema_max_decay','ema_power','ema_inv_gamma']
    for key in matched_keys:
        assert CFG[key]==PAIRED_RGB_CONTRACT['config'][key], f'RGB/Event downstream recipe mismatch: {key}'
    for key in ['action_mean','action_std','proprio_mean','proprio_std','train_demos']:
        assert canonical_hash(NORMALIZATION[key])==canonical_hash(PAIRED_RGB_CONTRACT['normalization'][key]), f'Paired normalization mismatch: {key}'
checkpoint_path=SOURCE_RUN/CHECKPOINT_RELATIVE
assert file_hash(checkpoint_path)==EXPECTED_CHECKPOINT_SHA256, 'Exact selected best checkpoint unavailable or changed.'
# Only load the user's own verified historical checkpoint. No remote pickle downloads.
saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
assert saved['run_id']==SOURCE_RUN_ID and saved['config_hash']==EXPECTED_CONFIG_HASH
assert saved['data_fingerprint']==EXPECTED_DATA_FINGERPRINT and saved['global_step']==1500
assert saved['schema']==EXPECTED_SCHEMA and canonical_hash(saved['normalization'])==canonical_hash(NORMALIZATION)
assert canonical_hash(saved['config'])==canonical_hash(CFG)
EMA_WEIGHTS=saved['ema'];del saved
SPLIT=json.loads((PROJECT_ROOT/'configs/training/d50_rgb_event_train_val.json').read_text())
train_ids=SPLIT['train_demos'];val_ids=SPLIT['validation_demos']
assert len(train_ids)==40 and len(val_ids)==10 and set(train_ids).isdisjoint(val_ids)
assert NORMALIZATION['train_demos']==train_ids
historical_audit=json.loads((SOURCE_RUN/'evidence/data_audit.json').read_text())
assert historical_audit['raw_hash']==EXPECTED_RAW_SHA256, 'Historical raw identity changed.'
paired_identity=canonical_hash({'raw':historical_audit['raw_hash'],'split':historical_audit['split'],
    'audit':historical_audit['per_demo'],'excluded':historical_audit['excluded']})
if MODALITY=='events':
    assert paired_identity==PAIRED_SAMPLE_FINGERPRINT==historical_audit['paired_sample_fingerprint']
    event_identity=canonical_hash({'paired_sample_fingerprint':paired_identity,'modality':'events',
        'event_audit':historical_audit['event_audit'],'representation':CFG['event_channels']})
    assert event_identity==EXPECTED_DATA_FINGERPRINT==historical_audit['event_data_fingerprint'], 'Events data identity changed.'
else:
    assert paired_identity==EXPECTED_DATA_FINGERPRINT, 'RGB data identity changed.'
assert canonical_hash(SPLIT)==canonical_hash(historical_audit['split'])
expected_packed={r['demo_id']:r['packed_sha256'] for r in historical_audit['per_demo']}
bounds=[]
for demo in train_ids:
    path=PACKED_ROOT/demo/'rgb_policy_samples.npz'
    manifest=json.loads((PACKED_ROOT/demo/'manifest.json').read_text())
    assert manifest['demo_id']==demo and file_hash(path)==manifest['sha256']==expected_packed[demo]
    with np.load(path,allow_pickle=False) as z: bounds.append(z['action_targets'].reshape(-1,7))
train_actions=np.concatenate(bounds)
RAW_LO=train_actions.min(0);RAW_HI=train_actions.max(0)
del bounds,train_actions
ACT_MEAN=np.asarray(NORMALIZATION['action_mean'],dtype=np.float32)
ACT_STD=np.asarray(NORMALIZATION['action_std'],dtype=np.float32)
PROP_MEAN=np.asarray(NORMALIZATION['proprio_mean'],dtype=np.float32)
PROP_STD=np.asarray(NORMALIZATION['proprio_std'],dtype=np.float32)
assert ACT_STD.shape==(7,) and PROP_STD.shape==(36,) and (ACT_STD>0).all() and (PROP_STD>0).all()
SCENE_DEMO=sorted(train_ids)[0]  # training-only calibration states; distinct from fixed historical XML template
with h5py.File(RAW_PATH,'r') as f:
    ENV_META=json.loads(f['data'].attrs['env_args'])
    ORIGINAL_XML=f[f'data/{SCENE_TEMPLATE_DEMO}'].attrs['model_file']
    CALIBRATION_STATES=np.asarray(f[f'data/{SCENE_DEMO}/states'])
    CALIBRATION_ACTIONS=np.asarray(f[f'data/{SCENE_DEMO}/actions'])
if isinstance(ORIGINAL_XML,bytes): ORIGINAL_XML=ORIGINAL_XML.decode()
# Embed recorded scene assets locally without changing raw files.
import xml.etree.ElementTree as ET
root=ET.fromstring(ORIGINAL_XML)
asset_root=Path(robosuite.__file__).resolve().parent/'models/assets'
for element in root.iter():
    name=element.get('file')
    if name:
        marker='/robosuite/models/assets/'
        if marker in name: name=str(asset_root/name.split(marker,1)[1])
        assert Path(name).is_file(), f'Unresolved XML asset: {name}'
        element.set('file',name)
SCENE_XML=ET.tostring(root,encoding='unicode')
assert ENV_META['env_name']=='Lift' and ENV_META['env_kwargs']['robots']==['Panda']
assert ENV_META['env_kwargs']['control_freq']==20 and ENV_META['env_kwargs']['lite_physics'] is False
assert ENV_META['env_version']=='1.5.0', 'Unexpected source environment; review before using this pilot.'
ctrl=ENV_META['env_kwargs']['controller_configs']
assert ctrl['type']=='BASIC' and set(ctrl['body_parts'])=={'right'}
arm=ctrl['body_parts']['right']
assert arm['type']=='OSC_POSE' and arm['control_delta'] is True
assert arm['input_ref_frame']=='world' and arm['impedance_mode']=='fixed'
np.testing.assert_allclose(arm['output_max'],[.05,.05,.05,.5,.5,.5])
assert arm['input_min']==-1 and arm['input_max']==1 and arm['gripper']['type']=='GRIP'
if MODALITY=='events':
    EVENT_OPTIONS['device']=json.loads((PROJECT_ROOT/'data/processed/events'/SCENE_DEMO/'manifest.json').read_text())['v2e']['device']
PROTOCOL={'mode':MODE,'repair_options':REPAIR_OPTIONS if MODE=='renderer_repair' else None,'event_render_policy':EVENT_RENDER_POLICY,'modality':MODALITY,'event_options':dict(EVENT_OPTIONS) if MODALITY=='events' else None,
    'version':CODE_VERSION,'code_sha256':CODE_SHA256,'pilot':PILOT,'source_run_id':SOURCE_RUN_ID,
    'checkpoint_sha256':EXPECTED_CHECKPOINT_SHA256,'config_hash':EXPECTED_CONFIG_HASH,
    'data_fingerprint':EXPECTED_DATA_FINGERPRINT,'scene_demo':SCENE_TEMPLATE_DEMO,'calibration_demo':SCENE_DEMO,
    'source_scene_xml_sha256':hashlib.sha256(ORIGINAL_XML.encode()).hexdigest(),
    'environment_metadata':ENV_META,'runtime':RUNTIME,'raw_train_min':RAW_LO.tolist(),'raw_train_max':RAW_HI.tolist(),
    'sampler':'100_step_DDPM_eps_FP32_EMA_train_range_x0','initial_history':('empty previous/current Event windows at reset; then consecutive causal 50ms windows'
        if MODALITY=='events' else 'duplicate reset observation; then consecutive 20Hz observations'),
    'action_alignment':'targets[t:t+16]; execute indices 0:8 (not 1:9)',
    'reset_protocol':'fixed historical demo154 XML used in stored D50 rendering; fresh seeded task reset, no recorded demo initial state',
    'event_source_method':'causal endpoint interpolation at 200Hz in a separate renderer simulator; not measured physical events',
    'scope':'development pilot; not final test; fixed scene geometry limits scope'}
if MODE=='paired_pilot':
    PROTOCOL['paired_training_sources']=SOURCES
    PROTOCOL['pairing']='new RGB pilot on template154; exact initial states verified for Events; no historical score reuse'
PROTOCOL_HASH=canonical_hash(PROTOCOL)
PILOT_ID=PILOT['tag']+'_'+PROTOCOL_HASH[:12]
OUTPUT=PROJECT_ROOT/'results'/('renderer_repair' if MODE=='renderer_repair' else 'closed_loop_development')/PILOT_ID
OUTPUT.mkdir(parents=True,exist_ok=True)
if (OUTPUT/'protocol.json').exists(): assert canonical_hash(json.loads((OUTPUT/'protocol.json').read_text()))==PROTOCOL_HASH
else: atomic_json(OUTPUT/'protocol.json',PROTOCOL)
SESSION=OUTPUT/'sessions'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
SESSION.mkdir(parents=True,exist_ok=False)
atomic_json(SESSION/'runtime.json',RUNTIME)
(SESSION/'recorded_scene.xml').write_text(ORIGINAL_XML)
(SESSION/'controller_mapping.md').write_text('7 actions: right OSC_POSE position delta xyz, orientation delta xyz, then right gripper command. Pass raw normalized controller input directly to env.step. Controller applies its own output scaling; do not scale twice. Runtime gates verify mapping.\n')
print('Pilot:',PILOT_ID,'\nOutputs:',OUTPUT)

RGB_REFERENCE_ID='rgb_d50_best1500_bounded_v1_d22a5883bdaa'
RGB_REFERENCE_DIR=Path(globals().get('PAIRED_RGB_DIR',PROJECT_ROOT/'results/closed_loop_development'/RGB_REFERENCE_ID))
RGB_REFERENCE=None
if MODALITY=='events' and MODE in ['pilot','paired_pilot']:
    assert MODE!='paired_pilot' or 'PAIRED_RGB_DIR' in globals(), 'Paired mode requires the new RGB pilot, never the historical one.'
    assert (RGB_REFERENCE_DIR/'pilot_summary.json').is_file(), 'A completed RGB pilot on the same scene is required.'
    RGB_REFERENCE=json.loads((RGB_REFERENCE_DIR/'pilot_summary.json').read_text())
    reference_protocol=json.loads((RGB_REFERENCE_DIR/'protocol.json').read_text())
    RGB_REFERENCE_ID=RGB_REFERENCE['pilot_id']
    assert RGB_REFERENCE['modality']=='rgb'
    assert RGB_REFERENCE['preflight']=='PASS'
    assert RGB_REFERENCE['status']=='COMPLETED' and RGB_REFERENCE['completed_episodes']==10
    assert canonical_hash(reference_protocol)==RGB_REFERENCE['protocol_hash']
    assert RGB_REFERENCE['checkpoint_sha256']==SOURCES['rgb']['sha256']
    assert reference_protocol['source_scene_xml_sha256']==PROTOCOL['source_scene_xml_sha256']
    for key in ['environment_seeds','policy_seeds','max_control_steps','execution_horizon','success_hold_steps']:
        assert reference_protocol['pilot'][key]==PILOT[key]
    for episode in RGB_REFERENCE['episodes']:
        assert episode['status']=='COMPLETED'
        for name,digest in episode['artifacts'].items():
            path=RGB_REFERENCE_DIR/name
            assert path.resolve().is_relative_to(RGB_REFERENCE_DIR.resolve())
            assert path.is_file() and file_hash(path)==digest, 'Historical RGB pilot artifact is incomplete.'
    atomic_json(SESSION/'paired_rgb_reference.json',{'pilot_id':RGB_REFERENCE_ID,'status':'VERIFIED',
      'protocol_hash':RGB_REFERENCE['protocol_hash'],'checkpoint_sha256':RGB_REFERENCE['checkpoint_sha256'],
      'any_success':RGB_REFERENCE['successes_any'],'sustained_success':RGB_REFERENCE['successes_sustained'],
      'scope':'new template154 RGB pilot reused within this paired evaluation' if MODE=='paired_pilot' else 'explicit RGB reference verified'})

def verify_paired_initial_state(seed,initial):
    if MODALITY!='events':return
    episode=next(e for e in RGB_REFERENCE['episodes'] if e['environment_seed']==seed)
    name=next(n for n in episode['artifacts'] if n.endswith('initial_state.npy'))
    reference=np.load(RGB_REFERENCE_DIR/name,allow_pickle=False)
    np.testing.assert_allclose(initial,reference,atol=1e-9,rtol=0,
      err_msg='Events reset differs from paired RGB reset. Episode blocked.')


# MODULE 03_model.py
# Custom project architecture retained; NOT an exact official Diffusion Policy reproduction.
RESNET_FEATURE_DIM = 512
PROPRIO_DIM = 2 * CFG['state_per_observation']
ACTION_DIM = CFG['action_dim']
UNET_DIMS = CFG['unet_dims']
TIME_EMBED_DIM = CFG['time_embed_dim']
UNET_KERNEL = CFG['unet_kernel']
GROUP_NORM_GROUPS = CFG['group_norm_groups']

def freeze_resnet_except_layer4(backbone):
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for parameter in backbone.layer4.parameters():
        parameter.requires_grad = True
    return backbone


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(
            weights=None
        )
        backbone.fc = nn.Identity()
        freeze_resnet_except_layer4(backbone)
        self.backbone = backbone
        self.output_dim = RESNET_FEATURE_DIM
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, x):
        features = self.backbone(x)
        return features.flatten(1)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half = self.dim // 2
        device = timesteps.device
        exponent = (
            -math.log(10000.0)
            * torch.arange(half, device=device).float()
            / max(half - 1, 1)
        )
        embedding = (
            timesteps.float().unsqueeze(1)
            * torch.exp(exponent).unsqueeze(0)
        )
        embedding = torch.cat(
            [torch.sin(embedding), torch.cos(embedding)],
            dim=1,
        )
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class FiLMResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, kernel_size=5, groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.film = nn.Linear(cond_dim, 2 * out_ch)
        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        scale, shift = self.film(cond).chunk(2, dim=1)
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = F.silu(h)
        return h + self.skip(x)


class ConditionalUNet1D(nn.Module):
    def __init__(self, action_dim, cond_dim=256):
        super().__init__()
        c1, c2, c3 = UNET_DIMS

        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(TIME_EMBED_DIM),
            nn.Linear(TIME_EMBED_DIM, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.input_proj = nn.Conv1d(action_dim, c1, 1)
        self.down1 = FiLMResBlock1D(c1, c1, cond_dim, UNET_KERNEL, GROUP_NORM_GROUPS)
        self.downsample1 = nn.Conv1d(c1, c2, 3, stride=2, padding=1)
        self.down2 = FiLMResBlock1D(c2, c2, cond_dim, UNET_KERNEL, GROUP_NORM_GROUPS)
        self.downsample2 = nn.Conv1d(c2, c3, 3, stride=2, padding=1)
        self.down3 = FiLMResBlock1D(c3, c3, cond_dim, UNET_KERNEL, GROUP_NORM_GROUPS)
        self.mid = FiLMResBlock1D(c3, c3, cond_dim, UNET_KERNEL, GROUP_NORM_GROUPS)
        self.up2 = nn.Conv1d(c3 + c2, c2, 3, padding=1)
        self.up2_block = FiLMResBlock1D(c2, c2, cond_dim, UNET_KERNEL, GROUP_NORM_GROUPS)
        self.up1 = nn.Conv1d(c2 + c1, c1, 3, padding=1)
        self.up1_block = FiLMResBlock1D(c1, c1, cond_dim, UNET_KERNEL, GROUP_NORM_GROUPS)
        self.out_norm = nn.GroupNorm(GROUP_NORM_GROUPS, c1)
        self.out = nn.Conv1d(c1, action_dim, 1)

    def forward(self, noisy_actions, condition, timesteps):
        cond = condition + self.time_embed(timesteps)
        x = self.input_proj(noisy_actions.transpose(1, 2))
        skip1 = self.down1(x, cond)
        x = self.downsample1(skip1)
        skip2 = self.down2(x, cond)
        x = self.downsample2(skip2)
        x = self.down3(x, cond)
        x = self.mid(x, cond)
        x = F.interpolate(x, size=skip2.shape[-1], mode='nearest')
        x = self.up2(torch.cat([x, skip2], dim=1))
        x = self.up2_block(x, cond)
        x = F.interpolate(x, size=skip1.shape[-1], mode='nearest')
        x = self.up1(torch.cat([x, skip1], dim=1))
        x = self.up1_block(x, cond)
        x = F.silu(self.out_norm(x))
        x = self.out(x)
        return x.transpose(1, 2)


class RGBConditionedDiffusionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.agentview_encoder = ResNet18Encoder()
        self.wrist_encoder = ResNet18Encoder()
        fusion_in = 4 * RESNET_FEATURE_DIM + PROPRIO_DIM
        self.condition = nn.Sequential(
            nn.Linear(fusion_in, 1024),
            nn.SiLU(),
            nn.Linear(1024, 256),
        )
        self.unet = ConditionalUNet1D(ACTION_DIM, cond_dim=256)

    def encode_observation(self, batch):
        agent = batch['agentview']
        wrist = batch['robot0_eye_in_hand']
        batch_size = agent.shape[0]

        agent_flat = agent.reshape(
            batch_size * 2, 3, agent.shape[-2], agent.shape[-1]
        )
        wrist_flat = wrist.reshape(
            batch_size * 2, 3, wrist.shape[-2], wrist.shape[-1]
        )

        agent_features = self.agentview_encoder(agent_flat).reshape(
            batch_size, 2, -1
        )
        wrist_features = self.wrist_encoder(wrist_flat).reshape(
            batch_size, 2, -1
        )

        fused = torch.cat(
            [
                agent_features[:, 0], agent_features[:, 1],
                wrist_features[:, 0], wrist_features[:, 1],
                batch['proprio'],
            ],
            dim=1,
        )
        return self.condition(fused)

    def predict_noise(self, batch, noisy_actions, timesteps):
        condition = self.encode_observation(batch)
        return self.unet(noisy_actions, condition, timesteps)


if MODE in ['pilot','paired_pilot']:
    MODEL=RGBConditionedDiffusionPolicy().to(DEVICE).eval().requires_grad_(False)
    MODEL.load_state_dict(EMA_WEIGHTS,strict=True)
    del EMA_WEIGHTS
    assert all(not m.training for m in MODEL.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm))
    print('Loaded immutable best EMA, step 1500. No optimizer constructed.')
else:
    MODEL=None
    del EMA_WEIGHTS
    print('Renderer repair mode: policy inference is disabled; checkpoint identity verified only.')


# MODULE 04_policy_and_environment.py
CAMERAS=['agentview','robot0_eye_in_hand']
ROBOT_JOINTS=[f'robot0_joint{i}' for i in range(1,8)]+['gripper0_right_finger_joint1','gripper0_right_finger_joint2']

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def make_env(seed):
    seed_all(int(seed))
    kwargs=copy.deepcopy(ENV_META['env_kwargs'])
    kwargs.update(has_renderer=False,has_offscreen_renderer=True,use_camera_obs=False,
        camera_names=CAMERAS,camera_heights=84,camera_widths=84,
        horizon=PILOT['max_control_steps'],ignore_done=True,hard_reset=False)
    # Source control_delta=True has the same meaning as v1.5 input_type='delta'.
    kwargs['controller_configs']['body_parts']['right']['input_type']='delta'
    from robosuite.environments.manipulation.lift import Lift
    if 'seed' in inspect.signature(Lift.__init__).parameters: kwargs['seed']=int(seed)
    env=robosuite.make(ENV_META['env_name'],**kwargs)
    try:
        env.reset()
        env.reset_from_xml_string(SCENE_XML)
        # Keep the training scene's geometry/cameras, but RANDOMIZE a new initial condition.
        env.hard_reset=False
        env.deterministic_reset=False
        env.reset()
        assert not env.deterministic_reset
        return env
    except BaseException:
        env.close();raise

def verify_environment(env):
    assert len(env.robots)==1 and env.action_dim==7
    assert env.control_freq==20
    assert env.sim.model.nq==16 and env.sim.model.nv==15
    qpos_ids=[];qvel_ids=[]
    for name in ROBOT_JOINTS:
        j=env.sim.model.joint_name2id(name)
        qpos_ids.append(int(env.sim.model.jnt_qposadr[j]));qvel_ids.append(int(env.sim.model.jnt_dofadr[j]))
    assert qpos_ids==list(range(9)) and qvel_ids==list(range(9)), 'Robot-only state layout differs from training.'
    for cam in CAMERAS: assert env.sim.model.camera_name2id(cam)>=0
    robot=env.robots[0]
    assert list(robot.arms)==['right']
    composite=robot.composite_controller
    parts=getattr(composite,'part_controllers',None)
    if parts is None: parts=getattr(robot,'part_controllers',None)
    assert parts is not None and 'right' in parts, 'Cannot verify active OSC controller.'
    controller=parts['right']
    assert 'OperationalSpace' in type(controller).__name__ and controller.control_dim==6
    assert controller.input_type=='delta' and controller.input_ref_frame=='world'
    assert controller.impedance_mode=='fixed'
    assert len(robot.gripper)==1 and robot.gripper['right'].dof==1
    np.testing.assert_allclose(controller.input_min,-np.ones(6))
    np.testing.assert_allclose(controller.input_max,np.ones(6))
    np.testing.assert_allclose(controller.output_max,[.05,.05,.05,.5,.5,.5])
    np.testing.assert_allclose(controller.output_min,[-.05,-.05,-.05,-.5,-.5,-.5])
    lo,hi=map(np.asarray,env.action_spec)
    np.testing.assert_allclose(lo,-np.ones(7));np.testing.assert_allclose(hi,np.ones(7))
    assert np.all(RAW_LO>=lo-1e-6) and np.all(RAW_HI<=hi+1e-6)
    splits=getattr(composite,'_action_split_indexes',getattr(robot,'_action_split_indexes',{}))
    if splits:
        assert 'right' in splits and int(splits['right'][0])==0 and int(splits['right'][1]) in [6,7]
        for key,span in splits.items():
            if 'gripper' in key: assert tuple(span)==(6,7)
    return {'status':'PASS','action_order':['dx','dy','dz','drotation_x','drotation_y','drotation_z','gripper'],
        'controller':type(controller).__name__,'input_type':controller.input_type,'reference_frame':controller.input_ref_frame,
        'action_spec_low':lo,'action_spec_high':hi,'action_splits':splits,'qpos_ids':qpos_ids,'qvel_ids':qvel_ids,
        'source_env_version':ENV_META['env_version'],'runtime_env_version':robosuite.__version__}

def get_observation(env):
    # Identical vertical orientation and image geometry to Notebook 02/04.
    result={cam:np.ascontiguousarray(env.sim.render(height=84,width=84,camera_name=cam)[::-1]) for cam in CAMERAS}
    for cam in CAMERAS: assert result[cam].shape==(84,84,3) and result[cam].dtype==np.uint8
    result['proprio']=np.r_[np.asarray(env.sim.data.qpos)[:9],np.asarray(env.sim.data.qvel)[:9]].astype(np.float32)
    assert result['proprio'].shape==(18,) and np.isfinite(result['proprio']).all()
    return result

def preprocess_observations(previous,current):
    out={};c=CFG['crop_size'];offset=(84-c)//2
    for cam in CAMERAS:
        if MODALITY=='rgb':
            images=np.stack([previous[cam],current[cam]])[:,offset:offset+c,offset:offset+c].astype(np.float32)/255.
            images=(images-np.asarray(NORMALIZATION['image_mean'],dtype=np.float32))/np.asarray(NORMALIZATION['image_std'],dtype=np.float32)
            out[cam]=np.ascontiguousarray(images.transpose(0,3,1,2))
        else:
            events=np.stack([previous[cam],current[cam]])[:,:,offset:offset+c,offset:offset+c].astype(np.float32)
            assert events.shape==(2,3,c,c) and np.isfinite(events).all() and (events>=0).all()
            stats=NORMALIZATION['event_pixel_zscore'][cam]
            mean=np.asarray(stats['mean'],dtype=np.float32)[None,:,None,None]
            std=np.asarray(stats['std'],dtype=np.float32)[None,:,None,None]
            assert mean.shape==std.shape==(1,3,1,1) and (std>0).all()
            out[cam]=np.ascontiguousarray((events-mean)/std)
    out['proprio']=((np.r_[previous['proprio'],current['proprio']]-PROP_MEAN)/PROP_STD).astype(np.float32)
    return out

def cosine_betas(n):
    def abar(t):return math.cos((t+.008)/1.008*math.pi/2)**2
    return np.asarray([min(1-abar((i+1)/n)/abar(i/n),.999) for i in range(n)],dtype=np.float64)
BETAS=torch.tensor(cosine_betas(100),dtype=torch.float32,device=DEVICE)
ALPHAS=1-BETAS;ABAR=ALPHAS.cumprod(0);PREV=torch.cat([torch.ones(1,device=DEVICE),ABAR[:-1]])
VARIANCE=(BETAS*(1-PREV)/(1-ABAR)).clamp_min(1e-20)
LOWER=torch.tensor((RAW_LO-ACT_MEAN)/ACT_STD,device=DEVICE)
UPPER=torch.tensor((RAW_HI-ACT_MEAN)/ACT_STD,device=DEVICE)
MEAN_T=torch.tensor(ACT_MEAN,device=DEVICE);STD_T=torch.tensor(ACT_STD,device=DEVICE)

@torch.inference_mode()
def predict_plan(previous,current,generator):
    batch={k:torch.from_numpy(v).unsqueeze(0).to(DEVICE) for k,v in preprocess_observations(previous,current).items()}
    condition=MODEL.encode_observation(batch)  # once per plan, reused for all 100 DDPM steps
    x=torch.randn((1,16,7),generator=generator,device=DEVICE)
    projected=[]
    for t in range(99,-1,-1):
        ts=torch.full((1,),t,device=DEVICE,dtype=torch.long)
        eps=MODEL.unet(x,condition,ts)
        x0=(x-(1-ABAR[t]).sqrt()*eps)/ABAR[t].sqrt()
        projected.append(((x0<LOWER)|(x0>UPPER)).float().mean())
        x0=torch.maximum(LOWER,torch.minimum(UPPER,x0))
        if t==0: x=x0
        else:
            x=(BETAS[t]*PREV[t].sqrt()/(1-ABAR[t]))*x0+((1-PREV[t])*ALPHAS[t].sqrt()/(1-ABAR[t]))*x
            x=x+VARIANCE[t].sqrt()*torch.randn(x.shape,generator=generator,device=DEVICE)
    actions=(x[0]*STD_T+MEAN_T).cpu().numpy()
    assert actions.shape==(16,7) and np.isfinite(actions).all(), 'Invalid sampled action plan.'
    assert np.all(actions>=RAW_LO-1e-5) and np.all(actions<=RAW_HI+1e-5)
    return actions,torch.stack(projected).cpu().numpy()

def first_actions(plan):
    # Dataset targets are actions[t:t+16], so prediction[0] is action t.
    assert plan.shape==(16,7)
    return plan[:PILOT['execution_horizon']]

def video_frame(env,seed,step,success,stream=None):
    r=PILOT['video_resolution']
    views=[np.ascontiguousarray(env.sim.render(height=r,width=r,camera_name=cam)[::-1]) for cam in CAMERAS]
    pixels=np.concatenate(views,axis=1)
    if MODALITY=='events':
        displays=[]
        for cam in CAMERAS:
            a=stream.current[cam] if stream is not None else np.zeros((3,84,84),dtype=np.float32)
            # Fixed display scales: red ON, blue OFF, green time surface. Display only, never model input.
            rgb=np.stack([np.clip(a[0]/3,0,1),np.clip(a[2],0,1),np.clip(a[1]/3,0,1)],axis=-1)
            displays.append(np.asarray(Image.fromarray((rgb*255).astype(np.uint8)).resize((r,r))))
        pixels=np.concatenate([pixels,np.concatenate(displays,axis=1)],axis=0)
    frame=Image.fromarray(pixels)
    draw=ImageDraw.Draw(frame);draw.rectangle((0,0,2*r,23),fill='black')
    draw.text((5,6),f'{MODALITY.upper()} DEV seed {seed} | step {step} | Lift={success} | agentview / wrist',fill='white')
    if MODALITY=='events':
        draw.rectangle((0,r,2*r,r+23),fill='black')
        draw.text((5,r+6),'EVENTS: ON red / OFF blue / time green (display only)',fill='white')
    return np.asarray(frame)

def model_hash():
    h=hashlib.sha256()
    for name,value in MODEL.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
MODEL_HASH_BEFORE=model_hash() if MODE in ['pilot','paired_pilot'] else None


def graphics_identity():
    # Diagnostic only: never alter render flags, scene geometry or physics for a faster score.
    report={'mujoco_gl':os.environ.get('MUJOCO_GL'),'gpu':RUNTIME.get('gpu'),
        'nvidia_driver':RUNTIME.get('nvidia_driver'),'scope':'current offscreen OpenGL context; not a speedup claim'}
    try:
        from OpenGL import GL
        for name,constant in [('vendor',GL.GL_VENDOR),('renderer',GL.GL_RENDERER),('version',GL.GL_VERSION)]:
            value=GL.glGetString(constant)
            report[name]=value.decode(errors='replace') if isinstance(value,bytes) else str(value)
    except Exception as exc:report['query_error']=str(exc)
    return report


# MODULE 04a_event_stream.py
# Events use the exact pinned V2E emulator and the Notebook03 half-open window contract.
# A separate simulator is used for interpolated rendering: live physics is never restored backwards.
EventEmulator=None
cv2=None
EVENT_DEVICE=None

def ensure_v2e():
    if MODALITY!='events':return None
    global EventEmulator, cv2, EVENT_DEVICE
    folder=PROJECT_ROOT/'data/processed/events'/SCENE_DEMO
    manifest=json.loads((folder/'manifest.json').read_text())
    settings=manifest['v2e']
    assert settings['resolved_commit']==V2E_COMMIT and settings['emulator_seed']==42
    assert settings['pos_thres']==settings['neg_thres']==.2
    assert all(settings[k]==0 for k in ['sigma_thres','cutoff_hz','leak_rate_hz','shot_noise_rate_hz','refractory_period_s'])
    EVENT_DEVICE=settings['device']
    assert EVENT_DEVICE in ['cpu','cuda'], 'Unsupported historical V2E device.'
    repo=Path('/content/v2e-notebook07-'+V2E_COMMIT[:12])
    if not repo.exists():subprocess.check_call(['git','clone','https://github.com/SensorsINI/v2e.git',str(repo)])
    subprocess.check_call(['git','-C',str(repo),'checkout','--quiet',V2E_COMMIT])
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==V2E_COMMIT
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip(), 'Pinned V2E source has local modifications.'
    requirements={'cv2':'opencv-python-headless','argcomplete':'argcomplete','engineering_notation':'engineering-notation',
      'screeninfo':'screeninfo','easygui':'easygui','plyer':'plyer','dv_processing':'dv-processing','scipy':'scipy','tqdm':'tqdm'}
    missing=[package for module,package in requirements.items() if importlib.util.find_spec(module) is None]
    if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
    sys.path.insert(0,str(repo));importlib.invalidate_caches()
    import cv2
    from v2ecore.emulator import EventEmulator
    assert Path(inspect.getfile(EventEmulator)).resolve().is_relative_to(repo.resolve()), 'Another V2E emulator is already imported. Use a fresh runtime.'
    return {'commit':V2E_COMMIT,'device':EVENT_DEVICE,'opencv':cv2.__version__,'settings':settings}

def new_emulator():
    # V2E seeds process-global RNGs in some versions. Keep evaluation RNGs separate.
    py=random.getstate();nr=np.random.get_state();tr=torch.get_rng_state();cr=torch.cuda.get_rng_state_all()
    try:
        return EventEmulator(pos_thres=.2,neg_thres=.2,sigma_thres=0.,cutoff_hz=0.,leak_rate_hz=0.,
          shot_noise_rate_hz=0.,refractory_period_s=0.,seed=42,output_folder=None,
          output_width=84,output_height=84,device=EVENT_DEVICE)
    finally:
        random.setstate(py);np.random.set_state(nr);torch.set_rng_state(tr);torch.cuda.set_rng_state_all(cr)

def build_event_tensor(events,window_start,window_end):
    tensor=np.zeros((3,84,84),dtype=np.float32)
    selected=events[(events[:,0]>=window_start)&(events[:,0]<window_end)]
    if len(selected)==0:return tensor
    xs=selected[:,1].astype(np.int64);ys=selected[:,2].astype(np.int64);polarity=selected[:,3]
    assert ((xs>=0)&(xs<84)&(ys>=0)&(ys<84)).all() and np.isin(polarity,[-1,1]).all()
    pos=polarity>0;neg=polarity<0
    positive=np.zeros((84,84),dtype=np.float32);negative=positive.copy()
    np.add.at(positive,(ys[pos],xs[pos]),1.);np.add.at(negative,(ys[neg],xs[neg]),1.)
    tensor[0]=np.log1p(positive);tensor[1]=np.log1p(negative)
    latest=np.full((84,84),-np.inf,dtype=np.float64)
    np.maximum.at(latest,(ys,xs),selected[:,0]);valid=np.isfinite(latest)
    tensor[2,valid]=np.exp(-(window_end-latest[valid])/EVENT_OPTIONS['tau_s'])
    assert np.isfinite(tensor).all()
    return tensor

def slerp_quaternion(q0,q1,alpha):
    q0=np.asarray(q0,dtype=np.float64);q1=np.asarray(q1,dtype=np.float64)
    q0=q0/np.linalg.norm(q0);q1=q1/np.linalg.norm(q1)
    dot=float(np.dot(q0,q1))
    if dot<0:q1=-q1;dot=-dot
    dot=np.clip(dot,-1.,1.)
    if dot>.9995:
        q=q0+alpha*(q1-q0);return q/np.linalg.norm(q)
    theta=np.arccos(dot)
    return np.sin((1-alpha)*theta)/np.sin(theta)*q0+np.sin(alpha*theta)/np.sin(theta)*q1

def interpolated_state(a,b,alpha):
    assert a.shape==b.shape==(32,) and 0<=alpha<=1
    s=a+alpha*(b-a)
    s[13:17]=slerp_quaternion(a[13:17],b[13:17],alpha)
    return s

class EventStream:
    def __init__(self):
        self.emulators={c:new_emulator() for c in CAMERAS}
        self.pending={c:np.empty((0,4),dtype=np.float32) for c in CAMERAS}
        self.last_frame_time=None;self.frame_count=0;self.generated={c:0 for c in CAMERAS}
        self.current={c:np.zeros((3,84,84),dtype=np.float32) for c in CAMERAS}
        self.previous={c:v.copy() for c,v in self.current.items()}
        self.last_counts={c:0 for c in CAMERAS};self.last_emulation_seconds=0.
    def feed(self,images,timestamp):
        assert self.last_frame_time is None or timestamp>self.last_frame_time, 'Duplicate/out-of-order source frame.'
        if self.last_frame_time is not None:np.testing.assert_allclose(timestamp-self.last_frame_time,.005,atol=1e-9,rtol=0)
        started=time.perf_counter()
        for cam in CAMERAS:
            image=images[cam];assert image.shape==(84,84,3) and image.dtype==np.uint8
            events=self.emulators[cam].generate_events(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),float(timestamp))
            if events is not None and len(events):
                events=np.asarray(events,dtype=np.float32)
                assert events.ndim==2 and events.shape[1]==4 and np.isfinite(events).all()
                assert np.all(np.diff(events[:,0])>=-1e-7), 'Non-monotone event times.'
                self.pending[cam]=np.concatenate([self.pending[cam],events]);self.generated[cam]+=len(events)
        self.last_emulation_seconds=time.perf_counter()-started
        self.last_frame_time=float(timestamp);self.frame_count+=1
    def windows(self,end):
        assert self.last_frame_time is not None and abs(self.last_frame_time-end)<1e-8
        for cam in CAMERAS:
            events=self.pending[cam]
            self.previous[cam]=build_event_tensor(events,end-.10,end-.05)
            self.current[cam]=build_event_tensor(events,end-.05,end)
            self.last_counts[cam]=int(np.sum((events[:,0]>=end-.05)&(events[:,0]<end)))
            # Retain two windows and the exact-boundary events needed by the next window.
            self.pending[cam]=events[events[:,0]>=end-.10]
        return self.previous,self.current
    def close(self):
        for emulator in self.emulators.values():
            cleanup=getattr(emulator,'cleanup',None)
            if callable(cleanup):cleanup()
        self.emulators.clear()

def raw_proprio(env):
    return np.r_[np.asarray(env.sim.data.qpos)[:9],np.asarray(env.sim.data.qvel)[:9]].astype(np.float32)

def policy_observation(env,stream=None):
    if MODALITY=='rgb':return get_observation(env)
    assert stream is not None
    return {**{c:stream.current[c].copy() for c in CAMERAS},'proprio':raw_proprio(env)}

def advance_event_stream(renderer_env,stream,a,b,step):
    # Only completed control endpoints are used. No look-ahead into future actions/states.
    np.testing.assert_allclose(b[0]-a[0],.05,atol=1e-7,rtol=0)
    render_seconds=0.;emulation_seconds=0.
    for sub in range(1,11):
        start=time.perf_counter()
        state=interpolated_state(a,b,sub/10)
        renderer_env.sim.set_state_from_flattened(state);renderer_env.sim.forward()
        images=get_observation(renderer_env)
        render_seconds+=time.perf_counter()-start
        timestamp=((step-1)*10+sub)/200.
        stream.feed(images,timestamp);emulation_seconds+=stream.last_emulation_seconds
    stream.last_render_seconds=render_seconds;stream.last_v2e_seconds=emulation_seconds
    stream.windows(step/20.)
    return {c:stream.last_counts[c] for c in CAMERAS}

def event_preview(previous,current,path):
    fig,axes=plt.subplots(4,3,figsize=(9,9))
    for row,(cam,suffix) in enumerate([(c,s) for c in CAMERAS for s in ['previous','current']]):
        a=previous[cam] if suffix=='previous' else current[cam]
        for ch,title in enumerate(['positive log1p','negative log1p','time surface']):
            axes[row,ch].imshow(a[ch],cmap='magma');axes[row,ch].set_title(f'{cam} {suffix}\n{title}',fontsize=8);axes[row,ch].axis('off')
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)

try:
    V2E_RUNTIME=ensure_v2e()
except Exception as exc:
    atomic_json(SESSION/'preflight_failure.json',{'status':'FAILED','stage':'event_dependency_initialization',
      'error':str(exc),'traceback':traceback.format_exc(),'scored_episodes_started':False})
    raise
if MODALITY=='events':
    assert NORMALIZATION['event_channels']==['positive','negative','time_surface']
    assert NORMALIZATION['event_transform']=='already_log1p_polarity_and_exponential_time_surface'
    assert CFG['event_window_ms']==50 and CFG['time_surface_tau_s']==.05
    # Verify all selected processed event artifacts against the training audit.
    source_devices={}
    for item in historical_audit['event_audit']:
        demo=item['demo_id'];folder=PROJECT_ROOT/'data/processed/events'/demo
        mpath=folder/'manifest.json';assert file_hash(mpath)==item['event_manifest_sha256']
        m=json.loads(mpath.read_text())
        assert m['v2e']['resolved_commit']==V2E_COMMIT and m['v2e']['device'] in ['cpu','cuda']
        source_devices[demo]=m['v2e']['device']
        record=next(r for r in historical_audit['per_demo'] if r['demo_id']==demo)
        for cam in CAMERAS:assert file_hash(folder/f'{cam}_events.npz')==record['event_array_hashes'][cam]
        assert file_hash(folder/'timestamps.npy')==record['event_times_sha256']
    V2E_RUNTIME['historical_source_devices']=source_devices
    V2E_RUNTIME['device_selection_scope']='match calibration-demo source device; all source device identities recorded'
    atomic_json(SESSION/'event_runtime.json',V2E_RUNTIME)
print('Observation modality:',MODALITY,'; online Events source:',V2E_RUNTIME)


# MODULE 04b_event_gates.py
def event_render_decision(rows, policy):
    assert policy in ['strict','exploratory']
    assert rows and all(np.isfinite(r[k]) for r in rows for k in
        ['mae','relative_rms_error','fraction_changed','normalized_crop_rms'])
    failed=[r for r in rows if not (r['mae']<=.02 and r['fraction_changed']<=.02 and r['relative_rms_error']<=.25)]
    return {'status':'PASS' if not failed else 'DOMAIN_SHIFT',
        'compatibility_pass':not failed,'failed_rows':failed,
        'scoring_allowed':not failed or policy=='exploratory','policy':policy,
        'scope':'Exploratory mode permits behavioral diagnosis, not source equivalence or a controlled modality claim.'}

def validate_event_inputs():
    """Pinned-V2E streaming tensors vs saved Notebook03 source windows; regenerated-render check."""
    assert MODALITY=='events'
    source=PROJECT_ROOT/'data/processed/high_rate_replay'/SCENE_DEMO
    source_manifest=json.loads((source/'manifest.json').read_text())
    times_path=source/'timestamps.npy'
    assert file_hash(times_path)==source_manifest['sha256']['timestamps']
    times=np.load(times_path,allow_pickle=False)
    frames={}
    for cam in CAMERAS:
        path=source/f'{cam}_rgb.npz'
        assert file_hash(path)==source_manifest['sha256'][f'{cam}_rgb']
        with np.load(path,allow_pickle=False) as z:
            assert z.files==['frames'] or 'frames' in z.files
            frames[cam]=z['frames'].copy()
        assert frames[cam].shape==(len(times),84,84,3)
    folder=PROJECT_ROOT/'data/processed/events'/SCENE_DEMO
    ends=np.load(folder/'timestamps.npy',allow_pickle=False)[:EVENT_OPTIONS['calibration_windows']]
    assert len(ends)==EVENT_OPTIONS['calibration_windows'] and ends[0]>=.1-1e-9
    reference={}
    for cam in CAMERAS:
        with np.load(folder/f'{cam}_events.npz',allow_pickle=False) as z:
            reference[cam]={s:z[s][:len(ends)].copy() for s in ['previous','current']}
    stream=None;renderer=None;rows=[];render_rows=[]
    try:
        stream=EventStream();next_window=0
        for i,timestamp in enumerate(times):
            stream.feed({c:frames[c][i] for c in CAMERAS},float(timestamp))
            if next_window<len(ends) and abs(timestamp-ends[next_window])<1e-9:
                previous,current=stream.windows(float(ends[next_window]))
                for cam in CAMERAS:
                    for suffix,values in [('previous',previous),('current',current)]:
                        expected=reference[cam][suffix][next_window]
                        delta=np.abs(values[cam]-expected)
                        rows.append({'window':next_window,'timestamp':float(timestamp),'camera':cam,
                          'which':suffix,'max_abs_error':float(delta.max()),'mean_abs_error':float(delta.mean())})
                        np.testing.assert_allclose(values[cam],expected,atol=EVENT_OPTIONS['source_tensor_atol'],rtol=0,
                          err_msg='Pinned V2E streaming differs from stored training tensors. Do not bypass this gate.')
                next_window+=1
                if next_window==len(ends):break
        assert next_window==len(ends)
        event_preview(previous,current,SESSION/'offline_online_event_windows.png')
        # Numerical preprocessing must exactly agree with Notebook05 center-crop normalization.
        packed_path=PACKED_ROOT/SCENE_DEMO/'rgb_policy_samples.npz'
        assert file_hash(packed_path)==expected_packed[SCENE_DEMO]
        with np.load(packed_path,allow_pickle=False) as z:
            sample=int(np.flatnonzero(np.isclose(z['control_timestamps'],ends[-1],atol=1e-9))[0])
            old={'proprio':z['proprio_previous'][sample,CFG['packed_state_indices']],**previous}
            cur={'proprio':z['proprio_current'][sample,CFG['packed_state_indices']],**current}
        actual=preprocess_observations(old,cur)
        for cam in CAMERAS:
            stats=NORMALIZATION['event_pixel_zscore'][cam]
            expected=(np.stack([reference[cam]['previous'][-1],reference[cam]['current'][-1]])[:,:,4:80,4:80]
              -np.asarray(stats['mean'],dtype=np.float32)[None,:,None,None])/np.asarray(stats['std'],dtype=np.float32)[None,:,None,None]
            np.testing.assert_allclose(actual[cam],expected,atol=1e-5,rtol=0)
        atomic_json(SESSION/'event_source_parity.json',{'status':'PASS','scope':'six early training windows, two cameras, previous/current, source RGB -> pinned V2E -> tensors -> normalization','rows':rows})
        stream.close();stream=EventStream();renderer=make_env(1996);verify_environment(renderer)
        # Render a continuous prefix from recorded endpoints, not pixels interpolated between frames.
        max_frame=int(round(ends[-1]*200))
        tensor_rows=[];regenerated_index=0
        for i in range(max_frame+1):
            timestamp=float(times[i])
            # Exact Notebook02 integer interpolation indexing; fixed 200Hz source timestamps.
            state=repair_state_at_frame(CALIBRATION_STATES,i,times,'notebook02_index')
            renderer.sim.set_state_from_flattened(state);renderer.sim.forward()
            regenerated=get_observation(renderer)
            for cam in CAMERAS:
                error=np.abs(regenerated[cam].astype(float)-frames[cam][i].astype(float))
                render_rows.append({'frame':i,'camera':cam,'mae_uint8':float(error.mean())})
                assert error.mean()<=EVENT_OPTIONS['render_mae_limit'], 'Intermediate renderer differs from Notebook02.'
            stream.feed(regenerated,timestamp)
            if regenerated_index<len(ends) and abs(timestamp-ends[regenerated_index])<1e-9:
                regen_previous,regen_current=stream.windows(float(ends[regenerated_index]))
                tensor_rows.extend(repair_tensor_rows(regen_previous,regen_current,reference,regenerated_index,NORMALIZATION))
                regenerated_index+=1
        assert regenerated_index==len(ends)
        event_preview(regen_previous,regen_current,SESSION/'regenerated_event_windows.png')
        write_csv(SESSION/'high_rate_render_calibration.csv',render_rows)
        decision=event_render_decision(tensor_rows,EVENT_RENDER_POLICY)
        atomic_json(SESSION/'event_render_parity.json',{**decision,'render_mae_limit_uint8':EVENT_OPTIONS['render_mae_limit'],
          'event_relative_rms_limit':.25,'event_channel_mae_limit':.02,'event_changed_pixel_fraction_limit':.02,
          'limits_scope':'predeclared compatibility limits; not model-performance thresholds',
          'max_render_mae_uint8':max(r['mae_uint8'] for r in render_rows),'event_rows':tensor_rows})
        assert decision['scoring_allowed'], (
            'Renderer domain shift detected; all channel metrics saved in event_render_parity.json. '
            'Strict mode blocks scoring. Exploratory mode is a separate, documented diagnostic protocol.')
        if not decision['compatibility_pass']:
            print('WARNING: renderer DOMAIN_SHIFT; exploratory trials do not establish modality-only differences.')
        return old,cur
    finally:
        if stream is not None:stream.close()
        if renderer is not None:renderer.close()

def validate_physics_parity():
    """Extra rendering/emulation must not perturb the authoritative environment trajectory."""
    actions=np.asarray([[.1,0,0,0,0,0,-.2],[0,.1,0,0,0,0,.2],[0,0,.1,0,0,0,0]],dtype=np.float32)
    base=None;live=None;renderer=None;stream=None
    try:
        base=make_env(1995);verify_environment(base);baseline=[base.sim.get_state().flatten().copy()]
        for a in actions:base.step(a);baseline.append(base.sim.get_state().flatten().copy())
        base.close();base=None
        live=make_env(1995);verify_environment(live)
        renderer=make_env(1995);verify_environment(renderer)
        np.testing.assert_allclose(live.sim.get_state().flatten(),baseline[0],atol=1e-9,rtol=0)
        np.testing.assert_allclose(renderer.sim.get_state().flatten(),baseline[0],atol=1e-9,rtol=0)
        stream=EventStream();stream.feed(get_observation(live),0.)
        errors=[]
        for step,a in enumerate(actions,1):
            before=live.sim.get_state().flatten().copy();live.step(a);after=live.sim.get_state().flatten().copy()
            advance_event_stream(renderer,stream,before,after,step)
            np.testing.assert_array_equal(live.sim.get_state().flatten(),after)
            np.testing.assert_allclose(after,baseline[step],atol=EVENT_OPTIONS['physics_parity_atol'],rtol=0)
            errors.append(float(np.max(np.abs(after-baseline[step]))))
        atomic_json(SESSION/'event_physics_parity.json',{'status':'PASS','steps':len(actions),
          'max_state_errors':errors,'rendering_environment':'separate simulator; never rewind scored physics',
          'atol':EVENT_OPTIONS['physics_parity_atol']})
        return {'status':'PASS','max_state_error':max(errors)}
    finally:
        for env in [base,live,renderer]:
            if env is not None:env.close()
        if stream is not None:stream.close()


# MODULE 04c_renderer_repair.py
# Renderer diagnosis uses training trajectories only. No policy trials or optimizer updates.
# Never change acceptance thresholds based on candidate performance.
def repair_xml(raw_xml):
    if isinstance(raw_xml,bytes):raw_xml=raw_xml.decode()
    root=ET.fromstring(raw_xml)
    for element in root.iter():
        name=element.get('file')
        if name:
            marker='/robosuite/models/assets/'
            if marker in name:name=str(asset_root/name.split(marker,1)[1])
            assert Path(name).is_file(), f'Unresolved scene asset: {name}'
            element.set('file',name)
    return ET.tostring(root,encoding='unicode')

def repair_state_at_frame(states,frame,timestamps,method):
    if method=='notebook02_index':
        control=frame//10;alpha=(frame%10)/10.
    else:
        timestamp=float(timestamps[frame])
        control=min(int(np.floor(timestamp*20+1e-10)),len(states)-1)
        alpha=float(np.clip((timestamp-states[control,0])/.05,0,1))
    if control>=len(states)-1:return states[-1].copy()
    return interpolated_state(states[control],states[control+1],alpha)

def repair_model_snapshot(env):
    result={'nq':int(env.sim.model.nq),'nv':int(env.sim.model.nv),'arrays':{}}
    # Visual and physical differences are separate evidence, not automatically considered harmless.
    for name in ['cam_pos','cam_quat','cam_fovy','geom_size','geom_pos','geom_rgba','geom_matid',
                 'mat_rgba','mat_texid','tex_data','mesh_vert','mesh_face','geom_bodyid','cam_bodyid',
                 'body_pos','body_mass','body_inertia','geom_friction']:
        try:
            a=np.asarray(getattr(env.sim.model,name)).copy()
            result['arrays'][name]={'shape':list(a.shape),'sha256':hashlib.sha256(a.tobytes()).hexdigest(),
                                    'values':a.tolist() if a.size<10000 else None}
        except AttributeError:result['arrays'][name]={'available':False}
    return result

def repair_make_environment(scene_xml,construction):
    if construction=='notebook02':
        # Exact constructor visible in uploaded Notebook02; original script reused demo_154 XML.
        env=robosuite.make('Lift',robots='Panda',has_renderer=False,has_offscreen_renderer=True,
            use_camera_obs=True,camera_names=CAMERAS,camera_heights=84,camera_widths=84,
            control_freq=20,reward_shaping=False)
        try:
            env.reset();env.reset_from_xml_string(scene_xml);env.sim.reset();env.sim.forward()
            return env
        except BaseException:env.close();raise
    # Same constructor/reset sequence as the historical Notebook07, with an explicit scene XML.
    seed_all(1996);kwargs=copy.deepcopy(ENV_META['env_kwargs'])
    kwargs.update(has_renderer=False,has_offscreen_renderer=True,use_camera_obs=False,
        camera_names=CAMERAS,camera_heights=84,camera_widths=84,
        horizon=300,ignore_done=True,hard_reset=False)
    kwargs['controller_configs']['body_parts']['right']['input_type']='delta'
    from robosuite.environments.manipulation.lift import Lift
    if 'seed' in inspect.signature(Lift.__init__).parameters:kwargs['seed']=1996
    env=robosuite.make('Lift',**kwargs)
    try:
        env.reset();env.reset_from_xml_string(scene_xml)
        env.hard_reset=False;env.deterministic_reset=False;env.reset()
        return env
    except BaseException:env.close();raise

class RepairCachedRenderer:
    """One persistent official MuJoCo Renderer; benchmark only, not production auto-selection."""
    def __init__(self,env):
        self.env=env
        model=getattr(env.sim.model,'_model',None);data=getattr(env.sim.data,'_data',None)
        assert model is not None and data is not None, 'Native model/data handles unavailable; candidate unsupported.'
        self.data=data
        context=getattr(env.sim,'_render_context_offscreen',None)
        self.option=getattr(context,'vopt',None)
        assert self.option is not None, 'Wrapper scene options unavailable; refuse unmatched native defaults.'
        self.renderer=mujoco.Renderer(model,height=84,width=84)
    def images(self):
        result={}
        for camera in CAMERAS:
            self.renderer.update_scene(self.data,camera=camera,scene_option=self.option)
            # Official Renderer returns top-origin RGB; wrapper outputs are vertically flipped explicitly.
            result[camera]=np.ascontiguousarray(self.renderer.render().copy())
        return result
    def close(self):self.renderer.close()

def repair_tensor_rows(previous,current,reference,index,normalization):
    rows=[]
    for camera in CAMERAS:
        for which,values in [('previous',previous),('current',current)]:
            a=values[camera];b=reference[camera][which][index]
            assert a.shape==b.shape==(3,84,84) and np.isfinite(a).all()
            for channel in range(3):
                delta=a[channel]-b[channel];error=np.abs(delta)
                rms=float(np.sqrt(np.mean(delta**2)));reference_rms=float(np.sqrt(np.mean(b[channel]**2)))
                normalized_rms=float(np.sqrt(np.mean(delta[4:80,4:80]**2))/normalization['event_pixel_zscore'][camera]['std'][channel])
                rows.append({'window':int(index),'camera':camera,'which':which,'channel':channel,
                    'mae':float(error.mean()),'max_abs':float(error.max()),'rms':rms,'reference_rms':reference_rms,
                    'relative_rms_error':rms/max(reference_rms,1e-3),'fraction_changed':float(np.mean(error>1e-5)),
                    'changed_pixels':int(np.sum(error>1e-5)),'normalized_crop_rms':normalized_rms})
    return rows

def repair_load_training_case(demo):
    assert demo in train_ids, 'Renderer diagnosis must use training trajectories only.'
    source=PROJECT_ROOT/'data/processed/high_rate_replay'/demo
    manifest=json.loads((source/'manifest.json').read_text())
    assert manifest['demo_id']==demo
    times_path=source/'timestamps.npy';assert file_hash(times_path)==manifest['sha256']['timestamps']
    times=np.load(times_path,allow_pickle=False);frames={}
    for camera in CAMERAS:
        path=source/f'{camera}_rgb.npz';assert file_hash(path)==manifest['sha256'][f'{camera}_rgb']
        with np.load(path,allow_pickle=False) as z:frames[camera]=z['frames'].copy()
        assert frames[camera].shape==(len(times),84,84,3)
    folder=PROJECT_ROOT/'data/processed/events'/demo
    event_manifest=json.loads((folder/'manifest.json').read_text())
    assert event_manifest['v2e']['device']==EVENT_DEVICE, 'Additional training case uses a different historical V2E device.'
    ends=np.load(folder/'timestamps.npy',allow_pickle=False)[:REPAIR_OPTIONS['calibration_windows']]
    assert len(ends)==REPAIR_OPTIONS['calibration_windows']
    reference={}
    for camera in CAMERAS:
        with np.load(folder/f'{camera}_events.npz',allow_pickle=False) as z:
            reference[camera]={which:z[which][:len(ends)].copy() for which in ['previous','current']}
    with h5py.File(RAW_PATH,'r') as f:
        states=np.asarray(f[f'data/{demo}/states'],dtype=np.float64)
        raw_xml=f[f'data/{demo}'].attrs['model_file']
    max_frame=int(round(float(ends[-1])*200))
    assert np.isclose(times[0],0,atol=1e-9) and max_frame<len(times)
    return {'demo':demo,'states':states,'times':times,'frames':frames,'ends':ends,
        'reference':reference,'xml':repair_xml(raw_xml),'raw_xml':raw_xml.decode() if isinstance(raw_xml,bytes) else raw_xml,
        'max_frame':max_frame,'manifest':manifest}

def repair_check_source(case):
    stream=None;rows=[]
    try:
        stream=EventStream();index=0
        for i in range(case['max_frame']+1):
            timestamp=float(case['times'][i]);stream.feed({c:case['frames'][c][i] for c in CAMERAS},timestamp)
            if index<len(case['ends']) and abs(timestamp-case['ends'][index])<1e-9:
                previous,current=stream.windows(float(case['ends'][index]))
                for camera in CAMERAS:
                    for which,values in [('previous',previous),('current',current)]:
                        error=np.abs(values[camera]-case['reference'][camera][which][index])
                        rows.append({'window':index,'camera':camera,'which':which,'max_abs':float(error.max())})
                index+=1
        assert index==len(case['ends'])
        report={'status':'PASS' if all(r['max_abs']<=EVENT_OPTIONS['source_tensor_atol'] for r in rows) else 'FAILED',
            'demo':case['demo'],'rows':rows}
        atomic_json(SESSION/f"source_parity_{case['demo']}.json",report)
        assert report['status']=='PASS', 'Saved RGB -> V2E parity failed; renderer comparison cannot isolate the source.'
        return report
    finally:
        if stream is not None:stream.close()

def repair_candidate(case,candidate,template_xml):
    folder=SESSION/'candidates'/case['demo']/candidate['id'];folder.mkdir(parents=True,exist_ok=True)
    env=None;stream=None;cached=None;pixels=[];event_rows=[];render_seconds=0.;v2e_seconds=0.;frame_times=[]
    record={'candidate':candidate,'demo':case['demo'],'status':'RUNNING','started_utc':now()}
    try:
        xml=template_xml if candidate['scene']=='template' else case['xml']
        env=repair_make_environment(xml,candidate['construction'])
        atomic_json(folder/'model_snapshot.json',repair_model_snapshot(env))
        # Native candidate reads the wrapper options only after an explicit wrapper warmup.
        get_observation(env)
        if candidate['backend']=='cached_native':cached=RepairCachedRenderer(env)
        stream=EventStream();index=0
        for i in range(case['max_frame']+1):
            started=time.perf_counter()
            state=repair_state_at_frame(case['states'],i,case['times'],candidate['interpolation'])
            env.sim.set_state_from_flattened(state);env.sim.forward()
            np.testing.assert_allclose(env.sim.get_state().flatten(),state,atol=1e-9,rtol=0)
            images=cached.images() if cached is not None else get_observation(env)
            elapsed=time.perf_counter()-started;render_seconds+=elapsed;frame_times.append(elapsed)
            for camera in CAMERAS:
                difference=np.abs(images[camera].astype(np.float32)-case['frames'][camera][i].astype(np.float32))
                pixels.append({'frame':i,'camera':camera,'mae_uint8':float(difference.mean()),
                    'max_abs_uint8':float(difference.max()),'fraction_changed':float(np.mean(difference>0))})
                if i in [0,10,case['max_frame']]:
                    montage=np.concatenate([case['frames'][camera][i],images[camera]],axis=1)
                    Image.fromarray(montage).resize((672,336)).save(folder/f'{camera}_frame_{i:04d}.png')
            stream.feed(images,float(case['times'][i]));v2e_seconds+=stream.last_emulation_seconds
            if index<len(case['ends']) and abs(case['times'][i]-case['ends'][index])<1e-9:
                previous,current=stream.windows(float(case['ends'][index]))
                event_rows.extend(repair_tensor_rows(previous,current,case['reference'],index,NORMALIZATION));index+=1
        assert index==len(case['ends'])
        decision=event_render_decision(event_rows,'strict')
        pixel_pass=all(p['mae_uint8']<=EVENT_OPTIONS['render_mae_limit'] for p in pixels)
        record.update(status='PASS' if pixel_pass and decision['compatibility_pass'] else 'DOMAIN_SHIFT',
            pixel_pass=pixel_pass,event_decision=decision,max_pixel_mae=max(p['mae_uint8'] for p in pixels),
            max_normalized_crop_rms=max(p['normalized_crop_rms'] for p in event_rows),
            max_event_mae=max(p['mae'] for p in event_rows),render_seconds=render_seconds,v2e_seconds=v2e_seconds,
            frames_per_camera=len(frame_times),mean_render_pair_seconds=float(np.mean(frame_times)),
            median_render_pair_seconds=float(np.median(frame_times)),timing_scope='state restore + forward + two 84px renders; warmup excluded; one prefix, not rollout latency')
        event_preview(previous,current,folder/'last_event_windows.png')
        np.savez_compressed(folder/'last_event_tensors.npz',**{f'{c}_{w}':v[c] for w,v in [('previous',previous),('current',current)] for c in CAMERAS})
    except Exception as exc:
        record.update(status='FAILED',error=str(exc),traceback=traceback.format_exc())
    finally:
        for resource in [cached,stream,env]:
            if resource is not None:
                try:resource.close()
                except Exception as exc:record.setdefault('cleanup_warnings',[]).append(str(exc))
        write_csv(folder/'pixels.csv',pixels);write_csv(folder/'event_channels.csv',event_rows)
        record.update(ended_utc=now(),completed_render_frames=len(frame_times))
        atomic_json(folder/'candidate_summary.json',record)
    print(case['demo'],candidate['id'],record['status'],'pixel MAE:',record.get('max_pixel_mae'),
          'normalized RMS:',record.get('max_normalized_crop_rms'),'render pair sec:',record.get('median_render_pair_seconds'))
    return record

def repair_select_candidate(records):
    acceptable=[r for r in records if r['status']=='PASS']
    # Restore scientific compatibility first. Timing only breaks ties among compatible candidates.
    return min(acceptable,key=lambda r:(r['max_event_mae'],r['mean_render_pair_seconds'])) if acceptable else None

def run_renderer_repair():
    assert MODALITY=='events'
    with h5py.File(RAW_PATH,'r') as f:raw_template=f[f"data/{REPAIR_OPTIONS['historical_template_demo']}"].attrs['model_file']
    template=repair_xml(raw_template)
    template_text=raw_template.decode() if isinstance(raw_template,bytes) else raw_template
    (SESSION/'historical_template.xml').write_text(template_text)
    case=repair_load_training_case(SCENE_DEMO);repair_check_source(case)
    candidates=[
        {'id':'pilot_scene_legacy_time','scene':'per_demo','construction':'pilot','interpolation':'legacy_time','backend':'wrapper'},
        {'id':'pilot_scene_index','scene':'per_demo','construction':'pilot','interpolation':'notebook02_index','backend':'wrapper'},
        {'id':'per_demo_notebook02','scene':'per_demo','construction':'notebook02','interpolation':'notebook02_index','backend':'wrapper'},
        {'id':'template_notebook02','scene':'template','construction':'notebook02','interpolation':'notebook02_index','backend':'wrapper'},
        {'id':'template_pilot_reset','scene':'template','construction':'pilot','interpolation':'notebook02_index','backend':'wrapper'},
        {'id':'template_cached_native','scene':'template','construction':'notebook02','interpolation':'notebook02_index','backend':'cached_native'}]
    atomic_json(SESSION/'candidate_plan.json',{'candidates':candidates,'acceptance':'Original pixel and event thresholds unchanged; all six windows; training only.',
        'source_code_hypothesis':'Uploaded Notebook02 loads demo_154 XML once before D50 loop. Historical execution is not proven by static code alone.',
        'template_raw_xml_sha256':hashlib.sha256(template_text.encode()).hexdigest(),
        'per_demo_raw_xml_sha256':hashlib.sha256(case['raw_xml'].encode()).hexdigest()})
    records=[repair_candidate(case,c,template) for c in candidates]
    selected=repair_select_candidate(records);verification=[]
    if selected:
        # Independent training trajectories; never use held-out/final evaluation episodes.
        same_device=[d for d in train_ids if d!=SCENE_DEMO and V2E_RUNTIME['historical_source_devices'][d]==EVENT_DEVICE]
        verify_ids=list(dict.fromkeys([same_device[(len(same_device)-1)//2],same_device[-1]]))[:REPAIR_OPTIONS['verification_demos']] if same_device else []
        for demo in verify_ids:
            extra=repair_load_training_case(demo);repair_check_source(extra)
            verification.append(repair_candidate(extra,selected['candidate'],template))
    status='CANDIDATE_VERIFIED_ON_TRAIN_PREFIXES' if selected and len(verification)==REPAIR_OPTIONS['verification_demos'] and all(v['status']=='PASS' for v in verification) else 'UNRESOLVED'
    summary={'status':status,'mode':MODE,'utc':now(),'protocol_hash':PROTOCOL_HASH,'source_run_id':SOURCE_RUN_ID,
        'checkpoint_sha256':EXPECTED_CHECKPOINT_SHA256,'new_training_performed':False,'scored_episodes':0,
        'candidate_records':records,'selected_candidate':selected,'independent_training_verification':verification,
        'historical_events_pilot':{'id':'events_d50_best1500_bounded_v2_e34f7a52ade4','completed':10,'successes_any':2,'successes_sustained':1,'renderer_status':'DOMAIN_SHIFT'},
        'limitations':['A compatible training-prefix candidate is not a closed-loop rollout or a proven universal fix.',
            'Model geometry/physical differences must be reviewed before changing a rollout scene.',
            'If the scene/controller/runtime protocol changes, rerun a fresh paired RGB/Event development pilot.',
            'No raw/processed data or historical artifacts were changed. Cached backend timing is not a production speedup claim.'],
        'output_directory':str(OUTPUT)}
    legacy_path=SESSION/'candidates'/SCENE_DEMO/'pilot_scene_legacy_time'/'model_snapshot.json'
    model_differences={}
    if legacy_path.exists():
        legacy=json.loads(legacy_path.read_text())['arrays']
        for record in records:
            snapshot=SESSION/'candidates'/SCENE_DEMO/record['candidate']['id']/'model_snapshot.json'
            if snapshot.exists():
                arrays=json.loads(snapshot.read_text())['arrays']
                model_differences[record['candidate']['id']]=[name for name,values in arrays.items()
                    if values.get('sha256')!=legacy.get(name,{}).get('sha256')]
    summary['model_array_differences_against_legacy']=model_differences
    wrapper=next((r for r in records if r['candidate']['id']=='template_notebook02' and r['status']=='PASS'),None)
    native=next((r for r in records if r['candidate']['id']=='template_cached_native' and r['status']=='PASS'),None)
    summary['cached_renderer_benchmark']=None
    if wrapper and native:
        summary['cached_renderer_benchmark']={
            'median_pair_speedup_vs_matching_wrapper':wrapper['median_render_pair_seconds']/max(native['median_render_pair_seconds'],1e-12),
            'scope':'one identical training prefix; both candidates pass original compatibility gates; not deployed or a rollout speed claim'}
    atomic_json(OUTPUT/'renderer_repair_summary.json',summary)
    report=f"# Notebook07 renderer repair\n\nStatus: {status}\n\nNo training or scored episodes. Original thresholds retained.\n\n"
    report+='| Candidate | Status | Max pixel MAE | Max normalized event RMS | Render pair median (s) |\n|---|---|---:|---:|---:|\n'
    for r in records:report+=f"| {r['candidate']['id']} | {r['status']} | {r.get('max_pixel_mae')} | {r.get('max_normalized_crop_rms')} | {r.get('median_render_pair_seconds')} |\n"
    report+='\nReview model snapshots, image montages, all-window channel metrics and independent training-prefix results before adoption. No production renderer is changed automatically.\n'
    (OUTPUT/'renderer_repair_summary.md').write_text(report)
    return summary

def repair_bundle():
    path=OUTPUT/'renderer_repair_bundle.zip'
    temporary=Path('/content')/('renderer-repair-'+uuid.uuid4().hex+'.zip')
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(OUTPUT.rglob('*')):
                if p.is_file() and p.suffix!='.zip':z.write(p,str(p.relative_to(OUTPUT)))
        shutil.copyfile(temporary,path)
        assert file_hash(path)==file_hash(temporary)
    finally:temporary.unlink(missing_ok=True)
    print('Upload:',path)

if MODE=='renderer_repair':
    try:
        REPAIR_SUMMARY=run_renderer_repair()
        print('Renderer repair diagnostic ended:',REPAIR_SUMMARY['status'])
    except Exception as exc:
        atomic_json(OUTPUT/'renderer_repair_failure.json',{'status':'FAILED','error':str(exc),
            'traceback':traceback.format_exc(),'scored_episodes':0,'new_training_performed':False})
        print('Diagnosis failed; evidence retained. No scoring or training.')
    finally:repair_bundle()


# MODULE 05_preflight.py
def preflight():
    env=None
    try:
        env=make_env(1998)
        audit=verify_environment(env)
        get_observation(env)
        atomic_json(SESSION/'graphics_identity.json',graphics_identity())
        atomic_json(SESSION/'compiled_scene_snapshot.json',repair_model_snapshot(env))
        initial_a=env.sim.get_state().flatten().copy()
        assert not env._check_success(), 'Reset already successful; invalid pilot initial condition.'
        env.close();env=make_env(1998)
        np.testing.assert_allclose(env.sim.get_state().flatten(),initial_a,rtol=0,atol=1e-9)
        env.close();env=make_env(1999)
        initial_b=env.sim.get_state().flatten().copy()
        assert not np.allclose(initial_a,initial_b), 'Different seeds produced identical initial states.'
        # Pixel, state, proprioception and action alignment calibration against a packed training sample.
        with np.load(PACKED_ROOT/SCENE_DEMO/'rgb_policy_samples.npz',allow_pickle=False) as z:
            t=int(round(float(z['control_timestamps'][0])*20))
            assert 1<=t and t+16<=len(CALIBRATION_ACTIONS)
            np.testing.assert_allclose(z['action_targets'][0],CALIBRATION_ACTIONS[t:t+16],rtol=1e-6,atol=1e-6)
            observations=[];render_errors=[]
            for index,suffix in [(t-1,'previous'),(t,'current')]:
                env.sim.set_state_from_flattened(CALIBRATION_STATES[index]);env.sim.forward()
                np.testing.assert_allclose(env.sim.get_state().flatten(),CALIBRATION_STATES[index],rtol=0,atol=1e-9)
                obs=get_observation(env);observations.append(obs)
                packed_state=z['proprio_'+suffix][0]
                expected=packed_state[list(range(9))+list(range(16,25))]
                np.testing.assert_allclose(obs['proprio'],expected,atol=1e-6,rtol=1e-6)
                for cam in CAMERAS:
                    reference=z[cam+'_'+suffix][0]
                    mae=float(np.abs(obs[cam].astype(float)-reference.astype(float)).mean())
                    render_errors.append({'camera':cam,'observation':suffix,'mae_uint8':mae})
                    montage=Image.fromarray(np.concatenate([reference,obs[cam]],axis=1)).resize((672,336))
                    montage.save(SESSION/f'calibration_{cam}_{suffix}.png')
            atomic_json(SESSION/'render_calibration.json',render_errors)
            assert max(r['mae_uint8'] for r in render_errors)<=PILOT['render_match_mae_limit'], 'Rendered images do not match training inputs. Inspect calibration images; rollouts blocked.'
            normalized={'proprio':((np.r_[observations[0]['proprio'],observations[1]['proprio']]-PROP_MEAN)/PROP_STD).astype(np.float32)}
            packed_prop=np.r_[z['proprio_previous'][0,CFG['packed_state_indices']],z['proprio_current'][0,CFG['packed_state_indices']]]
            np.testing.assert_allclose(normalized['proprio'],(packed_prop-PROP_MEAN)/PROP_STD,atol=1e-5,rtol=1e-5)
        if MODALITY=='events':
            observations=validate_event_inputs()
            audit['event_source_parity']='PASS'
            render_report=json.loads((SESSION/'event_render_parity.json').read_text())
            audit['event_render_parity']=render_report['status']
            audit['source_equivalence_verified']=render_report['compatibility_pass']
            audit['event_physics_parity']=validate_physics_parity()
        if MODALITY=='events':
            paired=make_env(PILOT['environment_seeds'][0])
            try:
                verify_paired_initial_state(PILOT['environment_seeds'][0],paired.sim.get_state().flatten().copy())
            finally:paired.close()
            audit['paired_rgb_reset']='PASS'
        # Real GPU inference smoke: independent generator, no model updates or rollout-seed consumption.
        gen=torch.Generator(device=DEVICE).manual_seed(90415)
        plan,projection=predict_plan(*observations,gen)
        assert first_actions(plan).shape==(8,7)
        np.savez_compressed(SESSION/'preflight_plan.npz',actions=plan,projection_fraction=projection)
        env.close();env=make_env(1997)
        verify_environment(env)
        before=float(env.sim.data.time)
        probe=np.zeros(7,dtype=np.float32) # disposable environment; not one of the 10 policy trials
        env.step(probe)
        np.testing.assert_allclose(float(env.sim.data.time)-before,.05,atol=1e-7,rtol=0)
        assert np.isfinite(env.sim.get_state().flatten()).all()
        # Confirm video encoding actually works before spending resources on trials.
        probe_video=Path('/content')/('nb07-video-probe-'+uuid.uuid4().hex+'.mp4')
        with imageio.get_writer(str(probe_video),fps=10,codec='libx264',macro_block_size=1) as writer:
            for _ in range(2):writer.append_data(video_frame(env,1997,1,False))
        assert probe_video.stat().st_size>0
        shutil.copyfile(probe_video,SESSION/'video_probe.mp4');probe_video.unlink()
        audit.update(status=('QUALIFIED_DOMAIN_SHIFT' if audit.get('event_render_parity')=='DOMAIN_SHIFT' else 'PASS'),reset_reproducible=True,different_seeds_differ=True,render_errors=render_errors,
            temporal_alignment='PASS: t-1,t -> actions[t:t+16], execute first 8',
            simulated_control_dt=.05,fp32_policy_forward='PASS',video_encoding='PASS')
        return audit
    finally:
        if env is not None:env.close()

if MODE in ['pilot','paired_pilot']:
    try:
        PREFLIGHT=preflight()
        atomic_json(SESSION/'preflight.json',PREFLIGHT)
        print('Preflight:',PREFLIGHT['status'],'| renderer policy:',EVENT_RENDER_POLICY)
    except Exception as exc:
        atomic_json(SESSION/'preflight_failure.json',{'status':'FAILED','error':str(exc),'traceback':traceback.format_exc()})
        print('Preflight failed; no pilot episodes were run. Share:',SESSION)
        raise
else:
    print('Pilot preflight skipped: renderer-repair mode; no trials scheduled.')


# MODULE 06_rollouts.py
def verified_episode(seed):
    pointer=OUTPUT/f'episode_{seed}.json'
    if not pointer.exists():return None
    info=json.loads(pointer.read_text())
    assert info['protocol_hash']==PROTOCOL_HASH and info['environment_seed']==seed
    if info['status']!='COMPLETED':return None
    for name,digest in info['artifacts'].items():
        p=OUTPUT/name
        assert p.resolve().is_relative_to(OUTPUT.resolve())
        if not p.is_file() or file_hash(p)!=digest:
            print('Previous episode artifact incomplete/corrupt; new attempt will be preserved:',seed)
            return None
    return info

def run_episode(seed,policy_seed):
    attempt=OUTPUT/'episodes'/f'seed_{seed}'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True,exist_ok=False)
    local=Path('/content')/('nb07-'+uuid.uuid4().hex);local.mkdir()
    rows=[];plans=[];projection_rows=[];states=[];latencies=[];observed_stats=[];event_elapsed=0.
    env=None;renderer_env=None;stream=None;writer=None;any_success=False;consecutive=0;max_consecutive=0;first_success=None
    result={'protocol_hash':PROTOCOL_HASH,'source_run_id':SOURCE_RUN_ID,'checkpoint_step':1500,
        'environment_seed':int(seed),'policy_seed':int(policy_seed),'status':'RUNNING','started_utc':now(),
        'modality':MODALITY,'closed_loop':True,'scope':'development_fixed_scene_seeded_reset'}
    atomic_json(attempt/'status.json',result)
    wall=time.perf_counter();step=0
    try:
        env=make_env(seed);verify_environment(env)
        initial=env.sim.get_state().flatten().copy()
        assert not env._check_success(), 'Unexpected initial success.'
        verify_paired_initial_state(seed,initial)
        states.append(initial)
        np.save(local/'initial_state.npy',initial)
        if MODALITY=='events':
            renderer_env=make_env(seed);verify_environment(renderer_env)
            np.testing.assert_allclose(renderer_env.sim.get_state().flatten(),initial,atol=1e-9,rtol=0)
            stream=EventStream();stream.feed(get_observation(env),0.)
        current=policy_observation(env,stream);previous={k:v.copy() for k,v in current.items()}
        observed_stats=[];event_elapsed=0.
        atomic_json(local/'initial_observation.json',{'modality':MODALITY,'history':PROTOCOL['initial_history']})
        generator=torch.Generator(device=DEVICE).manual_seed(int(policy_seed))
        writer=imageio.get_writer(str(local/'rollout.mp4'),fps=20/PILOT['video_stride'],codec='libx264',quality=7,macro_block_size=1)
        writer.append_data(video_frame(env,seed,0,False,stream))
        Image.fromarray(video_frame(env,seed,0,False,stream)).save(local/'initial.png')
        low,high=map(np.asarray,env.action_spec)
        stop_reason='horizon'
        peak_height=-float('inf')
        while step<PILOT['max_control_steps']:
            torch.cuda.synchronize();t0=time.perf_counter()
            plan,projection=predict_plan(previous,current,generator)
            torch.cuda.synchronize();latencies.append(time.perf_counter()-t0)
            plans.append(plan);projection_rows.append(projection)
            for action_index,requested in enumerate(first_actions(plan)):
                if step>=PILOT['max_control_steps']:break
                assert np.isfinite(requested).all()
                assert np.all(requested>=low-1e-5) and np.all(requested<=high+1e-5)
                action=np.clip(requested,low,high) # only float-roundoff correction within 1e-5
                previous=current
                before_state=env.sim.get_state().flatten().copy()
                _,reward,done,info=env.step(action)
                step+=1
                state=env.sim.get_state().flatten().copy();assert np.isfinite(state).all()
                states.append(state)
                event_seconds=0.;counts={}
                if stream is not None:
                    start_event=time.perf_counter()
                    counts=advance_event_stream(renderer_env,stream,before_state,state,step)
                    event_seconds=time.perf_counter()-start_event;event_elapsed+=event_seconds
                    np.testing.assert_array_equal(env.sim.get_state().flatten(),state)
                current=policy_observation(env,stream)
                if stream is not None:
                    previous={**{c:stream.previous[c].copy() for c in CAMERAS},'proprio':previous['proprio']}
                    for cam in CAMERAS:
                        for ch in range(3):
                            stat=NORMALIZATION['event_pixel_zscore'][cam]
                            channel=current[cam][ch];z=(channel-stat['mean'][ch])/stat['std'][ch]
                            observed_stats.append({'step':step,'camera':cam,'channel':ch,
                              'active_fraction':float(np.mean(channel>0)),'mean_z':float(z.mean()),'std_z':float(z.std()),
                              'max_abs_z':float(np.max(np.abs(z))),'events_in_current_window':counts[cam]})
                success=bool(env._check_success())
                any_success=any_success or success
                if success and first_success is None:first_success=step
                consecutive=consecutive+1 if success else 0
                max_consecutive=max(max_consecutive,consecutive)
                cube_height=float(env.sim.data.body_xpos[env.cube_body_id][2])
                table_height=float(env.model.mujoco_arena.table_offset[2])
                # Object state is diagnostic output only; it is NEVER included in model inputs.
                row={'step':step,'simulation_elapsed_s':step/20,'plan_index':len(plans)-1,'plan_action_index':action_index,
                    'reward':float(reward),'success':success,'consecutive_success':consecutive,
                    'cube_height':cube_height,'cube_above_table':cube_height-table_height,
                    'plan_inference_seconds':latencies[-1],'event_pipeline_seconds':event_seconds,
                    'event_render_seconds':getattr(stream,'last_render_seconds',0.),
                    'v2e_seconds':getattr(stream,'last_v2e_seconds',0.),'roundoff_adjusted':bool(np.any(action!=requested))}
                row.update({f'action_{j}':float(v) for j,v in enumerate(action)});rows.append(row)
                if cube_height>peak_height:
                    peak_height=cube_height
                    # Save only the current peak thumbnail locally; final one is copied at episode end.
                    Image.fromarray(video_frame(env,seed,step,success,stream)).save(local/'peak.png')
                if step%PILOT['video_stride']==0:writer.append_data(video_frame(env,seed,step,success,stream))
                if consecutive>=PILOT['success_hold_steps']:
                    stop_reason='sustained_success';break
                if done:
                    stop_reason='environment_done';break
            if stop_reason!='horizon':break
        if step%PILOT['video_stride']:writer.append_data(video_frame(env,seed,step,any_success,stream))
        result.update(status='COMPLETED',stop_reason=stop_reason)
    except KeyboardInterrupt:
        result.update(status='ABORTED',stop_reason='user_interrupt')
    except Exception as exc:
        result.update(status='FAILED',stop_reason=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        if writer is not None:
            try:writer.close()
            except Exception as exc:result.update(status='FAILED',stop_reason='video_finalize',error=str(exc))
        if stream is not None:
            try:stream.close()
            except Exception as exc:result['event_close_warning']=str(exc)
        if renderer_env is not None:
            try:renderer_env.close()
            except Exception as exc:result['renderer_close_warning']=str(exc)
        if env is not None:
            try:env.close()
            except Exception as exc:result['close_warning']=str(exc)
    write_csv(local/'steps.csv',rows)
    if MODALITY=='events':write_csv(local/'event_input_stats.csv',observed_stats)
    np.savez_compressed(local/'trajectory.npz',states=np.asarray(states),plans=np.asarray(plans),
        projection_fraction=np.asarray(projection_rows),inference_seconds=np.asarray(latencies))
    result.update(ended_utc=now(),steps=step,simulation_seconds=step/20,any_success=any_success,
        first_success_step=first_success,sustained_success=max_consecutive>=PILOT['success_hold_steps'],
        max_consecutive_success_steps=max_consecutive,wall_seconds=time.perf_counter()-wall,
        event_pipeline_seconds=event_elapsed,event_frames_per_camera=stream.frame_count if stream is not None else 0,
        plans=len(plans),mean_inference_seconds=float(np.mean(latencies)) if latencies else None,
        max_cube_above_table=max((r['cube_above_table'] for r in rows),default=None))
    result['artifacts']={}
    for p in sorted(local.iterdir()):
        target=attempt/p.name
        shutil.copyfile(p,target)
        assert file_hash(target)==file_hash(p), 'Evidence copy checksum failed.'
        result['artifacts'][str(target.relative_to(OUTPUT))]=file_hash(target)
    if result['status']=='COMPLETED':
        assert any(n.endswith('rollout.mp4') for n in result['artifacts']), 'Missing episode video.'
    atomic_json(attempt/'status.json',result)
    # Pointer publication only after verified per-episode copies. Earlier attempt folders are retained.
    atomic_json(OUTPUT/f'episode_{seed}.json',result)
    shutil.rmtree(local)
    return result

if MODE in ['pilot','paired_pilot']:
    assert PREFLIGHT['status']=='PASS' or (EVENT_RENDER_POLICY=='exploratory' and PREFLIGHT['status']=='QUALIFIED_DOMAIN_SHIFT')
    RESULTS=[]
    for seed,policy_seed in zip(PILOT['environment_seeds'],PILOT['policy_seeds']):
        existing=verified_episode(seed)
        if existing is not None:
            print('Reusing verified completed episode:',seed)
            RESULTS.append(existing)
        else:
            print('Running development episode:',seed,'policy seed:',policy_seed)
            result=run_episode(seed,policy_seed);RESULTS.append(result)
            print('Episode',seed,':',result['status'],'success=',result['any_success'],
                'sustained=',result['sustained_success'],'steps=',result['steps'])
        atomic_json(OUTPUT/'pilot_progress.json',{'protocol_hash':PROTOCOL_HASH,'utc':now(),'episodes':RESULTS})
        if RESULTS[-1]['status']=='ABORTED':break
        if sum(r['status']=='FAILED' for r in RESULTS)>=3:
            print('Stopping after three execution failures. Review errors before another run.');break
    assert model_hash()==MODEL_HASH_BEFORE, 'Policy weights/buffers changed during evaluation.'
else:
    RESULTS=[]
    print('Rollouts skipped: renderer-repair mode.')


# MODULE 07_summary.py
def wilson_interval(k,n,z=1.959963984540054):
    if n==0:return None
    p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]
if MODE in ['pilot','paired_pilot']:
    completed=[r for r in RESULTS if r['status']=='COMPLETED']
    successes=sum(r['any_success'] for r in completed)
    sustained=sum(r['sustained_success'] for r in completed)
    SUMMARY={'pilot_id':PILOT_ID,'modality':MODALITY,'protocol_hash':PROTOCOL_HASH,'utc':now(),
        'status':'COMPLETED' if len(completed)==10 else 'PARTIAL',
        'source_run_id':SOURCE_RUN_ID,'checkpoint_step':1500,'checkpoint_sha256':EXPECTED_CHECKPOINT_SHA256,
        'sampler':'EMA FP32 DDPM100 train_range_x0','planned_episodes':10,'completed_episodes':len(completed),
        'failed_episodes':sum(r['status']=='FAILED' for r in RESULTS),'aborted_episodes':sum(r['status']=='ABORTED' for r in RESULTS),
        'successes_any':successes,'successes_sustained':sustained,
        'success_rate_completed':successes/len(completed) if completed else None,
        'descriptive_wilson_95pct':wilson_interval(successes,len(completed)),
        'interval_scope':'descriptive binomial interval over pilot initializations, conditional on one trained model/fixed scene; excludes training-seed/model-selection uncertainty',
        'primary_success':'any env._check_success during <=300 control steps',
        'secondary_success':'10 consecutive successful control steps; stop then',
        'preflight':PREFLIGHT['status'],'event_render_policy':EVENT_RENDER_POLICY,
        'source_equivalence_verified':PREFLIGHT.get('source_equivalence_verified',True),'weights_unchanged':True,'new_training_performed':False,
        'final_test_performed':False,'elapsed_session_seconds':time.perf_counter()-START_PERF,
        'output_directory':str(OUTPUT),'episodes':RESULTS,
        'limitations':['10 development trials only; final evaluation seeds untouched.',
            'Fixed historical demo154 scene geometry used in stored D50 rendering with fresh randomized placements; no recorded demonstration initial states.',
            'Runtime robosuite 1.5.1 vs recorded metadata 1.5.0, matching Notebook02 renderer stack; not exact original-paper reproduction.',
            PROTOCOL['initial_history'],
            'Empirical training action bounds constrain the sampler; this is a documented inference intervention.',
            'Rollout is simulated-time, not a real-time 20Hz latency certification.',
            'Partial/failed execution is not silently counted as task failure or removed from reporting.']}
    if not SUMMARY['source_equivalence_verified']:
        SUMMARY['limitations'].append('Known renderer domain shift. Exploratory results are not a controlled modality comparison.')
    PAIRED_ROWS=[]
    if MODALITY=='events':
        reference_by_seed={e['environment_seed']:e for e in RGB_REFERENCE['episodes']}
        for episode in RESULTS:
            r=reference_by_seed[episode['environment_seed']]
            PAIRED_ROWS.append({'environment_seed':episode['environment_seed'],'policy_seed':episode['policy_seed'],
              'rgb_status':r['status'],'event_status':episode['status'],
              'rgb_any_success':r['any_success'],'event_any_success':episode['any_success'] if episode['status']=='COMPLETED' else None,
              'rgb_sustained_success':r['sustained_success'],'event_sustained_success':episode['sustained_success'] if episode['status']=='COMPLETED' else None,
              'rgb_steps':r['steps'],'event_steps':episode['steps'],
              'rgb_wall_seconds':r['wall_seconds'],'event_wall_seconds':episode['wall_seconds']})
        paired_complete=[r for r in PAIRED_ROWS if r['event_status']=='COMPLETED']
        counts={name:sum(bool(r['rgb_any_success'])==a and bool(r['event_any_success'])==b for r in paired_complete)
          for name,a,b in [('both_success',True,True),('rgb_only',True,False),('events_only',False,True),('both_fail',False,False)]}
        SUMMARY['paired_rgb']={'pilot_id':RGB_REFERENCE_ID,'historical_result_reused':MODE!='paired_pilot','modality_only_comparison_valid':SUMMARY['source_equivalence_verified'],'counts':counts,
          'paired_completed':len(paired_complete),'rgb_successes_all_10':RGB_REFERENCE['successes_any'],
          'scope':'new fixed-template paired development comparison; identical initial states; natural reset histories differ; not final evaluation'}
        SUMMARY['events_pipeline']={'v2e':V2E_RUNTIME,'source_options':EVENT_OPTIONS,
          'method':'causal interpolation between completed 20Hz endpoints in a separate renderer; synthesized, not measured events',
          'initial_event_windows':'empty; no pre-reset or fabricated movement; first plan relies on robot state and empty visual history',
          'total_pipeline_seconds_completed':sum(r.get('event_pipeline_seconds',0.) for r in completed)}
        SUMMARY['limitations'].extend(['Events use interpolated endpoint rendering, not physically sampled 200Hz sensor frames.',
          'At reset Events history is empty; RGB duplicates the reset image. No warmup actions alter paired initial states.',
          'Six early training-window parity checks do not prove agreement for every future rollout state.'])
        write_csv(OUTPUT/'paired_rgb_events.csv',PAIRED_ROWS)
    atomic_json(OUTPUT/'pilot_summary.json',SUMMARY)
    flat=[{k:v for k,v in r.items() if k not in ['artifacts','traceback']} for r in RESULTS]
    write_csv(OUTPUT/'pilot_results.csv',flat)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for r in completed:
        csv_name=next(n for n in r['artifacts'] if n.endswith('steps.csv'))
        with open(OUTPUT/csv_name) as f:rows=list(csv.DictReader(f))
        axes[0].plot([float(x['simulation_elapsed_s']) for x in rows],[float(x['cube_above_table']) for x in rows],label=str(r['environment_seed']))
    axes[0].axhline(.04,color='black',ls='--',label='Lift success height threshold')
    axes[0].set(xlabel='Simulated seconds',ylabel='Cube center above table (m)',title='All completed development episodes')
    if completed:axes[0].legend(fontsize=7,ncol=3)
    axes[1].bar(['Any success','Sustained','Task failure','Execution error'],[successes,sustained,len(completed)-successes,SUMMARY['failed_episodes']])
    axes[1].set(ylabel='Episode count',title='Development only; not final evaluation',ylim=(0,10))
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUTPUT/'pilot_dashboard.png',dpi=170);fig.savefig(OUTPUT/'pilot_dashboard.pdf');plt.close(fig)
    report=f'''# Notebook 07 — {MODALITY.upper()} development rollout pilot

    Status: {SUMMARY['status']}. Completed {len(completed)}/10; any Lift success {successes}/{len(completed)}; sustained success {sustained}/{len(completed)}.
    Execution errors: {SUMMARY['failed_episodes']}; interrupted: {SUMMARY['aborted_episodes']}.
    Source: {SOURCE_RUN_ID}, best EMA at 1500. No new training.
    Preflight passed. Policy state unchanged. Final evaluation seeds 1000–1049 were not used.
    Output: {OUTPUT}

    This is a development result on a fixed scene with seeded randomized resets. Review both successful and failed videos, initial states, inference latency and controller logs before deciding whether to freeze the D50 recipe or make another controlled change. A successful pilot does not complete the RGB-vs-Events study, prove generalization, or justify extending training past the selected best checkpoint.

    If the pipeline and behavior are satisfactory, freeze this development configuration and plan matched RGB/Event experiments and D100/D200 expansion with equal optimizer-update budgets. If failures persist, classify approach/grasp/lift/drop and inspect timing/inputs before selecting another training intervention. Infrastructure errors must be fixed before interpreting a task success rate.
    '''
    if MODALITY=='events':
        report+=f'\nPaired RGB reference: {RGB_REFERENCE["successes_any"]}/10. Events: {successes}/{len(completed)} completed. Paired counts: {SUMMARY["paired_rgb"]["counts"]}. This is not final evaluation or a significance claim.\n'
        report+=f'\nRenderer compatibility: {PREFLIGHT["event_render_parity"]}; policy: {EVENT_RENDER_POLICY}. Source-stream, normalization, physics and paired reset gates remain mandatory.\n'
        if not PREFLIGHT['source_equivalence_verified']:
            report+='\nKNOWN RENDERER DOMAIN SHIFT: behavioral diagnosis only; RGB/Event differences cannot be attributed to modality alone.\n'
    (OUTPUT/'pilot_summary.md').write_text(report,encoding='utf8')
    base=f'''Notebook 07 update — development pilot only.
    PILOT_ID: {PILOT_ID}
    Status: {SUMMARY['status']}; completed {len(completed)}/10; any Lift success {successes}/{len(completed)}; sustained success {sustained}/{len(completed)}; execution errors {SUMMARY['failed_episodes']}.
    Source: {SOURCE_RUN_ID} best EMA at step 1500, with no additional training; DDPM100 FP32 sampling with train_range_x0.
    Initialization seeds: 2000–2009; sampling seeds: 3000–3009. Final evaluation seeds 1000–1049 were not used.
    The scene uses the fixed historical demo154 XML with fresh randomized resets, never a recorded demonstration initial state. Training-only trajectory states are used for calibration. This is not final evaluation.
    Preflight status: {PREFLIGHT['status']}; renderer policy: {EVENT_RENDER_POLICY}; source equivalence verified: {SUMMARY['source_equivalence_verified']}. Camera calibration against a training sample, normalization, 36D robot-only state, OSC delta/world controller, 20Hz control and actions[t:t+16] -> indices 0:8 alignment were checked.
    Each trial has video, CSV, state and prediction records. Historical RGB and Events training sources were not modified. Modality: {MODALITY}. Initial history: {PROTOCOL['initial_history']}. Outputs: {OUTPUT}.
    If source equivalence is false, record known renderer domain shift and do not attribute RGB/Event differences to modality alone. Read pilot_summary.json, protocol.json and trial videos before drawing conclusions. Do not claim that the study is complete or that overfitting is resolved from this pilot.
    '''
    (OUTPUT/'PROMPT_Project_Control_EN.md').write_text(base+'\nUpdate project status, the experiment log and decisions without overwriting historical records. Separate task failure from infrastructure errors; link this evaluation to its source training run. Events training is already COMPLETED; this evaluation has its own status. D100/D200 and encoder changes remain pending.\n',encoding='utf8')
    (OUTPUT/('PROMPT_Notebook05_EN.md' if MODALITY=='events' else 'PROMPT_Notebook04_EN.md')).write_text(base+'\nKeep one active notebook per stage. Notebook07 is development evaluation and never performs training. Source training best1500 remains immutable; pilot success does not remove observed overfitting. Use the new template154 RGB reference; historical demo10 pilot scores are context only. Analyze event activity, normalization, timing and videos before attributing failures to Stage4 or dataset size.\n',encoding='utf8')
    # Small evidence bundle excludes MP4 and per-step state archives. Full bundle includes all ten videos.
    for bundle_name,full in [('pilot_evidence.zip',False),('pilot_full_bundle.zip',True)]:
        tmp=Path('/content')/(bundle_name+'.'+uuid.uuid4().hex)
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(OUTPUT.rglob('*')):
                if not p.is_file() or p.suffix=='.zip' or '.tmp-' in p.name:continue
                if not full and p.suffix in ['.mp4','.npz','.npy']:continue
                archive.write(p,str(p.relative_to(OUTPUT)))
        dst=OUTPUT/bundle_name
        shutil.copyfile(tmp,dst);assert file_hash(tmp)==file_hash(dst);tmp.unlink()
    print(report)
    from IPython.display import display, Video
    print('Share pilot_evidence.zip plus videos, or pilot_full_bundle.zip (all videos):',OUTPUT)
    # Show one success and one failure when present, without filling notebook output with all videos.
    shown=set()
    for desired in [True,False]:
        candidates=[r for r in completed if r['any_success']==desired]
        if candidates:
            r=candidates[0];video=next(n for n in r['artifacts'] if n.endswith('rollout.mp4'))
            if video not in shown:display(Video(str(OUTPUT/video),embed=True,width=640));shown.add(video)
else:
    print('Renderer diagnostic complete. Upload renderer_repair_bundle.zip; no pilot summary was replaced.')


# MODULE 08_paired_orchestrator.py
# A second isolated evaluation namespace prevents RGB/Event model or preprocessing contamination.
def split_distributed_modules(source):
    import re
    starts=list(re.finditer(r'^# MODULE ([A-Za-z0-9_]+\.py)\n',source,re.MULTILINE))
    modules={m.group(1):source[m.end():starts[i+1].start() if i+1<len(starts) else len(source)]
        for i,m in enumerate(starts)}
    required=['01_setup.py','02_artifacts.py','03_model.py','04_policy_and_environment.py',
        '04a_event_stream.py','04b_event_gates.py','04c_renderer_repair.py','05_preflight.py','06_rollouts.py','07_summary.py']
    assert all(name in modules for name in required), 'Distributed evaluator modules missing.'
    return [(name,modules[name]) for name in required]

def execute_event_phase(modules,rgb_directory,rgb_contract,distributed_source,identity):
    ctx={'__name__':'__main__','EVAL_MODALITY':'events','PAIRED_RGB_DIR':str(rgb_directory),
        'PAIRED_RGB_CONTRACT':copy.deepcopy(rgb_contract)}
    error=None
    try:
        for name,code in modules:
            exec(compile(code,name,'exec'),ctx)
            if name=='02_artifacts.py':
                (ctx['SESSION']/'notebook07_source.py').write_text(distributed_source,encoding='utf8')
                ctx['atomic_json'](ctx['SESSION']/'source_identity.json',identity)
    except KeyboardInterrupt:
        error={'status':'ABORTED','stage':'event_phase','error':'user_interrupt','traceback':traceback.format_exc()}
    except Exception as exc:
        error={'status':'FAILED','stage':'event_phase','error':str(exc),'traceback':traceback.format_exc()}
    finally:
        if ctx.get('MODEL') is not None:
            ctx['MODEL']=None;gc.collect();torch.cuda.empty_cache()
    return ctx,error

def paired_result_rows(rgb_summary,event_summary):
    rgb={e['environment_seed']:e for e in rgb_summary['episodes']}
    events={e['environment_seed']:e for e in (event_summary or {}).get('episodes',[])}
    rows=[]
    for seed,r in sorted(rgb.items()):
        e=events.get(seed)
        if e is not None:assert e['policy_seed']==r['policy_seed'], 'Paired policy seeds differ.'
        complete=e is not None and r['status']==e['status']=='COMPLETED'
        rows.append({'environment_seed':seed,'policy_seed':r['policy_seed'],
            'rgb_status':r['status'],'event_status':e['status'] if e else 'NOT_RUN',
            'rgb_any_success':r['any_success'] if r['status']=='COMPLETED' else None,
            'event_any_success':e['any_success'] if e and e['status']=='COMPLETED' else None,
            'rgb_sustained_success':r['sustained_success'] if r['status']=='COMPLETED' else None,
            'event_sustained_success':e['sustained_success'] if e and e['status']=='COMPLETED' else None,
            'paired_completed':complete})
    return rows

def paired_counts(rows):
    finished=[r for r in rows if r['paired_completed']]
    return {name:sum(bool(r['rgb_any_success'])==a and bool(r['event_any_success'])==b for r in finished)
        for name,a,b in [('both_success',True,True),('rgb_only',True,False),('events_only',False,True),('both_fail',False,False)]}

def create_paired_bundle(pair_dir,rgb_dir,event_dir):
    temporary=Path('/content')/('paired-pilot-'+uuid.uuid4().hex+'.zip')
    destination=pair_dir/'paired_pilot_full_bundle.zip'
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for label,folder in [('paired',pair_dir),('rgb',rgb_dir),('events',event_dir)]:
                if folder is None or not Path(folder).is_dir():continue
                for path in sorted(Path(folder).rglob('*')):
                    if path.is_file() and path.suffix!='.zip':
                        archive.write(path,label+'/'+str(path.relative_to(folder)))
        shutil.copyfile(temporary,destination)
        assert file_hash(destination)==file_hash(temporary)
    finally:temporary.unlink(missing_ok=True)
    print('Upload:',destination)

if MODE=='paired_pilot' and MODALITY=='rgb':
    RGB_PHASE_SUMMARY=copy.deepcopy(SUMMARY)
    RGB_PHASE_DIRECTORY=OUTPUT
    RGB_TRAINING_CONTRACT={'config':copy.deepcopy(CFG),'normalization':copy.deepcopy(NORMALIZATION)}
    pair_protocol={'version':CODE_VERSION,'code_sha256':CODE_SHA256,'sources':SOURCES,
        'scene_template_demo':SCENE_TEMPLATE_DEMO,'scene_xml_sha256':PROTOCOL['source_scene_xml_sha256'],
        'environment_seeds':PILOT['environment_seeds'],'policy_seeds':PILOT['policy_seeds'],
        'max_control_steps':PILOT['max_control_steps'],'execution_horizon':PILOT['execution_horizon'],
        'success_hold_steps':PILOT['success_hold_steps'],'sampler':PROTOCOL['sampler'],
        'runtime':RUNTIME,'event_render_policy':EVENT_RENDER_POLICY,
        'scope':'new paired development pilot; fixed historical D50 template; no final test or training'}
    pair_hash=canonical_hash(pair_protocol)
    PAIR_DIRECTORY=PROJECT_ROOT/'results/paired_development'/('d50_template154_'+pair_hash[:12])
    PAIR_DIRECTORY.mkdir(parents=True,exist_ok=True)
    if (PAIR_DIRECTORY/'protocol.json').exists():assert canonical_hash(json.loads((PAIR_DIRECTORY/'protocol.json').read_text()))==pair_hash
    else:atomic_json(PAIR_DIRECTORY/'protocol.json',pair_protocol)
    identity=json.loads((SESSION/'source_identity.json').read_text())
    source_snapshot=(SESSION/'notebook07_source.py').read_text(encoding='utf8')
    assert hashlib.sha256(source_snapshot.encode()).hexdigest()==identity['distributed_source_sha256']
    EVENT_PHASE_CONTEXT={};EVENT_PHASE_ERROR=None
    # One GPU model at a time; completed RGB evidence is immutable and independently verified by Events.
    MODEL=None;gc.collect();torch.cuda.empty_cache()
    if RGB_PHASE_SUMMARY['status']=='COMPLETED':
        print('RGB phase completed. Starting fresh paired Events evaluation on the same template154 scene.')
        EVENT_PHASE_CONTEXT,EVENT_PHASE_ERROR=execute_event_phase(split_distributed_modules(source_snapshot),
            RGB_PHASE_DIRECTORY,RGB_TRAINING_CONTRACT,source_snapshot,identity)
    else:
        print('RGB phase incomplete; Events deferred. Resume this same notebook/configuration with Run All.')
    event_summary=EVENT_PHASE_CONTEXT.get('SUMMARY')
    event_directory=EVENT_PHASE_CONTEXT.get('OUTPUT')
    if EVENT_PHASE_ERROR:atomic_json(PAIR_DIRECTORY/'event_phase_failure.json',EVENT_PHASE_ERROR)
    pair_rows=paired_result_rows(RGB_PHASE_SUMMARY,event_summary)
    pair_complete=RGB_PHASE_SUMMARY['status']=='COMPLETED' and event_summary is not None and event_summary['status']=='COMPLETED' and EVENT_PHASE_ERROR is None
    if pair_complete:
        assert RGB_PHASE_SUMMARY['preflight']==event_summary['preflight']=='PASS'
        assert event_summary['source_equivalence_verified'] is True
        assert sum(r['paired_completed'] for r in pair_rows)==10
    PAIR_SUMMARY={'pair_id':PAIR_DIRECTORY.name,'protocol_hash':pair_hash,'utc':now(),
        'status':'COMPLETED' if pair_complete else 'PARTIAL','scene_template_demo':SCENE_TEMPLATE_DEMO,
        'scene_xml_sha256':pair_protocol['scene_xml_sha256'],'rgb':RGB_PHASE_SUMMARY,'events':event_summary,
        'paired_counts':paired_counts(pair_rows),'paired_completed':sum(r['paired_completed'] for r in pair_rows),
        'execution_error':EVENT_PHASE_ERROR,'new_training_performed':False,'final_test_performed':False,
        'comparison_scope':'paired development, one training seed, fixed historical scene; no superiority/significance claim',
        'historical_results_context_only':{'rgb_demo10_any':3,'events_demo10_any':2,'events_demo10_sustained':1},
        'limitations':['The demo154 scene differs physically from the old demo10 pilot; old scores are not pooled.',
            'RGB repeats the reset image; Events starts with empty visual history.',
            'Strict source/render parity is checked on training prefixes, not every possible future state.',
            'This evaluates existing historical fixed-template D50 models; it does not repair/rewrite Notebook02 data.'],
        'output_directory':str(PAIR_DIRECTORY)}
    atomic_json(PAIR_DIRECTORY/'paired_summary.json',PAIR_SUMMARY)
    write_csv(PAIR_DIRECTORY/'paired_results.csv',pair_rows)
    report=f"# New template154 paired development pilot\n\nStatus: {PAIR_SUMMARY['status']}\n\nRGB any/sustained: {RGB_PHASE_SUMMARY['successes_any']}/{RGB_PHASE_SUMMARY['successes_sustained']} out of {RGB_PHASE_SUMMARY['completed_episodes']}.\n"
    if event_summary:report+=f"\nEvents any/sustained: {event_summary['successes_any']}/{event_summary['successes_sustained']} out of {event_summary['completed_episodes']}.\n"
    report+=f"\nPaired counts: {PAIR_SUMMARY['paired_counts']}\n\nFixed historical demo154 scene, ten development resets, existing EMA best1500 models. Old pilot scores are context only. No final-test or significance claim.\n"
    (PAIR_DIRECTORY/'paired_summary.md').write_text(report)
    prompt=report+'\nUpdate Notebook07/Project Control context using paired_summary.json. Keep old experiments immutable. Check exact initial-state pairing, strict input parity, videos and execution failures before conclusions. Notebook02 per-demo scene provenance remains future work; no data regeneration or retraining occurred.\n'
    (PAIR_DIRECTORY/'PROMPT_Notebook07_EN.md').write_text(prompt)
    (PAIR_DIRECTORY/'PROMPT_Project_Control_EN.md').write_text(prompt)
    create_paired_bundle(PAIR_DIRECTORY,RGB_PHASE_DIRECTORY,event_directory)
    print(report)

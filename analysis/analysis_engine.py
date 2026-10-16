"""CPU-only analysis of the completed, locked Notebook 08 evaluation.

This module consumes a pinned evidence archive. It never trains, loads weights,
renders a simulator, searches historical runs, or changes source evidence.
"""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import collections
import csv
import hashlib
import io
import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VERSION = 'notebook09-locked-final600-cpu-analysis-v1'
STUDY_ID = 'd200_tail_final_normal_dark_v1_fed681d7b085'
STUDY_HASH = 'fed681d7b0855888f5c8c80dbf6cc40d43b8678f453e24085ee104eb5d1a864b'
PROTOCOL_HASH = '6a86e71354c8ae309fa15a52aa3bfca70d19e5166d2bd76238ff8ccba6722ffa'
CORE_SHA256 = 'e88a08e3552519567a90165d463737345ba9f04df1fcf1e15bbeb5914b8a8315'
SEEDS = (42, 43, 44)
CONDITIONS = ('normal', 'dark_gain025')
MODALITIES = ('rgb', 'events')
SCOPE = ('Conditional on six fixed trained models and 50 shared reset/noise pairs. '
         'Synthetic noiseless V2E, fixed scene, standard reset region, digital gain0.25. '
         'Not physical low-light sensor performance or training-population inference.')


def sha(data):
    return hashlib.sha256(data).hexdigest()


ENGINE_SHA256 = globals().get('ENGINE_SHA256') or sha(Path(__file__).read_bytes())


def canonical_hash(obj):
    return sha(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def cpu_gate():
    """Block an allocated accelerator before mounting or reading Drive."""
    executable = shutil.which('nvidia-smi')
    if executable:
        result = subprocess.run([executable, '-L'], capture_output=True, text=True, timeout=8)
        require(not (result.returncode == 0 and 'GPU ' in result.stdout),
                'GPU allocated. Select Runtime > Change runtime type > Hardware accelerator: None, then Run all.')
    require(not os.environ.get('COLAB_TPU_ADDR'), 'TPU allocated. Select a CPU runtime for this notebook.')
    return {'status': 'PASS', 'accelerator_required': False, 'python': sys.version,
            'platform': platform.platform(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'matplotlib': matplotlib.__version__}


def load_archive(core_path):
    """Read one file once; subsequent validation operates only on local bytes."""
    core_path = Path(core_path)
    print('Reading one evidence ZIP; no project-wide scan:', core_path, flush=True)
    payload = core_path.read_bytes()
    require(sha(payload) == CORE_SHA256, 'Core evidence ZIP differs from the exact reviewed final600 archive. Stop; do not bypass this check.')
    archive = zipfile.ZipFile(io.BytesIO(payload))
    names = archive.namelist()
    require(len(names) == len(set(names)), 'Duplicate archive member names.')
    for name in names:
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe archive member path.')
    require(sum(i.file_size for i in archive.infolist()) < 1024 ** 3, 'Unexpected expanded evidence size.')
    require(archive.testzip() is None, 'ZIP CRC failure.')
    manifest = json.loads(archive.read('bundle_manifest.json'))
    members = {entry['path']: entry for entry in manifest['files']}
    require(len(members) == len(manifest['files']), 'Duplicate manifest member.')
    require(set(names) == set(members) | {'bundle_manifest.json'}, 'Manifest coverage differs from ZIP members.')
    hashes = {}
    for index, (name, entry) in enumerate(members.items(), 1):
        data = archive.read(name)
        digest = sha(data)
        require(len(data) == entry['bytes'] and digest == entry['sha256'], 'Manifest hash/size failure: ' + name)
        hashes[name] = digest
        if index % 2000 == 0:
            print('Local archive payloads verified:', index, '/', len(members), flush=True)
    return archive, hashes, {'core_sha256': CORE_SHA256, 'core_bytes': len(payload),
                             'verified_manifest_payloads': len(members), 'zip_crc': 'PASS'}


def read_json(archive, name):
    return json.loads(archive.read(name))


def artifact(episode, filename):
    matches = [name for name in episode['artifacts'] if name.endswith('/' + filename)]
    require(len(matches) == 1, 'Missing/ambiguous artifact: ' + filename)
    return matches[0]


def validate_evidence(archive, hashes, audit):
    protocol = read_json(archive, 'protocol.json')
    summary = read_json(archive, 'final_evaluation_summary.json')
    freeze = read_json(archive, 'protocol_freeze.json')
    started = read_json(archive, 'scoring_started.json')
    execution = read_json(archive, 'execution_summary.json')
    require(canonical_hash(protocol) == PROTOCOL_HASH == summary['protocol_hash'], 'Protocol identity mismatch.')
    require(canonical_hash(protocol['study_identity']) == STUDY_HASH == summary['study_hash'], 'Study identity mismatch.')
    require(summary['queue_id'] == STUDY_ID and summary['status'] == 'COMPLETED', 'Study is not the completed final600.')
    require(summary['planned_episodes'] == summary['completed_episodes'] == 600, 'Incomplete final evaluation.')
    require(execution['status'] == 'COMPLETED' and execution['scored_episodes'] == 600, 'Execution completion mismatch.')
    require(not execution['training_performed'] and not summary['training_performed'] and summary['historical_imports'] == 0,
            'Unexpected training or historical trial imports.')
    require(canonical_hash(freeze['payload']) == freeze['freeze_sha256'] == started['freeze_sha256'], 'Freeze hash mismatch.')
    require(freeze['payload']['study_identity'] == protocol['study_identity'], 'Frozen study changed.')
    require(freeze['payload']['analysis_plan'] == protocol['analysis_plan'], 'Predeclared analysis plan changed.')
    require(freeze['created_utc'] < started['started_utc'], 'Scoring preceded protocol freeze.')
    for filename in ['preflight.json', 'new_seed_role_audit.json', 'six_checkpoint_identities.json']:
        require(read_json(archive, 'session_evidence/' + filename)['status'] == 'PASS', 'Failed recorded preflight: ' + filename)
    options = protocol['study_identity']['options']
    require(options['training_seeds'] == list(SEEDS) and options['environment_seeds'] == list(range(1000, 1050)), 'Seed schedule mismatch.')
    require(options['policy_seeds'] == list(range(4000, 4050)), 'Policy-noise schedule mismatch.')
    require(options['conditions'] == [{'id': 'normal', 'kind': 'none'}, {'id': 'dark_gain025', 'kind': 'source_gain', 'gain': .25}], 'Condition mismatch.')
    require(options['success_hold_steps'] == 10 and options['max_control_steps'] == 300 and options['execution_horizon'] == 8,
            'Endpoint/control contract mismatch.')
    names = sorted(name for name in hashes if name.startswith('episodes/') and name.endswith('.json'))
    episodes = [read_json(archive, name) for name in names]
    recorded_guard_contract = episodes[0]['acquisition_guard']['contract']
    require(recorded_guard_contract['version'] == 'fixed-three-native-draws-one-level-dual-v2e-v2'
            and recorded_guard_contract['rgb_max_abs_uint8'] == recorded_guard_contract['gray_max_abs_uint8'] == 1
            and recorded_guard_contract['window_tensor_max_abs'] == 1e-5, 'Unexpected frozen acquisition contract.')
    expected = set(itertools.product(SEEDS, CONDITIONS, MODALITIES, range(1000, 1050)))
    keys = [(e['training_seed'], e['condition_id'], e['modality'], e['environment_seed']) for e in episodes]
    require(len(episodes) == len(set(keys)) == 600 and set(keys) == expected, 'Duplicate or missing final trial.')
    initial = {}
    rows = []
    omitted = collections.Counter()
    total_steps = reference_checks = tensor_checks = 0
    max_event_difference = 0.0
    reset_max_difference = 0.0
    for index, e in enumerate(episodes, 1):
        require(e['status'] == 'COMPLETED' and e['protocol_hash'] == PROTOCOL_HASH, 'Episode identity/status mismatch.')
        require(e['policy_seed'] == e['environment_seed'] + 3000 and e['stage'] == 'best', 'Episode seed/stage mismatch.')
        source = protocol['study_identity']['sources'][str(e['training_seed'])][e['modality']]
        require(e['checkpoint_step'] == source['best']['step'] == 4500, 'Checkpoint step mismatch.')
        require(e['checkpoint_sha256'] == source['best']['sha256'] and e['source_run_id'] == source['run_id'], 'Wrong checkpoint ownership.')
        condition = next(c for c in options['conditions'] if c['id'] == e['condition_id'])
        require(canonical_hash(condition) == e['condition_hash'] == e['stress_audit']['condition_hash'], 'Condition hash mismatch.')
        require(e['stress_audit']['condition'] == condition, 'Unexpected stress condition.')
        require(e['weights_unchanged'] and e['render_gate'] == 'PASS' and e['presentation']['status'] == 'PASS', 'Failed episode audit.')
        guard = e['acquisition_guard']
        require(guard['status'] == 'PASS', 'Failed acquisition guard.')
        require(guard['contract'] == recorded_guard_contract, 'Acquisition guard contract changed.')
        require(guard['control_endpoints_checked'] == e['steps'] and guard['tensor_checks'] == 4 * e['steps'], 'Event guard coverage mismatch.')
        require(guard['max_abs_tensor_difference'] <= 1e-5, 'Dual V2E mismatch.')
        max_event_difference = max(max_event_difference, guard['max_abs_tensor_difference'])
        tensor_checks += guard['tensor_checks']
        for name, digest in e['artifacts'].items():
            if name in hashes:
                require(hashes[name] == digest, 'Episode artifact digest mismatch: ' + name)
                reference_checks += 1
            else:
                omitted[PurePosixPath(name).name] += 1
        controls = list(csv.DictReader(io.StringIO(archive.read(artifact(e, 'steps.csv')).decode())))
        require(len(controls) == e['steps'] and 1 <= len(controls) <= 300, 'Step count mismatch.')
        consecutive = max_consecutive = 0
        any_success = False
        first_success = None
        for step, row in enumerate(controls, 1):
            require(int(row['step']) == step and abs(float(row['simulation_elapsed_s']) - step / 20) < 1e-8, 'Control timing mismatch.')
            success = float(row['cube_above_table']) > .04
            require((row['success'] == 'True') == success, 'Height predicate mismatch.')
            consecutive = consecutive + 1 if success else 0
            max_consecutive = max(max_consecutive, consecutive)
            any_success |= success
            if success and first_success is None:
                first_success = step
            require(int(row['consecutive_success']) == consecutive, 'Consecutive-success count mismatch.')
            require(int(row['plan_action_index']) == (step - 1) % 8, 'Action horizon alignment mismatch.')
            require(all(math.isfinite(float(row['action_' + str(k)])) for k in range(7)), 'Non-finite action.')
            require(max(abs(float(row['action_' + str(k)])) for k in range(7)) <= 1.000001, 'Controller command outside declared bounds.')
        require(e['any_success'] == any_success and e['sustained_success'] == (max_consecutive >= 10), 'Reported outcome differs from step rescore.')
        require(e['max_consecutive_success_steps'] == max_consecutive and e['first_success_step'] == first_success, 'Episode success summary mismatch.')
        require((e['steps'] == 300 and not e['sustained_success']) or (e['sustained_success'] and consecutive == 10), 'Unexpected episode stopping rule.')
        require(abs(e['simulation_seconds'] - e['steps'] / 20) < 1e-8, 'Simulated duration mismatch.')
        state = np.load(io.BytesIO(archive.read(artifact(e, 'initial_state.npy'))), allow_pickle=False)
        seed = e['environment_seed']
        require(np.isfinite(state).all(), 'Non-finite initial state.')
        if seed in initial:
            difference = float(np.max(np.abs(state - initial[seed])))
            reset_max_difference = max(reset_max_difference, difference)
            require(difference <= 1e-9, 'Paired initial-state mismatch.')
        else:
            initial[seed] = state
        reset = read_json(archive, artifact(e, 'reset_contract.json'))
        require(reset == e['reset_contract'] and reset['status'] == 'PASS' and reset['environment_seed'] == seed, 'Reset audit mismatch.')
        require(reset['x_range_m'] == reset['y_range_m'] == [-.03, .03] and reset['rotation_axis'] == 'z', 'Reset population changed.')
        total_steps += len(controls)
        rows.append({key: e[key] for key in ['key', 'training_seed', 'condition_id', 'modality', 'environment_seed', 'policy_seed',
                                            'steps', 'any_success', 'sustained_success', 'wall_seconds']})
        rows[-1]['time_to_sustained_success_s'] = e['steps'] / 20 if e['sustained_success'] else np.nan
        if index % 100 == 0:
            print('Independent step rescore and paired-reset checks:', index, '/600', flush=True)
    frame = pd.DataFrame(rows).sort_values(['training_seed', 'condition_id', 'modality', 'environment_seed']).reset_index(drop=True)
    original = pd.read_csv(io.BytesIO(archive.read('episode_summary.csv'))).set_index('key').sort_index()
    independent = frame.set_index('key').sort_index()
    for col in ['training_seed', 'condition_id', 'modality', 'environment_seed', 'policy_seed', 'steps', 'any_success', 'sustained_success']:
        require(original[col].equals(independent[col]), 'Published episode CSV mismatch: ' + col)
    audit.update(status='PASS', episodes=600, rescored_control_steps=total_steps,
                 matching_included_episode_artifact_hashes=reference_checks,
                 paired_initial_state_max_abs_difference=reset_max_difference,
                 dual_event_tensor_checks=tensor_checks, max_dual_event_difference=max_event_difference,
                 episodes_with_bounded_gray_differences=sum(e['acquisition_guard']['ambiguous_camera_reads'] > 0 for e in episodes),
                 total_bounded_gray_camera_reads=sum(e['acquisition_guard']['ambiguous_camera_reads'] for e in episodes),
                 artifacts_not_present_in_core=dict(omitted),
                 scope='Core ZIP and all included payloads independently checked. Dense audit parts, weight tensor bytes, full trajectories/query inputs and all videos are not present in the core ZIP.')
    progress = read_json(archive, 'progress.json')
    audit['source_status_observations'] = {'progress_json': progress, 'final_summary_status': summary['status'],
        'note': 'progress.json is a stale RUNNING snapshot at 600/600; final summary, execution summary and independent600-trial rescore establish completion. Source files were not changed.'}
    return frame, episodes, protocol, audit


def compute_statistics(frame, archive):
    groups = frame.groupby(['training_seed', 'condition_id', 'modality'], sort=False).agg(
        trials=('key', 'size'), any_success=('any_success', 'sum'), sustained_success=('sustained_success', 'sum')).reset_index()
    groups['sustained_rate'] = groups['sustained_success'] / groups['trials']
    groups['any_rate'] = groups['any_success'] / groups['trials']
    published = read_json(archive, 'final_evaluation_summary.json')['groups']
    for g in published:
        row = groups[(groups.training_seed == g['training_seed']) & (groups.condition_id == g['condition_id']) & (groups.modality == g['modality'])].iloc[0]
        require([int(row.trials), int(row.any_success), int(row.sustained_success)] == [g['completed'], g['any_success'], g['sustained_success']], 'Published group table mismatch.')
    aggregate = groups.groupby(['condition_id', 'modality']).agg(trials=('trials', 'sum'), sustained_success=('sustained_success', 'sum'),
        any_success=('any_success', 'sum'), mean_seed_rate=('sustained_rate', 'mean'), sample_sd_seed_rate=('sustained_rate', 'std')).reset_index()
    lookup = frame.set_index(['environment_seed', 'training_seed', 'condition_id', 'modality'])['sustained_success']
    cube = np.array([lookup.loc[(r, t, c, m)] for r,t,c,m in itertools.product(range(1000, 1050), SEEDS, CONDITIONS, MODALITIES)],
                    dtype=float).reshape(50, 3, 2, 2)
    mean_rates = cube.mean(axis=(0, 1))
    rng = np.random.default_rng(98439)
    reset_indices = rng.integers(0, 50, size=(4000, 50))
    boot_means = cube[reset_indices].mean(axis=(1, 2))
    differences = boot_means[:, :, 1] - boot_means[:, :, 0]
    point = mean_rates[:, 1] - mean_rates[:, 0]
    contrast_names = ['events_minus_rgb_normal', 'events_minus_rgb_dark', 'events_minus_rgb_change_dark_vs_normal']
    draws = [differences[:,0], differences[:,1], differences[:,1] - differences[:,0]]
    estimates = [point[0], point[1], point[1] - point[0]]
    contrasts = []
    recorded = read_json(archive, 'replication_statistics.json')['conditional_paired_reset_bootstrap']['contrasts']
    for name, distribution, estimate in zip(contrast_names, draws, estimates):
        low, high = np.quantile(distribution, [.025, .975])
        require(np.allclose([estimate,low,high], [recorded[name]['mean_rate_difference'], *recorded[name]['descriptive_reset_bootstrap_95pct']], atol=1e-12, rtol=0), 'Published bootstrap does not reproduce.')
        contrasts.append({'contrast': name, 'difference_percentage_points': float(100*estimate),
                          'conditional_95pct_low_pp': float(100*low), 'conditional_95pct_high_pp': float(100*high),
                          'scope': SCOPE})
    paired = []
    changes = []
    for t in SEEDS:
        for c in CONDITIONS:
            a = cube[:,SEEDS.index(t),CONDITIONS.index(c),0].astype(bool)
            b = cube[:,SEEDS.index(t),CONDITIONS.index(c),1].astype(bool)
            paired.append({'training_seed':t,'condition_id':c,'both_success':int((a&b).sum()),
                'rgb_only':int((a&~b).sum()),'events_only':int((~a&b).sum()),'both_fail':int((~a&~b).sum())})
        for m in MODALITIES:
            before = cube[:,SEEDS.index(t),0,MODALITIES.index(m)].astype(bool)
            after = cube[:,SEEDS.index(t),1,MODALITIES.index(m)].astype(bool)
            changes.append({'training_seed':t,'modality':m,'dark_minus_normal_pp':float(100*(after.mean()-before.mean())),
                'lost_successes':int((before&~after).sum()),'gained_successes':int((~before&after).sum())})
    times = frame[frame.sustained_success].groupby(['training_seed','condition_id','modality']).agg(
        successful_trials=('key','size'), median_simulated_s=('time_to_sustained_success_s','median')).reset_index()
    return {'groups':groups,'aggregate':aggregate,'contrasts':pd.DataFrame(contrasts),
            'paired':pd.DataFrame(paired),'changes':pd.DataFrame(changes),'conditional_success_times':times}


def create_figures(stats, archive, output):
    folder = output / 'figures'
    folder.mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.dpi':200,'pdf.fonttype':42})
    colors = {'rgb':'#2364AA','events':'#DF7D19'}
    def save(fig, name, note):
        fig.text(.02,.012,note,ha='left',fontsize=8,color='#455563')
        fig.tight_layout(rect=(0,.055,1,1))
        fig.savefig(folder/(name+'.png'),bbox_inches='tight')
        fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,c,label in zip(axes,CONDITIONS,['Normal','Digital attenuation: gain 0.25']):
        for offset,m in [(-.18,'rgb'),(.18,'events')]:
            values = stats['groups'].query('condition_id == @c and modality == @m').set_index('training_seed').loc[list(SEEDS)]
            bars=ax.bar(np.arange(3)+offset,100*values.sustained_rate,width=.34,color=colors[m],label=m.upper())
            for bar,n in zip(bars,values.sustained_success):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{n}/50',ha='center',fontsize=9)
        ax.set(title=label,xticks=np.arange(3),xticklabels=SEEDS,xlabel='Training seed',ylim=(0,109))
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].set_ylabel('Sustained Lift success (%)');axes[1].legend(loc='lower right')
    save(fig,'01_success_by_training_seed','All 600 locked trials. Height >0.04m for 10 consecutive 20Hz steps. No best-seed selection.')
    fig,ax=plt.subplots(figsize=(8,4.8))
    for offset,m in [(-.18,'rgb'),(.18,'events')]:
        values=stats['aggregate'].query('modality == @m').set_index('condition_id').loc[list(CONDITIONS)]
        bars=ax.bar(np.arange(2)+offset,100*values.mean_seed_rate,width=.34,color=colors[m],label=m.upper(),
                    yerr=100*values.sample_sd_seed_rate,capsize=5,error_kw={'elinewidth':1.3})
        for bar,rate in zip(bars,values.mean_seed_rate):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()/2,f'{100*rate:.1f}%',ha='center',color='white',weight='bold')
    ax.set(xticks=[0,1],xticklabels=['Normal','Digital gain 0.25'],ylabel='Mean sustained success (%)',ylim=(0,110),title='Equal-weight mean across three trained pairs')
    ax.legend();save(fig,'02_mean_and_training_seed_sd','Error bars: sample SD across 3 trained models per modality, not confidence intervals. Shared 50 reset/noise pairs.')
    fig,ax=plt.subplots(figsize=(8,4.8))
    for m in MODALITIES:
        rows=stats['changes'].query('modality == @m').set_index('training_seed').loc[list(SEEDS)]
        ax.plot(SEEDS,rows.dark_minus_normal_pp,'o-',color=colors[m],label=m.upper(),linewidth=2)
    ax.axhline(0,color='#777777',linestyle='--',linewidth=1);ax.set(xticks=SEEDS,xlabel='Training seed',ylabel='Dark minus normal (percentage points)',title='Sensitivity to fixed digital attenuation')
    ax.legend();save(fig,'03_condition_change','Paired comparisons use the same reset/noise ID within each trained model. Synthetic noiseless event generation.')
    fig,ax=plt.subplots(figsize=(10,4.8));bottom=np.zeros(6);paired=stats['paired'];labels=[f"{r.training_seed}\n{'Normal' if r.condition_id=='normal' else 'Gain0.25'}" for r in paired.itertuples()]
    for name,color in [('both_success','#4A8B74'),('rgb_only',colors['rgb']),('events_only',colors['events']),('both_fail','#A8AFB6')]:
        vals=paired[name].to_numpy();ax.bar(range(6),vals,bottom=bottom,color=color,label=name.replace('_',' ').title());bottom+=vals
    ax.set(xticks=range(6),xticklabels=labels,ylabel='Paired reset/noise IDs',ylim=(0,60),title='Paired outcomes for each trained pair and condition');ax.legend(ncol=4,fontsize=9)
    save(fig,'04_paired_outcomes','Each bar contains 50 paired trials. Discordant outcomes, including RGB-only and Events-only successes, are retained.')
    fig,ax=plt.subplots(figsize=(10,4.6));rows=stats['contrasts'];x=rows.difference_percentage_points.to_numpy();lo=rows.conditional_95pct_low_pp.to_numpy();hi=rows.conditional_95pct_high_pp.to_numpy()
    ax.errorbar(x,np.arange(3),xerr=[x-lo,hi-x],fmt='o',color='#355F80',capsize=5)
    ax.axvline(0,color='#777777',linestyle='--');ax.set(yticks=range(3),yticklabels=['Events - RGB: normal','Events - RGB: gain 0.25 (primary)','Difference in condition changes'],xlabel='Percentage points',title='Predeclared contrasts: paired-reset bootstrap');ax.invert_yaxis()
    save(fig,'05_conditional_paired_reset_intervals','95% descriptive intervals: 4000 shared-reset bootstrap draws. Conditional on 6 fixed models; excludes training-population uncertainty.')
    resets=pd.read_csv(io.BytesIO(archive.read('reset_population.csv')))
    fig,axes=plt.subplots(1,2,figsize=(10,4.7));sc=axes[0].scatter(resets.cube_x_world_m*100,resets.cube_y_world_m*100,c=resets.cube_yaw_deg,cmap='hsv',vmin=0,vmax=360,s=40,edgecolor='white')
    axes[0].set(xlabel='Cube world X (cm)',ylabel='Cube world Y (cm)',title='All 50 held-out initial placements',xlim=(-3.2,3.2),ylim=(-3.2,3.2));axes[0].set_aspect('equal');fig.colorbar(sc,ax=axes[0],label='Initial yaw (degrees)')
    axes[1].hist(resets.cube_yaw_deg,bins=np.linspace(0,360,13),color='#355F80',edgecolor='white');axes[1].set(xlabel='Cube yaw around Z (degrees)',ylabel='Reset count',title='Observed yaw coverage',xlim=(0,360))
    save(fig,'06_reset_population','Fixed TRAIN XML scene; standard XY region approximately+/-3cm. Robot initialization:Gaussian magnitude0.02. No wider-position OOD claim.')


def markdown_table(frame):
    columns=list(frame.columns)
    rows=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for values in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{v:.3f}' if isinstance(v,float) else str(v) for v in values)+' |')
    return '\n'.join(rows)


def write_report(frame, episodes, protocol, audit, stats, output, archive):
    for name,table in stats.items():table.to_csv(output/(name+'.csv'),index=False)
    frame.to_csv(output/'independent_episode_summary.csv',index=False)
    sources=[]
    for seed in SEEDS:
        for modality in MODALITIES:
            source=protocol['study_identity']['sources'][str(seed)][modality]
            sources.append({'training_seed':seed,'modality':modality,'run_id':source['run_id'],'config_hash':source['config_hash'],
                            'checkpoint_step':source['best']['step'],'checkpoint_sha256':source['best']['sha256']})
    pd.DataFrame(sources).to_csv(output/'model_artifact_ownership.csv',index=False)
    atomic_json(output/'independent_audit.json',audit)
    atomic_json(output/'source_protocol.json',protocol)
    for name in ['reset_population.csv','replication_statistics.json','protocol_freeze.json']:
        (output/('source_'+name)).write_bytes(archive.read(name))
    report=f'''# Locked final evaluation: RGB versus synthetic Events

Analysis version: `{VERSION}`. Study: `{STUDY_ID}`.
Source study status: COMPLETED 600/600; independent step-level verification: PASS.
No training, checkpoint loading, rendering or additional trials were performed by Notebook09.

## Primary result

Normal: RGB 148/150 (98.7%); Events 107/150 (71.3%).
Digital gain 0.25: RGB 88/150 (58.7%); Events 110/150 (73.3%).
Primary Events-minus-RGB dark contrast: +14.67 percentage points.
RGB dark-minus-normal change: -40.00pp; Events: +2.00pp. Difference in changes: +42.00pp.
The observed Events dark advantage is +26pp, +14pp, +4pp in training pairs 42,43,44 respectively.

## All 12 predeclared groups

{markdown_table(stats['groups'])}

## Equal-weight aggregation and training-seed variability

{markdown_table(stats['aggregate'])}

Rates equal pooled counts here because every group has 50 trials. These 150 observations per modality/condition are not 150 independently trained policies. Sample SD across three training seeds is descriptive and is not a confidence interval.

## Predeclared conditional paired-reset contrasts

{markdown_table(stats['contrasts'].drop(columns='scope'))}

Bootstrap: 4000 draws; summary seed 98439; the SAME sampled reset IDs are used jointly across all six models and both conditions. Published intervals were reproduced to numerical precision. Intervals condition on these six models and exclude training-population, model-selection, scene and physical-sensor uncertainty. No population-level significance or universal superiority claim is made.

## Paired outcomes

{markdown_table(stats['paired'])}

## Methods and reset contract

- Six frozen SMALL FP32 EMA4500 checkpoints; paired training seeds 42/43/44.
- Dataset D200 TAIL: 160 training and 40 validation demos; 7460/1806 windows.
- Final reset seeds 1000-1049; diffusion-noise seeds 4000-4049, paired across models/conditions.
- One fixed TRAIN XML scene: demo_10; seeded fresh reset, not a recorded held-out state.
- UniformRandomSampler XY ranges [-0.03,+0.03]m about reference [0,0,0.8]; random yaw about Z; Gaussian robot initialization magnitude 0.02.
- Control 20Hz; prediction 16; execute indices 0:8; 100-step DDPM with the shared training-range x0 projection.
- Sustained Lift: height above table >0.04m for 10 consecutive control observations; stop then or at 300 steps. Any Lift is secondary. This is a height predicate, not a direct attachment/contact test.
- Condition: unchanged upstream digital gain 0.25 from reset, selected on separate disclosed development evidence. Ideal noiseless synthetic V2E; not calibrated lux, HDR or a physical sensor experiment.
- All 600 assigned valid trials remain in the denominator. No final-cohort tuning, retraining, condition adjustment or best-seed selection.

## Independent verification and practical limitations

Verified {audit['verified_manifest_payloads']} manifest payloads, 600 episode identities, {audit['matching_included_episode_artifact_hashes']} included episode-artifact references and{audit['rescored_control_steps']} control rows. Full initial states match across all 12 groups for each reset, maximum absolute difference{audit['paired_initial_state_max_abs_difference']}. Recorded dual-event checks: {audit['dual_event_tensor_checks']}; maximum recorded tensor difference{audit['max_dual_event_difference']}. Bounded grayscale differences occurred in {audit['episodes_with_bounded_gray_differences']} episodes ({audit['total_bounded_gray_camera_reads']} camera reads); these are permitted by the frozen one-level amplitude contract and did not change the compared Event window tensors. Do not claim all RGB/grayscale pixels were always exact.

Core verification does not independently reload checkpoint tensor bytes, decode all 600 video sets, or inspect omitted dense audits/trajectory/query inputs. Their original ownership/digests remain in source episode pointers. The separately reviewed selected-video archive contains eight trials chosen before outcomes and16 decodable native/slow clips; it is an illustration, not an aggregate-success sample.

`progress.json` remains a stale RUNNING snapshot at 600/600. Final summary and execution summary report COMPLETED and agree with independent rescoring. This is retained as a reporting issue; source files were not modified. The Colab log reports a runtime-release REQUEST after verified completion; that message alone does not prove billing stopped.

## Claim supported by this study

Under this fixed synthetic protocol, RGB had higher normal-condition success, whereas the synthetic-Event policy was more robust to fixed digital attenuation and achieved higher average sustained Lift success under gain0.25 across the three evaluated trained pairs. The magnitude varied by training seed.

## Completion and next work

The locked final study is complete. Continue with documentation, selected-video presentation and final-project writing on CPU. LARGE, matched-rate, a different encoder, physical-sensor noise or wider reset positions would require a separately identified study and untouched evaluation data. They are optional future work, not prerequisites for reporting these results.
'''
    (output/'final_analysis_report.md').write_text(report)
    return report


def build_local_analysis(core_path, work_root=None):
    runtime=cpu_gate()
    archive,hashes,audit=load_archive(core_path)
    try:
        frame,episodes,protocol,audit=validate_evidence(archive,hashes,audit)
        stats=compute_statistics(frame,archive)
        identity={'version':VERSION,'engine_sha256':ENGINE_SHA256,'core_sha256':CORE_SHA256,'study_hash':STUDY_HASH,
                  'protocol_hash':PROTOCOL_HASH,'analysis_plan':protocol['analysis_plan']}
        analysis_id='d200_tail_final600_analysis_v1_'+canonical_hash(identity)[:12]
        root=Path(work_root) if work_root else Path(tempfile.mkdtemp(prefix='nb09-cpu-'))
        root.mkdir(parents=True,exist_ok=True)
        session=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
        output=root/analysis_id/session;output.mkdir(parents=True,exist_ok=False)
        engine_source=globals().get('ENGINE_SOURCE') or Path(__file__).read_text()
        require(sha(engine_source.encode()) == ENGINE_SHA256, 'Analysis engine code identity mismatch.')
        (output/'analysis_engine.py').write_text(engine_source)
        create_figures(stats,archive,output)
        report=write_report(frame,episodes,protocol,audit,stats,output,archive)
        summary={'status':'COMPLETED','analysis_id':analysis_id,'session_id':session,'analysis_version':VERSION,
                 'source_study_id':STUDY_ID,'engine_sha256':ENGINE_SHA256,'core_sha256':CORE_SHA256,'study_hash':STUDY_HASH,
                 'protocol_hash':PROTOCOL_HASH,'verified_episodes':600,'rescored_control_steps':audit['rescored_control_steps'],
                 'training_performed':False,'new_simulations':0,'gpu_used':False,'scope':SCOPE,
                 'created_utc':datetime.now(timezone.utc).isoformat(),'runtime':runtime,
                 'publication_status':'LOCAL_COMPLETE_NOT_YET_PUBLISHED_TO_DRIVE'}
        atomic_json(output/'analysis_summary.json',summary)
        inventory=[{'path':str(f.relative_to(output)),'bytes':f.stat().st_size,'sha256':sha(f.read_bytes())}
                   for f in sorted(output.rglob('*')) if f.is_file()]
        atomic_json(output/'analysis_manifest.json',{'files':inventory,'source_core_sha256':CORE_SHA256})
        bundle=root/(analysis_id+'_'+session+'_analysis_bundle.zip')
        with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for f in sorted(output.rglob('*')):
                if f.is_file():z.write(f,f.relative_to(output))
        with zipfile.ZipFile(bundle) as z:require(z.testzip() is None,'Analysis ZIP CRC failed.')
        print('LOCAL ANALYSIS COMPLETED:600/600 verified; no GPU or simulation.',flush=True)
        return {'summary':summary,'output':output,'bundle':bundle,'bundle_sha256':sha(bundle.read_bytes()),
                'report':report,'stats':stats,'audit':audit}
    finally:
        archive.close()


def publish_analysis(result, project_root):
    """Write only a new analysis session, then publish a small completion receipt."""
    summary=result['summary'];destination=Path(project_root)/'results/final_analysis'/summary['analysis_id']/'sessions'/summary['session_id']
    destination.mkdir(parents=True,exist_ok=False)
    bundle_target=destination/'final_analysis_bundle.zip'
    temporary=destination/('final_analysis_bundle.zip.tmp-'+uuid.uuid4().hex)
    shutil.copyfile(result['bundle'],temporary)
    require(sha(temporary.read_bytes())==result['bundle_sha256'],'Published ZIP readback mismatch. Local bundle retained; do not rerun simulations.')
    os.replace(temporary,bundle_target)
    (destination/'final_analysis_report.md').write_text(result['report'])
    receipt=dict(summary,publication_status='DRIVE_BUNDLE_VERIFIED',bundle_sha256=result['bundle_sha256'],
                 bundle_bytes=bundle_target.stat().st_size,bundle_path=str(bundle_target))
    atomic_json(destination/'analysis_summary.json',receipt)
    result['published_directory']=str(destination)
    result['published_bundle']=str(bundle_target)
    print('Upload this analysis bundle:',bundle_target,flush=True)
    return receipt

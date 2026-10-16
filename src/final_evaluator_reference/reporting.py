"""Descriptive development results; failed/incomplete attempts never count as task failures."""
def wilson(successes,n):
    if n==0: return None
    z=1.959963984540054;p=successes/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0.,center-radius),min(1.,center+radius)]

def make_bundle(folder,path,include_videos=False,allowed_artifacts=None):
    folder=Path(folder);path=Path(path);temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    manifest=[]
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
            for file in sorted(folder.rglob('*')):
                if not file.is_file() or file==temporary or file.suffix in ['.ckpt','.pt','.zip'] or '.tmp-' in file.name: continue
                relative=str(file.relative_to(folder))
                if allowed_artifacts is not None:
                    if len(Path(relative).parts)>1 and relative not in allowed_artifacts: continue
                # Small bundle excludes videos and full source arrays. Policy trajectory NPZ is selected only in full bundle.
                if file.suffix=='.mp4' and not include_videos: continue
                if file.suffix=='.npz' and not (include_videos and file.name=='trajectory.npz'): continue
                if file.suffix not in ['.json','.jsonl','.csv','.md','.py','.log','.png','.pdf','.npy','.npz','.mp4','.xml']: continue
                if file.stat().st_size>40*1024*1024 and not include_videos: continue
                z.write(file,relative);manifest.append({'path':relative,'sha256':sha_file(file),'bytes':file.stat().st_size})
            z.writestr('bundle_manifest.json',json.dumps({'files':manifest,'weights_included':False,
                'videos_included':include_videos,'scope':'development evidence; omitted arrays/checkpoints remain on Drive'},indent=2))
        os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)
    return path

def summarize(root,protocol,rows,expected,final=False,make_bundles=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);protocol_hash=canonical(protocol)
    # Refresh from durable verified pointers, including a worker that completed just before parent interruption.
    durable=[verified_episode(root,protocol_hash,job) for job in schedule(protocol['options'])]
    durable=[r for r in durable if r is not None]
    # Recheck all durable pairs before counting or bundling outcomes.
    paired_checked=[]
    for row in durable:
        verify_paired_reset(row,row,paired_checked);paired_checked.append(row)
    summaries=[]
    for stage in ['best']:
        for modality in ['rgb','events']:
            group=modality+'_'+stage;selected=[r for r in durable if r['group']==group]
            n=len(selected);any_count=sum(r['any_success'] for r in selected);hold_count=sum(r['sustained_success'] for r in selected)
            summaries.append({'group':group,'modality':modality,'stage':stage,'checkpoint_label':'validation_best_and_final4500','checkpoint_step':SOURCES[modality][stage]['step'],
                'completed_episodes':n,'planned_episodes':len(protocol['options']['environment_seeds']),
                'any_success':any_count,'sustained_success':hold_count,
                'any_rate':any_count/n if n else None,'sustained_rate':hold_count/n if n else None,
                'sustained_descriptive_wilson_95pct':wilson(hold_count,n),
                'total_wall_minutes':sum(r['wall_seconds'] for r in selected)/60,
                'mean_inference_seconds':float(np.mean([r['mean_inference_seconds'] for r in selected])) if n else None,
                'total_source_minutes':sum(r['source_pipeline_seconds'] for r in selected)/60})
    paired=[]
    for outcome in ['any_success','sustained_success']:
        paired.append(paired_counts(durable,'rgb_best','events_best',outcome))
    report={'status':'COMPLETED' if len(durable)==expected else 'PARTIAL','protocol_hash':protocol_hash,
        'pilot_id':root.name,'completed_episodes':len(durable),'planned_episodes':expected,'training_performed':False,
        'primary_endpoint':'Lift condition true for 10 consecutive 20Hz control steps; sustained_success',
        'secondary_endpoint':'Lift condition true at least once; any_success','groups':summaries,'paired_comparisons':paired,
        'episodes':durable,'updated_utc':utc(),
        'dataset_id':DATASET_ID,'data_fingerprint':DATA_HASH,'normalization_hash':NORM_HASH,
        'selected_checkpoints':SOURCES,'historical_episodes_imported':0,
        'limitations':['Development seeds 2000-2009 were previously used in historical pilots; this is not a new locked final test.',
            'One training seed, one fixed training-scene geometry; reset initial conditions are newly seeded, not recorded held-out states.',
            'Ten matched pairs at4500 updates; best and final are the same selected checkpoint for each modality and are not scored twice. Development only, no significance or winner claim.',
            'Wilson intervals describe the chosen development reset sample conditional on these models/protocol; not training-seed/model-selection uncertainty.',
            'Bounded x0 is a shared explicit inference intervention; in-range actions do not establish action accuracy or Lift success.',
            'Both modalities run matched 200Hz synthetic source acquisition for fidelity; wall time is not native modality inference cost.',
            'Rollout v3 bounds native RGB/gray differences by one uint8 level, audits all counts, and requires persistent dual V2E window equivalence. Rendering is not claimed bit-exact. Only completions under this exact D200 protocol are reused. Historical D50 episodes are never imported. Failed attempts are excluded.',
            'History at reset differs from packed training startup (first sample t=.10); duplicate RGB / zero Events is explicitly declared.',
            'COMPLETED is execution status. Lift success tests height, not a verified gripper attachment.']}
    atomic_json(root/'pilot_summary.json',report)
    write_csv(root/'group_summary.csv',summaries)
    keys=['key','group','environment_seed','policy_seed','checkpoint_step','checkpoint_sha256','steps','any_success',
        'sustained_success','max_consecutive_success_steps','first_success_step','wall_seconds','source_pipeline_seconds',
        'render_seconds','v2e_seconds','mean_inference_seconds','max_cube_above_table','render_gate','weights_unchanged']
    keys+=['execution_origin','origin_protocol_hash']
    write_csv(root/'episode_summary.csv',[{k:r.get(k) for k in keys}|{'execution_origin':r.get('origin',{}).get('kind','CURRENT_RELEASE_EXECUTION'),
        'origin_protocol_hash':r.get('origin',{}).get('protocol_hash',protocol_hash)} for r in durable])
    lines=['# Notebook07 — D200 schema3 tail paired4500 development evaluation','',
        f"- Status: {report['status']}; verified episodes: {len(durable)}/{expected}",
        '- Historical episode imports: 0. Selected best checkpoints are also the fixed-budget final4500 models.',
        f'- Pilot ID: `{root.name}`',f'- Protocol hash: `{protocol_hash}`','- No training was performed. Final seeds 1000–1049 were not used.',
        '- Primary endpoint: sustained Lift for 10 consecutive control steps (0.5 simulated seconds).','',
        '| Model/checkpoint | Completed/planned | Any Lift | Sustained Lift |','|---|---:|---:|---:|']
    for group in summaries:
        lines.append(f"| {group['group']} @{group['checkpoint_step']} | {group['completed_episodes']}/{group['planned_episodes']} | {group['any_success']} | {group['sustained_success']} |")
    lines+=['','## Paired comparisons','',json.dumps(paired,indent=2),'','## Interpretation limits','']+['- '+x for x in report['limitations']]
    lines+=['','## Next decision','',
        'Review sustained Lift, paired resets, videos, source parity, action projection and failure traces. Compare the two selected4500 policies on matched reset seeds; both are validation-best and fixed-budget-final. A successful D200 pilot supports continued development; it does not finish the paired three-training-seed, fifty-rollout final study. Poor behavior requires diagnosis before more training. No training or architecture change occurs automatically.']
    (root/'pilot_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    if durable:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        labels=[r['group']+f"\n(n={r['completed_episodes']})" for r in summaries];xs=np.arange(len(summaries))
        for delta,key,label in [(-.17,'any_rate','Any Lift'),(.17,'sustained_rate','Sustained Lift')]:
            axes[0].bar(xs+delta,[r[key] if r[key] is not None else 0 for r in summaries],width=.34,label=label)
        axes[0].set(xticks=xs,xticklabels=labels,ylim=(0,1),ylabel='Observed development success fraction',title='Matched development counts; n=10 each, not final evaluation');axes[0].legend()
        for group in [r['group'] for r in summaries]:
            selected=[r for r in durable if r['group']==group]
            if selected: axes[1].scatter([r['environment_seed'] for r in selected],[r['wall_seconds']/60 for r in selected],label=group)
        axes[1].set(xlabel='Development reset seed',ylabel='Wall minutes / episode',title='Policy + matched source + video');axes[1].legend(fontsize=8)
        for ax in axes: ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(root/'pilot_results.png',dpi=160);fig.savefig(root/'pilot_results.pdf');plt.close(fig)
    if make_bundles:
        allowed={name for r in durable for name in r['artifacts']}
        make_bundle(root,root/'pilot_evidence.zip',False,allowed)
        make_bundle(root,root/'pilot_full_bundle.zip',True,allowed)
    return report

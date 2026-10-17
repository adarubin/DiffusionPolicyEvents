from pathlib import Path
import csv, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser(description='Plot the included historical training logs without models or a GPU.')
p.add_argument('--output',type=Path)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
OUT=a.output or root/'results/training'
OUT.mkdir(parents=True,exist_ok=True)
with (root/'results/training/validation_history_all_six.csv').open(encoding='utf-8',newline='') as f:
 rows=list(csv.DictReader(f))
assert len(rows)==54
runs={(m,s):{'validation':[r for r in rows if r['modality']==m and int(r['training_seed'])==s]} for m in ['rgb','events'] for s in [42,43,44]}
assert all([int(x['step']) for x in r['validation']]==list(range(500,4501,500)) for r in runs.values())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
colors={'rgb':'#1768ac','events':'#cf671b'};styles={42:'-',43:'--',44:':'}
fig,axs=plt.subplots(1,2,figsize=(13,5.5))
for (mod,seed),r in sorted(runs.items()):
 x=np.array([int(z['step']) for z in r['validation']]);y=np.array([float(z['ema_demo_mean']) for z in r['validation']])
 for ax in axs:ax.plot(x,y,styles[seed],color=colors[mod],lw=2,marker='o',ms=3,label=f'{"RGB" if mod=="rgb" else "Events"} seed {seed}')
for ax in axs:
 ax.set_xlabel('Successful optimizer updates');ax.set_ylabel('EMA validation epsilon-MSE (lower is better)');ax.grid(alpha=.2)
axs[0].set_title('Full recorded validation history');axs[0].set_xlim(400,4600);axs[0].legend(fontsize=9,ncol=2)
axs[1].set_title('Late training: improvement continues');axs[1].set_xlim(2950,4550);axs[1].set_ylim(.1465,.1535)
fig.suptitle('All six final D200 TAIL models: validation improves through update 4500',fontsize=14,fontweight='bold',y=.985)
fig.text(.5,.025,'40 held-out demos, equal-demo means, fixed 3 noise draws. Six fixed training runs; no confidence interval or rollout-success curve.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.065,1,.94))
for ext in ['png','pdf']:fig.savefig(OUT/f'validation_learning_curves.{ext}')
plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(13,7.5),sharex=True,sharey=True)
for row,mod in enumerate(['rgb','events']):
 for col,seed in enumerate([42,43,44]):
  ax=axs[row,col];v=runs[(mod,seed)]['validation'];x=[int(z['step']) for z in v]
  ax.plot(x,[float(z['ema_demo_mean']) for z in v],color='#d87918',lw=2,label='EMA validation: 40 demos')
  ax.plot(x,[float(z['train_diag_ema']) for z in v],color='#14817b',lw=2,ls='--',label='EMA train diagnostic: subset')
  ax.set_title(f'{"RGB" if mod=="rgb" else "Events"} / training seed {seed}');ax.grid(alpha=.2)
  if row==1:ax.set_xlabel('Successful optimizer updates')
  if col==0:ax.set_ylabel('Epsilon-MSE')
axs[0,0].legend(fontsize=8)
fig.suptitle('Train diagnostic and held-out validation: separate estimators, same EMA checkpoint',fontsize=14,fontweight='bold',y=.98)
fig.text(.5,.025,'The train diagnostic uses a fixed subset and is not the stochastic online minibatch training loss. Validation loss is not robot success.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.94))
for ext in ['png','pdf']:fig.savefig(OUT/f'train_validation_diagnostics.{ext}')
plt.close(fig)

print('Regenerated figures from54historical validation records; no training or simulations.')

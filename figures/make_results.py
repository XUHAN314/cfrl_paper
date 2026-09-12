import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
# deployment-sampler accuracy (%) per task; lists = seeds
tasks=["Countdown","Sudoku","K&K","Game24"]
base=[42.2,66.2,67.5,45.0]
grpo=[[47.2],[70.0,70.4,63.0],[70.2,70.0,74.6],[46.8,48.0,46.3]]
jpc=[[48.4,47.3],[69.9,70.1,69.2],[70.3,70.3,70.8],[56.8,60.6,58.8]]
loop=[54.4,None,94.9,58.3]
ar=[53.9,None,None,32.2]
fig,ax=plt.subplots(figsize=(5.6,2.2))
w=0.2; x=np.arange(len(tasks))
def bar(off,vals,color,label):
    m=[np.mean(v) if isinstance(v,list) else v for v in vals]; e=[np.std(v) if isinstance(v,list) and len(v)>1 else 0 for v in vals]
    xs=[xi+off for xi,v in zip(x,vals) if v is not None]; ms=[mi for mi,v in zip(m,vals) if v is not None]; es=[ei for ei,v in zip(e,vals) if v is not None]
    ax.bar(xs,ms,w,color=color,label=label,yerr=es,capsize=2,error_kw=dict(lw=0.8))
    for xi,mi in zip(xs,ms): ax.text(xi,mi+1.2,f"{mi:.1f}",ha="center",va="bottom",fontsize=6)
bar(-1.5*w,base,"#bbbbbb","SFT base")
bar(-0.5*w,grpo,"#e08a3c","GRPO (score-function)")
bar(0.5*w,jpc,"#2b6cb0","JPC, 2D critic (ours)")
bar(1.5*w,loop,"#1a9c5b","JPC + amortisation loop")
for xi,v in zip(x,ar):
    if v is not None: ax.plot([xi-2*w,xi+2*w],[v,v],ls="--",lw=1,color="k"); ax.text(xi+2*w,v,"AR+PPO",fontsize=6,va="center",ha="left")
ax.set_xticks(x); ax.set_xticklabels(tasks); ax.set_ylabel("accuracy at deployment sampler (%)"); ax.set_ylim(20,100)
ax.legend(fontsize=6,frameon=False,ncol=2,loc="upper left")
fig.tight_layout(); fig.savefig("fig_results.pdf"); fig.savefig("fig_results.png",dpi=200); print("fig_results ok")
# loop attribution (Game24, deployment / ODE64)
fig,ax=plt.subplots(figsize=(3.4,2.1))
labels=["SFT\nbase","RL policy\n(2D-JPC)","loop from RL\n4 rounds","loop from base\n(no RL) 2 rounds","gold deriv.\nsame problems"]
dep=[45.0,55.2,58.3,48.0,53.8]; ode=[35.2,41.2,52.7,41.0,40.0]
xx=np.arange(len(labels)); ax.bar(xx-0.18,dep,0.36,color="#1a9c5b",label="deployment (γ56@64)"); ax.bar(xx+0.18,ode,0.36,color="#7fc9a5",label="cheap (ODE-64)")
for xi,d,o in zip(xx,dep,ode): ax.text(xi-0.18,d+0.8,f"{d:.1f}",ha="center",fontsize=5.5); ax.text(xi+0.18,o+0.8,f"{o:.1f}",ha="center",fontsize=5.5)
ax.set_xticks(xx); ax.set_xticklabels(labels,fontsize=5.2,rotation=12); ax.set_ylim(30,65); ax.set_ylabel("Game24 accuracy (%)"); ax.legend(fontsize=5.5,frameon=False,loc="upper left")
fig.tight_layout(); fig.savefig("fig_loop_attrib.pdf"); fig.savefig("fig_loop_attrib.png",dpi=200); print("fig_loop ok")

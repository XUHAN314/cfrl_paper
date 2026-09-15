import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
# deployment-sampler accuracy (%) per task; lists = seeds
tasks=["Countdown","Sudoku","K&K","Game24"]
base=[42.2,66.2,67.5,45.0]
grpo=[[43.1,42.8,43.3],[70.1,72.8,66.2],[70.2,70.0,74.6],[46.8,48.6,47.1]]
scal=[[51.2,49.2,44.3],[53.1,58.1,63.3],[69.5,72.0,72.0],[55.2,43.7,50.6]]
jpc=[[47.2,46.0,47.5],[69.5,69.7,70.7],[79.5,79.1,79.0],[59.3,58.5,58.8]]
loop=[52.1,71.2,98.8,57.3]
ar=[53.9,None,None,32.2]
fig,ax=plt.subplots(figsize=(5.6,2.2))
w=0.16; x=np.arange(len(tasks))
def bar(off,vals,color,label):
    m=[np.mean(v) if isinstance(v,list) else v for v in vals]; e=[np.std(v) if isinstance(v,list) and len(v)>1 else 0 for v in vals]
    xs=[xi+off for xi,v in zip(x,vals) if v is not None]; ms=[mi for mi,v in zip(m,vals) if v is not None]; es=[ei for ei,v in zip(e,vals) if v is not None]
    ax.bar(xs,ms,w,color=color,label=label,yerr=es,capsize=2,error_kw=dict(lw=0.8))
    for xi,mi in zip(xs,ms): ax.text(xi,mi+1.2,f"{mi:.1f}",ha="center",va="bottom",fontsize=6)
bar(-2*w,base,"#bbbbbb","SFT base")
bar(-1*w,grpo,"#e08a3c","GRPO (score-function)")
bar(0,scal,"#8fb3e0","JPC, scalar critic")
bar(1*w,jpc,"#2b6cb0","JPC, 2D critic (ours)")
bar(2*w,loop,"#1a9c5b","2D policy + amortisation loop")
for xi,v in zip(x,ar):
    if v is not None: ax.plot([xi-2.5*w,xi+2.5*w],[v,v],ls="--",lw=1,color="k"); ax.text(xi+2.5*w,v,"AR+PPO",fontsize=6,va="center",ha="left")
ax.set_xticks(x); ax.set_xticklabels(tasks); ax.set_ylabel("accuracy at deployment sampler (%)"); ax.set_ylim(20,100)
ax.legend(fontsize=6,frameon=False,ncol=2,loc="upper left")
fig.tight_layout(); fig.savefig("fig_results.pdf"); fig.savefig("fig_results.png",dpi=200); print("fig_results ok")
# loop attribution (Game24, deployment / ODE64)
fig,ax=plt.subplots(figsize=(3.4,2.1))
labels=["SFT\nbase","RL policy\n(scalar JPC)","loop from RL\n4 rounds","loop from base\n(no RL) 2 rounds","gold deriv.\nsame problems"]
dep=[45.0,55.2,58.3,48.0,53.8]; ode=[35.2,41.2,52.7,41.0,40.0]
xx=np.arange(len(labels)); ax.bar(xx-0.18,dep,0.36,color="#1a9c5b",label="deployment (γ56@64)"); ax.bar(xx+0.18,ode,0.36,color="#7fc9a5",label="cheap (ODE-64)")
for xi,d,o in zip(xx,dep,ode): ax.text(xi-0.18,d+0.8,f"{d:.1f}",ha="center",fontsize=5.5); ax.text(xi+0.18,o+0.8,f"{o:.1f}",ha="center",fontsize=5.5)
ax.set_xticks(xx); ax.set_xticklabels(labels,fontsize=5.2,rotation=12); ax.set_ylim(30,65); ax.set_ylabel("Game24 accuracy (%)"); ax.legend(fontsize=5.5,frameon=False,loc="upper left")
fig.tight_layout(); fig.savefig("fig_loop_attrib.pdf"); fig.savefig("fig_loop_attrib.png",dpi=200); print("fig_loop ok")

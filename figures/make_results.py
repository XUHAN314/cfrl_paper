import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"xtick.labelsize":9,"ytick.labelsize":8.5})
# deployment-sampler accuracy (%) per task; lists = seeds
tasks=["Countdown","Sudoku","K&K","Game24"]
base=[42.2,66.2,67.5,45.0]
grpo=[[43.1,42.8,43.3],[70.1,72.8,66.2],[70.2,70.0,74.6],[46.8,48.6,47.1]]
scal=[[48.8,48.6,48.2],[68.7,67.8,65.8],[77.8,77.5,79.2],[51.6,52.1,52.0]]
jpc=[[47.2,46.0,47.5],[69.5,69.7,70.7],[79.5,79.1,79.0],[59.3,58.5,58.8]]
loop=[52.3,70.8,98.8,58.1]
ar=[53.9,1.4,93.2,32.2]
fig,ax=plt.subplots(figsize=(7.6,2.9))
w=0.155; x=np.arange(len(tasks))
def bar(off,vals,color,label):
    m=[np.mean(v) if isinstance(v,list) else v for v in vals]; e=[np.std(v) if isinstance(v,list) and len(v)>1 else 0 for v in vals]
    xs=[xi+off for xi,v in zip(x,vals) if v is not None]; ms=[mi for mi,v in zip(m,vals) if v is not None]; es=[ei for ei,v in zip(e,vals) if v is not None]
    ax.bar(xs,ms,w,color=color,label=label,yerr=es,capsize=2,error_kw=dict(lw=0.8))
    for xi,mi,ei in zip(xs,ms,es): ax.text(xi,mi+ei+1.0,f"{mi:.1f}",ha="center",va="bottom",fontsize=6.8)
bar(-2*w,base,"#bbbbbb","SFT base")
bar(-1*w,grpo,"#e08a3c","GRPO (score-function)")
bar(0,scal,"#8fb3e0","JPC, scalar critic")
bar(1*w,jpc,"#2b6cb0","JPC, 2D critic (ours)")
bar(2*w,loop,"#1a9c5b","2D policy + amortisation loop")
place={0:("left",1.6),1:("right",0),2:("left",1.6),3:("right",0)}  # label above the line at the group's left edge, or to the right of the group
for xi,v in zip(x,ar):
    if v is None: continue
    ax.plot([xi-2.5*w,xi+2.5*w],[v,v],ls="--",lw=1.2,color="k")
    side,dy=place[int(xi)]
    if side=="left": ax.text(xi-2.5*w,v+dy,f"GPT-2 + PPO  {v:.1f}",fontsize=6.8,va="bottom",ha="left")
    else: ax.text(xi+2.6*w,v,f"GPT-2 + PPO  {v:.1f}",fontsize=6.8,va="center",ha="left")
ax.set_xticks(x); ax.set_xticklabels(tasks); ax.set_ylabel("deployment-sampler accuracy (%)"); ax.set_ylim(0,104); ax.set_xlim(-0.5,3.95)
ax.legend(fontsize=7.5,frameon=False,ncol=5,loc="lower center",bbox_to_anchor=(0.5,-0.24),handlelength=1.4,columnspacing=1.2)
fig.tight_layout(rect=(0,0.02,1,0.99)); fig.savefig("fig_results.pdf"); fig.savefig("fig_results.png",dpi=200); print("fig_results ok")

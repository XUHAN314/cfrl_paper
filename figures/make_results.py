import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"xtick.labelsize":11.5,"ytick.labelsize":11.5,"axes.labelsize":12})
# deployment-sampler accuracy (%) per task; lists = seeds
tasks=["Countdown","Sudoku","K&K","Game24"]
base=[42.2,66.2,67.5,45.0]
grpo=[[43.1,42.8,43.3],[70.1,72.8,66.2],[70.2,70.0,74.6],[46.8,48.6,47.1]]
scal=[[48.5,47.4,48.6],[65.5,67.3,66.0],[77.8,77.1,78.0],[50.9,52.5,50.7]]
jpc=[[50.1,50.2,50.4],[70.8,71.3,71.0],[79.7,79.2,80.0],[58.2,59.3,58.9]]
loop=[50.2,71.4,99.1,None]  # no Game24 loop: the set split leaves no pool
ar=[53.9,1.4,93.2,32.2]
fig,ax=plt.subplots(figsize=(7.6,3.4))
w=0.155; x=np.arange(len(tasks))
def bar(off,vals,color,label):
    m=[np.mean(v) if isinstance(v,list) else v for v in vals]; e=[np.std(v) if isinstance(v,list) and len(v)>1 else 0 for v in vals]
    xs=[xi+off for xi,v in zip(x,vals) if v is not None]; ms=[mi for mi,v in zip(m,vals) if v is not None]; es=[ei for ei,v in zip(e,vals) if v is not None]
    ax.bar(xs,ms,w,color=color,label=label,yerr=es,capsize=2,error_kw=dict(lw=0.8))
    for xi,mi,ei in zip(xs,ms,es): ax.text(xi,mi+ei+1.0,f"{mi:.1f}",ha="center",va="bottom",fontsize=8.4,zorder=5,bbox=dict(facecolor="white",edgecolor="none",pad=0.3))
bar(-2*w,base,"#bbbbbb","SFT base")
bar(-1*w,grpo,"#e08a3c","CFRL-GRPO")
bar(0,scal,"#8fb3e0","CFRL-SVG-Scalar")
bar(1*w,jpc,"#2b6cb0","CFRL-SVG-2D (ours)")
bar(2*w,loop,"#1a9c5b","2D policy + amortisation loop")
place={0:("left",4.2),1:("right",0),2:("left",1.6),3:("right",0)}  # label above the line at the group's left edge, or to the right of the group
for xi,v in zip(x,ar):
    if v is None: continue
    ax.plot([xi-2.5*w,xi+2.5*w],[v,v],ls="--",lw=1.2,color="k")
    side,dy=place[int(xi)]
    if side=="left": ax.text(xi-2.5*w,v+dy,f"GPT-2 + PPO  {v:.1f}",fontsize=9.5,va="bottom",ha="left")
    else: ax.text(xi+2.6*w,v,f"GPT-2 + PPO  {v:.1f}",fontsize=9.5,va="center",ha="left")
ax.set_xticks(x); ax.set_xticklabels(tasks); ax.set_ylabel("deployment-sampler accuracy (%)"); ax.set_ylim(0,104); ax.set_xlim(-0.5,3.95)
ax.legend(fontsize=10,frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.17),handlelength=1.4,columnspacing=1.2)
fig.subplots_adjust(left=0.09,right=0.99,top=0.97,bottom=0.34); import os; _d=os.path.dirname(os.path.abspath(__file__)); fig.savefig(os.path.join(_d,"fig_results.pdf"),bbox_inches="tight"); print("fig_results ok")

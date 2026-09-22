import re, glob, json, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8.5,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
R="/cluster/tufts/liulab/xhan03/flow-nlp-rl"
def parse(log):
    it_v=[]; buf=[]; it=0
    for l in open(log):
        m=re.match(r"\[(\d+)\] V=([-\d.]+) ",l)
        if m: it=int(m.group(1)); it_v.append((it,float(m.group(2))))
        m=re.search(r"\[joint\] buf \d+ outcome ([\d.]+)",l)
        if m: buf.append((it,float(m.group(1))))
    return np.array(it_v),np.array(buf)
def evals(prefix,task,key):
    out=[]
    for d in glob.glob(f"{R}/outputs/{prefix}_ckpt*"):
        ck=int(d.rsplit("ckpt",1)[1])
        for f in glob.glob(d+"/*/*_metrics.json"):
            sub=os.path.basename(os.path.dirname(f))
            if key in sub:
                m=json.load(open(f)); v=m.get("all",m); a=v.get("solved",v.get("accuracy")) if isinstance(v,dict) else v
                out.append((ck,float(a)*100 if a<=1 else float(a)))
    return np.array(sorted(out))
panels=[("Sudoku, 2D critic lr 1e-6 (seed 1)","slurm/logs/cd-pathwise-3390784.out","eval_sd_sd_jpc_tok_lr1e6","gamma28",66.2),
        ("Game24, 2D critic lr 1e-6 (seed 2)","slurm/logs/cd-pathwise-3402644.out","eval_g24_g24_jpc_tok_lr1e6_s2","gamma56",45.0)]
fig,axes=plt.subplots(1,2,figsize=(5.6,2.2))
for ax,(title,log,pref,key,base) in zip(axes,panels):
    v,b=parse(f"{R}/{log}"); e=evals(pref,None,key)
    ax2=ax.twinx(); ax2.spines["top"].set_visible(False)
    if len(v): ax2.plot(v[:,0],np.convolve(v[:,1],np.ones(15)/15,mode="same"),color="#c0392b",lw=0.8,alpha=0.8,label="critic value $V_\\phi$ (train batch)")
    if len(b): ax.plot(b[:,0],b[:,1]*100,color="#7f8c8d",lw=1,label="replay-buffer success (train)")
    if len(e): ax.plot(e[:,0],e[:,1],"o-",color="#2b6cb0",lw=1.2,ms=3,label="held-out accuracy (deployment)")
    ax.axhline(base,color="k",ls=":",lw=0.8); ax.set_title(title,fontsize=8); ax.set_xlabel("iteration"); ax.set_ylabel("accuracy (%)")
    ax2.set_ylabel("$V_\\phi$",color="#c0392b"); ax2.tick_params(axis="y",colors="#c0392b")
    if len(e):
        pk=e[np.argmax(e[:,1])]; ax.annotate("peak / early-stop",xy=(pk[0],pk[1]),xytext=(pk[0]+80,pk[1]-9),fontsize=7.5,arrowprops=dict(arrowstyle="->",lw=0.6))
H={}
for a in fig.axes:
    for h,l in zip(*a.get_legend_handles_labels()): H.setdefault(l,h)
fig.legend(list(H.values()),list(H.keys()),fontsize=7.5,frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=(0,0.08,1,1)); fig.savefig(f"{R}/paper/figures/fig_curves.pdf"); fig.savefig(f"{R}/paper/figures/fig_curves.png",dpi=200); print("curves ok", len(v), len(b), len(e))

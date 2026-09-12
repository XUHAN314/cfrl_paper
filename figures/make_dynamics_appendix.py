"""Appendix: training dynamics of 2D-critic JPC on Countdown and K&K (same layout as the main-text figure)."""
import re, glob, json, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":7.5,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
R="/cluster/tufts/liulab/xhan03/flow-nlp-rl"
def parse(log):
    it_v=[]; buf=[]; it=0
    for l in open(log):
        m=re.match(r"\[(\d+)\] V=([-\d.]+) ",l)
        if m: it=int(m.group(1)); it_v.append((it,float(m.group(2))))
        m=re.search(r"\[joint\] buf \d+ outcome ([\d.]+)",l)
        if m: buf.append((it,float(m.group(1))))
    return np.array(it_v),np.array(buf)
def evals(prefix,key):
    out=[]
    for d in glob.glob(f"{R}/outputs/{prefix}_ckpt*"):
        ck=int(d.rsplit("ckpt",1)[1])
        for f in glob.glob(d+"/*/*_metrics.json"):
            if key in os.path.basename(os.path.dirname(f)):
                m=json.load(open(f)); v=m.get("all",m); a=v.get("solved",v.get("accuracy")) if isinstance(v,dict) else v
                out.append((ck,float(a)*100 if a<=1 else float(a)))
    return np.array(sorted(out))
fig,axes=plt.subplots(1,2,figsize=(6.4,2.3))
panels=[("Countdown, 2D critic, lr 1e-6","slurm/logs/cd-pathwise-3404243.out","eval_cd5_cd5_jpc_tok_lr1e6","gamma56",42.2,(150,-6)),
        ("K&K, 2D critic, lr 1e-6","slurm/logs/cd-pathwise-3396157.out","eval_kk_kk_jpc_tok_lr1e6","gamma56",67.5,(60,-4))]
H={}
for ax,(title,log,pref,key,base,off) in zip(axes,panels):
    v,b=parse(f"{R}/{log}"); e=evals(pref,key); ax2=ax.twinx(); ax2.spines["top"].set_visible(False)
    ax2.plot(v[:,0],np.convolve(v[:,1],np.ones(15)/15,mode="same"),color="#c0392b",lw=0.8,alpha=0.85,label="critic value $V_\\phi$ (train batch)")
    if len(b): ax.plot(b[:,0],b[:,1]*100,color="#7f8c8d",lw=1,label="replay-buffer success (train)")
    ax.plot(e[:,0],e[:,1],"o-",color="#2b6cb0",lw=1.2,ms=3,label="held-out accuracy (deployment)")
    ax.axhline(base,color="k",ls=":",lw=0.8); ax.set_title(title,fontsize=7.5); ax.set_xlabel("iteration"); ax.set_ylabel("accuracy (%)")
    ax2.set_ylabel("$V_\\phi$",color="#c0392b",labelpad=1); ax2.tick_params(axis="y",colors="#c0392b",labelsize=6)
    pk=e[np.argmax(e[:,1])]; ax.annotate("peak (early-stop)",xy=(pk[0],pk[1]),xytext=(pk[0]+off[0],pk[1]+off[1]),fontsize=6,arrowprops=dict(arrowstyle="->",lw=0.6))
    for a in (ax,ax2):
        for h,l in zip(*a.get_legend_handles_labels()): H.setdefault(l,h)
    print(title, "evals", e.tolist(), "V range", v[:,1].min(), v[:,1].max(), "buf n", len(b))
fig.legend(list(H.values()),list(H.keys()),fontsize=6.5,frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(0.5,-0.01))
fig.tight_layout(rect=(0,0.08,1,1)); fig.savefig("fig_dynamics_cd_kk.pdf"); fig.savefig("fig_dynamics_cd_kk.png",dpi=200); print("ok")

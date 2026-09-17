"""Appendix: training dynamics of 2D-critic JPC on Countdown and K&K (same layout as the main-text figure)."""
import re, glob, json, os, numpy as np, style; import matplotlib.pyplot as plt; style.apply()
plt.rcParams.update({"font.size":9,"axes.labelsize":9,"xtick.labelsize":8.5,"ytick.labelsize":8.5,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
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
fig,axes=plt.subplots(1,2,figsize=(7.2,2.9))
panels=[("Countdown (Algorithm 1, seed 3)","slurm/logs/cd5_K40_s3-3729236.out","eval_cd5_cd5_K40_s3","gamma56",42.2,(-450,-3)),
        ("K&K (Algorithm 1, seed 3)","slurm/logs/kk_T_s3-3730403.out","eval_kk_kk_T_s3","gamma56",67.5,(-330,-3))]
H={}
for ax,(title,log,pref,key,base,off) in zip(axes,panels):
    v,b=parse(f"{R}/{log}"); e=evals(pref,key); ax2=ax.twinx(); ax2.spines["top"].set_visible(False)
    ax2.plot(v[:,0],np.convolve(v[:,1],np.ones(15)/15,mode="same"),color=style.VAL,lw=1.2,alpha=0.9,label="critic value $V_\\phi$ (train batch)")
    if len(b): ax.plot(b[:,0],b[:,1]*100,color=style.TRUE,lw=1.4,label="replay-buffer success (train)")
    ax.plot(e[:,0],e[:,1],"o-",color=style.ACC,lw=1.8,ms=4.5,label="held-out accuracy (deployment)")
    ax.axhline(base,color=style.BASE,ls=":",lw=0.8); ax.set_title(title,fontsize=9); ax.set_xlabel("iteration"); ax.set_ylabel("accuracy (%)")
    ax2.set_ylabel("$V_\\phi$",color=style.VAL,labelpad=1); ax2.tick_params(axis="y",colors=style.VAL,labelsize=8)
    pk=e[np.argmax(e[:,1])]; ax.annotate("reported peak",xy=(pk[0],pk[1]),xytext=(pk[0]+off[0],pk[1]+off[1]),fontsize=8,arrowprops=dict(arrowstyle="->",lw=0.8))
    for a in (ax,ax2):
        for h,l in zip(*a.get_legend_handles_labels()): H.setdefault(l,h)
    print(title, "evals", e.tolist(), "V range", v[:,1].min(), v[:,1].max(), "buf n", len(b))
fig.legend(list(H.values()),list(H.keys()),fontsize=8,frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(0.5,-0.005))
fig.tight_layout(rect=(0,0.09,1,1)); fig.savefig("fig_dynamics_cd_kk.pdf"); fig.savefig("fig_dynamics_cd_kk.png",dpi=200); print("ok")

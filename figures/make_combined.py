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
fig,axes=plt.subplots(1,3,figsize=(7.2,2.3),gridspec_kw=dict(width_ratios=[1,1,1.35]))
panels=[("Sudoku, 2D critic, lr 1e-6","slurm/logs/cd-pathwise-3390784.out","eval_sd_sd_jpc_tok_lr1e6","gamma28",66.2,(120,-11)),
        ("Game24, 2D critic, lr 1e-6","slurm/logs/cd-pathwise-3402644.out","eval_g24_g24_jpc_tok_lr1e6_s2","gamma56",45.0,(-430,3))]
H={}
for ax,(title,log,pref,key,base,off) in zip(axes[:2],panels):
    v,b=parse(f"{R}/{log}"); e=evals(pref,key); ax2=ax.twinx(); ax2.spines["top"].set_visible(False)
    ax2.plot(v[:,0],np.convolve(v[:,1],np.ones(15)/15,mode="same"),color="#c0392b",lw=0.8,alpha=0.85,label="critic value $V_\\phi$ (train batch)")
    ax.plot(b[:,0],b[:,1]*100,color="#7f8c8d",lw=1,label="replay-buffer success (train)")
    ax.plot(e[:,0],e[:,1],"o-",color="#2b6cb0",lw=1.2,ms=3,label="held-out accuracy (deployment)")
    ax.axhline(base,color="k",ls=":",lw=0.8); ax.set_title(title,fontsize=7.5); ax.set_xlabel("iteration"); ax.set_ylabel("accuracy (%)")
    ax2.set_ylabel("$V_\\phi$",color="#c0392b",labelpad=1); ax2.tick_params(axis="y",colors="#c0392b",labelsize=6)
    pk=e[np.argmax(e[:,1])]; ax.annotate("peak (early-stop)",xy=(pk[0],pk[1]),xytext=(pk[0]+off[0],pk[1]+off[1]),fontsize=6,arrowprops=dict(arrowstyle="->",lw=0.6))
    for a in (ax,ax2):
        for h,l in zip(*a.get_legend_handles_labels()): H.setdefault(l,h)
ax=axes[2]; labels=["SFT\nbase","2D-critic\npolicy","loop\nfrom it\n(4 rounds)","loop\nfrom base\n(4 rounds)"]
dep=[45.0,59.3,58.1,46.1]; ode=[35.2,41.1,50.9,40.2]; xx=np.arange(4)
ax.bar(xx-0.19,dep,0.38,color="#1a9c5b",label="deployment ($\\gamma$56@64)"); ax.bar(xx+0.19,ode,0.38,color="#7fc9a5",label="cheap (ODE-64)")
for xi,d_,o in zip(xx,dep,ode): ax.text(xi-0.19,d_+0.6,f"{d_:.1f}",ha="center",fontsize=5.5); ax.text(xi+0.19,o+0.6,f"{o:.1f}",ha="center",fontsize=5.5)
ax.set_xticks(xx); ax.set_xticklabels(labels,fontsize=5.6); ax.set_ylim(30,67); ax.set_ylabel("Game24 accuracy (%)"); ax.set_title("Loop attribution (Game24)",fontsize=7.5); ax.legend(fontsize=6,frameon=False,loc="upper left")
fig.legend(list(H.values()),list(H.keys()),fontsize=6.5,frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(0.36,-0.01))
fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig("fig_dynamics_loop.pdf"); fig.savefig("fig_dynamics_loop.png",dpi=200); print("combined regenerated")

import re, glob, json, os, numpy as np, style; import matplotlib.pyplot as plt; style.apply()
plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"axes.labelsize":12,"xtick.labelsize":11,"ytick.labelsize":11})
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
fig,axes=plt.subplots(1,3,figsize=(7.6,3.0),gridspec_kw=dict(width_ratios=[1,1,1.6]))
panels=[("Sudoku (Algorithm 1, seed 2)","slurm/logs/sd_T_s2-3730399.out","eval_sd_sd_T_s2","gamma28",66.2,(120,-2.2)),
        ("Game24 (Algorithm 1, seed 2)","slurm/logs/g24_T_s2-3730401.out","eval_g24_g24_T_s2","gamma56",45.0,(-560,2.5))]
H={}
for ax,(title,log,pref,key,base,off) in zip(axes[:2],panels):
    v,b=parse(f"{R}/{log}"); e=evals(pref,key); ax2=ax.twinx(); ax2.spines["top"].set_visible(False)
    ax2.plot(v[:,0],np.convolve(v[:,1],np.ones(15)/15,mode="same"),color=style.VAL,lw=1.2,alpha=0.9,label="critic $V_\phi$ (train batch)")
    ax.plot(b[:,0],b[:,1]*100,color=style.TRUE,lw=1.4,label="replay-buffer success (train)")
    ax.plot(e[:,0],e[:,1],"o-",color=style.ACC,lw=1.8,ms=4.5,label="held-out accuracy (deploy.)")
    ax.axhline(base,color=style.BASE,ls=":",lw=1); ax.set_title(title,fontsize=10); ax.set_xlabel("iteration"); ax.set_ylabel("accuracy (%)")
    ax2.set_ylabel("$V_\\phi$",color=style.VAL,labelpad=1); ax2.tick_params(axis="y",colors=style.VAL,labelsize=11)
    pk=e[np.argmax(e[:,1])]; ax.annotate("reported peak",xy=(pk[0],pk[1]),xytext=(pk[0]+off[0],pk[1]+off[1]),fontsize=10.5,arrowprops=dict(arrowstyle="->",lw=0.8))
    for a in (ax,ax2):
        for h,l in zip(*a.get_legend_handles_labels()): H.setdefault(l,h)
ax=axes[2]; labels=["SFT\nbase","2D\npolicy","loop:\n2D","loop:\nbase","loop:\nGRR","gold\nlabels"]
dep=[45.0,59.3,56.0,46.1,50.5,49.2]; ode=[35.2,42.2,50.5,40.2,43.5,38.7]; xx=np.arange(len(labels))
ax.bar(xx-0.2,dep,0.4,color="#1a9c5b",label="deployment ($\\gamma$56@64)"); ax.bar(xx+0.2,ode,0.4,color="#7fc9a5",label="ODE-64")
for xi,d_,o in zip(xx,dep,ode): ax.text(xi-0.2,d_+0.7,f"{d_:.1f}",ha="center",fontsize=9); ax.text(xi+0.2,o+0.7,f"{o:.1f}",ha="center",fontsize=9)
ax.set_xticks(xx); ax.set_xticklabels(labels,fontsize=8.5,rotation=30,ha="right"); ax.set_ylim(30,70); ax.set_xlim(-0.55,5.55); ax.set_ylabel("Game24 accuracy (%)"); ax.set_title("Game24 loop and controls",fontsize=11); ax.legend(fontsize=7.5,frameon=False,loc="upper right",ncol=2,columnspacing=0.8,handlelength=1.2)
fig.legend(list(H.values()),list(H.keys()),fontsize=10.5,frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(0.5,-0.005))
fig.tight_layout(rect=(0,0.12,1,1)); fig.savefig("fig_dynamics_loop.pdf"); fig.savefig("fig_dynamics_loop.png",dpi=200); print("combined regenerated")

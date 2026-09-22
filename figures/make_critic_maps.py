"""Heat maps of per-position 2D-critic values along the trajectory for held-out examples."""
import sys, torch, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":7.5,"pdf.fonttype":42})
R="/cluster/tufts/liulab/xhan03/flow-nlp-rl"
def load(p): return torch.load(p, map_location="cpu")
def n_eff(toks):
    return (toks.index("</s>")+1) if "</s>" in toks else len(toks)
def auc(scores, labels):
    scores=np.asarray(scores,float); labels=np.asarray(labels,float); pos=scores[labels>0.5]; neg=scores[labels<0.5]
    if len(pos)==0 or len(neg)==0: return float("nan")
    order=np.argsort(np.concatenate([pos,neg])); ranks=np.empty(len(order)); ranks[order]=np.arange(1,len(order)+1)
    # average ranks for ties
    allv=np.concatenate([pos,neg]); 
    for v in np.unique(allv):
        m=allv==v
        if m.sum()>1: ranks[m]=ranks[m].mean()
    return float((ranks[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
def panel(ax, d, i, title, sudoku=False):
    lg=d["logits"][i]; gm=d["gm"][i].bool(); ks=d["ks"]; K=d["K"]
    # generated positions = where gm==1, aligned with ids after shift_left: take the first gen_len masked positions
    pos=gm.nonzero().squeeze(-1); toks=d["tokens"][i]; n=n_eff(toks)
    n=min(n,len(pos)); P=torch.sigmoid(lg[:,pos[:n]]).numpy()
    im=ax.imshow(P,aspect="auto",vmin=0,vmax=1,cmap="RdYlGn",origin="lower")
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([f"{k}/{K}" for k in ks],fontsize=7); ax.set_ylabel("denoising step $k$",fontsize=7.5)
    if sudoku:
        tg=d["target"][i].split(); hy=d["hyp"][i].split(); wrong=[j for j in range(min(len(tg),len(hy),n)) if tg[j]!=hy[j]]
        ax.set_xticks(range(0,81,9)); ax.set_xticklabels([f"row {j//9+1}" for j in range(0,81,9)],fontsize=7,rotation=45)
        for j in wrong: ax.axvline(j,color="k",lw=0.8,ls=":"); ax.plot(j,len(ks)-0.5,"kv",ms=3)
        if wrong: title=title.replace(" sample","")+f" ({len(wrong)} wrong cells)"
    else:
        ax.set_xticks(range(n)); ax.set_xticklabels([t.strip() or "·" for t in toks[:n]],fontsize=6.5,rotation=90)
    ax.set_title(title,fontsize=8); return im
tasks=[("g24_jpc_s2_1000.pt","Game24",False),("sd_jpc_500.pt","Sudoku",True)]
fig,axes=plt.subplots(2,2,figsize=(5.5,4.4),constrained_layout=True)
def pooled_final(d,i):
    gm=d["gm"][i].bool(); pos=gm.nonzero().squeeze(-1); n=min(n_eff(d["tokens"][i]),len(pos)); return float(torch.sigmoid(d["logits"][i][-1,pos[:n]]).mean())
def n_wrong(d,i):
    tg=d["target"][i].split(); hy=d["hyp"][i].split(); return sum(1 for j in range(min(len(tg),len(hy))) if tg[j]!=hy[j])
for r,(f,name,sud) in enumerate(tasks):
    d=load(f"{R}/outputs/critic_maps/{f}"); oc=d["outcome"].numpy()
    ok=np.where(oc>0.5)[0]; bad=np.where(oc<0.5)[0]
    # representative correct sample: median final pooled value among correct; incorrect: Sudoku -> fewest wrong cells, Game24 -> median pooled value
    vok=sorted(ok,key=lambda i:pooled_final(d,i)); iok=vok[len(vok)//2]
    if sud: ibad=min(bad,key=lambda i:n_wrong(d,i))
    else: vb=sorted(bad,key=lambda i:pooled_final(d,i)); ibad=vb[len(vb)//2]
    for c,(idx,lab) in enumerate([(iok,"correct"),(ibad,"incorrect")]):
        cond=d["cond"][idx]; ttl=f"{name}, {lab} sample" + ("" if sud else f"  ({cond.replace('->', chr(8594))})")
        im=panel(axes[r,c],d,idx,ttl,sudoku=sud)
fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.6,pad=0.01,label="$\\sigma(R_i)$: per-position success estimate")
fig.savefig("fig_critic_maps.pdf",bbox_inches="tight"); fig.savefig("fig_critic_maps.png",dpi=170,bbox_inches="tight"); print("critic maps ok")
# summary statistics: how R_i separates correct vs incorrect samples per step, and per-cell for Sudoku
for f,name,sud in tasks:
    d=load(f"{R}/outputs/critic_maps/{f}"); oc=d["outcome"].numpy(); P=torch.sigmoid(d["logits"]).numpy(); gm=d["gm"].numpy()
    mean=np.zeros((len(oc),P.shape[1]))
    for i in range(len(oc)):
        pos=np.nonzero(gm[i])[0]; n=min(n_eff(d["tokens"][i]),len(pos)); mean[i]=P[i][:,pos[:n]].mean(-1)
    print(name,"mean sigma(R) per k, correct vs incorrect:", [ (k, round(float(mean[oc>0.5,j].mean()),3), round(float(mean[oc<0.5,j].mean()),3)) for j,k in enumerate(d["ks"])])
    if sud:
        # per-cell: critic value at the last stored k vs cell correctness
        pos=[torch.tensor(g).bool().nonzero().squeeze(-1)[:81] for g in gm]
        vals=[];lab=[]
        for i in range(len(oc)):
            tg=d["target"][i].split(); hy=d["hyp"][i].split(); m=min(81,len(tg),len(hy))
            for j in range(m): vals.append(P[i,-1,pos[i][j]]); lab.append(float(tg[j]==hy[j]))
        vals=np.array(vals); lab=np.array(lab)
        print("Sudoku per-cell AUC of sigma(R_i) at last k vs cell correctness:", round(auc(vals,lab),3), "n cells", len(lab), "wrong frac", round(1-lab.mean(),3))
        # AUC per k, incorrect samples only (within-sample discrimination of the wrong cells)
        for j,k in enumerate(d["ks"]):
            vals=[];lab=[]
            for i in range(len(oc)):
                if oc[i]>0.5: continue
                tg=d["target"][i].split(); hy=d["hyp"][i].split(); m=min(81,len(tg),len(hy))
                for c in range(m): vals.append(P[i,j,pos[i][c]]); lab.append(float(tg[c]==hy[c]))
            print("   k",k,"per-cell AUC (incorrect samples only)",round(auc(vals,lab),3))
    # sequence-level AUC of pooled value per k
    print(name,"sequence AUC of pooled sigma(R) per k:",[(k,round(auc(mean[:,j],oc),3)) for j,k in enumerate(d["ks"])])

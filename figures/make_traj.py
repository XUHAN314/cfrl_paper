"""Figure: scalar vs 2D critic over training. Row 1: held-out accuracy at the deployment sampler per checkpoint (3 seeds thin, mean thick).
Row 2: AUC of the jointly trained head on the newest 256 replay-buffer states (mean over seeds per 250-iteration window, from the training logs)."""
import json, os, glob, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); import style; style.apply()
import matplotlib.pyplot as plt
R = "/cluster/tufts/liulab/xhan03/flow-nlp-rl"
TASKS = [("Countdown", "cd5", "cd5_K40_s%d", "cd5_K40S_s%d", "sde-steps64-cfg2-ts_uniform-gamma56.0-cond", "countdown_metrics.json", 42.2),
         ("Sudoku", "sd", "sd_T_s%d", "sd_TS_s%d", "sde-steps32-cfg2-ts_uniform-gamma28.0-cond", "sudoku_metrics.json", 66.2),
         ("K&K", "kk", "kk_T_s%d", "kk_TS_s%d", "sde-steps64-cfg2-ts_uniform-gamma56.0-cond", "kk_metrics.json", 67.5),
         ("Game24", "g24", "g24_T_s%d", "g24_TS_s%d", "sde-steps64-cfg2-ts_uniform-gamma56.0-cond", "countdown_metrics.json", 45.0)]
def acc(pre, run, c, sampler, mfile):
    f = f"{R}/outputs/eval_{pre}_{run}_ckpt{c}/{sampler}/{mfile}"
    if not os.path.exists(f): return None
    d = json.load(open(f)); return 100 * (d["all"]["solved"] if "all" in d else d["accuracy"])
gap = json.load(open(f"{R}/outputs/analysis/train_log_gap.json"))
fig, axes = plt.subplots(2, 4, figsize=(5.5, 3.4), sharex="col", gridspec_kw=dict(height_ratios=[1.35, 1], hspace=0.12, wspace=0.32))
C = {"2D": style.ACC, "scalar": style.VAL}
for j, (name, pre, p2d, psc, sampler, mfile, base) in enumerate(TASKS):
    ax, ax2 = axes[0, j], axes[1, j]
    for lab, pat in (("scalar", psc), ("2D", p2d)):
        curves = []
        for s in (1, 2, 3):
            xs, ys = [], []
            for c in range(250, 1501, 250):
                a = acc(pre, pat % s, c, sampler, mfile)
                if a is not None: xs.append(c); ys.append(a)
            if xs: ax.plot(xs, ys, color=C[lab], lw=0.7, alpha=0.3); curves.append(dict(zip(xs, ys)))
        allx = sorted(set(x for cv in curves for x in cv)); mean = [np.mean([cv[x] for cv in curves if x in cv]) for x in allx]
        n = [sum(x in cv for cv in curves) for x in allx]; keep = [i for i, k in enumerate(n) if k == 3]  # mean only where all seeds have the checkpoint
        ax.plot([allx[i] for i in keep], [mean[i] for i in keep], color=C[lab], lw=1.8, marker="o", ms=3, label=f"{lab} critic")
        # AUC row from the logs
        per_seed = gap[name][lab]; ws = sorted(set(int(w) for ws_, _, _ in per_seed for w in ws_))
        auc = [np.nanmean([ws_[str(w)]["auc"] for ws_, _, _ in per_seed if str(w) in ws_]) for w in ws]
        ax2.plot(ws, auc, color=C[lab], lw=1.6, marker="o", ms=2.5)
    ax.axhline(base, color=style.BASE, ls="--", lw=0.8)
    ax2.axhline(0.9, color=style.BASE, ls=":", lw=0.8)
    ax.set_title(name, fontsize=8.5, pad=3); ax2.set_xlabel("actor iteration", fontsize=7.5); ax2.set_ylim(0.85, 1.0); ax2.set_yticks([0.85, 0.9, 0.95, 1.0])
    ax2.set_xticks([500, 1000, 1500] if name in ("Countdown", "Game24") else [500, 1000]); ax2.tick_params(labelsize=7); ax.tick_params(labelsize=7)
    if j == 0: ax.set_ylabel("held-out accuracy (%)", fontsize=7.5); ax2.set_ylabel("critic AUC,\nnewest buffer states", fontsize=7.5)
axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.6, borderaxespad=0.2)
out = os.path.join(os.path.dirname(__file__), "fig_traj.pdf"); fig.savefig(out, bbox_inches="tight", pad_inches=0.02); print("wrote", out)

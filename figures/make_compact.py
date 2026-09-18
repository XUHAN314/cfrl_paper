"""Compact main-text figures for §6.
fig_estimator.pdf : half-split cosine vs N for the estimators of Table tab:snr (legend carries the SNR at N=256).
fig_frozen.pdf    : single panel, frozen critic on Game24 (value on the training batch, true success, held-out accuracy).
fig_dyn.pdf       : three panels: Countdown held-out accuracy scalar vs 2D over training; Countdown critic AUC on the newest
                    buffer states; Game24 critic exchange at iteration 250."""
import re, glob, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import style; style.apply()
import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__)); R = "/cluster/tufts/liulab/xhan03/flow-nlp-rl"
plt.rcParams.update({"font.size": 7.5})
# ---------------- estimator ----------------
N = [256, 1024, 2048]
series = [("SF, mean-centred $A$  (SNR 0.061)", [-0.03, -0.04, 0.00], style.VAL, "-", "o"),
          ("SF, GRPO $A$  (0.065)", [0.04, 0.02, -0.04], "#c9553d", "--", "s"),
          ("SF, $A{=}1$ control  (0.063)", [0.02, 0.04, 0.02], style.BASE, ":", "x"),
          ("pathwise, 2D critic  (0.177)", [0.79, 0.94, 0.97], style.ACC, "-", "o"),
          ("GPT-2 token GRPO, Countdown", [0.05, 0.17, 0.27], "#2a9d8f", "-", "^")]
fig, ax = plt.subplots(figsize=(2.15, 2.05))
for lab, ys, c, ls, mk in series: ax.plot(N, ys, ls, color=c, marker=mk, ms=3.2, lw=1.4, label=lab)
ax.axhline(0, color=style.BASE, lw=0.6); ax.set_xscale("log", base=2); ax.set_xticks(N); ax.set_xticklabels([str(n) for n in N])
ax.set_xlabel("rollouts $N$", fontsize=7.5); ax.set_ylabel("half-split cosine", fontsize=7.5); ax.set_ylim(-0.12, 1.02); ax.tick_params(labelsize=7)
ax.legend(fontsize=6.0, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.55), handlelength=1.6, borderaxespad=0.0, labelspacing=0.35)
fig.savefig(os.path.join(D, "fig_estimator.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# ---------------- frozen ----------------
def parse(log):
    it_v, it_t = [], []
    for l in open(log):
        m = re.match(r"\[(\d+)\] V=([-\d.]+) .*true_acc=([\d.]+)", l)
        if m: it = int(m.group(1)); it_v.append((it, float(m.group(2)))); it_t.append((it, float(m.group(3))))
    return np.array(it_v), np.array(it_t)
def evals(prefix, key="gamma56"):
    out = []
    for d in glob.glob(f"{R}/outputs/{prefix}_ckpt*"):
        ck = int(d.rsplit("ckpt", 1)[1])
        for f in glob.glob(d + "/*/*_metrics.json"):
            if key in os.path.basename(os.path.dirname(f)):
                m = json.load(open(f)); v = m.get("all", m); a = v.get("solved", v.get("accuracy")); out.append((ck, 100 * a))
    return np.array(sorted(out))
def smooth(x, w=5): return np.convolve(x, np.ones(w) / w, mode="same") if len(x) >= w else x
v, t = parse(f"{R}/slurm/logs/cd-pathwise-3447085.out"); e = evals("eval_g24_g24_abl_frozen")
fig, ax = plt.subplots(figsize=(2.0, 1.25)); ax2 = ax.twinx(); ax2.spines["top"].set_visible(False); ax2.grid(False)
ax.plot(t[:, 0], smooth(t[:, 1]), color=style.TRUE, lw=1.2); ax.plot(e[:, 0], e[:, 1], "o-", color=style.ACC, lw=1.4, ms=3)
ax2.plot(v[:, 0], smooth(v[:, 1]), color=style.VAL, lw=1.2); ax.axhline(45.0, color=style.BASE, ls=":", lw=0.8)
ax.set_xlabel("actor iteration", fontsize=7.5); ax.set_ylabel("accuracy (%)", fontsize=7.5); ax2.set_ylabel("$V_\\phi$ (logit)", color=style.VAL, fontsize=7.5)
ax.set_ylim(0, 78); ax.set_xlim(0, 1000); ax.tick_params(labelsize=7); ax2.tick_params(axis="y", colors=style.VAL, labelsize=7); ax2.spines["right"].set_color(style.VAL)
fig.savefig(os.path.join(D, "fig_frozen.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# ---------------- dynamics: Countdown acc, Countdown AUC, Game24 exchange ----------------
def acc(pre, run, c, sampler, mfile):
    f = f"{R}/outputs/eval_{pre}_{run}_ckpt{c}/{sampler}/{mfile}"
    if not os.path.exists(f): return None
    d = json.load(open(f)); return 100 * d["accuracy"]
gap = json.load(open(f"{R}/outputs/analysis/train_log_gap.json"))
fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.9), gridspec_kw=dict(wspace=0.55, width_ratios=[1.1, 1, 1.15]))
C = {"2D": style.ACC, "scalar": style.VAL}; S64 = "sde-steps64-cfg2-ts_uniform-gamma56.0-cond"
ax = axes[0]
for lab, pat in (("scalar", "cd5_K40S_s%d"), ("2D", "cd5_K40_s%d")):
    curves = []
    for s in (1, 2, 3):
        xs, ys = [], []
        for c in range(250, 1501, 250):
            a = acc("cd5", pat % s, c, S64, "countdown_metrics.json")
            if a is not None: xs.append(c); ys.append(a)
        ax.plot(xs, ys, color=C[lab], lw=0.6, alpha=0.3); curves.append(dict(zip(xs, ys)))
    allx = sorted(set(x for cv in curves for x in cv)); mean = [np.mean([cv[x] for cv in curves if x in cv]) for x in allx]
    ax.plot(allx, mean, color=C[lab], lw=1.7, marker="o", ms=2.8, label=f"{lab} critic")
ax.axhline(42.2, color=style.BASE, ls="--", lw=0.8); ax.set_title("Countdown: held-out accuracy", fontsize=7, pad=3); ax.set_ylabel("accuracy (%)", fontsize=7)
ax.set_xlabel("actor iteration", fontsize=7.5); ax.tick_params(labelsize=7); ax.set_xticks([500, 1000, 1500]); ax.legend(fontsize=6.3, frameon=False, loc="lower right", handlelength=1.5, borderaxespad=0.2)
ax = axes[1]
for lab in ("scalar", "2D"):
    per_seed = gap["Countdown"][lab]; ws = sorted(set(int(w) for ws_, _, _ in per_seed for w in ws_))
    auc = [np.nanmean([ws_[str(w)]["auc"] for ws_, _, _ in per_seed if str(w) in ws_]) for w in ws]
    ax.plot(ws, auc, color=C[lab], lw=1.6, marker="o", ms=2.5, label=f"{lab} critic")
ax.axhline(0.9, color=style.BASE, ls=":", lw=0.8); ax.set_title("Countdown: head AUC", fontsize=7, pad=3); ax.set_ylabel("AUC, newest buffer states", fontsize=7)
ax.set_ylim(0.85, 1.0); ax.set_yticks([0.85, 0.9, 0.95, 1.0]); ax.set_xticks([500, 1000, 1500]); ax.set_xlabel("actor iteration", fontsize=7.5); ax.tick_params(labelsize=7)
ax.text(0.04, 0.08, "floor $a_0$", transform=ax.transAxes, fontsize=6, color=style.BASE)
ax = axes[2]
its = [250, 500, 750, 1000, 1250, 1500]
for lab, ys, c, ls in [("scalar throughout", [47.7, 49.8, 51.6, 49.8, 49.9, 50.8], style.VAL, "-"), ("scalar $\\to$ 2D at 250", [47.7, 51.3, 51.5, 54.2, 55.8, 56.4], style.ACC, "--"),
                       ("2D throughout", [49.8, 52.3, 52.4, 56.1, 56.7, 57.3], style.ACC, "-"), ("2D $\\to$ scalar at 250", [49.8, 51.7, 51.8, 50.1, 50.1, 51.3], style.VAL, "--")]:
    ax.plot(its, ys, ls, color=c, lw=1.5, marker="o", ms=2.6, label=lab)
ax.axvline(250, color=style.BASE, ls=":", lw=0.8); ax.set_title("Game24: critics exchanged", fontsize=7, pad=3); ax.set_ylabel("accuracy (%)", fontsize=7)
ax.set_xlim(200, 1550); ax.set_ylim(46.5, 61); ax.set_xticks([250, 750, 1250]); ax.set_xlabel("actor iteration", fontsize=7.5); ax.tick_params(labelsize=7)
ax.legend(fontsize=5.8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, columnspacing=0.8, handlelength=1.6, borderaxespad=0.1, labelspacing=0.25)
fig.savefig(os.path.join(D, "fig_dyn.pdf"), bbox_inches="tight", pad_inches=0.02); print("compact figures written")

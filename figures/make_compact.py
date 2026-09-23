"""Compact main-text figures for §6 (round 2).
fig_estimator.pdf : half-split cosine vs N (Game24; GPT-2 row on Countdown), 4 series, legend below, fonts >= 7 pt at 0.40 linewidth.
fig_frozen.pdf    : frozen critic on Game24, twin axis (% left, critic value in logits right): true success of the training batch, held-out accuracy.
fig_dyn.pdf       : two Game24 panels, shared y: scalar vs 2D held-out trajectories (3 seeds) | critics exchanged at iteration 250."""
import re, glob, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import style; style.apply()
import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__)); R = "/cluster/tufts/liulab/xhan03/flow-nlp-rl"
F = 8.0; plt.rcParams.update({"font.size": F, "legend.fontsize": F, "axes.titlesize": F + 0.3, "axes.labelsize": F, "xtick.labelsize": F, "ytick.labelsize": F})
SC, TD = style.VAL, style.ACC   # orange = scalar critic, blue = 2D critic, everywhere in §6
# ---------------- estimator (placed at 0.385*5.5 = 2.1 in): two stacked panels ----------------
# top: half-split cosine vs N; bottom: held-out gain of ONE step along each direction (Game24, AdamW-normalised directions, paired 95% CI)
N = [256, 1024, 2048]; GRPO_C = "#c9553d"; GPT_C = "#2a9d8f"
_one = json.load(open(f"{R}/outputs/analysis/onestep_g24/results.json"))["eval"]["results"]
_fr = ["0.004", "0.008", "0.016", "0.032", "0.0625", "0.25", "0.5", "1"]; _x = [float(f) for f in _fr]
def _series(tag): 
    r = [_one[f"{tag}_eta{f}D"] for f in _fr]; return np.array([q["diff"] for q in r]), np.array([q["ci95"][0] for q in r]), np.array([q["ci95"][1] for q in r])
fig, (ax, bx) = plt.subplots(2, 1, figsize=(2.1, 2.8), gridspec_kw=dict(hspace=0.78, height_ratios=[1, 1.08]))
ax.plot(N, [0.04, 0.02, -0.04], "-", color=GRPO_C, marker="o", ms=3.0, lw=1.4, label="score-function")
ax.plot(N, [0.02, 0.04, 0.02], ":", color=style.BASE, marker="o", ms=3.0, lw=1.3, label="control / random")
ax.plot(N, [0.79, 0.94, 0.97], "-", color=TD, marker="o", ms=3.0, lw=1.6, label="pathwise (2D)")
ax.plot(N, [0.05, 0.17, 0.27], "-", color=GPT_C, marker="o", ms=3.0, lw=1.4, label="GPT-2 (Countdown)")
ax.axhline(0, color=style.BASE, lw=0.6); ax.set_xscale("log", base=2); ax.set_xticks(N); ax.set_xticklabels([str(n) for n in N]); ax.minorticks_off()
ax.set_xlabel("rollouts $N$", labelpad=1); ax.set_ylabel("half-split cosine"); ax.set_ylim(-0.12, 1.04); ax.set_yticks([0, 0.5, 1.0])
for tag, col, ls, lw, band in (("RAND", style.BASE, ":", 1.2, False), ("SF-1024+adam", GRPO_C, "-", 1.4, True), ("PW-1024+adam", TD, "-", 1.6, True)):
    d_, lo, hi = _series(tag)
    if band: bx.fill_between(_x, lo, hi, color=col, alpha=0.16, lw=0)
    bx.plot(_x, d_, ls, color=col, marker="o", ms=3.0, lw=lw)
bx.axhline(0, color=style.BASE, lw=0.6); bx.set_xscale("log", base=2); bx.set_xticks([1 / 256, 1 / 64, 1 / 16, 1 / 4, 1]); bx.set_xticklabels(["1/256", "1/64", "1/16", "1/4", "1"]); bx.minorticks_off()
bx.set_xlabel("length of one step (units of $\\Lambda$)", labelpad=1); bx.set_ylabel("$\\Delta$ accuracy (pts)"); bx.set_ylim(-4.2, 7.0); bx.set_yticks([-3, 0, 3, 6])
fig.align_ylabels([ax, bx])
fig.subplots_adjust(bottom=0.36, top=0.98, left=0.24, right=0.97)
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=1, frameon=False, fontsize=7.0, handlelength=1.2, labelspacing=0.2, handletextpad=0.4, bbox_to_anchor=(0.6, 0.0))
fig.savefig(os.path.join(D, "fig_estimator.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# ---------------- frozen critic (placed at 0.42*5.5 = 2.3 in) ----------------
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
def smooth_valid(x, y, w=5):
    if len(y) < w: return x, y
    return x[w // 2: len(y) - (w - 1 - w // 2)], np.convolve(y, np.ones(w) / w, mode="valid")
v, t = parse(f"{R}/slurm/logs/cd-pathwise-3447085.out"); e = evals("eval_g24_g24_abl_frozen")
fig, ax = plt.subplots(figsize=(1.95, 2.15)); ax2 = ax.twinx(); ax2.spines["top"].set_visible(False); ax2.grid(False)
xt, yt = smooth_valid(t[:, 0], t[:, 1]); xv, yv = smooth_valid(v[:, 0], v[:, 1])
h1, = ax.plot(xt, yt, color=style.TRUE, lw=1.3, label="batch success")
h2, = ax.plot(e[:, 0], e[:, 1], "o-", color="#212529", lw=1.4, ms=3, label="held-out acc.")
h3, = ax2.plot(xv, yv, color="#6a4c93", lw=1.4, label="critic $V_\\phi$ (logits, right)")
h4 = ax.axhline(45.0, color=style.BASE, ls=":", lw=0.8, label="SFT base")
ax.set_xlabel("actor iteration"); ax.set_ylabel("%"); ax.set_ylim(0, 100); ax.set_xlim(0, 1000); ax.set_yticks([0, 50, 100])
ax2.tick_params(axis="y", colors="#6a4c93"); ax2.spines["right"].set_color("#6a4c93"); ax2.set_ylim(-5, 100); ax2.set_yticks([0, 50, 100]); ax2.set_ylabel("logit", color="#6a4c93", labelpad=1)
fig.subplots_adjust(bottom=0.50, top=0.97, left=0.19, right=0.82)
fig.legend([h1, h2, h3, h4], ["batch success (train)", "held-out accuracy", "critic value (right axis)", "base"], loc="lower center", ncol=1, frameon=False, fontsize=7.2, handlelength=1.2, labelspacing=0.2, handletextpad=0.4, bbox_to_anchor=(0.52, 0.0))
fig.savefig(os.path.join(D, "fig_frozen.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# ---------------- Game24: trajectories | exchange (placed at 5.5 in) ----------------
def acc(pre, run, c, sampler, mfile):
    f = f"{R}/outputs/eval_{pre}_{run}_ckpt{c}/{sampler}/{mfile}"
    if not os.path.exists(f): return None
    return 100 * json.load(open(f))["accuracy"]
S64 = "sde-steps64-cfg2-ts_uniform-gamma56.0-cond"
fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 1.45), sharey=True, gridspec_kw=dict(wspace=0.08))
for lab, pat, col in (("scalar critic", "g24_TS_s%d", SC), ("2D critic", "g24_T_s%d", TD)):
    curves = []
    for s in (1, 2, 3):
        xs, ys = [], []
        for c in range(250, 1501, 250):
            a = acc("g24", pat % s, c, S64, "countdown_metrics.json")
            if a is not None: xs.append(c); ys.append(a)
        ax.plot(xs, ys, color=col, lw=0.7, alpha=0.55); curves.append(dict(zip(xs, ys)))
    allx = sorted(set(x for cv in curves for x in cv)); mean = [np.mean([cv[x] for cv in curves if x in cv]) for x in allx]
    ax.plot(allx, mean, color=col, lw=1.8, marker="o", ms=3, label=lab)
ax.axhline(45.0, color=style.BASE, ls="--", lw=0.8, label="base")
ax.legend(loc="lower right", ncol=3, frameon=False, fontsize=6.5, handlelength=1.5, borderaxespad=0.15, columnspacing=0.8, bbox_to_anchor=(1.0, 0.0))
ax.set_xlabel("actor iteration", labelpad=1); ax.set_ylabel("held-out acc. (%)")
ax.set_xticks([250, 500, 750, 1000, 1250, 1500])
its = [250, 500, 750, 1000, 1250, 1500]
for lab, ys, col, ls in [("scalar throughout", [47.7, 49.8, 51.6, 49.8, 49.9, 50.8], SC, "-"), ("scalar$\\to$2D at 250", [47.7, 51.3, 51.5, 54.2, 55.8, 56.4], TD, "--"),
                         ("2D throughout", [49.8, 52.3, 52.4, 56.1, 56.7, 57.3], TD, "-"), ("2D$\\to$scalar at 250", [49.8, 51.7, 51.8, 50.1, 50.1, 51.3], SC, "--")]:
    bx.plot(its, ys, ls, color=col, lw=1.6, marker="o", ms=3, label=lab)
bx.vlines(250, 45.5, 60, color=style.BASE, ls=":", lw=0.9); bx.axhline(45.0, color=style.BASE, ls="--", lw=0.8)
bx.set_xlabel("actor iteration", labelpad=1); bx.set_xticks([250, 500, 750, 1000, 1250, 1500]); bx.set_xlim(200, 1550)
bx.legend(loc="lower right", ncol=2, frameon=False, fontsize=6.2, handlelength=1.6, borderaxespad=0.15, columnspacing=0.8, labelspacing=0.2, bbox_to_anchor=(1.0, 0.0))
ax.set_ylim(39.5, 60); ax.set_yticks([40, 45, 50, 55, 60])
fig.savefig(os.path.join(D, "fig_dyn.pdf"), bbox_inches="tight", pad_inches=0.02); print("compact figures written")

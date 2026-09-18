"""Figure: (left) frozen against jointly trained critic on Game24 (2D critic, actor lr 1e-6, earlier recipe): critic value on the
training batch, true success of that batch, held-out accuracy of the frozen-critic run; (right) exchanging the critics at iteration 250
(Game24, seed 4): held-out accuracy at the deployment sampler for the four continuations."""
import re, glob, json, os, numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import style; style.apply()
import matplotlib.pyplot as plt
R = "/cluster/tufts/liulab/xhan03/flow-nlp-rl"
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
fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.15), gridspec_kw=dict(wspace=0.55))
# ---- left: frozen critic
v, t = parse(f"{R}/slurm/logs/cd-pathwise-3447085.out"); e = evals("eval_g24_g24_abl_frozen")
ax2 = ax.twinx(); ax2.spines["top"].set_visible(False); ax2.grid(False)
ax.plot(t[:, 0], smooth(t[:, 1]), color=style.TRUE, lw=1.3, label="true success, training batch")
ax.plot(e[:, 0], e[:, 1], "o-", color=style.ACC, lw=1.5, ms=3.5, label="held-out accuracy")
ax2.plot(v[:, 0], smooth(v[:, 1]), color=style.VAL, lw=1.3, label="critic value $V_\\phi$, training batch")
ax.axhline(45.0, color=style.BASE, ls=":", lw=0.9)
ax.set_title("critic frozen after pre-training", fontsize=8.5); ax.set_xlabel("actor iteration", fontsize=8); ax.set_ylabel("accuracy (%)", fontsize=8)
ax.set_ylim(0, 75); ax.set_xlim(0, 1000); ax.tick_params(labelsize=7); ax2.tick_params(axis="y", colors=style.VAL, labelsize=7)
ax2.set_ylabel("$V_\\phi$ (logit)", color=style.VAL, fontsize=8); ax2.spines["right"].set_color(style.VAL)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=6.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1, handlelength=1.4, borderaxespad=0.0)
# ---- right: critic exchange
its = [250, 500, 750, 1000, 1250, 1500]
curves = {"scalar critic throughout": ([47.7, 49.8, 51.6, 49.8, 49.9, 50.8], style.VAL, "-"),
          "scalar policy, 2D critic from 250": ([47.7, 51.3, 51.5, 54.2, 55.8, 56.4], style.ACC, "--"),
          "2D critic throughout": ([49.8, 52.3, 52.4, 56.1, 56.7, 57.3], style.ACC, "-"),
          "2D policy, scalar critic from 250": ([49.8, 51.7, 51.8, 50.1, 50.1, 51.3], style.VAL, "--")}
for lab, (ys, c, ls) in curves.items(): bx.plot(its, ys, ls, color=c, lw=1.6, marker="o", ms=3, label=lab)
bx.axvline(250, color=style.BASE, ls=":", lw=0.9); bx.axhline(45.0, color=style.BASE, ls=":", lw=0.9)
bx.set_title("critics exchanged at iteration 250", fontsize=8.5); bx.set_xlabel("actor iteration", fontsize=8); bx.set_ylabel("held-out accuracy (%)", fontsize=8)
bx.set_xlim(200, 1550); bx.set_ylim(46, 60); bx.set_xticks([250, 500, 750, 1000, 1250, 1500]); bx.tick_params(labelsize=7)
bx.legend(fontsize=6.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1, handlelength=1.8, borderaxespad=0.0)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_frozen_swap.pdf"); fig.savefig(out, bbox_inches="tight", pad_inches=0.02); print("wrote", out)

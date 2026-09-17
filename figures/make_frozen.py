"""Figure: frozen against jointly trained critic on Game24 (2D critic, actor lr 1e-6, early recipe).
Left: critic frozen after pre-training (run outputs/g24_abl_frozen, log cd-pathwise-3447085). Right: the identical run with
two head steps per actor step on the replay buffer (run g24_jpc_tok_lr1e6_s2, log cd-pathwise-3402644).
Grey: true success of the training batch (logged every 10 iterations); red: critic value on that batch; blue: held-out accuracy
at the deployment sampler (gamma56@64) of the saved checkpoints; dotted: SFT base (45.0)."""
import re, glob, json, os, numpy as np, style; import matplotlib.pyplot as plt
style.apply(); R = "/cluster/tufts/liulab/xhan03/flow-nlp-rl"
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
                m = json.load(open(f)); v = m.get("all", m); a = v.get("solved", v.get("accuracy")) if isinstance(v, dict) else v
                out.append((ck, float(a) * 100 if a <= 1 else float(a)))
    return np.array(sorted(out))
def smooth(x, w=5): return np.convolve(x, np.ones(w) / w, mode="same") if len(x) >= w else x
panels = [("critic frozen after pre-training", "slurm/logs/cd-pathwise-3447085.out", "eval_g24_g24_abl_frozen"),
          ("head refit after every actor step", "slurm/logs/cd-pathwise-3402644.out", "eval_g24_g24_jpc_tok_lr1e6_s2")]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3), sharey=True)
H = {}
for ax, (title, log, pref) in zip(axes, panels):
    v, t = parse(f"{R}/{log}"); e = evals(pref)
    ax2 = ax.twinx(); ax2.spines["top"].set_visible(False); ax2.grid(False)
    if len(t): h, = ax.plot(t[:, 0], smooth(t[:, 1]), color=style.TRUE, lw=1.3, label="true success of the training batch"); H.setdefault(h.get_label(), h)
    if len(e): h, = ax.plot(e[:, 0], e[:, 1], "o-", color=style.ACC, lw=1.5, ms=3.5, label="held-out accuracy (deployment sampler)"); H.setdefault(h.get_label(), h)
    if len(v): h, = ax2.plot(v[:, 0], smooth(v[:, 1]), color=style.VAL, lw=1.3, label="critic value $V_\\phi$ on the training batch"); H.setdefault(h.get_label(), h)
    h = ax.axhline(45.0, color=style.BASE, ls=":", lw=0.9, label="SFT base"); H.setdefault(h.get_label(), h)
    ax.set_title(title, fontsize=8.5); ax.set_xlabel("actor iteration"); ax.set_ylim(0, 75); ax.set_xlim(0, 1000)
    ax2.set_ylabel("$V_\\phi$ (logit)", color=style.VAL); ax2.tick_params(axis="y", colors=style.VAL); ax2.spines["right"].set_color(style.VAL)
axes[0].set_ylabel("accuracy (%)")
fig.legend(list(H.values()), list(H.keys()), fontsize=7, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig("fig_frozen.pdf"); fig.savefig("fig_frozen.png", dpi=200); print("frozen ok")

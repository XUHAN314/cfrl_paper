"""fig_samplers.pdf: held-out accuracy (test half) of the SFT base, the 2D-critic checkpoint the loop starts from and the model
after the loop, at four samplers of increasing cost (gamma4@16, gamma28@32, ODE-64, gamma56@64), Countdown and K&K.
Data: flow-nlp-rl/outputs/analysis/split_select/loop_numbers.json (flow_rl/analysis/loop_numbers.py)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import style; style.apply()
import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__)); R = "/cluster/tufts/liulab/xhan03/flow-nlp-rl"
F = 7.5; plt.rcParams.update({"font.size": F, "legend.fontsize": F, "axes.titlesize": F + 0.3, "axes.labelsize": F, "xtick.labelsize": F, "ytick.labelsize": F})
J = json.load(open(f"{R}/outputs/analysis/split_select/loop_numbers.json"))
KEYS = ["g4@16", "g28@32", "ode64", "g56@64"]; XL = ["16", "32", "ODE", "64"]
ROWS = {"cd5": [("base", "cd5:base"), ("2D-critic ckpt", "cd5:2D start (K40_s3 ckpt1500)"), ("+ loop", "cd5:cd5_ours2_r4 final")],
        "kk": [("base", "kk:base"), ("2D-critic ckpt", "kk:2D start (T_s3 ckpt1000)"), ("+ loop", "kk:kk_ours2_r5 final")]}
STY = {"base": dict(color=style.BASE, ls=":", lw=1.2), "2D-critic ckpt": dict(color=style.ACC, ls="-", lw=1.4), "+ loop": dict(color="#2a9d8f", ls="-", lw=1.6)}
fig, axes = plt.subplots(1, 2, figsize=(2.4, 1.6), gridspec_kw=dict(wspace=0.45))
for ax, (task, title) in zip(axes, (("cd5", "Countdown"), ("kk", "K&K"))):
    for lab, key in ROWS[task]:
        y = [J[key][k]["report"] for k in KEYS]
        ax.plot(range(4), y, marker="o", ms=2.8, label=lab, **STY[lab])
    ax.set_xticks(range(4)); ax.set_xticklabels(XL); ax.set_title(title, pad=2); ax.tick_params(axis="x", length=2, pad=1.5)
axes[0].set_ylabel("accuracy (%)"); axes[0].set_ylim(18, 56); axes[1].set_ylim(52, 102)
fig.text(0.58, 0.215, "sampler steps (ODE: 64 steps)", ha="center", va="top", fontsize=F - 0.5)
fig.subplots_adjust(bottom=0.37, top=0.90, left=0.19, right=0.98)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=F - 0.5, handlelength=1.3, columnspacing=0.8, handletextpad=0.4, bbox_to_anchor=(0.56, -0.01))
fig.savefig(os.path.join(D, "fig_samplers.pdf"), bbox_inches="tight", pad_inches=0.02); print("fig_samplers.pdf written")

"""Shared plot style for the paper's line figures: warm off-white panels with a light grid, one palette."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ACC = "#2a6f97"      # held-out accuracy
VAL = "#e76f51"      # critic value on the training batch
TRUE = "#8d99ae"     # true success of the training batch / replay buffer
BASE = "#495057"     # base model reference line
PANEL = "#fbf8f1"    # axes background
def apply():
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "axes.facecolor": PANEL, "axes.grid": True, "grid.color": "#ffffff", "grid.linewidth": 0.9, "axes.axisbelow": True,
                         "axes.edgecolor": "#adb5bd", "xtick.color": "#495057", "ytick.color": "#495057", "axes.labelcolor": "#212529"})

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

CM = 1 / 2.54
PRX_SINGLE = 8.5 * CM  # ~3.35 in
PRX_ONEHALF = 14.0 * CM  # ~5.51 in
PRX_DOUBLE = 17.8 * CM  # ~7.01 in


def prx_figsize(width="single", aspect=1.5):
    w = {"single": PRX_SINGLE, "1.5": PRX_ONEHALF, "double": PRX_DOUBLE}[width]
    return (w, w * aspect)

plt.style.use("../prx_quantum.mplstyle")

# Load the data from the csv files
csv_data = pd.read_csv("data.csv")
data = {"SK_100": {
    "RBQS_NQA": {
        "e_best": csv_data["SK 100 Best"].to_numpy(),
        "e_res": csv_data["SK 100 Final"].to_numpy(),
    },
}, "SK_200": {
    "DBQS_auto": {
        "e_best": csv_data["SK 200 Best"].to_numpy(),
        "e_res": csv_data["SK 200 Final"].to_numpy(),
    },
}}


# Histogram parameters
bins = 11
x_label, y_label = r"$\varepsilon$", "Counts"
log_bins = True

# Build common bin edges (log-spaced)
all_vals = np.concatenate([np.concatenate([v["e_res"], v["e_best"]]) for v in data["SK_200"].values()])
bin_edges = np.logspace(-10, 0, bins + 1) if log_bins else np.histogram_bin_edges(all_vals, bins=bins)


plt.rcParams["figure.figsize"] = prx_figsize("single", aspect=1.0)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
ax1.hist(
    data["SK_100"]["RBQS_NQA"]["e_res"],
    bins=bin_edges,
    alpha=0.5,
    label=r"$\varepsilon_Q$",
    hatch="\\\\\\\\",
)
ax1.hist(
    data["SK_100"]["RBQS_NQA"]["e_best"],
    bins=bin_edges,
    alpha=0.5,
    label=r"$\varepsilon_B$",
    hatch="////"
)
ax1.set_xscale("log")
ax1.set_ylabel("Counts")
ax1.legend(frameon=True, loc="upper right")
ax1.tick_params(axis="both", which="major")
ax2.hist(
    data["SK_200"]["DBQS_auto"]["e_res"],
    bins=bin_edges,
    alpha=0.5,
    label=r"$\varepsilon_Q$",
    hatch="\\\\\\\\",
)
ax2.hist(
    data["SK_200"]["DBQS_auto"]["e_best"],
    bins=bin_edges,
    alpha=0.5,
    label=r"$\varepsilon_B$",
    hatch="////"
)
ax1.text(
    0.95, 0.05, r"$N=100$",
    transform=ax1.transAxes,
    ha="right",
    color="black",
    path_effects=[pe.withStroke(linewidth=3, foreground="white")]
)
ax2.text(0.95, 0.05, r"$N=200$", transform=ax2.transAxes, 
    ha="right",
    color="black",
    path_effects=[pe.withStroke(linewidth=3, foreground="white")]
)
ax2.set_xscale("log")
ax2.set_ylabel("Counts")
ax2.set_xlabel(r"$\varepsilon$")
ax2.legend(frameon=True, loc="upper right")
ax2.tick_params(axis="both", which="major")
ax2.set_yticks([0, 2, 4, 6, 8, 10])
fig.tight_layout()
fig.savefig("res_energies.pdf")
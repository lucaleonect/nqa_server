import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

CM = 1 / 2.54
PRX_SINGLE = 8.5 * CM  # ~3.35 in
PRX_ONEHALF = 14.0 * CM  # ~5.51 in
PRX_DOUBLE = 17.8 * CM  # ~7.01 in


def prx_figsize(width="single", aspect=1.5):
    w = {"single": PRX_SINGLE, "1.5": PRX_ONEHALF, "double": PRX_DOUBLE}[width]
    return (w, w * aspect)

plt.style.use("../prx_quantum.mplstyle")

csb_data = pd.read_csv("data.csv")

data = {}
data["SK_100"] = {}
data["SK_200"] = {}
data["SK_100"]["cRBM_SR"] = {"e_best": csb_data["cRBM_SR_100"].values}
data["SK_100"]["cRBM_NQA"] = {"e_best": csb_data["cRBM_NQA_100"].values}
data["SK_100"]["RBQS_SR"] = {"e_best": csb_data["RBQS_SR_100"].values}
data["SK_100"]["RBQS_NQA"] = {"e_best": csb_data["RBQS_NQA_100"].values}
data["SK_200"]["cRBM_SR"] = {"e_best": csb_data["cRBM_SR_200"].values}
data["SK_200"]["cRBM_NQA"] = {"e_best": csb_data["cRBM_NQA_200"].values}
data["SK_200"]["RBQS_SR"] = {"e_best": csb_data["RBQS_SR_200"].values}
data["SK_200"]["RBQS_NQA"] = {"e_best": csb_data["RBQS_NQA_200"].values}


bins = 11
x_label, y_label = r"$\varepsilon$", "Counts"
log_bins = True

# Build common bin edges (log-spaced)
all_vals = np.concatenate([v["e_best"] for v in data["SK_200"].values()])
bin_edges = np.logspace(-10, 0, bins + 1) if log_bins else np.histogram_bin_edges(all_vals, bins=bins)


plt.rcParams["figure.figsize"] = prx_figsize("double", aspect=.5)
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
ax1, ax2 = axes[0]
ax1.set_title(r"SR")
ax2.set_title(r"NQA")
ax1.hist(
    data["SK_100"]["RBQS_SR"]["e_best"],
    bins=bin_edges,
    alpha=0.6,
    label=r"$RBQS$",
    color="green"
)
ax1.hist(
    data["SK_100"]["cRBM_SR"]["e_best"],
    bins=bin_edges,
    alpha=0.4,
    label=r"$cRBM$",
    color="red",
)
ax1.set_xscale("log")
ax1.set_ylabel("Counts")
ax1.legend(frameon=True)
ax1.tick_params(axis="both", which="major")
ax2.hist(
    data["SK_100"]["RBQS_NQA"]["e_best"],
    bins=bin_edges,
    alpha=0.6,
    label=r"$RBQS$",
    color="green"
)
ax2.hist(
    data["SK_100"]["cRBM_NQA"]["e_best"],
    bins=bin_edges,
    alpha=0.4,
    label=r"$cRBM$",
    color="red",
)
ax2.set_xscale("log")
ax2.legend(frameon=True)
ax2.tick_params(axis="both", which="major")
ax1.text(0.95, 0.05, r"$N=100$", transform=ax1.transAxes, ha="right")
ax2.text(0.95, 0.05, r"$N=100$", transform=ax2.transAxes, ha="right")
ax1, ax2 = axes[1]
ax1.hist(
    data["SK_200"]["RBQS_SR"]["e_best"],
    bins=bin_edges,
    alpha=0.6,
    label=r"$RBQS$",
    color="green"
)
ax1.hist(
    data["SK_200"]["cRBM_SR"]["e_best"],
    bins=bin_edges,
    alpha=0.4,
    label=r"$cRBM$",
    color="red",
)
ax1.set_xscale("log")
ax1.set_ylabel("Counts")
ax1.legend(frameon=True)
ax1.tick_params(axis="both", which="major")
ax2.hist(
    data["SK_200"]["RBQS_NQA"]["e_best"],
    bins=bin_edges,
    alpha=0.6,
    label=r"$RBQS$",
    color="green"
)
ax2.hist(
    data["SK_200"]["cRBM_NQA"]["e_best"],
    bins=bin_edges,
    alpha=0.4,
    label=r"$cRBM$",
    color="red",
)
ax2.set_xscale("log")
ax2.set_xlabel(r"$\varepsilon_{B}$")
ax1.set_xlabel(r"$\varepsilon_{B}$")
ax2.legend(frameon=True)
ax2.tick_params(axis="both", which="major")
ax1.text(0.95, 0.05, r"$N=200$", transform=ax1.transAxes, ha="right")
ax2.text(0.95, 0.05, r"$N=200$", transform=ax2.transAxes, ha="right")
fig.tight_layout()
fig.savefig("res_energies_app1.pdf")
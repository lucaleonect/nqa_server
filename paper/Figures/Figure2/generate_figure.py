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

# Load the data from the csv files
data = pd.read_csv("data.csv")

typical_energies_100 = {}
typical_energies_200 = {}
for algo in data["Algorithm"].unique():
    typical_energies_100[algo] = {}
    typical_energies_200[algo] = {}
    typical_energies_100[algo]["final"] = data[f"SK 100 Final"][data["Algorithm"] == algo].values[0]
    typical_energies_100[algo]["best"] = data[f"SK 100 Best"][data["Algorithm"] == algo].values[0]
    typical_energies_200[algo]["final"] = data[f"SK 200 Final"][data["Algorithm"] == algo].values[0]
    typical_energies_200[algo]["best"] = data[f"SK 200 Best"][data["Algorithm"] == algo].values[0]

plt.rcParams["figure.figsize"] = prx_figsize("single", aspect=1.3)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
# Top one is the scatter of the residual energies
ax1.plot([1e-14, 10], [1e-14, 10], color="black", linestyle="--", linewidth=1.2, alpha=0.3)
ax1.scatter(
    [typical_energies_100["cRBM_SR"]["final"]],
    [typical_energies_100["cRBM_SR"]["best"]],
    color="blue",
    label=r"cRBM (SR)",
    s=15,
)
ax1.scatter(
    [typical_energies_100["RBQS_SR"]["final"]],
    [typical_energies_100["RBQS_SR"]["best"]],
    color="red",
    label=r"RBQS (SR)",
    s=15,
)
ax1.scatter(
    [typical_energies_100["cRBM_NQA"]["final"]],
    [typical_energies_100["cRBM_NQA"]["best"]],
    color="blue",
    label=r"cRBM (VQA)",
    marker="x",
    s=15,
)
ax1.scatter(
    [typical_energies_100["RBQS_NQA"]["final"]],
    [typical_energies_100["RBQS_NQA"]["best"]],
    color="red",
    label=r"RBQS (NQA)",
    marker="x",
    s=15,
)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylabel(r"$[\varepsilon_B]_{\rm typ}$")
ax1.legend(frameon=True, loc="upper left")
ax1.tick_params(axis="both", which="major")
# Top one is the scatter of the residual energies
ax2.plot([1e-14, 10], [1e-14, 10], color="black", linestyle="--", linewidth=1.2, alpha=0.3)
ax2.scatter(
    [typical_energies_200["cRBM_SR"]["final"]],
    [typical_energies_200["cRBM_SR"]["best"]],
    color="blue",
    label="cRBM (SR)",
    s=15,
)
ax2.scatter(
    [typical_energies_200["RBQS_SR"]["final"]],
    [typical_energies_200["RBQS_SR"]["best"]],
    color="red",
    label=r"RBQS (SR)",
    s=15,
)
ax2.scatter(
    [typical_energies_200["cRBM_NQA"]["final"]],
    [typical_energies_200["cRBM_NQA"]["best"]],
    color="blue",
    label=r"cRBM (VQA)",
    marker="x",
    s=15,
)
ax2.scatter(
    [typical_energies_200["RBQS_NQA"]["final"]],
    [typical_energies_200["RBQS_NQA"]["best"]],
    color="red",
    label=r"RBQS (NQA)",
    marker="x",
    s=15,
)
ax2.scatter(
    [typical_energies_200["DBQS_auto"]["final"]],
    [typical_energies_200["DBQS_auto"]["best"]],
    color="purple",
    label=r"DBQS (NQA)",
    marker="x",
    s=15,
)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$[\varepsilon_Q]_{\rm typ}$")
ax2.set_ylabel(r"$[\varepsilon_B]_{\rm typ}$")
ax2.set_xlim(1e-11, 1)
ax2.set_ylim(1e-11, 1)
ax2.set_xticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])
ax2.set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])
ax2.legend(frameon=True, loc="upper left")
ax2.tick_params(axis="both", which="major")
# Write N=100 and N=200 on the bottom right corner of each subplot
ax1.text(0.95, 0.05, r"$N=100$", transform=ax1.transAxes, ha="right")
ax2.text(0.95, 0.05, r"$N=200$", transform=ax2.transAxes, ha="right")
fig.tight_layout()
fig.savefig("typical_energies.pdf")
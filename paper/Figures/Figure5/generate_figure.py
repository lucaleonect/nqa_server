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
data_points = pd.read_csv("tfsk_16_final_energies_a.csv")
data_points_b = pd.read_csv("tfsk_16_final_energies_b.csv")
data_ed = pd.read_csv("ed_energies_16.csv")

plt.rcParams["figure.figsize"] = prx_figsize("double", aspect=0.4)
# Share y axis with subplots in the same line
fig, axes = plt.subplots(2, 2, sharex="col", sharey="row", gridspec_kw={"width_ratios": [6, 4]})
(ax1, ax2, ax3, ax4) = axes.flatten()
i=0
# Get all the instance 0 data and save the as np.array for plotting
instance_0_data = data_points[data_points["instance"] == i]
instance_0_data = np.array(instance_0_data)[:, 1:]

instance_0_ed = data_ed[data_ed["instance"] == i]
instance_0_ed = np.array(instance_0_ed)[:, 1:]
print("ED Energies:", instance_0_ed)


for ed_energy in instance_0_ed:
    ax1.axhline(ed_energy, color="red", alpha=0.5, linewidth=1.0, linestyle="-", label="ED Energy")
    ax2.axhline(ed_energy, color="red", alpha=0.5, linewidth=1.0, linestyle="-", label="ED Energy")
ax1.scatter(
    instance_0_data[:, 1],
    instance_0_data[:, 0],
    c="black",
)
ax1.set_ylabel(r"$\langle H_T \rangle$")
ax2.hist(
    instance_0_data[:, 0],
    bins=51,
    orientation="horizontal",
    color="black",
    align="left",
)


ax1.text(-0.20, 1.05, "a)", transform=ax1.transAxes, fontweight="bold", va="top", ha="right")
ax2.text(-0.05, 1.05, "b)", transform=ax2.transAxes, fontweight="bold", va="top", ha="right")


i=4
# Get all the instance 0 data and save the as np.array for plotting
instance_0_data = data_points_b[data_points_b["instance"] == i]
instance_0_data = np.array(instance_0_data)[:, 1:]

instance_0_ed = data_ed[data_ed["instance"] == i]
instance_0_ed = np.array(instance_0_ed)[:, 1:]
print("ED Energies:", instance_0_ed)


for ed_energy in instance_0_ed:
    ax3.axhline(ed_energy, color="red", alpha=0.5, linewidth=1.0, linestyle="-", label="ED Energy")
    ax4.axhline(ed_energy, color="red", alpha=0.5, linewidth=1.0, linestyle="-", label="ED Energy")
ax3.scatter(
    instance_0_data[:, 1],
    instance_0_data[:, 0],
    c="black",
)
ax3.set_ylabel(r"$\langle H_T \rangle$")
ax4.hist(
    instance_0_data[:, 0],
    bins=51,
    orientation="horizontal",
    color="black",
    align="left",
)


ax3.text(-0.20, 1.05, "c)", transform=ax3.transAxes, fontweight="bold", va="top", ha="right")
ax4.text(-0.05, 1.05, "d)", transform=ax4.transAxes, fontweight="bold", va="top", ha="right")

ax3.set_xlabel(r"$\sigma^2$")
ax4.set_xlabel("Counts")
fig.tight_layout()
fig.savefig("figure_tfsk_16_i4.pdf")

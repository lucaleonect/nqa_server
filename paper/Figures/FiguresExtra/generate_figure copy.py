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


classical_gs_energies_100 = [
    -75.20181311460237,
    -71.87341830032535,
    -75.04394373228551,
    -75.31808585501405,
    -79.60962573360997,
    -73.01367758789947,
    -74.69668933726089,
    -72.85470668007179,
    -71.94677698248434,
    -73.3052934065986,
]


for i in range(10):
    data_points = pd.read_csv(f"tfsk_100_{i}_final_energies.csv")
    print(data_points.keys())
    plt.rcParams["figure.figsize"] = prx_figsize("single", aspect=0.8)
    # Share y axis with subplots in the same line
    # fig, axes = plt.subplots(1, 2, sharey="row", gridspec_kw={"width_ratios": [6, 4]})
    # ax1, ax3 = axes
    fig, axes = plt.subplots(2, 1)
    ax1, ax3 = axes.flatten()
    # Get all the instance 0 data and save the as np.array for plotting
    instance_0_data = data_points[data_points["Instance"] == i]
    instance_0_data = np.array(instance_0_data)[:, 1:]

    classical_energy = classical_gs_energies_100[i]
    ax1.axhline(classical_energy, color="red", linestyle="--", label="Classical Energy")
    ax3.axhline(classical_energy, color="red", linestyle="--", label="Classical Energy")
    ax1.scatter(
        instance_0_data[:, 1],
        instance_0_data[:, 0],
        c="black",
    )
    ax1.set_ylabel(r"$\langle H_T \rangle$")
    ax3.hist(
        instance_0_data[:, 0],
        bins=51,
        orientation="horizontal",
        color="black",
        align="left",
    )
    ax3.set_ylabel(r"$\langle H_T \rangle$")

    ax1.text(-0.20, 1.05, "a)", transform=ax1.transAxes, fontweight="bold", va="top", ha="right")
    ax3.text(-0.05, 1.05, "b)", transform=ax3.transAxes, fontweight="bold", va="top", ha="right")

    ax1.set_xlabel(r"$\sigma^2$")
    ax3.set_xlabel("Counts")
    fig.tight_layout()
    fig.savefig(f"figure_tfsk_100_i_{i}.pdf")


# Load the data from the csv files
data_ed = pd.read_csv("ed_energies_16.csv")

for i in range(10):
    data_points = pd.read_csv(f"tfsk_16_{i}_final_energies.csv")
    print(data_points.keys())
    plt.rcParams["figure.figsize"] = prx_figsize("double", aspect=0.3)
    # Share y axis with subplots in the same line
    fig, axes = plt.subplots(1, 2, sharey="row", gridspec_kw={"width_ratios": [6, 4]})
    ax1, ax3 = axes
    # Get all the instance 0 data and save the as np.array for plotting
    instance_0_data = data_points[data_points["Instance"] == i]
    instance_0_data = np.array(instance_0_data)[:, 1:]

    instance_0_ed = data_ed[data_ed["instance"] == i]
    instance_0_ed = np.array(instance_0_ed)[:, 1:]
    print("ED Energies:", instance_0_ed)

    for ed_energy in instance_0_ed:
        ax1.axhline(ed_energy, color="red", alpha=0.5, linewidth=1.0, linestyle="-", label="ED Energy")
        ax3.axhline(ed_energy, color="red", alpha=0.5, linewidth=1.0, linestyle="-", label="ED Energy")
    ax1.scatter(
        instance_0_data[:, 1],
        instance_0_data[:, 0],
        c="black",
    )
    ax1.set_ylabel(r"$\langle H_T \rangle$")
    ax3.hist(
        instance_0_data[:, 0],
        bins=51,
        orientation="horizontal",
        color="black",
        align="left",
    )

    ax1.text(-0.20, 1.05, "a)", transform=ax1.transAxes, fontweight="bold", va="top", ha="right")
    ax3.text(-0.05, 1.05, "b)", transform=ax3.transAxes, fontweight="bold", va="top", ha="right")

    ax1.set_xlabel(r"$\sigma^2$")
    ax3.set_xlabel("Counts")
    fig.tight_layout()
    fig.savefig(f"figure_tfsk_16_i_{i}.pdf")

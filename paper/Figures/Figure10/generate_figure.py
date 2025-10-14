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
data_fit = pd.read_csv("tfsk_100_band_fits.csv")
data_stats = pd.read_csv("tfsk_100_band_statistics.csv")
data_points = pd.read_csv("tfsk_100_final_energies.csv")
data_cl = pd.read_csv("ed_classical_energies_100.csv")

plt.rcParams["figure.figsize"] = prx_figsize("double", aspect=1.0)
# Share y axis with subplots in the same line
fig, axes = plt.subplots(10, 3, sharey="row", sharex="col", gridspec_kw={"width_ratios": [4, 4, 2]})
for i in range(10):
    (ax1, ax2, ax3) = axes[i]

    # Get all the instance 0 data and save the as np.array for plotting
    instance_0_data = data_points[data_points["instance"] == i]
    instance_0_data = np.array(instance_0_data)[:, 1:]

    instance_0_fit = data_stats[data_stats["instance"] == i]
    bands_fit_params = {}
    for band in range(2):
        try:
            bands_fit_params[band] = {}
            band_data = instance_0_fit[instance_0_fit["band"] == band]
            bands_fit_params[band]["var_fit"] = np.array(band_data[band_data["scaling"]=="variance"])[0][3:]
            bands_fit_params[band]["inverse_num_params"] = np.array(band_data[band_data["scaling"]=="inverse_num_params"])[0][3:]
        except IndexError:
            pass
    print(bands_fit_params)
    classical_energy = data_cl[data_cl["instance"] == i]["exact_energy"].values[0]
    ax1.axhline(classical_energy, color="blue", linewidth=1.5, linestyle=":")
    ax2.axhline(classical_energy, color="blue", linewidth=1.5, linestyle=":")
    ax3.axhline(classical_energy, color="blue", linewidth=1.5, linestyle=":")


    ax1.scatter(
        instance_0_data[:, 1],
        instance_0_data[:, 0],
        c="black",
    )
    x_fit = np.linspace(0, instance_0_data[:, 1].max(), 100)
    for band in range(2):
        try:
            y_fit = [bands_fit_params[band]["var_fit"][0] + bands_fit_params[band]["var_fit"][1]* x for x in x_fit]
            ax1.plot(x_fit, y_fit, label=f"Band {band} Var. Fit", linestyle="--",c="black")
            ax1.scatter(
                0,
                bands_fit_params[band]["var_fit"][0],
                marker="x",
                c="black",
            )
        except:
            pass
    ax1.set_ylabel(r"$\langle H_t \rangle$")
    ax1.set_xlim(0, x_fit.max())
    ax2.scatter(
        instance_0_data[:, 2]*(1e5),
        instance_0_data[:, 0],
        c="black",
    )
    x_fit = np.linspace(0, instance_0_data[:, 2].max()*(1e5), 100)
    for band in range(2):
        try:
            y_fit = [bands_fit_params[band]["inverse_num_params"][0] + bands_fit_params[band]["inverse_num_params"][1]* x for x in x_fit]
            ax2.plot(x_fit, y_fit, label=f"Band {band} Inv. Fit", linestyle="--",c="black")
            ax2.scatter(
                0,
                bands_fit_params[band]["inverse_num_params"][0],
                marker="x",
                c="black",
            )
        except:
            pass
    ax2.set_xlim(0, x_fit.max())
    ax3.hist(
        instance_0_data[:, 0],
        bins=11,
        orientation="horizontal",
        color="black",
    )
ax1.set_xlabel(r"$\sigma^2$")
ax2.set_xlabel(r"$N_\text{params}^{-1} \times 10^5$")
ax3.set_xlabel("Counts")
fig.tight_layout()
fig.savefig("figure_tfsk_100_all.pdf")

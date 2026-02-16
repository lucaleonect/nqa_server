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
data_fit = pd.read_csv("tfsk_16_band_fits.csv")
data_stats = pd.read_csv("tfsk_16_band_statistics.csv")
data_points = pd.read_csv("tfsk_16_final_energies.csv")
data_ed = pd.read_csv("ed_energies_16.csv")

plt.rcParams["figure.figsize"] = prx_figsize("double", aspect=1.)
# Share y axis with subplots in the same line
fig, axes = plt.subplots(10, 3, sharey="row", gridspec_kw={"width_ratios": [4, 4, 2]})
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

    instance_0_ed = data_ed[data_ed["instance"] == i]
    instance_0_ed = np.array(instance_0_ed)[:, 1:]
    print("ED Energies:", instance_0_ed)


    # ED lines in all panels
    for ed_energy in instance_0_ed:
        ax1.axhline(ed_energy, color="red", alpha=0.5, linewidth=1.5, linestyle="-", label="ED Energy")
        ax2.axhline(ed_energy, color="red", alpha=0.5, linewidth=1.5, linestyle="-", label="ED Energy")
        ax3.axhline(ed_energy, color="red", alpha=0.5, linewidth=1.5, linestyle="-", label="ED Energy")
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
                s=30,
                clip_on=False,
            )
        except:
            pass
    ax1.set_ylabel(r"$\langle H_T \rangle$")
    ax1.set_xlim(0, x_fit.max())
    ax2.scatter(
        instance_0_data[:, 2],
        instance_0_data[:, 0],
        c="black",
    )
    x_fit = np.linspace(0, instance_0_data[:, 2].max(), 100)
    for band in range(2):
        try:
            y_fit = [bands_fit_params[band]["inverse_num_params"][0] + bands_fit_params[band]["inverse_num_params"][1]* x for x in x_fit]
            ax2.plot(x_fit, y_fit, label=f"Band {band} Inv. Fit", linestyle="--",c="black")
            ax2.scatter(
                0,
                bands_fit_params[band]["inverse_num_params"][0],
                marker="x",
                c="black",
                s=30,
                clip_on=False,
            )
        except:
            pass
    ax2.set_xlim(0, x_fit.max())
    ax3.hist(
        instance_0_data[:, 0],
        bins=21,
        orientation="horizontal",
        color="black",
    )
ax1.set_xlabel(r"$\sigma^2$")
ax2.set_xlabel(r"$N_\text{params}^{-1}$")
ax3.set_xlabel("Counts")
fig.tight_layout()
fig.savefig("figure_tfsk_16_all.pdf")

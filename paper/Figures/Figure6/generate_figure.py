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
global_histo_data = pd.read_csv("data.csv", header=None).to_numpy()

errors = np.array([e for e, r in global_histo_data])
ranks = np.array([r for e, r in global_histo_data])

bins = np.linspace(errors.min(), errors.max(), 102)
bin_indices = np.digitize(errors, bins) - 1
bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

n_bins = len(bins) - 1
counts = np.zeros(n_bins)
mean_ranks = np.full(n_bins, np.nan)
for b in range(n_bins):
    mask = bin_indices == b
    counts[b] = mask.sum()
    if mask.any():
        mean_ranks[b] = ranks[mask].mean()

norm = plt.Normalize(vmin=np.nanmin(mean_ranks), vmax=np.nanmax(mean_ranks))
cmap = plt.cm.viridis_r

plt.rcParams["figure.figsize"] = prx_figsize("single", aspect=0.6)
fig, ax = plt.subplots()
for b in range(n_bins):
    if counts[b] > 0:
        color = cmap(norm(mean_ranks[b]))
        ax.bar(
            (bins[b] + bins[b + 1]) / 2,
            counts[b],
            width=(bins[1] - bins[0]),
            color=color,
            edgecolor="none",
        )

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Mean Trial Rank")

ax.axvline(0, color="red", linestyle="--")
plt.xlabel(r"$\langle H_T \rangle - E_0^{\rm cl}$")
ax.set_ylabel("Count")
plt.savefig("fig_tfsk_color.pdf")
plt.close()

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import os
import ast
from matplotlib.patches import ConnectionPatch  # added
import matplotlib.patheffects as pe  # NEW: for text outline
from qubo import QUBO_to_Ising

CM = 1 / 2.54
PRX_SINGLE = 8.5 * CM  # ~3.35 in
PRX_ONEHALF = 14.0 * CM  # ~5.51 in
PRX_DOUBLE = 17.8 * CM  # ~7.01 in


def prx_figsize(width="single", aspect=1.5):
    w = {"single": PRX_SINGLE, "1.5": PRX_ONEHALF, "double": PRX_DOUBLE}[width]
    return (w, w * aspect)


plt.style.use("../prx_quantum.mplstyle")


def main():
    # il sample deve essere un dizionario con valori
    Q = np.load("./matrix_Q_j5_op5_pruned.npy")
    print("Q shape: ", Q.shape)
    n_spins = Q.shape[0]

    J, h, C = QUBO_to_Ising(Q)
    J = J * 4
    h = h / np.sqrt(h.size)
    print("J shape: ", J.shape)
    print("h shape: ", h.shape)
    print("C shape: ", C.shape)
    print(J.min(), J.max())
    print(h.min(), h.max())

    plt.rcParams["figure.figsize"] = prx_figsize("single", aspect=0.7)
    fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [0.9, 0.1]})
    vmax_j = np.max(np.abs(J))
    vmax_h = np.max(np.abs(h))
    vmax = max(vmax_j, vmax_h)
    vmin = 0
    base_cmap = plt.get_cmap("RdBu")
    half_cmap = mcolors.LinearSegmentedColormap.from_list(
        "RdBu_half",
        base_cmap(np.linspace(0.5, 1.0, 256)),
    )
    im = ax[0].pcolor(J, vmin=vmin, vmax=vmax, cmap=half_cmap)
    ax[0].set_title(r"$4J_{ij}$")
    ax[0].set_ylabel("j")
    ax[1].pcolor(h.reshape(1, -1), vmin=vmin, vmax=vmax, cmap=half_cmap)
    # No ticks on the y axis of h
    ax[1].set_yticks([])
    ax[1].set_ylabel(r"$h_i/\sqrt{N}$")  # , rotation=45)
    ax[1].yaxis.set_label_coords(-0.08, -0.15)
    ax[1].set_xlabel("i")

    # Shared colorbar to the right of both subplots (show only non-negative ticks)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.linspace(0, vmax, 2))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    plt.savefig("figure_JH_jssp.pdf")


if __name__ == "__main__":
    main()

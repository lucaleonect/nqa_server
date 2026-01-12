import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import os
import ast
from matplotlib.patches import ConnectionPatch  # added
import matplotlib.patheffects as pe            # NEW: for text outline
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
    print("J shape: ", J.shape)
    print("h shape: ", h.shape)
    print("C shape: ", C.shape)


    plt.rcParams["figure.figsize"] = prx_figsize("single", aspect=0.7)
    fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [0.9, 0.1]})
    vmax_j = np.max(np.abs(J))
    vmin_j = -vmax_j
    vmin_h = -np.max(np.abs(h))
    vmax_h = np.max(np.abs(h))
    ax[0].pcolor(J, vmin=vmin_j, vmax=vmax_j, cmap="RdBu")
    ax[0].set_ylabel("Spin index")
    ax[1].pcolor(h.reshape(1,-1), vmin=vmin_h, vmax=vmax_h, cmap="RdBu")
    # No ticks on the y axis of h
    ax[1].set_yticks([])
    ax[1].set_xlabel("Spin index")

    plt.savefig("figure_JH_jssp.pdf")
    J = np.load("./J_matrix_sk200_i0.npy")
    h = np.zeros(J.shape[0])
    print("J shape: ", J.shape)
    print("h shape: ", h.shape)


    plt.rcParams["figure.figsize"] = prx_figsize("single", aspect=0.7)
    fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [0.9, 0.1]})
    vmax_j = np.max(np.abs(J))
    vmin_j = -vmax_j
    vmin_h = -1
    vmax_h = 1
    ax[0].pcolor(J, vmin=vmin_j, vmax=vmax_j, cmap="RdBu")
    ax[0].set_ylabel("Spin index")
    ax[1].pcolor(h.reshape(1,-1), vmin=vmin_h, vmax=vmax_h, cmap="RdBu")
    # No ticks on the y axis of h
    ax[1].set_yticks([])
    ax[1].set_xlabel("Spin index")

    plt.savefig("figure_JH_sk.pdf")

if __name__ == "__main__":
    main()

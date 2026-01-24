import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm

CM = 1 / 2.54
PRX_SINGLE = 8.5 * CM
PRX_ONEHALF = 14.0 * CM
PRX_DOUBLE = 17.8 * CM


def prx_figsize(width="single", aspect=1.5):
    w = {"single": PRX_SINGLE, "1.5": PRX_ONEHALF, "double": PRX_DOUBLE}[width]
    return (w, w * aspect)


def load_csv_2d(path):
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1:          # e.g. one row -> shape (2,)
        arr = arr.reshape(1, -1)
    return arr

def load_csv(path):
    return np.loadtxt(path, delimiter=",")



def load_vec_csv(path):
    return np.loadtxt(path, delimiter=",").reshape(-1)


def s_tag(s: float) -> str:
    return f"s{int(round(100 * s)):03d}"


def main():
    data_dir = "trajectory_pca_data"

    # Style (requires this file to exist)
    plt.style.use("../prx_quantum.mplstyle")

    # Load shared arrays
    A = load_csv(os.path.join(data_dir, "A.csv"))
    B = load_csv(os.path.join(data_dir, "B.csv"))
    anneals = load_vec_csv(os.path.join(data_dir, "anneals.csv"))
    levels = load_vec_csv(os.path.join(data_dir, "levels.csv"))
    line_levels = load_vec_csv(os.path.join(data_dir, "line_levels.csv"))

    # Load all surfaces + trajectories in anneals order
    surfaces = []
    trajs = []
    for s in anneals:
        tag = s_tag(float(s))
        ls = load_csv(os.path.join(data_dir, f"loss_surface_{tag}.csv"))
        tr = load_csv_2d(os.path.join(data_dir, f"traj_{tag}.csv"))   # <-- FIX
        surfaces.append(ls)
        trajs.append(tr)

    # --- Plot: same layout and logic as your original script ---
    plt.rcParams["figure.figsize"] = prx_figsize("double", aspect=0.7)

    n_rows = len(anneals)
    fig, axes = plt.subplots(n_rows // 2, 2, sharex=True, sharey=True, gridspec_kw={"hspace": 0, "wspace": 0})
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.08, hspace=0, wspace=0)

    cf0 = None
    s_labels = []

    # Right column: last 3 anneals
    for row, s in enumerate(anneals[3:]):
        ax_land = axes[row, 1]
        loss_surface = surfaces[row + 3]
        traj_xy = trajs[row + 3]

        if row < (n_rows // 2) - 1:
            ax_land.tick_params(labelbottom=False)

        cf = ax_land.contourf(
            A,
            B,
            loss_surface,
            levels=levels,
            cmap="viridis",
            norm=BoundaryNorm(levels, ncolors=256, clip=True),
        )
        ax_land.contour(A, B, loss_surface, levels=line_levels, colors="k", linewidths=0.35, alpha=0.45)
        ax_land.set_xlabel(r"$\alpha$", fontsize=10)

        if traj_xy.shape[0] > 1:
            segs = np.stack([traj_xy[:-1], traj_xy[1:]], axis=1)
            lc = LineCollection(segs, colors="white", linewidth=1.5, alpha=0.95)
            ax_land.add_collection(lc)

        ax_land.scatter(traj_xy[0, 0], traj_xy[0, 1], c="red", edgecolor="k", label=r"$\widetilde{\theta}(0)$", zorder=3)
        ax_land.scatter(traj_xy[-1, 0], traj_xy[-1, 1], c="lime", edgecolor="k", label=r"$\widetilde{\theta}(s)$", zorder=3)

        if cf0 is None:
            cf0 = cf
        s_labels.append((float(np.clip(s, 0.0, 1.0)), ax_land))

    # Left column: first 3 anneals
    for row, s in enumerate(anneals[:3]):
        ax_land = axes[row, 0]
        loss_surface = surfaces[row]
        traj_xy = trajs[row]

        if row < (n_rows // 2) - 1:
            ax_land.tick_params(labelbottom=False)

        cf = ax_land.contourf(
            A,
            B,
            loss_surface,
            levels=levels,
            cmap="viridis",
            norm=BoundaryNorm(levels, ncolors=256, clip=True),
        )
        ax_land.contour(A, B, loss_surface, levels=line_levels, colors="k", linewidths=0.35, alpha=0.45)

        ax_land.set_xlabel(r"$\alpha$", fontsize=10)
        ax_land.set_ylabel(r"$\beta$")
        ax_land.tick_params(axis="both", which="major", labelsize=8)

        if traj_xy.shape[0] > 1:
            segs = np.stack([traj_xy[:-1], traj_xy[1:]], axis=1)
            lc = LineCollection(segs, colors="white", linewidth=1.5, alpha=0.95)
            ax_land.add_collection(lc)

        ax_land.scatter(traj_xy[0, 0], traj_xy[0, 1], c="red", edgecolor="k", label=r"$\widetilde{\theta}(0)$", zorder=3)
        ax_land.scatter(traj_xy[-1, 0], traj_xy[-1, 1], c="lime", edgecolor="k", label=r"$\widetilde{\theta}(s)$", zorder=3)

        if cf0 is None:
            cf0 = cf
        s_labels.append((float(np.clip(s, 0.0, 1.0)), ax_land))

    # Ensure bottom row shows x tick labels
    for col in range(axes.shape[1]):
        axes[-1, col].tick_params(axis="x", bottom=True, labelbottom=True)

    # Shared colorbar (horizontal, on top)
    land_axes = axes.ravel().tolist()
    cbar = fig.colorbar(
        cf0,
        ax=land_axes,
        orientation="horizontal",
        location="top",
        fraction=0.03,
        aspect=40,
        spacing="uniform",
    )
    cbar.ax.tick_params()
    cbar.set_ticks(levels[:: max(1, len(levels) // 6)])
    cbar.ax.minorticks_off()
    cbar.ax.set_xticklabels([f"{tick:.2f}" for tick in cbar.ax.get_xticks()])
    cbar.set_label(r"$E(s;\widetilde{{\boldsymbol{\theta}}}(\alpha,\beta))$", fontsize=10)

    # Row labels
    fig.canvas.draw()
    for s_val, ax_l in s_labels:
        pl = ax_l.get_position()
        x_right = pl.x0 + pl.width - 0.06
        y_top = pl.y0 + pl.height - 0.04
        fig.text(x_right, y_top, f"s = {s_val:.2f}", ha="center", va="bottom", fontweight="bold")

    fig.savefig("trajectory_PCA_paper.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.show()


if __name__ == "__main__":
    main()

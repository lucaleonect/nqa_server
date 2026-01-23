import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import os
import ast
from matplotlib.patches import ConnectionPatch  # added
import matplotlib.patheffects as pe            # NEW: for text outline

CM = 1 / 2.54
PRX_SINGLE = 8.5 * CM  # ~3.35 in
PRX_ONEHALF = 14.0 * CM  # ~5.51 in
PRX_DOUBLE = 17.8 * CM  # ~7.01 in


def prx_figsize(width="single", aspect=1.5):
    w = {"single": PRX_SINGLE, "1.5": PRX_ONEHALF, "double": PRX_DOUBLE}[width]
    return (w, w * aspect)


plt.style.use("../prx_quantum.mplstyle")

with open("./JSS_info.json", "r") as f:
    info = json.load(f)

JSS_df = pd.DataFrame.from_dict(info["df"], orient="index").T
print(JSS_df)
JSS_n = info["n"]
JSS_o = info["o"]
JSS_m = info["m"]
JSS_timespan = info["timespan"]
JSS_idx_to_delete = info["idx_to_delete"]


# NEW: Build per-operation probability distributions (superposition) from pruned variables
def aqc_probs_to_distributions(sample, normalize_per_op=True):
    """
    sample: dict {1..n_spins -> p in [0,1]} for kept variables after pruning.
    Returns:
      - op_distributions: list of np.array (length JSS_timespan) per operation with probabilities
      - op_machines: list of machine index per operation (1-based)
      - op_durations: list of durations per operation
      - op_names: list of (job, op) tuples aligned with operations
      - expected_starts: list of expected start time per operation (float, np.nan if sum prob == 0)
    """
    # Re-insert pruned variables
    total_vars = int(np.sum(JSS_o) * JSS_timespan)
    entire_sample = np.zeros(total_vars, dtype=float)

    # build kept indices once
    delete_set = set(JSS_idx_to_delete)
    idx_kept = [i for i in range(total_vars) if i not in delete_set]

    # enforce key order 1..len(sample) to keep correct layout
    sample_list = [float(sample[i + 1]) for i in range(len(sample))]
    entire_sample[np.array(idx_kept, dtype=int)] = np.array(sample_list, dtype=float)

    # Slice per operation distributions
    op_distributions = [entire_sample[i : i + JSS_timespan].astype(float) for i in range(0, total_vars, JSS_timespan)]

    if normalize_per_op:
        for k, arr in enumerate(op_distributions):
            s = float(np.sum(arr))
            if s > 0:
                op_distributions[k] = arr / s

    # Map operation index -> machine and duration using JSS_df
    op_machines = [-1] * int(np.sum(JSS_o))
    df = JSS_df.T.unstack()[JSS_df.T.unstack().notna()]
    for i in range(len(op_machines)):
        op_machines[i] = int(df.iloc[i][0])  # machine is first element

    # Names, durations
    op_names = []
    op_durations = []
    for n in range(JSS_n):
        for o in range(JSS_o[n]):
            op_names.append((n + 1, o + 1))
            op_durations.append(int(JSS_df.iloc[n, o][1]))

    # Expected start times
    expected_starts = []
    times = np.arange(JSS_timespan, dtype=float)
    for dist in op_distributions:
        s = float(np.sum(dist))
        if s > 0:
            expected_starts.append(float(np.dot(times, dist)))
        else:
            expected_starts.append(np.nan)

    return op_distributions, op_machines, op_durations, op_names, expected_starts


# NEW: Plot superposition of schedules with translucent bars per possible start time
def gantt_superposition(
    op_distributions,
    op_machines,
    op_durations,
    op_names,
    directory,
    alpha_max=0.85,
    prob_threshold=0.01,
    normalize_alpha_per_op=True,
    show_expected=True,
    title="",
):
    """
    Draws a Gantt-like chart where each operation is rendered as multiple semi-transparent bars,
    one for each possible start time with probability > prob_threshold. Opacity ~ probability.

    Saves figure to 'gantt_superposition_jssp.png' in directory.
    """
    # Colors per job
    colors = [
        "#332288",
        "#DDCC77",
        "#117733",
        "#c1121f",
        "#CC6677",
        "#88CCEE",
        "#44AA99",
        "#999933",
        "#AA4499",
        "#117733",
        "#882255",
        "#6699CC",
        "#888888",
    ]
    color_map = {job: colors[(job - 1) % len(colors)] for job in range(1, JSS_n + 1)}

    # Build helper dicts
    jobs = [name[0] for name in op_names]
    job_colors = [color_map[j] for j in jobs]

    # Compute per-op alpha normalization (to avoid over-dark if probabilities sum != 1)
    if normalize_alpha_per_op:
        per_op_scale = [np.max(d) if np.max(d) > 0 else 1.0 for d in op_distributions]
    else:
        per_op_scale = [1.0] * len(op_distributions)

    plt.rcParams["figure.figsize"] = prx_figsize("single", aspect=0.7)
    fig, ax = plt.subplots()

    # Draw translucent bars for each possible start time
    for i, (dist, m, dur, name, color, scale) in enumerate(
        zip(op_distributions, op_machines, op_durations, op_names, job_colors, per_op_scale)
    ):
        m_row = int(m)  # machine row (1-based)
        for t, p in enumerate(dist):
            if p <= prob_threshold:
                continue
            alpha = alpha_max * (p / scale if scale > 0 else 0.0)
            ax.barh(m_row, dur, left=t, color=color, alpha=np.clip(alpha, 0.0, 1.0), edgecolor=None)

    # Optionally overlay expected start time as outline
    # Also approximate expected makespan for reference
    expected_starts = []
    for dist in op_distributions:
        s = float(np.sum(dist))
        if s > 0:
            expected_starts.append(float(np.dot(np.arange(JSS_timespan, dtype=float), dist)))
        else:
            expected_starts.append(np.nan)

    if show_expected:
        for i, (e_t, m, dur) in enumerate(zip(expected_starts, op_machines, op_durations)):
            if not np.isnan(e_t):
                ax.barh(int(m), dur, left=e_t, fill=False, edgecolor="black")

    # Approx expected makespan (very rough): max_m max_{ops on m}(E[start]+duration)
    approx_makespan = 0.0
    for m in range(1, JSS_m + 1):
        latest_on_m = 0.0
        for i, m_i in enumerate(op_machines):
            if int(m_i) == m and not np.isnan(expected_starts[i]):
                latest_on_m = max(latest_on_m, expected_starts[i] + op_durations[i])
        approx_makespan = max(approx_makespan, latest_on_m)

    # Styling
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_yticks(list(range(1, JSS_m + 1)))
    ax.set_yticklabels(list(range(1, JSS_m + 1)))
    step = max(1, int(np.ceil(JSS_timespan / 10)))
    ax.set_xticks(list(range(0, JSS_timespan + 1, step)))
    ax.set_xticklabels(list(range(0, JSS_timespan + 1, step)))
    ax.tick_params(axis="both", which="major")
    ax.set_xlim(0, JSS_timespan)
    ax.set_ylim(0.5, JSS_m + 0.5)

    # Legend by job
    # unique_jobs = list(range(1, JSS_n + 1))
    # handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[j], alpha=0.8) for j in unique_jobs]
    # labels = [f"Job {j}" for j in unique_jobs]
    # ax.legend(handles, labels, loc="upper right")

    # # Hint line for approx expected makespan
    # if approx_makespan > 0:
    #     ax.axvline(x=approx_makespan, linestyle="--", color="gray", lw=2)
    #     ax.text(
    #         approx_makespan + 0.1,
    #         ax.get_ylim()[1] * 0.5,
    #         f"~E[makespan] ≈ {approx_makespan:.1f}",
    #         color="gray",
    #         rotation=90,
    #         va="center",
    #     )
    _label_kwargs = {"fontsize": 8, "color": "black", "ha": "center", "va": "center"}
    for dist, m, dur, name in zip(op_distributions, op_machines, op_durations, op_names):
        if np.sum(dist) <= 0:
            continue
        t_star = int(np.argmax(dist))           # most probable start
        x = t_star + 0.5 * float(dur)           # center of the bar
        y = int(m)
        job_id, op_id = name
        text = "{op}".format(job=job_id, op=op_id)  # now shows op number by default
        txt = ax.text(x, y, text, **_label_kwargs)
        # white stroke to keep readable on color
        txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])


    if title:
        ax.set_title(title)

    # Add "b)" on the top left corner of the figure
    ax.text(
        -0.1,
        1.05,
        "b)",
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="right",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )

    plt.savefig(os.path.join(directory, "gantt_superposition_jssp.pdf"))
    plt.close()
    return True


def main():
    # il sample deve essere un dizionario con valori
    # Q = np.load("./matrix_Q_j5_op5_pruned.npy")
    Q = np.loadtxt("./matrix_Q_j5_op5_pruned.csv", delimiter=",")
    print("Q shape: ", Q.shape)
    n_spins = Q.shape[0]

    data = np.load("./data.npz", allow_pickle=True)
    # Interpret magnetizations as probabilities in [0,1] for a single superposition Gantt (optional)
    probs = ((data["magnetizations"][-1] + 1) / 2)[:n_spins]
    prob_dict = {i + 1: float(probs[i]) for i in range(len(probs))}

    # Build distributions and plot superposition Gantt
    op_dists, op_machs, op_durs, op_names, exp_starts = aqc_probs_to_distributions(prob_dict, normalize_per_op=True)
    print("Plotting superposition Gantt...")
    gantt_superposition(
        op_dists,
        op_machs,
        op_durs,
        op_names,
        directory="",
        alpha_max=0.8,
        prob_threshold=0.02,
        normalize_alpha_per_op=True,
        show_expected=True,
    )


if __name__ == "__main__":
    main()

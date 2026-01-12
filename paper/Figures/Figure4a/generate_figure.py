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


def main():
    # il sample deve essere un dizionario con valori
    Q = np.load("./matrix_Q_j5_op5_pruned.npy")
    print("Q shape: ", Q.shape)
    n_spins = Q.shape[0]

    data = np.load("./data.npz", allow_pickle=True)
    # Interpret magnetizations as probabilities in [0,1] for a single superposition Gantt (optional)
    probs = ((data["magnetizations"][0] + 1) / 2)[:n_spins]
    prob_dict = {i + 1: float(probs[i]) for i in range(len(probs))}

    # Build distributions and plot superposition Gantt

    # --- Main figure with three insets (early/mid/late) linked to the energy curve ---
    plt.rcParams["figure.figsize"] = prx_figsize("single", aspect=.7)

    plt.figure()

    plt.plot(np.linspace(0,1,data["target_energy"][:, 0].size), data["target_energy"][:, 0],label=r"$\langle H_T \rangle$",color="tab:blue")
    plt.plot(np.linspace(0,1,data["target_energy"][:, 0].size), data["annealing_energy"][:, 0],label=r"$\langle H_A \rangle$",color="tab:red")
    plt.plot(np.linspace(0,1,data["target_energy"][:, 0].size), data["catalyst_energy"][:, 0],label=r"$\langle H_C \rangle$",color="tab:green")
    sol_found_at = 0
    best_energy_thus_far = np.minimum.accumulate(data["target_energy"][:, 2])
    for i in range(data["target_energy"][:, 0].size):
        if best_energy_thus_far[i] < best_energy_thus_far[sol_found_at]:
            sol_found_at = i
    print("Solution found at iteration:", sol_found_at)
    print("Target energy at solution:", data["target_energy"][sol_found_at, 0])
    print("Best target energy:", best_energy_thus_far)
    plt.scatter(sol_found_at/len(data["target_energy"][:, 0]), data["target_energy"][sol_found_at, 0], marker="*", color="black", s=100, zorder=3)
    plt.plot(np.linspace(0,1,data["target_energy"][:, 0].size), best_energy_thus_far, color="tab:blue", linestyle="--")
    plt.xlabel(r"$s$")
    plt.ylabel("Energies")
    plt.legend()

    plt.savefig("figure.pdf")


if __name__ == "__main__":
    main()

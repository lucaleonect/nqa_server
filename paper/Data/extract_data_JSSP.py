import pandas as pd
import numpy as np
import json

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



with open("./JSSP/5x5/pmax7/instance2/JSS_info.json", "r") as f:
    info = json.load(f)

JSS_df = pd.DataFrame.from_dict(info["df"], orient="index").T
JSS_n = info["n"]
JSS_o = info["o"]
JSS_m = info["m"]
JSS_timespan = info["timespan"]
JSS_idx_to_delete = info["idx_to_delete"]

Q = np.load("./JSSP/5x5/pmax7/instance2/matrix_Q_j5_op5_pruned.npy")
n_spins = Q.shape[0]

data = np.load("./JSSP/best_trial_data_5x5_pmax7_i2.npz", allow_pickle=True)

probs = ((data["magnetizations"][0] + 1) / 2)[:n_spins]
prob_dict = {i + 1: float(probs[i]) for i in range(len(probs))}

op_dists, op_machs, op_durs, op_names, _ = aqc_probs_to_distributions(prob_dict, normalize_per_op=True)
print(op_dists)
print(f"Types: {type(op_dists)=}, {type(op_machs)=}, {type(op_durs)=}, {type(op_names)=}")
print(f"Lengths: {len(op_dists)=}, {len(op_machs)=}, {len(op_durs)=}, {len(op_names)=}")
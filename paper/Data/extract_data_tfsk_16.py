import numpy as np
import pandas as pd
import optuna
from sklearn.cluster import DBSCAN

DB_STORAGE_TFSK = "sqlite:///SK_16/tfsk_db.db"
DBSCAN_MIN_NUM_POINTS = 5
DBSCAN_EPS = 0.01
NUM_SUBSAMPLES = 100


data = {"TFSK": {}}

for i in range(10):
    data["TFSK"][f"Instance_{i}"] = {}
    study = optuna.load_study(study_name=f"DBM_SK16_instance_{i}", storage=DB_STORAGE_TFSK)
    completed_trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.user_attrs["Final Energy Var"] < 1e-2
        and t.user_attrs["Final Energy"] - study.best_value < 2
    ]
    data["TFSK"][f"Instance_{i}"]["min_energy"] = min(t.user_attrs["Final Energy"] for t in completed_trials)
    data["TFSK"][f"Instance_{i}"]["final_energies"] = np.array(
        [t.user_attrs["Final Energy"] for t in completed_trials]
    )
    data["TFSK"][f"Instance_{i}"]["variances"] = np.array([t.user_attrs["Final Energy Var"] for t in completed_trials])
    data["TFSK"][f"Instance_{i}"]["inverse_num_params"] = np.array(
        [1.0 / t.user_attrs["Num Params"] for t in completed_trials]
    )
    data["TFSK"][f"Instance_{i}"]["bands"] = {}
    data["TFSK"][f"Instance_{i}"]["bands_ip"] = {}


for i in range(10):
    energies = data["TFSK"][f"Instance_{i}"]["final_energies"]
    variances = data["TFSK"][f"Instance_{i}"]["variances"]
    inv_params = data["TFSK"][f"Instance_{i}"]["inverse_num_params"]

    # DBSCAN clustering in energy axis
    X = np.vstack((np.zeros_like(energies), energies)).T
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_NUM_POINTS).fit(X)
    raw_labels = db.labels_

    # Relabel clusters by mean energy
    cluster_means = {lbl: np.mean(energies[raw_labels == lbl]) for lbl in set(raw_labels) if lbl != -1}
    sorted_labels = {old: new for new, old in enumerate(sorted(cluster_means, key=cluster_means.get))}
    labels = np.array([sorted_labels.get(lbl, -1) for lbl in raw_labels])

    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        v = variances[mask]
        e = energies[mask]

        data["TFSK"][f"Instance_{i}"][f"band_{label}"] = {}

        # Variance-based fit
        mean_intercept = None
        std_intercept = None
        coeffs = np.polyfit(v, e, 1)
        fit_fn = np.poly1d(coeffs)
        data["TFSK"][f"Instance_{i}"][f"band_{label}"]["var_global_fit"] = np.array([min(e), max(e), *coeffs])

        for n in range(NUM_SUBSAMPLES):
            # Randomly sample half of the points in the cluster
            # Take a random permutation of the indices and take the first half
            indices = np.random.permutation(len(e))[: len(e) // 2]
            v_sample = v[indices]
            e_sample = e[indices]
            coeffs_sample = np.polyfit(v_sample, e_sample, 1)
            data["TFSK"][f"Instance_{i}"][f"band_{label}"][f"var_subsample_fit_{n}"] = np.array(
                [min(e_sample), max(e_sample), *coeffs_sample]
            )

        intercepts = np.array(
            [data["TFSK"][f"Instance_{i}"][f"band_{label}"][f"var_subsample_fit_{n}"][3] for n in range(NUM_SUBSAMPLES)]
        )
        mean_intercept = np.mean(intercepts)
        std_intercept = np.std(intercepts)

        slopes = np.array(
            [data["TFSK"][f"Instance_{i}"][f"band_{label}"][f"var_subsample_fit_{n}"][2] for n in range(NUM_SUBSAMPLES)]
        )
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)

        data["TFSK"][f"Instance_{i}"][f"band_{label}"][f"var_statistics"] = np.array(
            [mean_intercept, std_intercept, mean_slope, std_slope]
        )

        # Inverse-params fit
        v = inv_params[mask]
        mean_intercept = None
        std_intercept = None
        coeffs = np.polyfit(v, e, 1)
        fit_fn = np.poly1d(coeffs)
        data["TFSK"][f"Instance_{i}"][f"band_{label}"]["ip_global_fit"] = np.array([min(e), max(e), *coeffs])
        for n in range(NUM_SUBSAMPLES):
            indices = np.random.permutation(len(e))[: len(e) // 2]
            v_sample = v[indices]
            e_sample = e[indices]
            coeffs_sample = np.polyfit(v_sample, e_sample, 1)
            data["TFSK"][f"Instance_{i}"][f"band_{label}"][f"ip_subsample_fit_{n}"] = np.array(
                [min(e_sample), max(e_sample), *coeffs_sample]
            )
        intercepts = np.array(
            [data["TFSK"][f"Instance_{i}"][f"band_{label}"][f"ip_subsample_fit_{n}"][3] for n in range(NUM_SUBSAMPLES)]
        )
        mean_intercept = np.mean(intercepts)
        std_intercept = np.std(intercepts)
        slopes = np.array(
            [data["TFSK"][f"Instance_{i}"][f"band_{label}"][f"ip_subsample_fit_{n}"][2] for n in range(NUM_SUBSAMPLES)]
        )
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        data["TFSK"][f"Instance_{i}"][f"band_{label}"][f"ip_statistics"] = np.array(
            [mean_intercept, std_intercept, mean_slope, std_slope]
        )

        print(f"Instance {i}, Band {label}")
        print(data["TFSK"][f"Instance_{i}"][f"band_{label}"][f"var_statistics"])
        print(data["TFSK"][f"Instance_{i}"][f"band_{label}"][f"ip_statistics"])




# Save data to CSV
rows_fits = []
rows_stats = []

for instance_key, inst_dict in data["TFSK"].items():
    # Skip if bands not yet populated
    for band_key in [k for k in inst_dict.keys() if k.startswith("band_")]:
        try:
            instance_idx = int(instance_key.split("_")[1])
        except (IndexError, ValueError):
            instance_idx = instance_key
        try:
            band_label = int(band_key.split("_")[1])
        except (IndexError, ValueError):
            band_label = band_key

        band_data = inst_dict[band_key]

        # Variance scaling fits
        if "var_global_fit" in band_data:
            min_e, max_e, slope, intercept = band_data["var_global_fit"]
            rows_fits.append(
                {
                    "instance": instance_idx,
                    "band": band_label,
                    "scaling": "variance",
                    "sample_type": "global",
                    "sample_index": None,
                    "min_energy": min_e,
                    "max_energy": max_e,
                    "slope": slope,
                    "intercept": intercept,
                }
            )
        for n in range(NUM_SUBSAMPLES):
            key = f"var_subsample_fit_{n}"
            if key in band_data:
                min_e, max_e, slope, intercept = band_data[key]
                rows_fits.append(
                    {
                        "instance": instance_idx,
                        "band": band_label,
                        "scaling": "variance",
                        "sample_type": "subsample",
                        "sample_index": n,
                        "min_energy": min_e,
                        "max_energy": max_e,
                        "slope": slope,
                        "intercept": intercept,
                    }
                )
        if "var_statistics" in band_data:
            mean_intercept, std_intercept, mean_slope, std_slope = band_data["var_statistics"]
            rows_stats.append(
                {
                    "instance": instance_idx,
                    "band": band_label,
                    "scaling": "variance",
                    "mean_intercept": mean_intercept,
                    "std_intercept": std_intercept,
                    "mean_slope": mean_slope,
                    "std_slope": std_slope,
                }
            )

        # Inverse-parameters scaling fits
        if "ip_global_fit" in band_data:
            min_e, max_e, slope, intercept = band_data["ip_global_fit"]
            rows_fits.append(
                {
                    "instance": instance_idx,
                    "band": band_label,
                    "scaling": "inverse_num_params",
                    "sample_type": "global",
                    "sample_index": None,
                    "min_energy": min_e,
                    "max_energy": max_e,
                    "slope": slope,
                    "intercept": intercept,
                }
            )
        for n in range(NUM_SUBSAMPLES):
            key = f"ip_subsample_fit_{n}"
            if key in band_data:
                min_e, max_e, slope, intercept = band_data[key]
                rows_fits.append(
                    {
                        "instance": instance_idx,
                        "band": band_label,
                        "scaling": "inverse_num_params",
                        "sample_type": "subsample",
                        "sample_index": n,
                        "min_energy": min_e,
                        "max_energy": max_e,
                        "slope": slope,
                        "intercept": intercept,
                    }
                )
        if "ip_statistics" in band_data:
            mean_intercept, std_intercept, mean_slope, std_slope = band_data["ip_statistics"]
            rows_stats.append(
                {
                    "instance": instance_idx,
                    "band": band_label,
                    "scaling": "inverse_num_params",
                    "mean_intercept": mean_intercept,
                    "std_intercept": std_intercept,
                    "mean_slope": mean_slope,
                    "std_slope": std_slope,
                }
            )

df_fits = pd.DataFrame(rows_fits)
df_stats = pd.DataFrame(rows_stats)

# Write CSV files
df_fits.to_csv("tfsk_16_band_fits.csv", index=False)
df_stats.to_csv("tfsk_16_band_statistics.csv", index=False)

print("Saved tfsk_16_band_fits.csv and tfsk_16_band_statistics.csv")

# Also save all the final energies, variances, and inverse num params for each instance in a separate CSV
rows_final = []
for instance_key, inst_dict in data["TFSK"].items():
    try:
        instance_idx = int(instance_key.split("_")[1])
    except (IndexError, ValueError):
        instance_idx = instance_key
    final_energies = inst_dict.get("final_energies", [])
    variances = inst_dict.get("variances", [])
    inv_params = inst_dict.get("inverse_num_params", [])
    for e, v, ip in zip(final_energies, variances, inv_params):
        rows_final.append(
            {
                "instance": instance_idx,
                "final_energy": e,
                "variance": v,
                "inverse_num_params": ip,
            }
        )
df_final = pd.DataFrame(rows_final)
df_final.to_csv("tfsk_16_final_energies.csv", index=False)
import numpy as np
import optuna
import openjij as oj
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

plt.rcParams["mathtext.fontset"] = "cm"
ojij_reads = 1000

DB_STORAGE_SK_100 = "sqlite:///Benchmarks/SK_100/full_comparison_100.db"
DB_STORAGE_SK_200_1 = "sqlite:///Benchmarks/SK_200/full_comparison_200.db"
DB_STORAGE_SK_200_2 = "sqlite:///Benchmarks/SK_200/sk_200_data_v2.db"
DB_STORAGE_TFSK = "sqlite:///Benchmarks/Optuna_QSK_paper/tfsk_db.db"

data = {}
data["SK_100"] = {
    "cRBM_SR": {"e_res": [], "e_best": []},
    "cRBM_NQA": {"e_res": [], "e_best": []},
    "RBQS_SR": {"e_res": [], "e_best": []},
    "RBQS_NQA": {"e_res": [], "e_best": []},
}
data["SK_200"] = {
    "cRBM_SR": {"e_res": [], "e_best": []},
    "cRBM_NQA": {"e_res": [], "e_best": []},
    "RBQS_SR": {"e_res": [], "e_best": []},
    "RBQS_NQA": {"e_res": [], "e_best": []},
    "cRBM_SR_cata": {"e_res": [], "e_best": []},
    "cRBM_NQA_cata": {"e_res": [], "e_best": []},
    "RBQS_SR_cata": {"e_res": [], "e_best": []},
    "RBQS_NQA_cata": {"e_res": [], "e_best": []},
    "cRBM_auto": {"e_res": [], "e_best": []},
    "DBQS_auto": {"e_res": [], "e_best": []},
}
data["TFSK"] = {}

exact_energies_100 = []

for i in range(10):
    print(f"\nproblem {i}")
    J_MATRIX_PATH = f"problems/SK_full_norm/num_spins_100/instance_{i}/J_matrix.npy"
    J_matrix = np.load(J_MATRIX_PATH)
    N = J_matrix.shape[0]
    h = {v: 0 for v in range(N)}
    J = {(p, q): 2 * J_matrix[p, q] for p in range(N) for q in range(p + 1, N)}
    sampler = oj.SASampler()
    response = sampler.sample_ising(h, J, num_reads=ojij_reads)
    energies = np.array([state.T @ J_matrix @ state for state in response.states])
    EXACT_ENERGY = energies.min()
    exact_energies_100 += [EXACT_ENERGY]
    print(f"openjij gs energy {EXACT_ENERGY}")

    def handle_study(study_path: str, key: str):
        try:
            study = optuna.load_study(study_name=study_path, storage=DB_STORAGE_SK_100)
            final_energy = study.best_trial.user_attrs["Final Energy"]
            data["SK_100"][key]["e_res"].append(np.abs(final_energy - EXACT_ENERGY) / np.abs(EXACT_ENERGY) + 1e-10)
            if final_energy < EXACT_ENERGY:
                print(f"Better energy than reference found. Δ = {final_energy - EXACT_ENERGY}")
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            best_energy = min(t.user_attrs["Best Target Energy"] for t in completed)
            data["SK_100"][key]["e_best"].append(np.abs(best_energy - EXACT_ENERGY) / np.abs(EXACT_ENERGY) + 1e-10)
            if best_energy < EXACT_ENERGY:
                print(f"Better energy than reference found. Δ = {best_energy - EXACT_ENERGY}")
            print(f"{key} completed trials: {len(completed)}")
            print(f"{key}: final = {data['SK_100'][key]['e_res'][-1]:.3e}, best = {data['SK_100'][key]['e_best'][-1]:.3e}")
        except Exception as exc:
            print(f"{key} - study not found or failed to load ({exc}). Skipping …")
            pass

    base = f"problems/SK_full_norm/num_spins_100/instance_{i}"
    handle_study(f"{base}/cRBM_SR/", "cRBM_SR")
    handle_study(f"{base}/cRBM_NQA/", "cRBM_NQA")
    handle_study(f"{base}/RBQS_SR/", "RBQS_SR")
    handle_study(f"{base}/RBQS_NQA/", "RBQS_NQA")

for i in range(10):
    print(f"\nproblem {i}")
    J_MATRIX_PATH = f"problems/SK_full_norm/num_spins_200/instance_{i}/J_matrix.npy"
    J_matrix = np.load(J_MATRIX_PATH)
    N = J_matrix.shape[0]
    h = {v: 0 for v in range(N)}
    J = {(p, q): 2 * J_matrix[p, q] for p in range(N) for q in range(p + 1, N)}

    sampler = oj.SASampler()
    response = sampler.sample_ising(h, J, num_reads=ojij_reads)
    energies = np.array([state.T @ J_matrix @ state for state in response.states])
    EXACT_ENERGY = energies.min()
    print(f"openjij gs energy {EXACT_ENERGY}")
    def handle_study(study_path: str, key: str):
        try:
            study = optuna.load_study(study_name=study_path, storage=DB_STORAGE_SK_200_1)
            final_energy = study.best_trial.user_attrs["Final Energy"]
            data["SK_200"][key]["e_res"].append(np.abs(final_energy - EXACT_ENERGY) / np.abs(EXACT_ENERGY) + 1e-10)
            if final_energy < EXACT_ENERGY:
                print(f"Better energy than reference found. Δ = {final_energy - EXACT_ENERGY}")
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            best_energy = min(t.user_attrs["Best Target Energy"] for t in completed)
            data["SK_200"][key]["e_best"].append(np.abs(best_energy - EXACT_ENERGY) / np.abs(EXACT_ENERGY) + 1e-10)
            if best_energy < EXACT_ENERGY:
                print(f"Better energy than reference found. Δ = {best_energy - EXACT_ENERGY}")
            print(f"{key} completed trials: {len(completed)}")
            print(f"{key}: final = {data['SK_200'][key]['e_res'][-1]:.3e}, best = {data['SK_200'][key]['e_best'][-1]:.3e}")
        except Exception as exc:
            print(f"{key} – study not found or failed to load ({exc}). Skipping …")
            pass
    def handle_study_fa(study_path: str, key: str):
        try:
            study = optuna.load_study(study_name=study_path, storage=DB_STORAGE_SK_200_2)
            final_energy = study.best_trial.user_attrs["Final Energy"]
            data["SK_200"][key]["e_res"].append(np.abs(final_energy - EXACT_ENERGY) / np.abs(EXACT_ENERGY) + 1e-10)
            if final_energy < EXACT_ENERGY:
                print(f"Better energy than reference found. Δ = {final_energy - EXACT_ENERGY}")
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            best_energy = min(t.user_attrs["Best Target Energy"] for t in completed)
            data["SK_200"][key]["e_best"].append(np.abs(best_energy - EXACT_ENERGY) / np.abs(EXACT_ENERGY) + 1e-10)
            if best_energy < EXACT_ENERGY:
                print(f"Better energy than reference found. Δ = {best_energy - EXACT_ENERGY}")
            print(f"{key} completed trials: {len(completed)}")
            print(f"{key}: final = {data['SK_200'][key]['e_res'][-1]:.3e}, best = {data['SK_200'][key]['e_best'][-1]:.3e}")
        except Exception as exc:
            print(f"{key} – study not found or failed to load ({exc}). Skipping …")
            pass

    base = f"problems/SK_full_norm/num_spins_200/instance_{i}"
    handle_study(f"{base}/cRBM_SR_mTPE/", "cRBM_SR")
    handle_study(f"{base}/cRBM_NQA_mTPE/", "cRBM_NQA")
    handle_study(f"{base}/RBQS_SR_mTPE/", "RBQS_SR")
    handle_study(f"{base}/RBQS_NQA_mTPE/", "RBQS_NQA")
    handle_study(f"{base}/cRBM_SR_mTPE_cata/", "cRBM_SR_cata")
    handle_study(f"{base}/cRBM_NQA_mTPE_cata/", "cRBM_NQA_cata")
    handle_study(f"{base}/RBQS_SR_mTPE_cata/", "RBQS_SR_cata")
    handle_study(f"{base}/RBQS_NQA_mTPE_cata/", "RBQS_NQA_cata")
    handle_study_fa(f"{base}/cRBM_auto_v2/", "cRBM_auto")
    handle_study_fa(f"{base}/DBQS_auto_v2/", "DBQS_auto")


typical_energies_100 = {}
for algo, metrics in data["SK_100"].items():
    typical_energies_100[algo] = {}
    log_energy_err = np.exp(np.mean(np.log(np.array(metrics["e_res"]))))
    print(f"{algo} typical final energy: {log_energy_err:.3e}")
    typical_energies_100[algo]["final"] = log_energy_err
    log_energy_err = np.exp(np.mean(np.log(np.array(metrics["e_best"]))))
    print(f"{algo} typical best energy: {log_energy_err:.3e}")
    typical_energies_100[algo]["best"] = log_energy_err

typical_energies_200 = {}
for algo, metrics in data["SK_200"].items():
    typical_energies_200[algo] = {}
    log_energy_err = np.exp(np.mean(np.log(np.array(metrics["e_res"]))))
    print(f"{algo} typical final energy: {log_energy_err:.3e}")
    typical_energies_200[algo]["final"] = log_energy_err
    log_energy_err = np.exp(np.mean(np.log(np.array(metrics["e_best"]))))
    print(f"{algo} typical best energy: {log_energy_err:.3e}")
    typical_energies_200[algo]["best"] = log_energy_err

# Figure with two subplots, same axis scales, vertical orientation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 4), sharex=True, sharey=True)
# Top one is the scatter of the residual energies
ax1.scatter(
    [typical_energies_100["cRBM_SR"]["final"], ],
    [typical_energies_100["cRBM_SR"]["best"], ],
    color="blue",
    label="cRBM (SR)",
)
ax1.scatter(
    [typical_energies_100["RBQS_SR"]["final"], ],
    [typical_energies_100["RBQS_SR"]["best"], ],
    color="red",
    label="RBQS (SR)",
)
ax1.scatter(
    [typical_energies_100["cRBM_NQA"]["final"], ],
    [typical_energies_100["cRBM_NQA"]["best"], ],
    color="blue",
    label="cRBM (VQA)",
    marker="x",
)
ax1.scatter(
    [typical_energies_100["RBQS_NQA"]["final"], ],
    [typical_energies_100["RBQS_NQA"]["best"], ],
    color="red",
    label="RBQS (NQA)",
    marker="x",
)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylabel(r"$\epsilon_B$", fontsize=9)
ax1.legend(frameon=True, fontsize=6 ,loc='upper left')
ax1.tick_params(axis='both', which='major', labelsize=8)
# Top one is the scatter of the residual energies
ax2.scatter(
    [typical_energies_200["cRBM_SR"]["final"], ],
    [typical_energies_200["cRBM_SR"]["best"], ],
    color="blue",
    label="cRBM (SR)",
)
ax2.scatter(
    [typical_energies_200["RBQS_SR"]["final"], ],
    [typical_energies_200["RBQS_SR"]["best"], ],
    color="red",
    label="RBQS (SR)",
)
ax2.scatter(
    [typical_energies_200["cRBM_NQA"]["final"], ],
    [typical_energies_200["cRBM_NQA"]["best"], ],
    color="blue",
    label="cRBM (VQA)",
    marker="x",
)
ax2.scatter(
    [typical_energies_200["RBQS_NQA"]["final"], ],
    [typical_energies_200["RBQS_NQA"]["best"], ],
    color="red",
    label="RBQS (NQA)",
    marker="x",
)
plt.scatter(
    [typical_energies_200["DBQS_auto"]["final"], ],
    [typical_energies_200["DBQS_auto"]["best"], ],
    color="purple",
    label="DBQS (NQA)",
    marker="x",
)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$\epsilon_Q$", fontsize=9)
ax2.set_ylabel(r"$\epsilon_B$", fontsize=9)
ax2.legend(frameon=True, fontsize=6 ,loc='upper left')
ax2.tick_params(axis='both', which='major', labelsize=8)
fig.tight_layout()
fig.savefig("typical_energies.pdf")


bins = 11
x_label, y_label = r"$\varepsilon$", "Counts"
log_bins = True

# Build common bin edges (log-spaced)
all_vals = np.concatenate([np.concatenate([v["e_res"], v["e_best"]]) for v in data["SK_200"].values()])
bin_edges = np.logspace(-10, 0, bins + 1) if log_bins else np.histogram_bin_edges(all_vals, bins=bins)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 3), sharex=True, sharey=True)
ax1.hist(data["SK_100"]["RBQS_NQA"]["e_res"], bins=bin_edges, alpha=0.6, color="orange", edgecolor="black", label=r"$\varepsilon_Q$")
ax1.hist(
    data["SK_100"]["RBQS_NQA"]["e_best"],
    bins=bin_edges,
    histtype="step",
    linewidth=1.6,
    hatch="\\\\\\\\",
    color="red",
    label=r"$\varepsilon_B$",
)
ax1.set_xscale("log")
ax1.set_ylabel("Counts", fontsize=9)
ax1.legend(frameon=True, fontsize=9, loc='upper right')
ax1.tick_params(axis='both', which='major', labelsize=8)
ax2.hist(data["SK_200"]["DBQS_auto"]["e_res"], bins=bin_edges, alpha=0.6, color="orange", edgecolor="black", label=r"$\varepsilon_Q$")
ax2.hist(
    data["SK_200"]["DBQS_auto"]["e_best"],
    bins=bin_edges,
    histtype="step",
    linewidth=1.6,
    hatch="\\\\\\\\",
    color="red",
    label=r"$\varepsilon_B$",
)
ax2.set_xscale("log")
ax2.set_ylabel("Counts", fontsize=9)
ax2.set_xlabel(r"$\varepsilon$", fontsize=10)
ax2.legend(frameon=True, fontsize=9, loc='upper right')
ax2.tick_params(axis='both', which='major', labelsize=8)
fig.tight_layout()
fig.savefig("res_energies.pdf")




################################### QUANTUM STUFF HERE ######################################





for i in range(10):
    data["TFSK"][f"Instance_{i}"] = {}
    study = optuna.load_study(study_name=f"DBM_SK_instances_{i}", storage=DB_STORAGE_TFSK)
    completed_trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.user_attrs["Final Energy Var"] < 5e-2
        and t.user_attrs["Final Energy"] - study.best_value < 10.0
    ]
    data["TFSK"][f"Instance_{i}"]["min_energy"] = min(t.user_attrs["Final Energy"] for t in completed_trials)
    data["TFSK"][f"Instance_{i}"]["relative_final_energies"] = np.array(
        [(t.user_attrs["Final Energy"]) for t in completed_trials]
    )
    data["TFSK"][f"Instance_{i}"]["variances"] = np.array([t.user_attrs["Final Energy Var"] for t in completed_trials])
    data["TFSK"][f"Instance_{i}"]["inverse_num_params"] = np.array(
        [1.0 / t.user_attrs["Num Params"] for t in completed_trials]
    )
    data["TFSK"][f"Instance_{i}"]["bands"] = []
    data["TFSK"][f"Instance_{i}"]["bands_ip"] = []


for i in range(10):
    k = min(len(data["TFSK"][f"Instance_{i}"]["relative_final_energies"]), 50)
    y_lim_sup = np.partition(data["TFSK"][f"Instance_{i}"]["relative_final_energies"], k)[k] + 1

    spliced_energies = np.arange(
        np.floor(min(data["TFSK"][f"Instance_{i}"]["relative_final_energies"])),
        y_lim_sup,
        0.2,
    )
    for e1, e2 in zip(spliced_energies[:-1], spliced_energies[1:]):
        mask = (data["TFSK"][f"Instance_{i}"]["relative_final_energies"] >= e1) & (
            data["TFSK"][f"Instance_{i}"]["relative_final_energies"] < e2
        )
        if np.sum(mask) >= 5:
            v = data["TFSK"][f"Instance_{i}"]["variances"][mask]
            e = data["TFSK"][f"Instance_{i}"]["relative_final_energies"][mask]
            if len(v) > 1:
                coeffs = np.polyfit(v, e, 1, w=1/np.sqrt(v+1e-3))
                fit_fn = np.poly1d(coeffs)
            data["TFSK"][f"Instance_{i}"]["bands"].append((e1, e2, coeffs))

            v = data["TFSK"][f"Instance_{i}"]["inverse_num_params"][mask]
            e = data["TFSK"][f"Instance_{i}"]["relative_final_energies"][mask]
            if len(v) > 1:
                coeffs = np.polyfit(v, e, 1)
                fit_fn = np.poly1d(coeffs)
            data["TFSK"][f"Instance_{i}"]["bands_ip"].append((e1, e2, coeffs))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 2), sharey=True)
    ax1.scatter(
        data["TFSK"][f"Instance_{i}"]["variances"],
        data["TFSK"][f"Instance_{i}"]["relative_final_energies"],
        color="black",
        s=8,
        alpha=0.6,
    )
    for e1, e2, coeffs in data["TFSK"][f"Instance_{i}"]["bands"]:
        x = np.linspace(0, 5e-2, 100)
        y = coeffs[0] * x + coeffs[1]
        ax1.plot(x, y, color="black", linestyle="--", linewidth=1.2)
        ax1.scatter(
            [0],
            [coeffs[1]],
            color="black",
            marker="x",
            s=20,
            zorder=5,
        )
    ax1.set_ylim(data["TFSK"][f"Instance_{i}"]["min_energy"]-1e-1, y_lim_sup+1)
    ax1.set_xlim(-1e-3, 5e-2)
    ax1.set_xlabel(r"$\sigma^2$", fontsize=10)
    ax1.set_ylabel(r"$\langle H \rangle$", fontsize=10)
    # horizontal line for the classical energy

    ax2.scatter(
        data["TFSK"][f"Instance_{i}"]["inverse_num_params"],
        data["TFSK"][f"Instance_{i}"]["relative_final_energies"],
        color="black",
        s=8,
        alpha=0.6,
    )
    for e1, e2, coeffs in data["TFSK"][f"Instance_{i}"]["bands_ip"]:
        x = np.linspace(0, 0.00015, 100)
        y = coeffs[0] * x + coeffs[1]
        ax2.plot(x, y, color="black", linestyle="--", linewidth=1.2)
        ax2.scatter([0,], [coeffs[1],], color="black", marker="x", s=20, zorder=5)
    ax2.set_xlim(-1e-6, 0.00015)
    ax2.set_xticks([0, 5e-5, 1e-4])
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0e}"))
    ax2.set_xlabel(r"$1/N_{\text{params}}$", fontsize=10)

    # ax3 should be a histogram of the relative final energies (vertical orientation, i. e.: normal histogram rotated by 90 degrees)
    ax3.hist(
        data["TFSK"][f"Instance_{i}"]["relative_final_energies"],
        bins=51,
        orientation="horizontal",
        color="gray",
        edgecolor="black",
        alpha=0.7,
    )
    ax3.set_xlabel("Counts", fontsize=10)

    ax1.axhline(y=exact_energies_100[i], color="blue", linestyle=":", linewidth=1.2)
    ax2.axhline(y=exact_energies_100[i], color="blue", linestyle=":", linewidth=1.2)
    ax3.axhline(y=exact_energies_100[i], color="blue", linestyle=":", linewidth=1.2)

    fig.tight_layout()
    fig.savefig(f"tfsk_bands_{i}.pdf")
    fig.show()
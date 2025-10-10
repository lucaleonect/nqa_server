# Get the data from the various db files to generate the figures and save them in the related csv files

import optuna
import openjij as oj
import numpy as np
import pandas as pd

ojij_reads = 1000

DB_STORAGE_SK_100 = "sqlite:///SK_100/sk_100.db"
DB_STORAGE_TFSK = "sqlite:///SK_100/tfsk_db.db"
DB_STORAGE_SK_200_1 = "sqlite:///SK_200/sk_200_v1.db"
DB_STORAGE_SK_200_2 = "sqlite:///SK_200/sk_200_v2.db"

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

exact_energies_100 = []

for i in range(10):
    print(f"\nproblem {i}")
    J_MATRIX_PATH = f"./SK_100/instances/instance_{i}/J_matrix.npy"
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
    J_MATRIX_PATH = f"./SK_200/instances/instance_{i}/J_matrix.npy"
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

# Figure 2 (typical energies scatter plot)
save_path = "../Figures/Figure2/data.csv"
df_100 = pd.DataFrame(typical_energies_100).T
df_100.index.name = "Algorithm"
df_100.columns = [f"SK 100 {col.title()}" for col in df_100.columns]

df_200 = pd.DataFrame(typical_energies_200).T
df_200.index.name = "Algorithm"
df_200.columns = [f"SK 200 {col.title()}" for col in df_200.columns]

df = pd.concat([df_100, df_200], axis=1)
df.to_csv(save_path)

# Figure 3 (Both errors for SK_100 RBQS_NQA and SK_200 DBQS_auto)
save_path = "../Figures/Figure3/data.csv"
error_best_100 = data["SK_100"]["RBQS_NQA"]["e_best"]
error_res_100 = data["SK_100"]["RBQS_NQA"]["e_res"]
error_best_200 = data["SK_200"]["DBQS_auto"]["e_best"]
error_res_200 = data["SK_200"]["DBQS_auto"]["e_res"]
df = pd.DataFrame({
    "SK 100 Best": error_best_100,
    "SK 100 Final": error_res_100,
    "SK 200 Best": error_best_200,
    "SK 200 Final": error_res_200,
})
df.to_csv(save_path)

# Figure 7 (histograms of all the best energy errors)
error_best_100_cRBM_SR = data["SK_100"]["cRBM_SR"]["e_best"]
error_best_100_cRBM_NQA = data["SK_100"]["cRBM_NQA"]["e_best"]
error_best_100_RBQS_SR = data["SK_100"]["RBQS_SR"]["e_best"]
error_best_100_RBQS_NQA = data["SK_100"]["RBQS_NQA"]["e_best"]
error_best_200_cRBM_SR = data["SK_200"]["cRBM_SR"]["e_best"]
error_best_200_cRBM_NQA = data["SK_200"]["cRBM_NQA"]["e_best"]
error_best_200_RBQS_SR = data["SK_200"]["RBQS_SR"]["e_best"]
error_best_200_RBQS_NQA = data["SK_200"]["RBQS_NQA"]["e_best"]
save_path = "../Figures/Figure7/data.csv"
df = pd.DataFrame({
    "cRBM_SR_100": error_best_100_cRBM_SR,
    "cRBM_NQA_100": error_best_100_cRBM_NQA,
    "RBQS_SR_100": error_best_100_RBQS_SR,
    "RBQS_NQA_100": error_best_100_RBQS_NQA,
    "cRBM_SR_200": error_best_200_cRBM_SR,
    "cRBM_NQA_200": error_best_200_cRBM_NQA,
    "RBQS_SR_200": error_best_200_RBQS_SR,
    "RBQS_NQA_200": error_best_200_RBQS_NQA,
})
df.to_csv(save_path, index=False)

# Figure 8 (histogram of SK_200 errors with cata)
error_best_200_cRBM_SR_cata = data["SK_200"]["cRBM_SR_cata"]["e_best"]
error_best_200_cRBM_NQA_cata = data["SK_200"]["cRBM_NQA_cata"]["e_best"]
error_best_200_RBQS_SR_cata = data["SK_200"]["RBQS_SR_cata"]["e_best"]
error_best_200_RBQS_NQA_cata = data["SK_200"]["RBQS_NQA_cata"]["e_best"]
save_path = "../Figures/Figure8/data.csv"
df = pd.DataFrame({
    "cRBM_SR_cata_200": error_best_200_cRBM_SR_cata,
    "cRBM_NQA_cata_200": error_best_200_cRBM_NQA_cata,
    "RBQS_SR_cata_200": error_best_200_RBQS_SR_cata,
    "RBQS_NQA_cata_200": error_best_200_RBQS_NQA_cata,
})
df.to_csv(save_path, index=False)


import numpy as np
import optuna

DB_STORAGE_TFSK = "sqlite:///Benchmarks/Optuna_QSK_paper/tfsk_db.db"

data = {}

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
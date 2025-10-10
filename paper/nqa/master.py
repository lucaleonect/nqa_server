import numpy as np
import time
import optuna
import subprocess
from itertools import chain

optuna.logging.set_verbosity(optuna.logging.WARNING)

DB_STORAGE = f"sqlite:///optuna_database.db"
NUM_TRIALS = 10
NUM_WORKERS = 1
STUDY_NAME = f"test/"
J_MATRIX_PATH = f"test/J_matrix.npy"


default_args = {
    "cuda_device": 0,
    "max_runtime": 60 * 60,
    "J_matrix_path": J_MATRIX_PATH,
    "vqa_num_warmup_steps": 1,
    "vqa_num_updates_per_step": 1,
    "vqa_num_finetuning_steps": 1,
    "vqa_annealing_field_scale": 1.0,
    "sr_diagonal_shift": 1e-4,
    "dbqs_num_hidden_layers": 1,
    "dbqs_unit_density_per_layer": 1.0,
    "dbqs_param_dtype": "complex",
    "mcmc_num_samples": 2**7,
    "mcmc_num_thermalization_steps": 2**10,
    "mcmc_num_sweep_steps": 2**7,
}
trial_args_settings = {
    "vqa_num_annealing_steps": ("int", 1e3, 1e5, "log"),
    "sgd_learning_rate": ("float", 1e-4, 1e0, "log"),
    "sgd_momentum": ("float", 0.0, 0.99, "linear"),
}
dynamic_args = {
    "target_energy": None,
    "save_path": None,
}
flag_args = {
    "tqdm",
    # "vqa_no_annealing",
    "vqa_use_catalyst",
    # "cRBM",
    "mcmc_persistent_markov_chains",
    # "mcmc_metropolis_hastings",
}

def start_annealer(cfg: dict):
    cmd = [
        "python3", 
        "new_worker.py",
        *chain.from_iterable((f"--{k}", str(v)) for k, v in cfg.items()),
        *[f"--{f}" for f in flag_args],
    ]
    return subprocess.Popen(cmd)


def objective(trial):
    trial_args = {}
    for arg in trial_args_settings:
        arg_type, low, high, scale = trial_args_settings[arg]
        if arg_type == "int":
            trial_args[arg] = trial.suggest_int(arg, low, high, log=(scale == "log"))
        elif arg_type == "float":
            trial_args[arg] = trial.suggest_float(arg, low, high, log=(scale == "log"))

    dynamic_args = {}
    dynamic_args["save_path"] = f"{STUDY_NAME}/test_{trial.number}"
    try:
        dynamic_args["target_energy"] = trial.study.best_value
    except:
        dynamic_args["target_energy"] = 100

    annealer_args = {**default_args, **trial_args, **dynamic_args}
    new_simulation = start_annealer(annealer_args)
    new_simulation.wait()  # Unfortunately this doesn't work, as the process finishes immediately after the job is submitted

    while True:
        try:
            with open(f"{dynamic_args['save_path']}/finished.txt", "r") as f:
                break
        except FileNotFoundError:
            time.sleep(15)

    # if failed.txt exists, return None
    try:
        with open(f"{dynamic_args['save_path']}/failed.txt", "r") as f:
            print(f"Trial {trial.number} failed.")
            return None
    except FileNotFoundError:
        pass

    out_data = np.load(f"{annealer_args['save_path']}/data.npz")
    obj_vaue = out_data["target_energy"][-1, 0]
    print(
        f"Trial {trial.number} completed. Objective values: {obj_vaue:.2f}, {out_data["target_energy"][-1, 0]:.2f}, {out_data["best_energy_so_far"][-1]:.2f}, {out_data["runtime"]}"
    )

    trial.set_user_attr("Final Energy", out_data["target_energy"][-1, 0])
    trial.set_user_attr("Best Target Energy", out_data["best_energy_so_far"][-1])
    trial.set_user_attr("Num Params", int(out_data["num_params"]))
    trial.set_user_attr("Runtime", float(out_data["runtime"]))
    trial.set_user_attr("Best Config", str(np.array((out_data["best_config"] + 1) / 2)))

    return obj_vaue

def main():
    optuna_sampler = optuna.samplers.TPESampler()
    try:
        study = optuna.create_study(
            study_name=STUDY_NAME, storage=DB_STORAGE, direction="minimize", sampler=optuna_sampler
        )
        for arg in default_args:
            study.set_user_attr(arg, default_args[arg])
        study.set_user_attr("Trial Args", trial_args_settings)

    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_STORAGE, sampler=optuna_sampler)

    study.optimize(
        objective,
        n_trials=NUM_TRIALS,
        n_jobs=NUM_WORKERS,
    )

if __name__ == "__main__":
    main()

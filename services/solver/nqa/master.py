import os
import argparse
import numpy as np
import time
import optuna
import subprocess
from itertools import chain

parser = argparse.ArgumentParser(description="Neural Quantum Annealer.")
parser.add_argument("--J_matrix_path", type=str, default=None)
parser.add_argument("--h_vector_path", type=str, default=None)
parser.add_argument("--g_vector_path", type=str, default=None)
parser.add_argument("--db_storage_path", type=str, default=None)
parser.add_argument("--study_name", type=str, default=None)
parser.add_argument("--num_trials", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--cuda_device", type=int, default=0)
parser.add_argument("--trial_max_runtime", type=int, default=60 * 10)
args = parser.parse_args()

if args.J_matrix_path is None:
    raise ValueError("J_matrix_path must be specified.")

if args.study_name is None:
    args.study_name = f"study_{int(time.time())}"
args.study_name = f"studies/{args.study_name}"

if args.db_storage_path is None:
    args.db_storage_path = f"{args.study_name}/optuna_db.db"

STUDY_NAME = args.study_name
os.makedirs(STUDY_NAME, exist_ok=True)
DB_STORAGE = f"sqlite:///{args.db_storage_path}"
NUM_TRIALS = args.num_trials
NUM_WORKERS = args.num_workers

default_args = {
    "cuda_device": args.cuda_device,
    "max_runtime": args.trial_max_runtime,
    "J_matrix_path": args.J_matrix_path,
    "vqa_num_warmup_steps": 10,
    "vqa_num_updates_per_step": 1,
    "vqa_num_finetuning_steps": 100,
    "dbqs_param_dtype": "complex",
    "mcmc_num_thermalization_steps": 2**7,
}
if args.h_vector_path is not None:
    default_args["h_vector_path"] = args.h_vector_path
if args.g_vector_path is not None:
    default_args["g_vector_path"] = args.g_vector_path
trial_args_settings = {
    "vqa_num_annealing_steps": ("int", 1e3, 1e4, "log"),
    "vqa_annealing_field_scale": ("float", 1e-1, 1e1, "log"),
    "vqa_catalyst_field_scale": ("float", 1e-1, 1e1, "log"),
    "sgd_learning_rate": ("float", 1e-4, 1e0, "log"),
    "sgd_momentum": ("float", 0.0, 0.9, "linear"),
    "sr_diagonal_shift": ("float", 1e-9, 1e0, "log"),
    "dbqs_num_hidden_layers": ("int", 1, 3, "linear"),
    "dbqs_unit_density_per_layer": ("float", 0.25, 4, "log"),
    "mcmc_num_samples": ("int", 2**3, 2**7, "log"),
    "mcmc_num_sweep_steps": ("int", 2**2, 2**7, "log"),
}
dynamic_args = {
    "target_energy": None,
    "save_path": None,
}
flag_args = {
}

def start_annealer(cfg: dict):
    cmd = [
        "python3",
        "worker.py",
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
    # optuna_sampler = optuna.samplers.CmaEsSampler()
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

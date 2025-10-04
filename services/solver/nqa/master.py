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
parser.add_argument("--energy_shift", type=float, default=0.0)
parser.add_argument("--db_storage_path", type=str, default=None)
parser.add_argument("--study_name", type=str, default=None)
parser.add_argument("--num_trials", type=int, default=100)
parser.add_argument("--vqa_num_annealing_steps_min", type=int, default=1000)
parser.add_argument("--vqa_num_annealing_steps_max", type=int, default=10000)
parser.add_argument("--vqa_num_updates_per_step_min", type=int, default=1)
parser.add_argument("--vqa_num_updates_per_step_max", type=int, default=5)
parser.add_argument("--vqa_annealing_field_scale_min", type=float,default=1e-1)
parser.add_argument("--vqa_annealing_field_scale_max", type=float,default=1e1)
parser.add_argument("--vqa_catalyst_field_scale_min", type=float,default=1e-1)
parser.add_argument("--vqa_catalyst_field_scale_max", type=float,default=1e1)
parser.add_argument("--sgd_learning_rate_min", type=float,default=1e-3)
parser.add_argument("--sgd_learning_rate_max", type=float,default=1e0)
parser.add_argument("--sgd_momentum_min", type=float,default=0)
parser.add_argument("--sgd_momentum_max", type=float,default=0.9)
parser.add_argument("--sr_diagonal_shift_min", type=float,default=1e-9)
parser.add_argument("--sr_diagonal_shift_max", type=float,default=1e-2)
parser.add_argument("--dbqs_num_hidden_layers", type=int,default=2)
parser.add_argument("--dbqs_unit_density_per_layer_min", type=float,default=0.25)
parser.add_argument("--dbqs_unit_density_per_layer_max", type=float,default=4.0)
parser.add_argument("--mcmc_num_samples_min", type=int,default=2**3)
parser.add_argument("--mcmc_num_samples_max", type=int,default=2**6)
parser.add_argument("--mcmc_num_sweep_steps_min", type=int,default=2**2)
parser.add_argument("--mcmc_num_sweep_steps_max", type=int,default=2**6)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--cuda_device", type=int, default=0)
parser.add_argument("--trial_max_runtime", type=int, default=60 * 30)
parser.add_argument("--target_objective_value", type=float, default=-1e5)

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

TARGET_OBJECTIVE_VALUE = args.target_objective_value

default_args = {
    "cuda_device": args.cuda_device,
    "max_runtime": args.trial_max_runtime,
    "J_matrix_path": args.J_matrix_path,
    "vqa_num_warmup_steps": 10,
    "vqa_num_finetuning_steps": 100,
    "dbqs_param_dtype": "complex",
    "mcmc_num_thermalization_steps": 2**7,
    "dbqs_num_hidden_layers": args.dbqs_num_hidden_layers,
    "energy_shift": args.energy_shift,
}
if args.h_vector_path is not None:
    default_args["h_vector_path"] = args.h_vector_path
if args.g_vector_path is not None:
    default_args["g_vector_path"] = args.g_vector_path
trial_args_settings = {
    "vqa_num_annealing_steps": ("int", args.vqa_num_annealing_steps_min, args.vqa_num_annealing_steps_max, "log"),
    "vqa_num_updates_per_step": ("int", args.vqa_num_updates_per_step_min, args.vqa_num_updates_per_step_max, "linear"),
    "vqa_annealing_field_scale": ("float", args.vqa_annealing_field_scale_min, args.vqa_annealing_field_scale_max, "log"),
    "vqa_catalyst_field_scale": ("float", args.vqa_catalyst_field_scale_min, args.vqa_catalyst_field_scale_max, "log"),
    "sgd_learning_rate": ("float", args.sgd_learning_rate_min, args.sgd_learning_rate_max, "log"),
    "sgd_momentum": ("float", args.sgd_momentum_min, args.sgd_momentum_max, "linear"),
    "sr_diagonal_shift": ("float", args.sr_diagonal_shift_min, args.sr_diagonal_shift_max, "log"),
    "dbqs_unit_density_per_layer": ("float", args.dbqs_unit_density_per_layer_min, args.dbqs_unit_density_per_layer_max, "log"),
    "mcmc_num_samples": ("int", args.mcmc_num_samples_min, args.mcmc_num_samples_max, "log"),
    "mcmc_num_sweep_steps": ("int", args.mcmc_num_sweep_steps_min, args.mcmc_num_sweep_steps_max, "log"),
}
dynamic_args = {
    "target_energy": None,
    "save_path": None,
}
flag_args = {
}

is_classical_target = (args.g_vector_path is None)  # If g_vector_path is not given, we assume it's a classical target Hamiltonian.

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
    obj_value = out_data["target_energy"][-1, 0]

    trial.set_user_attr("Final Energy", out_data["target_energy"][-1, 0].real)
    if is_classical_target:
        trial.set_user_attr("Best Target Energy", out_data["best_energy_so_far"][-1])
        trial.set_user_attr("Best Config", str(np.array((out_data["best_config"] + 1) / 2)))
    trial.set_user_attr("Final Energy Variance", out_data["target_energy"][-1, 1].real)
    trial.set_user_attr("Num Params", int(out_data["num_params"]))
    trial.set_user_attr("Runtime", float(out_data["runtime"]))

    return obj_value

def study_callback(study, trial):
    if study.best_value is not None and study.best_value < TARGET_OBJECTIVE_VALUE:
        print(f"Target objective value {TARGET_OBJECTIVE_VALUE} reached. Stopping study.")
        study.stop()


def main():
    optuna_sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=DB_STORAGE, direction="minimize", sampler=optuna_sampler
    )
    for arg in default_args:
        study.set_user_attr(arg, default_args[arg])
    study.set_user_attr("Trial Args", trial_args_settings)

    tpe_trials = NUM_TRIALS//5
    cmaes_trials = NUM_TRIALS - tpe_trials

    study.optimize(
        objective,
        n_trials=cmaes_trials,
        n_jobs=NUM_WORKERS,
        callbacks=[study_callback],
    )

    optuna_sampler = optuna.samplers.TPESampler()
    study = optuna.load_study(study_name=STUDY_NAME, storage=DB_STORAGE, sampler=optuna_sampler)
    study.optimize(
        objective,
        n_trials=tpe_trials,
        n_jobs=NUM_WORKERS,
        callbacks=[study_callback],
    )


if __name__ == "__main__":
    main()

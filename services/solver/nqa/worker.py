# region: argparse
import argparse

parser = argparse.ArgumentParser(
    prog="Neural Quantum Annealer",
    description="Runs NQA to solve a given SK spin glass problem.",
)

# General settings
parser.add_argument(
    "--cuda_device",
    type=str,
    default=None,
    help="CUDA device to use, e.g. '0' for the first GPU, '1' for the second, etc. If None, '0' will be used by default.",
)
parser.add_argument(
    "--max_runtime",
    type=int,
    default=60 * 60,
    help="Maximum runtime in seconds. After 10%% of the maximum runtime, the program will estimate the remaining runtime and halt if it exceeds the maximum runtime. Default is 6 hours (21600 seconds).",
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    help="Path to save the results. If None, a timestamped directory will be created in the current working directory.",
)
parser.add_argument(
    "--prng_seed",
    type=int,
    default=None,
    help="Random seed for reproducibility. If None, the current timestamp will be used as the seed.",
)
parser.add_argument(
    "--tqdm",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use tqdm to show progress bars.",
)

# Problem settings
parser.add_argument(
    "--J_matrix_path",
    type=str,
    default=None,
    help="Path to the J matrix file (numpy .npy format).",
)
parser.add_argument(
    "--h_vector_path",
    type=str,
    default=None,
    help="Path to the h vector file (numpy .npy format). If None and --h_value is not provided, a zero vector will be used.",
)
parser.add_argument(
    "--h_value",
    type=float,
    default=None,
    help="Fixed value for the magnetic field h. If provided, this will create a uniform field h_vector = h_value * ones(num_spins). Cannot be used together with --h_vector_path.",
)
parser.add_argument(
    "--g_vector_path",
    type=str,
    default=None,
    help="Path to the g vector file (numpy .npy format). If None and --g_value is not provided, a zero vector will be used.",
)
parser.add_argument(
    "--g_value",
    type=float,
    default=None,
    help="Fixed value for the transverse field g. If provided, this will create a uniform field g_vector = g_value * ones(num_spins). Cannot be used together with --g_vector_path.",
)
parser.add_argument(
    "--energy_shift",
    type=float,
    default=0.0,
    help="Constant to add to the energy. This is useful for shifting the energy scale, e.g. to avoid negative energies.",
)
parser.add_argument(
    "--target_energy",
    type=float,
    default=None,
    help="Target energy for the optimization. If None, jnp.inf will be used. If the final energy is above this value, the run will be considered failed and the data will be saved with minimal logging.",
)

# Annealer settings
parser.add_argument(
    "--vqa_num_annealing_steps",
    type=int,
    default=10000,
    help="Number of annealing steps for the variational quantum annealer. Default is 10000.",
)
parser.add_argument(
    "--vqa_num_warmup_steps",
    type=int,
    default=1,
    help="Number of warmup steps before the annealing starts. Default is 1.",
)
parser.add_argument(
    "--vqa_num_updates_per_step",
    type=int,
    default=1,
    help="Number of updates per annealing step. Default is 1.",
)
parser.add_argument(
    "--vqa_num_finetuning_steps",
    type=int,
    default=100,
    help="Number of finetuning steps after the annealing ends. Default is 100.",
)
parser.add_argument(
    "--vqa_annealing_field_scale",
    type=float,
    default=1,
    help="Scaling factor for the annealing field. Default is 1.",
)
parser.add_argument(
    "--vqa_catalyst_field_scale",
    type=float,
    default=1,
    help="Scaling factor for the catalyst field. Default is 1.",
)
parser.add_argument(
    "--vqa_no_catalyst",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Disable catalyst for the annealing.",
)

# Optimizer settings
parser.add_argument(
    "--sgd_learning_rate",
    type=float,
    default=1e-1,
    help="Learning rate for the SGD optimizer. Default is 1e-1.",
)
parser.add_argument(
    "--sgd_momentum",
    type=float,
    default=0.5,
    help="Momentum for the SGD optimizer. Default is 0.5.",
)

# SR settings
parser.add_argument(
    "--sr_prefactor",
    type=complex,
    default=1.0 + 0.0j,
    help="Prefactor for the natural gradients method.",
)
parser.add_argument(
    "--sr_diagonal_shift",
    type=float,
    default=1e-2,
    help="Diagonal shift applied to the FIM or NTK before computing its pseudoinverse. Default is 1e-2.",
)
parser.add_argument(
    "--sr_method",
    type=str,
    default=None,
    help="Method for the natural gradients. 'SR' to use standard SR, 'minSR' to use minSR or 'auto' to automatically chose the most convenient one depending on the other settings. Default is None, which uses the auto method.",
)

# Variational quantum state settings
parser.add_argument(
    "--dbqs_num_hidden_layers",
    type=int,
    default=2,
    help="Number of hidden layers in the Deep Boltzmann Quantum State. Default is 2.",
)
parser.add_argument(
    "--dbqs_unit_density_per_layer",
    type=float,
    default=1,
    help="Number of units density per layer in the Deep Boltzmann Quantum State. Default is 1, which means one unit per visible spin.",
)
parser.add_argument(
    "--dbqs_param_dtype",
    type=str,
    default="complex",
    help="Data type for the parameters of the Deep Boltzmann Quantum State. Accepted values are 'float' and 'complex'. Default is 'complex'.",
)
parser.add_argument(
    "dbqs_use_bias",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to use bias terms in the Deep Boltzmann Quantum State. Default is True.",
)

# Markov chain Monte Carlo sampler settings
parser.add_argument(
    "--mcmc_num_samples",
    type=int,
    default=2**7,
    help="Number of samples to generate to compute observables. Default is 128.",
)
parser.add_argument(
    "--mcmc_num_chains",
    type=int,
    default=None,
    help="Number of parallel Markov chains to run. If None, it will be set to the number of samples. If the number of samples is not divisible by the number of chains, it will be overridden to the next smallest number that is divisible by the number of chains.",
)
parser.add_argument(
    "--mcmc_num_thermalization_steps",
    type=int,
    default=2**7,
    help="Number of thermalization steps to run for each chain. Default is 128.",
)
parser.add_argument(
    "--mcmc_num_sweep_steps",
    type=int,
    default=2**4,
    help="Number of sweep steps to run for each chain. Default is 16.",
)
parser.add_argument(
    "--mcmc_disable_persistent_markov_chains",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Disable persistent Markov chains. Default is False.",
)

args = parser.parse_args()
# endregion

# region: imports and environment variables
import os

os.environ["JAX_ENABLE_X64"] = "True"  # Use double precision

if args.cuda_device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

import jax
import jax.numpy as jnp
import optax
import numpy as np
import time
import pickle
from utils.boltzmann_quantum_states import DeepBoltzmannQuantumState
from utils.operators import (
    build_local_tfsk_energy,
    build_local_parametric_hamiltonian,
    build_measurement_function,
)
from utils.tdvp import build_parametric_gradient_estimator
from utils.variational_annealer import VariationalAnnealer
from utils.serialization import serialize_data

# endregion


def main():
    start_time = int(time.time())
    if args.prng_seed is None:
        args.prng_seed = start_time
    if args.save_path is None:
        args.save_path = f"./data_{start_time}"
    if args.target_energy is None:
        args.target_energy = jnp.inf
    if args.mcmc_num_chains is None:
        args.mcmc_num_chains = args.mcmc_num_samples

    def print_log(msg: str):
        # print_log on log.txt (append if exists)
        with open("log.txt", "a") as f:
            f.write(f"{msg}\n")

    print_log("Args recap:")
    for arg_name, arg_value in vars(args).items():
        print_log(f"{arg_name}: {arg_value}")

    print_log(f"Debug info\n")
    print_log(f"JAX version: {jax.__version__}")
    print_log(f"JAX backend: {jax.default_backend()}")
    print_log(f"JAX devices: {jax.devices()}")

    os.makedirs(args.save_path, exist_ok=True)
    with open(args.save_path + f"/worker_args.pkl", "wb") as f:
        pickle.dump(args, f)

    prngkey = jax.random.PRNGKey(args.prng_seed)
    # region: load the problem
    J_matrix = np.load(args.J_matrix_path)
    num_spins = J_matrix.shape[0]

    # Handle h_vector and g_vector: check for conflicting options, then set appropriately
    if args.h_value is not None and args.h_vector_path is not None:
        raise ValueError("Cannot specify both --h_value and --h_vector_path. Please use only one option.")
    elif args.h_value is not None:
        h_vector = args.h_value * np.ones(num_spins)
    elif args.h_vector_path is not None:
        h_vector = np.load(args.h_vector_path)
    else:
        h_vector = np.zeros(num_spins)

    is_classical_target = False
    if args.g_value is not None and args.g_vector_path is not None:
        raise ValueError("Cannot specify both --g_value and --g_vector_path. Please use only one option.")
    elif args.g_value is not None:
        g_vector = args.g_value * np.ones(num_spins)
    elif args.g_vector_path is not None:
        g_vector = np.load(args.g_vector_path)
    else:
        g_vector = None
        is_classical_target = True
    # endregion

    # region: build bqs
    dbqs_units_per_layer = int(num_spins * args.dbqs_unit_density_per_layer)
    dbqs_layers = [dbqs_units_per_layer] * args.dbqs_num_hidden_layers

    if args.dbqs_param_dtype == "float":
        dtype = jnp.float64
    elif args.dbqs_param_dtype == "complex":
        dtype = jnp.complex128
    else:
        raise ValueError(f"Unknown data type: {args.dbqs_param_dtype}")

    prngkey, tempkey = jax.random.split(prngkey)
    vqs = DeepBoltzmannQuantumState(
        num_spins=num_spins,
        hidden_layers=dbqs_layers,
        prngkey=tempkey,
        num_samples=args.mcmc_num_samples,
        num_thermalization_steps=args.mcmc_num_thermalization_steps,
        num_sweep_steps=args.mcmc_num_sweep_steps,
        num_chains=args.mcmc_num_chains,
        dtype=dtype,
        use_bias=args.dbqs_use_bias,
    )
    # endregion

    # region: build the variational annealer
    # region: local energy function and tdvp gradient estimator
    local_energy_target = build_local_tfsk_energy(
        deep_boltzmann_quantum_state=vqs,
        J_matrix=J_matrix,
        h_vector=h_vector,
        g_vector=g_vector,
        energy_shift=args.energy_shift,
    )

    local_energy_annealing = (
        lambda params, spins: vqs.local_energy_sigma_x(params, spins) * args.vqa_annealing_field_scale
    )
    if not args.vqa_no_catalyst:
        local_energy_catalyst = (
            lambda params, spins: vqs.local_energy_sigma_y(params, spins) * args.vqa_catalyst_field_scale
        )
    else:
        local_energy_catalyst = None

    local_hamiltonian = build_local_parametric_hamiltonian(
        local_target_hamiltonian=local_energy_target,
        local_annealing_hamiltonian=local_energy_annealing,
        local_catalyst_hamiltonian=local_energy_catalyst,
    )
    gradient_estimator = build_parametric_gradient_estimator(
        deep_boltzmann_quantum_state=vqs,
        local_hamiltonian=local_hamiltonian,
        diag_shift=args.sr_diagonal_shift,
        method=args.sr_method,
        prefactor=args.sr_prefactor,
        return_aux=True,
    )
    observables_dict = {
        "target_energy": build_measurement_function(local_energy_target, return_best=is_classical_target),
        "annealing_energy": build_measurement_function(local_energy_annealing),
    }
    if not args.vqa_no_catalyst:
        observables_dict["catalyst_energy"] = build_measurement_function(local_energy_catalyst)
    # endregion

    optimizer = optax.sgd(args.sgd_learning_rate, momentum=args.sgd_momentum)
    times = jnp.linspace(0.0, 1.0, args.vqa_num_annealing_steps)
    if not args.vqa_no_catalyst:
        schedule = jax.vmap(lambda t: jnp.array([t, 1 - t, t * (1 - t)]))(times)
    else:
        schedule = jax.vmap(lambda t: jnp.array([t, 1 - t]))(times)

    variational_annealer = VariationalAnnealer(
        variational_quantum_state=vqs,
        parametric_gradient_estimator=gradient_estimator,
        optimizer=optimizer,
        annealing_schedule=schedule,
        persistent_chains=not args.mcmc_disable_persistent_markov_chains,
        observables_dict=observables_dict,
        num_warmup_steps=args.vqa_num_warmup_steps,
        num_updates_per_step=args.vqa_num_updates_per_step,
        num_finetuning_steps=args.vqa_num_finetuning_steps,
        use_tqdm=args.tqdm,
    )

    data = variational_annealer.run(
        prngkey,
        max_runtime=args.max_runtime,
    )

    if data is None:
        # make a failed.txt file
        print_log("Failed")
        os.makedirs(args.save_path, exist_ok=True)
        with open(f"{args.save_path}/failed.txt", "w") as f:
            f.write("Failed")

    if data is not None:
        serialize_data(
            data,
            args.save_path,
            minimal_logging=(data["target_energy"][-1][0] > args.target_energy),
            classical_target=is_classical_target,
        )

    with open(f"{args.save_path}/finished.txt", "w") as f:
        f.write("Natural termination")


if __name__ == "__main__":
    main()

import os
os.environ["JAX_ENABLE_X64"] = "True"  # Use double precision

import json
import numpy as np
import jax
import jax.numpy as jnp
import optax

from utils.variational_quantum_states.boltzmann_quantum_states import DeepBoltzmannQuantumState
from utils.operators import (
    build_local_longitudinal_field_ising_energy,
    build_local_parametric_hamiltonian,
    build_measurement_function,
)
from utils.tdvp import build_parametric_gradient_estimator
from utils.variational_annealer import VariationalAnnealer


# -----------------------------
# Utilities: saving CSV files
# -----------------------------
def save_csv(path, array_2d):
    np.savetxt(path, np.asarray(array_2d), delimiter=",")


def save_vec_csv(path, array_1d):
    np.savetxt(path, np.asarray(array_1d).reshape(-1, 1), delimiter=",")


def s_tag(s: float) -> str:
    # "s000", "s050", "s085", ...
    return f"s{int(round(100 * s)):03d}"


# -----------------------------
# Main: generate and save data
# -----------------------------
def main():
    out_dir = "trajectory_pca_data"
    os.makedirs(out_dir, exist_ok=True)

    prngkey = jax.random.PRNGKey(0)

    num_spins = 4
    dbqs_layers = [2]

    # Random symmetric coupling matrix J
    prngkey, tempkey = jax.random.split(prngkey)
    J_matrix = jax.random.normal(tempkey, (num_spins, num_spins))
    J_matrix = (J_matrix + J_matrix.T) / 2
    J_matrix = J_matrix.at[jnp.diag_indices(num_spins)].set(0.0)
    J_matrix = J_matrix / (2 * np.sqrt(num_spins))

    # Save J_matrix (for reference / reproducibility)
    save_csv(os.path.join(out_dir, "J_matrix.csv"), np.array(J_matrix))

    # Brute force solution (optional, prints only)
    J_np = np.asarray(J_matrix, dtype=np.float64)
    current_best_energy = float("inf")
    current_best_config = None
    for idx in range(2**num_spins):
        bits = np.array(list(np.binary_repr(idx, width=num_spins)), dtype=np.int8)
        config = (2 * bits - 1).astype(np.float64)
        energy = float(config @ J_np @ config)
        if energy < current_best_energy:
            current_best_energy = energy
            current_best_config = config

    print("Brute force solution:")
    print("Ground state energy:", current_best_energy)
    print("Ground state configuration:", current_best_config)

    # Build variational quantum state
    prngkey, tempkey = jax.random.split(prngkey)
    vqs = DeepBoltzmannQuantumState(
        num_spins=num_spins,
        layers=dbqs_layers,
        prngkey=tempkey,
        num_samples=2**8,
        num_thermalization_steps=10,
        num_sweep_steps=3,
        num_chains=2**8,
        dtype=jnp.complex128,
        use_bias=True,
    )

    local_energy_z_target = build_local_longitudinal_field_ising_energy(J_matrix=J_matrix)
    _local_energy_sigma_x = vqs.local_energy_sigma_x
    _local_energy_sigma_y = vqs.local_energy_sigma_y

    local_energy_target = lambda params, config: local_energy_z_target(params, config)
    local_energy_annealing = lambda params, spins: _local_energy_sigma_x(params, spins)
    local_energy_catalyst = lambda params, spins: 0  # _local_energy_sigma_y(params, spins)

    local_hamiltonian = build_local_parametric_hamiltonian(
        local_target_hamiltonian=local_energy_target,
        local_annealing_hamiltonian=local_energy_annealing,
        local_catalyst_hamiltonian=local_energy_catalyst,
    )

    gradient_estimator = build_parametric_gradient_estimator(
        variational_quantum_state=vqs,
        local_hamiltonian=local_hamiltonian,
        diag_shift=1e-2,
    )

    optimizer = optax.sgd(0.1, momentum=0.5)

    times = jnp.linspace(0, 1.0, 1000)
    observables_dict = {
        "target_energy": build_measurement_function(local_energy_target, return_best=True),
        "annealing_energy": build_measurement_function(local_energy_annealing),
        "catalyst_energy": build_measurement_function(local_energy_catalyst),
    }
    schedule = jax.vmap(lambda t: jnp.array([t, 1 - t, t * (1 - t)]))(times)

    variational_annealer = VariationalAnnealer(
        variational_quantum_state=vqs,
        parametric_gradient_estimator=gradient_estimator,
        optimizer=optimizer,
        annealing_schedule=schedule,
        persistent_chains=True,
        observables_dict=observables_dict,
        num_warmup_steps=1,
        num_updates_per_step=1,
        num_finetuning_steps=1,
        use_tqdm=True,
        log_every=1,
        log_params=True,
    )

    # Run annealer
    data = variational_annealer.run(prngkey)
    data = {key: np.array(value) for key, value in data.items()}
    print("Final energy: ",data["target_energy"][-1])
    params_trajectory = data["params"]
    W_orig = np.asarray(params_trajectory)
    assert W_orig.ndim == 2, "params_trajectory must be [T,P]"
    T_full, P = W_orig.shape

    # Save params trajectory (real+imag if needed)
    if np.iscomplexobj(W_orig):
        W_real = np.concatenate([W_orig.real, W_orig.imag], axis=1)
        complex_flag = 1
        P_orig = P
    else:
        W_real = W_orig
        complex_flag = 0
        P_orig = P

    save_csv(os.path.join(out_dir, "params_trajectory_real.csv"), W_real)

    # PCA directions in parameter space
    w_final_real = np.mean(W_real, axis=0)
    centered_full = W_real - w_final_real

    _, _, Vt = np.linalg.svd(centered_full, full_matrices=False)
    d1 = Vt[0]
    d2 = Vt[1] if Vt.shape[0] > 1 else np.eye(Vt.shape[1])[
        (np.argmax(np.abs(Vt[0])) + 1) % Vt.shape[1]
    ]

    # Project trajectory to PCA plane
    alphas_full = (W_real - w_final_real) @ d1
    betas_full = (W_real - w_final_real) @ d2

    min_extent = 1e-1
    pad_frac = 0.5

    a_min, a_max = float(alphas_full.min()), float(alphas_full.max())
    b_min, b_max = float(betas_full.min()), float(betas_full.max())
    a_span = max(a_max - a_min, min_extent)
    b_span = max(b_max - b_min, min_extent)
    a_min_pad, a_max_pad = a_min - pad_frac * a_span, a_max + pad_frac * a_span
    b_min_pad, b_max_pad = b_min - pad_frac * b_span, b_max + pad_frac * b_span

    num_points = 25
    a_lin = np.linspace(a_min_pad, a_max_pad, num_points)
    b_lin = np.linspace(b_min_pad, b_max_pad, num_points)
    A, B = np.meshgrid(a_lin, b_lin)

    save_csv(os.path.join(out_dir, "A.csv"), A)
    save_csv(os.path.join(out_dir, "B.csv"), B)

    grid_plane = w_final_real[None, :] + A.reshape(-1, 1) * d1 + B.reshape(-1, 1) * d2

    # Loss function (JAX-jitted) evaluated at grid_plane points
    @jax.jit
    def loss(params, s):
        tempkey = jax.random.PRNGKey(0)
        samples, _ = vqs.generate_samples(tempkey, params)
        loss_value = (
            s * observables_dict["target_energy"](params, samples)[0]
            + (1 - s) * observables_dict["annealing_energy"](params, samples)[0]
            + s * (1 - s) * observables_dict["catalyst_energy"](params, samples)[0]
        )
        return (loss_value).real

    # Helper to map real-vector back to complex params if needed
    if complex_flag == 1:
        def to_original(r):
            r = np.asarray(r)
            real_part = r[:P_orig]
            imag_part = r[P_orig:]
            return real_part + 1j * imag_part
    else:
        def to_original(r):
            return np.asarray(r)

    def foo(s):
        s_clamped = float(np.clip(s, 0.0, 1.0))
        step_idx = int(round(s_clamped * (T_full - 1)))
        W_trunc_real = W_real[: step_idx + 1]

        losses = [loss(to_original(g), s_clamped) for g in grid_plane]
        loss_surface = np.array(losses).reshape(A.shape)

        alphas_trunc = (W_trunc_real - w_final_real) @ d1
        betas_trunc = (W_trunc_real - w_final_real) @ d2
        traj_xy = np.stack([alphas_trunc, betas_trunc], axis=1)
        return loss_surface, traj_xy

    anneals = [0, 0.5, 0.6, 0.7, 0.85, 1.0]

    surfaces = []
    trajs = []
    mins = []
    maxs = []

    for s in anneals:
        ls, tr = foo(s)
        surfaces.append(ls)
        trajs.append(tr)
        mins.append(np.nanmin(ls))
        maxs.append(np.nanmax(ls))

        # Save per-s surface and trajectory
        tag = s_tag(s)
        save_csv(os.path.join(out_dir, f"loss_surface_{tag}.csv"), ls)
        save_csv(os.path.join(out_dir, f"traj_{tag}.csv"), tr)

    # Compute shared levels (exactly like your original script)
    sorted_extremes = np.sort(np.array((mins, maxs)), axis=None)
    nums_points = [3, 3, 5, 5, 7, 7, 7, 5, 5, 3, 3]
    levels = np.concatenate(
        [
            np.linspace(sorted_extremes[i], sorted_extremes[i + 1], nums_points[i])[:-1]
            for i in range(len(sorted_extremes) - 1)
        ]
    )
    line_levels = levels

    # Save shared plotting arrays
    save_vec_csv(os.path.join(out_dir, "anneals.csv"), np.array(anneals))
    save_vec_csv(os.path.join(out_dir, "levels.csv"), levels)
    save_vec_csv(os.path.join(out_dir, "line_levels.csv"), line_levels)

    # Save a minimal metadata CSV (still "CSV-only" compliant)
    meta = np.array(
        [
            ["num_spins", str(num_spins)],
            ["T_full", str(T_full)],
            ["P_original", str(P_orig)],
            ["complex_flag", str(complex_flag)],
            ["num_points_grid", str(num_points)],
        ],
        dtype=object,
    )
    np.savetxt(os.path.join(out_dir, "meta.csv"), meta, delimiter=",", fmt="%s")

    print(f"\nSaved CSV data to: {out_dir}/")
    print("Files include: A.csv, B.csv, levels.csv, loss_surface_sXXX.csv, traj_sXXX.csv, ...")


if __name__ == "__main__":
    main()

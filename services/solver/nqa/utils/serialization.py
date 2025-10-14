import numpy as np
import os
import matplotlib.pyplot as plt


def serialize_data(data, save_path, minimal_logging=True, classical_target=False):
    """Persist run outputs, handling multi-replica tensors correctly.

    Expects shapes (T, C, R) for observable time series where:
    - T: logged iterations
    - C: components (e.g., 0=mean, 1=variance, 2=best_value, 3:3+N=best_config spins when classical_target)
    - R: replicas
    """
    data = {key: np.array(value) for key, value in data.items()}

    # Derive sizes and ensure output directory exists
    os.makedirs(save_path, exist_ok=True)
    num_spins = int(data.get("num_spins", 0))

    # Infer num_replicas from last axis of a known series
    try:
        num_replicas = int(data["avg_energy"][-1].shape[-1])
    except Exception:
        # Fallback to target_energy if avg_energy unavailable
        num_replicas = int(np.asarray(data["target_energy"][-1, 0]).shape[-1])

    # Compute per-replica best trajectories and configs for classical targets
    best_energy_so_far = None
    best_config = None
    if classical_target:
        # best value per time and replica sits at component index 2
        best_vals = np.asarray(data["target_energy"])[:, 2, :]  # shape (T, R)
        best_energy_so_far = np.minimum.accumulate(best_vals, axis=0)  # (T, R)

        # Best configs are logged as components [3:3+num_spins], shape (T, num_spins, R)
        if num_spins > 0:
            best_cfg_series = np.asarray(data["target_energy"])[:, 3 : 3 + num_spins, :]  # (T, N, R)
            best_indices = np.argmin(best_vals, axis=0)  # (R,)
            # Gather per-replica best config across time -> (N, R)
            best_config = np.stack(
                [best_cfg_series[idx, :, r] for r, idx in enumerate(best_indices)], axis=1
            )

        # Expose arrays at top-level to avoid pickling dicts and support master.py
        if best_energy_so_far is not None:
            data["best_energy_so_far"] = best_energy_so_far
        if best_config is not None:
            data["best_config"] = best_config

    # Human-readable summary
    with open(f"{save_path}/results.txt", "w") as f:
        final_target_mean = np.asarray(data["target_energy"])[-1, 0]  # (R,)
        final_target_var = np.asarray(data["target_energy"])[-1, 1]  # (R,)
        final_anneal_mean = np.asarray(data["annealing_energy"])[-1, 0]  # (R,)
        f.write(f"Final target energy: {final_target_mean}\n")
        f.write(f"Final target energy variance: {final_target_var}\n")
        f.write(f"Final annealing energy: {final_anneal_mean}\n")
        if classical_target and best_energy_so_far is not None:
            f.write(f"Best target energy: {best_energy_so_far[-1].tolist()}\n")
            if best_config is not None:
                f.write(f"Best config per replica (spins x replicas):\n {best_config.tolist()}\n")
        f.write(f"Runtime: {data['runtime']}\n")
        hours, remainder = divmod(float(data["runtime"]), 3600)
        minutes, seconds = divmod(remainder, 60)
        f.write(f"Total time: {hours:.0f} hours, {minutes:.0f} minutes, {seconds:.0f} seconds")

    # Persist NPZ with pure arrays (avoid pickled dicts)
    np.savez(f"{save_path}/data.npz", **data)
    if minimal_logging:
        return 0

    os.makedirs(f"{save_path}/plots", exist_ok=True)

    plt.figure()
    plt.title("Energies")
    # avg_energy is (T, R)
    plt.plot(np.asarray(data["avg_energy"]).real, label="Instant energy", c="black")
    # target_energy and annealing_energy are (T, C, R) -> select mean (0)
    plt.plot(np.asarray(data["target_energy"])[:, 0], label="Target energy", c="blue")
    plt.plot(np.asarray(data["annealing_energy"])[:, 0].real, label="Annealing energy", c="red")
    if classical_target and best_energy_so_far is not None:
        for r in range(num_replicas):
            plt.plot(
                best_energy_so_far[:, r],
                label=f"Best target energy (replica {r})",
                linestyle="dashed",
            )
    if "catalyst_energy" in data:
        plt.plot(np.asarray(data["catalyst_energy"])[:, 0].real, label="Catalyst energy", c="green")
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
    # plt.legend()
    plt.savefig(f"{save_path}/plots/energies.png")
    plt.close()

    # plt.figure()
    # plt.title("Magnetizations")
    # plt.plot(data["magnetizations"][:, : data["num_spins"]])
    # plt.xlabel("Epoch")
    # plt.ylabel("Magnetization")
    # plt.savefig(f"{save_path}/plots/magnetizations.png")
    # plt.close()

    # plt.figure()
    # plt.title("Couplings")
    # plt.plot(data["couplings"])
    # plt.xlabel("Epoch")
    # plt.ylabel("Coupling")
    # plt.savefig(f"{save_path}/plots/couplings.png")
    # plt.close()


import numpy as np
import os
import matplotlib.pyplot as plt


def serialize_data(data, save_path, minimal_logging=True, classical_target=False):
    data = {key: np.array(value) for key, value in data.items()}
    
    os.makedirs(save_path, exist_ok=True)
    if classical_target:
        best_energy_so_far_list = np.minimum.accumulate(data["target_energy"][:, 2])
        best_config_idx = np.argmin(data["target_energy"][:, 2])
        best_config = data["target_energy"][best_config_idx, 3:][: data["num_spins"]]
        data["best_energy_so_far"] = best_energy_so_far_list
        data["best_config"] = best_config
    
    with open(f"{save_path}/results.txt", "w") as f:
        f.write(f"Final energy: {data['inst_energy'][-1]}\n")
        f.write(f"Final target energy: {data['target_energy'][-1, 0]}\n")
        f.write(f"Final target energy variance: {data['target_energy'][-1, 1]}\n")
        f.write(f"Final annealing energy: {data['annealing_energy'][-1, 0]}\n")
        if classical_target:
            f.write(f"Best target energy: {best_energy_so_far_list[-1]}\n")
            f.write(f"Best config:\n {best_config}\n")
        f.write(f"Runtime: {data['runtime']}\n")
        hours, remainder = divmod(data["runtime"], 3600)
        minutes, seconds = divmod(remainder, 60)
        f.write(f"Total time: {hours:.0f} hours, {minutes:.0f} minutes, {seconds:.0f} seconds")
    
    np.savez(f"{save_path}/data.npz", **data)
    if minimal_logging:
        # minimal_data = {
        #     "target_energy": data["target_energy"],  
        #     "runtime": data["runtime"],
        #     "num_params": data["optimized_params"].size,
        #     "best_config": best_config,
        # }
        # if classical_target:
        #     minimal_data["best_energy_so_far"] = best_energy_so_far_list
        # np.savez(f"{save_path}/data.npz", **minimal_data)
        return 0
    
    os.makedirs(f"{save_path}/plots", exist_ok=True)

    plt.figure()
    plt.title("Energies")
    plt.plot(data["avg_energy"].real, label="Instant energy", c="black")
    plt.plot(data["target_energy"][:, 0], label="Target energy", c="blue")
    plt.plot(data["annealing_energy"][:, 0].real, label="Annealing energy", c="red")
    if classical_target:
        plt.plot(best_energy_so_far_list, label="Best target energy", c="blue", linestyle="--")
    if "catalyst_energy" in data:
        plt.plot(data["catalyst_energy"][:, 0].real, label="Catalyst energy", c="green")
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
    plt.legend()
    plt.savefig(f"{save_path}/plots/energies.png")
    plt.close()

    plt.figure()
    plt.title("Magnetizations")
    plt.plot(data["magnetizations"][:, : data["num_spins"]])
    plt.xlabel("Epoch")
    plt.ylabel("Magnetization")
    plt.savefig(f"{save_path}/plots/magnetizations.png")
    plt.close()

    plt.figure()
    plt.title("Couplings")
    plt.plot(data["couplings"])
    plt.xlabel("Epoch")
    plt.ylabel("Coupling")
    plt.savefig(f"{save_path}/plots/couplings.png")
    plt.close()



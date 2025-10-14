import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import time
import optuna
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.cluster import DBSCAN
import scipy.stats

CM = 1 / 2.54
PRX_SINGLE = 8.5 * CM
PRX_ONEHALF = 14.0 * CM
PRX_DOUBLE = 17.8 * CM

def prx_figsize(width="single", aspect=1.5):
    w = {"single": PRX_SINGLE, "1.5": PRX_ONEHALF, "double": PRX_DOUBLE}[width]
    return (w, w * aspect)

def build_spin_glass_hamiltonian(J, h_field=0.0, dtype=np.float64):
    """
    Builds a transverse-field Ising spin glass Hamiltonian:
        H = - sum_{i,j} J[i,j] σ^z_i σ^z_j  - sum_i h_i σ^x_i

    Args:
        J : (N,N) array — symmetric coupling matrix (J_ii ignored)
        h_field : scalar or array of length N (transverse field)
        dtype : float or complex (default float64)

    Returns:
        H : quspin Hamiltonian object (sparse)
        basis : spin_basis_1d object
    """
    J = np.asarray(J, dtype=np.float64)
    N = J.shape[0]
    assert J.shape == (N, N), "J must be an NxN matrix"

    # transverse field (can be scalar or per-site array)
    if np.isscalar(h_field):
        h = np.full(N, float(h_field))
    else:
        h = np.asarray(h_field, dtype=float)
        assert h.shape == (N,), "h_field must be scalar or length N"

    # Interaction terms: -J_ij σz_i σz_j
    zz_list = [[J[i, j], i, j] for i in range(N) for j in range( N) if J[i, j] != 0.0]

    # Transverse field terms: -h_i σx_i
    x_list = [[h[i], i] for i in range(N) if h[i] != 0.0]

    static = [["zz", zz_list], ["x", x_list]]
    dynamic = []

    basis = spin_basis_1d(N)  

    H = hamiltonian(static, dynamic, basis=basis, dtype=dtype)
    return H, basis

def first_few_energy_states(J, h_field, sign_J=+1, which="SA", return_state=True, **eigsh_kwargs):
    """
    Compute first five non-degenerate energy states using sparse Lanczos (ARPACK via QuSpin wrapper).

    Args:
      J, h_field, sign_J : passed to build_tf_ising_hamiltonian
      which : 'SA' for smallest algebraic (good for ground state)
      return_state : if True, returns (E0, psi0)
      eigsh_kwargs : passed to H.eigsh (e.g. maxiter, tol)  """
    H, basis = build_spin_glass_hamiltonian(J, h_field)
    E, v = H.eigsh(k=5, which=which, **eigsh_kwargs)
    #use only one energy value for degenerate energy states
    E0 = np.real(E)
    non_degenerate = np.unique(np.round(E0, decimals=1))
    if return_state:
        psi0 = v
        return non_degenerate, psi0
    else:
        return non_degenerate

DB_STORAGE_TFSK = "sqlite:///optuna_16_QSK.db"

data = {"TFSK": {}}

# Load TFSK data from Optuna studies
for i in range(10):
    data["TFSK"][f"Instance_{i}"] = {}
    study = optuna.load_study(study_name=f"DBM_SK16_instance_{i}", storage=DB_STORAGE_TFSK)
    completed_trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.user_attrs["Final Energy Var"] < 1e-2
        and t.user_attrs["Final Energy"] - study.best_value < 2
    ]
    data["TFSK"][f"Instance_{i}"]["min_energy"] = min(t.user_attrs["Final Energy"] for t in completed_trials)
    data["TFSK"][f"Instance_{i}"]["relative_final_energies"] = np.array([t.user_attrs["Final Energy"] for t in completed_trials])
    data["TFSK"][f"Instance_{i}"]["variances"] = np.array([t.user_attrs["Final Energy Var"] for t in completed_trials])
    data["TFSK"][f"Instance_{i}"]["inverse_num_params"] = np.array([1.0 / t.user_attrs["Num Params"] for t in completed_trials])
    data["TFSK"][f"Instance_{i}"]["bands"] = []
    data["TFSK"][f"Instance_{i}"]["bands_ip"] = []

# Prepare figure
plt.rcParams["figure.figsize"] = prx_figsize("double", aspect=1.0)
fig, axes = plt.subplots(10, 3, sharex="col", sharey="row", gridspec_kw={"width_ratios": [4, 4, 2]})

for i in range(10):
    ax1, ax2, ax3 = axes[i]

    energies = data["TFSK"][f"Instance_{i}"]["relative_final_energies"]
    variances = data["TFSK"][f"Instance_{i}"]["variances"]
    inv_params = data["TFSK"][f"Instance_{i}"]["inverse_num_params"]
    # DBSCAN clustering in energy axis
    X = np.vstack((np.zeros_like(energies), energies)).T
    db = DBSCAN(eps=0.01, min_samples=4).fit(X)
    raw_labels = db.labels_

    # Relabel clusters by mean energy
    cluster_means = {lbl: np.mean(energies[raw_labels == lbl]) for lbl in set(raw_labels) if lbl != -1}
    sorted_labels = {old: new for new, old in enumerate(sorted(cluster_means, key=cluster_means.get))}
    labels = np.array([sorted_labels.get(lbl, -1) for lbl in raw_labels])

    positive_slope_text_added = False

    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        v = variances[mask]
        e = energies[mask]

        # Variance-based fit
        mean_intercept = None
        std_intercept = None
        if len(v) > 1:
            coeffs = np.polyfit(v, e, 1)
            fit_fn = np.poly1d(coeffs)
            if label in [0, 1] and not positive_slope_text_added:
                ax1.text(0.01, 0.95, f"Instance {i}: slope={coeffs[0]:.3f}, intercept={coeffs[1]:.3f}",
                         transform=ax1.transAxes, fontsize=6, verticalalignment="top")
                positive_slope_text_added = True
            data["TFSK"][f"Instance_{i}"]["bands"].append((min(e), max(e), coeffs))

        # Split extrapolation test (variance)
        if len(v) > 4:
            sort_idx = np.argsort(v)
            v_sorted, e_sorted = v[sort_idx], e[sort_idx]
            n = len(v_sorted)
            splits = [(v_sorted[:n//2], e_sorted[:n//2]), (v_sorted[n//2:], e_sorted[n//2:])] if n % 2 == 0 else \
                     [(v_sorted[np.random.permutation(n)[:n//2]], e_sorted[np.random.permutation(n)[:n//2]]) for _ in range(20)]
            intercepts = [np.polyfit(v_s, e_s, 1)[1] for v_s, e_s in splits if len(v_s) > 1]
            if intercepts:
                mean_intercept = np.mean(intercepts)
                std_intercept = np.std(intercepts)
                print(f"Instance {i} Band {label}: mean intercept={mean_intercept:.6f}, std={std_intercept:.6f}")

        # Inverse-params fit
        p = inv_params[mask]
        mean_intercept_ip = None
        std_intercept_ip = None
        if len(p) > 1:
            coeffs_ip = np.polyfit(p, e, 1)
            data["TFSK"][f"Instance_{i}"]["bands_ip"].append((min(e), max(e), coeffs_ip))

        # Split extrapolation test (inverse params)
        if len(p) > 4:
            sort_idx = np.argsort(p)
            p_sorted, e_sorted = p[sort_idx], e[sort_idx]
            n = len(p_sorted)
            splits = [(p_sorted[:n//2], e_sorted[:n//2]), (p_sorted[n//2:], e_sorted[n//2:])] if n % 2 == 0 else \
                     [(p_sorted[np.random.permutation(n)[:n//2]], e_sorted[np.random.permutation(n)[:n//2]]) for _ in range(20)]
            intercepts_ip = [np.polyfit(p_s, e_s, 1)[1] for p_s, e_s in splits if len(p_s) > 1]
            if intercepts_ip:
                mean_intercept_ip = np.mean(intercepts_ip)
                std_intercept_ip = np.std(intercepts_ip)
                print(f"Instance {i} Inv params Band {label}: mean intercept={mean_intercept_ip:.6f}, std={std_intercept_ip:.6f}")

        # Check compatibility within 2 std
        if label == 0 and mean_intercept is not None and mean_intercept_ip is not None:
            compatible = abs(mean_intercept - mean_intercept_ip) < 2 * (std_intercept + std_intercept_ip)
            print(f"  -> Band {label} intercepts are {'compatible' if compatible else 'NOT compatible'} within 2 std")

    # Plot variance vs energy
    ax1.scatter(variances, energies, color="black", s=8, alpha=0.6)
    for e1, e2, coeffs in data["TFSK"][f"Instance_{i}"]["bands"]:
        x = np.linspace(-0.001, 0.07, 100)
        y = coeffs[0]*x + coeffs[1]
        ax1.plot(x, y, "--", color="black", linewidth=1.2)
        ax1.scatter([0], [coeffs[1]], color="black", marker="x", s=20, zorder=5)
    ax1.set_xlim(-1e-3, 0.01)
    ax1.set_ylim(
        data["TFSK"][f"Instance_{i}"]["min_energy"] - 3e-1,
        data["TFSK"][f"Instance_{i}"]["min_energy"] + 2.8)
    ax1.set_ylabel(r"$\langle H_T \rangle$", fontsize=10)

    # Plot inverse params vs energy
    ax2.scatter(inv_params, energies, color="black", s=8, alpha=0.6)
    for e1, e2, coeffs in data["TFSK"][f"Instance_{i}"]["bands_ip"]:
        x = np.linspace(-3e-7, 0.0001, 100)
        y = coeffs[0]*x + coeffs[1]
        ax2.plot(x, y, "--", color="black", linewidth=1.2)
        ax2.scatter([0], [coeffs[1]], color="black", marker="x", s=20, zorder=5)
    ax2.set_xlim(-3e-7, 0.0001)
    ax2.set_ylim(data["TFSK"][f"Instance_{i}"]["min_energy"] - 3e-1,
         data["TFSK"][f"Instance_{i}"]["min_energy"] + 2.8)
    ax2.set_xticks([0, 3e-5, 6e-5, 9e-5])
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0e}"))

    # Histogram of energies
    ax3.hist(energies, bins=51, orientation="horizontal", color="black", alpha=0.5, linewidth=2, edgecolor="black")

ax1.set_xlabel(r"$\sigma^2$", fontsize=10)
ax2.set_xlabel(r"$1/N_{\rm params}$")
ax3.set_xlabel("Counts")

#plot the ED energies as horizontal lines
for i in range(10):
    exact_energies = first_few_energy_states(J=np.load(f"Instance_16/Instance_{i}/J_matrix.npy"), h_field=0.1, return_state=False)
    for e in exact_energies:
        for ax in axes[i]:
            ax.axhline(e, color="red", linestyle=":", linewidth=1.0)
fig.tight_layout()
fig.savefig("tfsk_bands_all_16_sites.pdf")
fig.show()
plt.close(fig)
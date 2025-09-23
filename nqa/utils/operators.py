import jax
import jax.numpy as jnp
from utils.boltzmann_quantum_states import DeepBoltzmannQuantumState
from typing import (
    Union,
    Callable,
    Optional,
)

CNumber = Union[float, complex]


def build_local_tfsk_energy(
    deep_boltzmann_quantum_state: DeepBoltzmannQuantumState,
    J_matrix: jnp.ndarray,
    h_vector: Optional[jnp.ndarray] = None,
    g_vector: Optional[jnp.ndarray] = None,
    energy_shift: Optional[float] = None,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Constructs a local energy function for the Transverse Field Sherrington-Kirkpatrick (TFSK) model.

    Args:
        deep_boltzmann_quantum_state (DeepBoltzmannQuantumState): The variational quantum state.
        J_matrix (jnp.ndarray): Coupling matrix for Z-basis interactions.
        h_vector (Optional[jnp.ndarray]): External field vector for Z-basis. Defaults to zero.
        g_vector (Optional[jnp.ndarray]): External field vector for X-basis. Defaults to None.
        energy_shift (Optional[float]): Constant energy shift. Defaults to zero.

    Returns:
        Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]: Function to compute local energy given parameters and configurations.
    """
    if h_vector is None:
        h_vector = jnp.zeros(J_matrix.shape[0])
    if energy_shift is None:
        energy_shift = 0.0

    num_spins = J_matrix.shape[0]

    def local_z_energy(
        params: jnp.ndarray,
        config: jnp.ndarray,
    ) -> CNumber:
        config = config[:num_spins]
        return config.T @ J_matrix @ config + config.T @ h_vector + energy_shift

    if g_vector is not None:

        def local_x_energy(
            params,
            config,
        ):
            return jnp.sum(g_vector * deep_boltzmann_quantum_state.local_sigma_xs(params, config))

        def local_energy(
            params: jnp.ndarray,
            config: jnp.ndarray,
        ) -> CNumber:
            return local_z_energy(params, config) + local_x_energy(params, config)

    else:

        def local_energy(
            params: jnp.ndarray,
            config: jnp.ndarray,
        ) -> CNumber:
            return local_z_energy(params, config)

    return local_energy


def build_local_parametric_hamiltonian(
    local_target_hamiltonian: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    local_annealing_hamiltonian: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    local_catalyst_hamiltonian: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Constructs a local parametric Hamiltonian by combining target, annealing, and optional catalyst Hamiltonians.

    Args:
        local_target_hamiltonian (Callable): Local target Hamiltonian.
        local_annealing_hamiltonian (Callable): Local annealing Hamiltonian.
        local_catalyst_hamiltonian (Optional[Callable]): Local catalyst Hamiltonian. Defaults to None.

    Returns:
        Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]: Function to compute parametric Hamiltonian given parameters, configurations, and couplings.
    """

    def local_parametric_hamiltonian(
        params: jnp.ndarray,
        config: jnp.ndarray,
        couplings: jnp.ndarray,
    ) -> CNumber:
        """
        Computes the local parametric Hamiltonian.

        Args:
            params (jnp.ndarray): The parameters of the variational quantum state.
            config (jnp.ndarray): The configuration.
            couplings (jnp.ndarray): The couplings.

        Returns:
            CNumber: The local parametric Hamiltonian.
        """
        total_energy = couplings[0] * local_target_hamiltonian(params, config)
        total_energy += couplings[1] * local_annealing_hamiltonian(params, config)
        if local_catalyst_hamiltonian is not None:
            total_energy += couplings[2] * local_catalyst_hamiltonian(params, config)
        return total_energy

    return local_parametric_hamiltonian


def build_measurement_function(
    local_operator: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    return_best: bool = False
) -> Callable[[jnp.ndarray, jnp.ndarray], Union[tuple, jnp.ndarray]]:
    """
    Constructs a measurement function for a given local operator.

    Args:
        local_operator (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]): Local operator to measure.
        return_best (bool, optional): Whether to return the best value and configuration. Defaults to False.

    Returns:
        Callable[[jnp.ndarray, jnp.ndarray], Union[tuple, jnp.ndarray]]: Function to compute mean, variance, and optionally the best value and configuration of the operator.
    """
    vmapd_local_operator = jax.vmap(local_operator, in_axes=(None, 0))

    def measure_operator(params, samples):
        values = vmapd_local_operator(params, samples)
        if return_best:
            _best_value = jnp.min(values, axis=0)
            _best_config = samples[jnp.argmin(values)]
            return jnp.mean(values, axis=0), jnp.var(values, axis=0), _best_value, *_best_config
        else:
            return jnp.mean(values, axis=0), jnp.var(values, axis=0)

    return jax.jit(measure_operator)

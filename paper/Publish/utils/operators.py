import jax
import jax.numpy as jnp
from typing import (
    Tuple,
    Optional,
    Any,
)
from utils.global_defs import (
    CNumber,
    LocalOperator,
    VariationalParameters,
    ParametricLocalOperator,
)

def build_local_tfi_energy(
    J_matrix: jnp.ndarray,
    h_vector: Optional[jnp.ndarray] = None,
    g_vector: Optional[jnp.ndarray] = None,
    energy_shift: Optional[float] = None,
    variational_quantum_state=None,
) -> LocalOperator:
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
        if variational_quantum_state is None:
            raise ValueError("variational_quantum_state must be provided if g_vector is not None")

        def local_x_energy(
            params,
            config,
        ):
            return jnp.sum(g_vector * variational_quantum_state.local_sigma_xs(params, config))

        def local_ising_energy(
            params: jnp.ndarray,
            config: jnp.ndarray,
        ) -> CNumber:
            return local_z_energy(params, config) + local_x_energy(params, config)

    else:

        def local_ising_energy(
            params: jnp.ndarray,
            config: jnp.ndarray,
        ) -> CNumber:
            return local_z_energy(params, config)

    return local_ising_energy

def build_qubo_energy(
    Q_matrix: jnp.ndarray,
) -> LocalOperator:
    num_spins = Q_matrix.shape[0]

    def local_qubo_energy(
        params: jnp.ndarray,
        config: jnp.ndarray,
    ) -> CNumber:
        config = (config[:num_spins] + 1)/2
        return config.T @ Q_matrix @ config

    return local_qubo_energy

def build_local_longitudinal_field_ising_energy(
    J_matrix: jnp.ndarray,
    h_vector: Optional[jnp.ndarray] = None,
    energy_shift: Optional[float] = None,
) -> LocalOperator:
    if h_vector is None:
        h_vector = jnp.zeros(J_matrix.shape[0])
    if energy_shift is None:
        energy_shift = 0.0
    
    num_spins = J_matrix.shape[0]

    def local_ising_energy(
        params: jnp.ndarray,
        config: jnp.ndarray,
    ) -> CNumber:
        config = config[:num_spins]
        return config.T @ J_matrix @ config + config.T @ h_vector + energy_shift

    return local_ising_energy

def build_local_parametric_hamiltonian(
    local_target_hamiltonian: LocalOperator,
    local_annealing_hamiltonian: LocalOperator,
    local_catalyst_hamiltonian: Optional[LocalOperator] = None,
) -> ParametricLocalOperator:
    """
    Builds the local parametric Hamiltonian given the local target, annealing and (optional) catalyst Hamiltonians.

    Args:
        local_target_hamiltonian (LocalOperator): The local target Hamiltonian.
        local_annealing_hamiltonian (LocalOperator): The local annealing Hamiltonian.
        local_catalyst_hamiltonian (Optional[LocalOperator], optional): The local catalyst Hamiltonian. Defaults to None.
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


def build_measurement_function(local_operator, return_best=False):
    """
    Builds a measurement function for the given local operator.
    That is a function that given the variational parameters and a set of samples
    computes the mean and variance of the local operator.
    If return_best is True, it also returns the best (lowest) value of the local 
    operator and the corresponding configuration.

    Args:
        local_operator (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]): The local operator.
        return_best (bool, optional): Whether to return the best value and configuration. Defaults to False.

    Returns:
        Callable[[jnp.ndarray, jnp.ndarray], Any]: The measurement function
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
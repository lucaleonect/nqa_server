import jax
import jax.numpy as jnp
from typing import Callable, Optional, Union
from utils.boltzmann_quantum_states import DeepBoltzmannQuantumState

VALID_TDVP_METHODS = ["auto", "SR", "minSR"]
CNumber = Union[float, complex]


def outer_product(
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """
    Computes the outer product of two given vectors (1d arrays), that is the matrix
    $$O_{i,j}= x_i \bar y_j$$

    Args:
        x (jnp.ndarray): First vector
        y (jnp.ndarary): Second vector

    Returns:
        jnp.ndarray: Outer product of x and y
    """
    return jax.vmap(jax.vmap(lambda x, y: jnp.conjugate(y) * x, in_axes=(0, None)), in_axes=(None, 0))(x, y)


def self_outer_product(
    x: jnp.ndarray,
) -> jnp.ndarray:
    """
    Computes the outer product of the given vector (1d array) with itself, that is the matrix
    $$O_{i,j}= x_i \bar x_j$$

    Args:
        x (jnp.ndarray): Vector to compute the self outer product of

    Returns:
        jnp.ndarray: self outer product of the given vector
    """
    return outer_product(x, x)


def vmapd_self_outerproduct(
    x: jnp.ndarray,
) -> jnp.ndarray:
    """
    Computes the outer products of the all given vectors with themselves, that is the tensor
    $$O_{n,i,j}= x_{n,i} \bar x_{n,j}$$

    Args:
        x (jnp.ndarray): Matrix of vectors to compute the self outer products of

    Returns:
        jnp.ndarray: self outer products of the given vectors arranged as a 3-tensor
    """
    return jax.vmap(self_outer_product)(x)


def multiply_by_scalar(
    a: CNumber,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """
    Computes the product of the vector x by the scalar a

    Args:
        a (CNumber): scalar to multiply
        x (jnp.ndarray): vector to multiply

    Returns:
        jnp.ndarray: product of the vector x by the scalar a

    Note:
        x must be a 1d array for this to make sense
    """
    return jax.vmap(lambda a, b: a * b, in_axes=(None, 0))(a, x)


def vmapd_multiply_by_scalar(
    a: jnp.ndarray,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """
    Vectorized version of multiply_by_scalar that takes as input a list of scalars a (1d array) and
    a list of vectors (2d array) of shapes (N,) and (N, ...) and returs the list of vectors multiplied by
    the corresponding scalar in the list a, that is the matrix
    [a[i]*x[i] for i in range(len(a))]

    Args:
        a (jnp.ndarray): list of scalars (1d array)
        x (jnp.ndarray): list of vectors (2d array)

    Returns:
        jnp.ndarray: list of vectors multiplied by the corresponding scalar [a[i]*x[i] for i in range(len(a))]
    """
    return jax.vmap(multiply_by_scalar, in_axes=(0, 0))(a, x)


def build_parametric_gradient_estimator(
    deep_boltzmann_quantum_state: DeepBoltzmannQuantumState,
    local_hamiltonian: Callable[[jnp.ndarray, jnp.ndarray], CNumber],
    prefactor: Optional[CNumber] = None,
    p_inv_rcond: Optional[float] = None,
    diag_shift: Optional[float] = None,
    method: Optional[str] = None,
    return_aux: Optional[bool] = None,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Builds a function that estimates the gradient of the energy expectation value with respect to the variational
    parameters of the given variational quantum state using the Time-Dependent Variational Principle

    Args:
        deep_boltzmann_quantum_state (DeepBoltzmannQuantumState): Variational quantum state describing the system
        local_hamiltonian (Callable[[jnp.ndarray, jnp.ndarray, CNumber], CNumber]): Local Hamiltonian operator
            Expected Callable with sinature (params, config, couplings)->local_energy
        prefactor (CNumber, optional): Prefactor for the gradient estimator. Defaults to 1.0.
        p_inv_rcond (float, optional): Pseudo-inverse rcond parameter for the matrix inversion. Defaults to 1e-10.
        diag_shift (float, optional): Diagonal shift for the matrix inversion. Defaults to 1e-3.
        method (str, optional): Method to use to compute the natural gradients. Defaults to "auto" that will choose the least expensive method.
        return_aux (bool, optional): Whether to return also the energy and energy variance. Defaults to False.

    Returns:
        Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]: Function that estimates the gradient of the energy expectation value
            with respect to the variational parameters of the given variational quantum state using the Time-Dependent Variational Principle
            (params, mc_samples, couplings)->(grad, avg_energy, energy_var)
    """
    if prefactor is None:
        prefactor = 1.0
    if p_inv_rcond is None:
        p_inv_rcond = 1e-10
    if diag_shift is None:
        diag_shift = 1e-2
    if method is None:
        method = "auto"
    if return_aux is None:
        return_aux = False

    logpsi = deep_boltzmann_quantum_state.logpsi
    is_holomorphic = deep_boltzmann_quantum_state.is_holomorphic
    dtype = deep_boltzmann_quantum_state.dtype

    _num_samples = deep_boltzmann_quantum_state.num_samples
    _num_params = deep_boltzmann_quantum_state.params.size

    vmapd_grad_logpsi = jax.vmap(jax.grad(logpsi, argnums=0, holomorphic=is_holomorphic), in_axes=(None, 0))
    vmapd_local_hamiltonian = jax.vmap(local_hamiltonian, in_axes=(None, 0, None))

    @jax.jit
    def minsr_estimate_gradients(
        params: jnp.ndarray,
        sample: jnp.ndarray,
        couplings: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes the gradients using the minSR method

        Args:
            params (jnp.ndarray): Parameters of the variational quantum state
            sample (jnp.ndarray): Monte carlo samples to estimate the gradients
            couplings (jnp.ndarray): couplings constant of the local Hamiltonian

        Returns:
            jnp.ndarray: Gradients of the energy expectation value with respect to the variational parameters.
        """
        d_logpsi = vmapd_grad_logpsi(params, sample).astype(dtype)
        d_logpsi = d_logpsi - jnp.average(d_logpsi, axis=0, keepdims=True)
        d_logpsi_dag = jnp.conjugate(d_logpsi).T
        local_energies = vmapd_local_hamiltonian(params, sample, couplings)
        avg_energy = jnp.average(local_energies)
        energy_var = jnp.var(local_energies)
        local_energies = local_energies - jnp.average(local_energies)
        T_matrix = d_logpsi @ d_logpsi_dag
        T_matrix = T_matrix + diag_shift * jnp.eye(T_matrix.shape[0], dtype=dtype)
        gradients = prefactor * d_logpsi_dag @ jnp.linalg.pinv(T_matrix, rtol=p_inv_rcond) @ local_energies

        if return_aux:
            return (
                gradients,
                avg_energy,
                energy_var,
            )
        else:
            return gradients

    @jax.jit
    def sr_estimate_gradients(
        params: jnp.ndarray,
        sample: jnp.ndarray,
        couplings: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes the gradients using the SR method

        Args:
            params (jnp.ndarray): Parameters of the variational quantum state
            sample (jnp.ndarray): Monte carlo samples to estimate the gradients
            couplings (jnp.ndarray): couplings constant of the local Hamiltonian

        Returns:
            jnp.ndarray: Gradients of the energy expectation value with respect to the variational parameters.
        """
        d_logpsi = vmapd_grad_logpsi(params, sample).astype(dtype)
        local_energies = vmapd_local_hamiltonian(params, sample, couplings)
        avg_energy = jnp.average(local_energies)
        energy_var = jnp.var(local_energies)

        S_matrix = jnp.average(vmapd_self_outerproduct(d_logpsi), axis=0) - self_outer_product(
            jnp.average(d_logpsi, axis=0)
        )
        S_matrix = S_matrix + diag_shift * jnp.eye(S_matrix.shape[0], dtype=dtype)
        inv_S_matrix = jnp.linalg.pinv(S_matrix, rcond=p_inv_rcond)
        Forces_array = jnp.average(
            (vmapd_multiply_by_scalar(local_energies, jnp.conjugate(d_logpsi))), axis=0
        ) - multiply_by_scalar(jnp.average(local_energies), jnp.average(jnp.conjugate(d_logpsi), axis=0))
        gradients = prefactor * jnp.matmul(inv_S_matrix, Forces_array)

        if return_aux:
            return (
                gradients,
                avg_energy,
                energy_var,
            )
        else:
            return gradients

    if method == "SR":
        return sr_estimate_gradients
    elif method == "minSR":
        return minsr_estimate_gradients
    elif method == "auto":
        if _num_samples <= _num_params:
            return minsr_estimate_gradients
        else:
            return sr_estimate_gradients

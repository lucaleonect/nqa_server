import jax
import jax.numpy as jnp
import numpy as np

from utils.boltzmann_quantum_states import DeepBoltzmannQuantumState
from utils.operators import build_local_tfsk_energy, build_local_parametric_hamiltonian
from utils.tdvp import build_parametric_gradient_estimator


def make_symmetric_J(key, n):
    m = jax.random.normal(key, (n, n))
    m = (m + m.T) / 2.0
    m = m.at[jnp.diag_indices(n)].set(0.0)
    return m


def random_configs(key, num_samples, num_units_list):
    keys = jax.random.split(key, len(num_units_list))
    layers = [2 * jax.random.bernoulli(k, p=0.5, shape=(num_samples, n)) - 1 for k, n in zip(keys, num_units_list)]
    return jnp.concatenate(layers, axis=1).astype(jnp.int64)


def setup_small_system(
    num_samples_for_build=128,
):
    prng = jax.random.PRNGKey(42)
    key_J, key_h, key_g, key_params = jax.random.split(prng, 4)

    n = 4
    J = make_symmetric_J(key_J, n)
    h = jax.random.normal(key_h, (n,)) * 0.2
    g = jax.random.normal(key_g, (n,)) * 0.3

    # Use a single hidden layer with n units for a compact model
    dbqs = DeepBoltzmannQuantumState(
        num_spins=n,
        hidden_layers=[n],
        prngkey=key_params,
        num_samples=num_samples_for_build,
        num_thermalization_steps=2,
        num_sweep_steps=2,
        num_chains=8,
        use_bias=True,
        initial_params_gain=1e-1,
    )

    # Local target: ZZ + Z field; Local annealing: X-field using g
    local_target = build_local_tfsk_energy(dbqs, J, h, g_vector=None)
    local_annealing = build_local_tfsk_energy(
        dbqs,
        jnp.zeros_like(J),
        jnp.zeros(J.shape[0]),
        g_vector=g,
    )
    local_param_h = build_local_parametric_hamiltonian(local_target, local_annealing)

    return dbqs, local_param_h, J, h, g


def test_tdvp_grad_shapes_and_types():
    dbqs, local_param_h, *_ = setup_small_system(num_samples_for_build=64)

    # Generate independent random configurations (fast and deterministic)
    key = jax.random.PRNGKey(0)
    num_samples = 256
    samples = random_configs(key, num_samples, dbqs.num_units_list)
    couplings = jnp.array([1.0, 1.0])

    # Build estimators
    sr_est = build_parametric_gradient_estimator(
        dbqs,
        local_param_h,
        p_inv_rcond=1e-10,
        diag_shift=1e-3,
        return_aux=True,
    )

    grad_sr, e_sr, v_sr = sr_est(dbqs.params, samples, couplings)

    assert grad_sr.shape == dbqs.params.shape
    assert grad_sr.dtype == dbqs.params.dtype

    # Energy stats should be finite scalars
    for x in [e_sr, v_sr]:
        assert jnp.ndim(x) == 0
        assert jnp.isfinite(x)


def test_tdvp_zero_couplings_gives_zero_grad():
    dbqs, local_param_h, *_ = setup_small_system(num_samples_for_build=64)

    key = jax.random.PRNGKey(3)
    samples = random_configs(key, 128, dbqs.num_units_list)
    couplings = jnp.array([0.0, 0.0])

    est = build_parametric_gradient_estimator(
        dbqs,
        local_param_h,
        diag_shift=1e-3,
        return_aux=False,
    )
    g = est(dbqs.params, samples, couplings)
    np.testing.assert_allclose(np.array(g), np.zeros_like(np.array(g)), rtol=0, atol=1e-10)


def test_tdvp_constant_energy_yields_zero_gradient():
    # With zero params and annealing-only Hamiltonian with constant local X energy,
    # local energies are constant across samples -> gradient should be zero.
    prng = jax.random.PRNGKey(5)
    n = 4
    g = jnp.array([0.3, -0.2, 0.1, 0.4])

    dbqs = DeepBoltzmannQuantumState(
        num_spins=n,
        hidden_layers=[n],
        prngkey=prng,
        num_samples=64,
        num_thermalization_steps=1,
        num_sweep_steps=1,
        num_chains=8,
        use_bias=True,
        initial_params_gain=1e-1,
    )

    # Zero out parameters -> visible biases and first-layer weights are zero
    zero_params = jnp.zeros_like(dbqs.params)

    # Define annealing-only Hamiltonian: X field with g; J,h=0
    local_target = build_local_tfsk_energy(dbqs, jnp.zeros((n, n)), jnp.zeros((n,)), g_vector=None)
    local_annealing = build_local_tfsk_energy(dbqs, jnp.zeros((n, n)), jnp.zeros((n,)), g_vector=g)
    local_param_h = build_local_parametric_hamiltonian(local_target, local_annealing)

    # Random sample set
    key_s = jax.random.PRNGKey(6)
    samples = random_configs(key_s, 256, dbqs.num_units_list)
    couplings = jnp.array([0.0, 1.0])

    est = build_parametric_gradient_estimator(
        dbqs,
        local_param_h,
        diag_shift=1e-3,
        return_aux=False,
    )
    g = est(zero_params, samples, couplings)
    # Expect exact zeros within numerical tolerance
    np.testing.assert_allclose(np.array(g), np.zeros_like(np.array(g)), rtol=0, atol=1e-10)

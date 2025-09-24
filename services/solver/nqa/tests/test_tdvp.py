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
    dtype=jnp.complex128,
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
        dtype=dtype,
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
        prefactor=1.0,
        p_inv_rcond=1e-10,
        diag_shift=1e-3,
        method="SR",
        return_aux=True,
    )
    minsr_est = build_parametric_gradient_estimator(
        dbqs,
        local_param_h,
        prefactor=1.0,
        p_inv_rcond=1e-10,
        diag_shift=1e-3,
        method="minSR",
        return_aux=True,
    )

    grad_sr, e_sr, v_sr = sr_est(dbqs.params, samples, couplings)
    grad_minsr, e_ms, v_ms = minsr_est(dbqs.params, samples, couplings)

    assert grad_sr.shape == dbqs.params.shape
    assert grad_minsr.shape == dbqs.params.shape
    assert grad_sr.dtype == dbqs.dtype
    assert grad_minsr.dtype == dbqs.dtype

    # Energy stats should be finite scalars
    for x in [e_sr, v_sr, e_ms, v_ms]:
        assert jnp.ndim(x) == 0
        assert jnp.isfinite(x)


def test_tdvp_sr_minsr_agree_reasonably():
    dbqs, local_param_h, *_ = setup_small_system(num_samples_for_build=256)

    key = jax.random.PRNGKey(1)
    num_samples = 1024
    samples = random_configs(key, num_samples, dbqs.num_units_list)
    couplings = jnp.array([1.0, 1.0])

    sr_est = build_parametric_gradient_estimator(
        dbqs,
        local_param_h,
        prefactor=1.0,
        p_inv_rcond=1e-12,
        diag_shift=1e-3,
        method="SR",
        return_aux=False,
    )
    minsr_est = build_parametric_gradient_estimator(
        dbqs,
        local_param_h,
        prefactor=1.0,
        p_inv_rcond=1e-12,
        diag_shift=1e-3,
        method="minSR",
        return_aux=False,
    )

    g_sr = sr_est(dbqs.params, samples, couplings)
    g_ms = minsr_est(dbqs.params, samples, couplings)

    # They should be close; allow moderate tolerance due to sampling noise
    np.testing.assert_allclose(np.array(g_sr), np.array(g_ms), rtol=5e-2, atol=5e-3)


def test_tdvp_auto_selects_expected_method():
    # Case 1: num_samples <= num_params -> auto should select minSR
    dbqs_small, local_param_h_small, *_ = setup_small_system(num_samples_for_build=16)
    key = jax.random.PRNGKey(2)
    samples = random_configs(key, 128, dbqs_small.num_units_list)
    couplings = jnp.array([0.7, 1.3])

    auto_est_small = build_parametric_gradient_estimator(
        dbqs_small,
        local_param_h_small,
        method="auto",
        diag_shift=1e-3,
        return_aux=False,
    )
    minsr_est_small = build_parametric_gradient_estimator(
        dbqs_small,
        local_param_h_small,
        method="minSR",
        diag_shift=1e-3,
        return_aux=False,
    )

    g_auto_small = auto_est_small(dbqs_small.params, samples, couplings)
    g_minsr_small = minsr_est_small(dbqs_small.params, samples, couplings)
    np.testing.assert_allclose(np.array(g_auto_small), np.array(g_minsr_small), rtol=1e-6, atol=1e-8)

    # Case 2: num_samples > num_params -> auto should select SR
    # Build a model with few params (e.g., hidden_layers=[2]) and a big build-sample
    prng = jax.random.PRNGKey(123)
    n = 4
    J = make_symmetric_J(prng, n)
    h = jnp.zeros((n,))
    g = jnp.ones((n,)) * 0.1

    dbqs_largeS = DeepBoltzmannQuantumState(
        num_spins=n,
        hidden_layers=[2],
        prngkey=prng,
        num_samples=4096,  # ensure > num_params
        num_thermalization_steps=1,
        num_sweep_steps=1,
        num_chains=32,
        dtype=jnp.complex128,
        use_bias=True,
        initial_params_gain=1e-1,
    )
    local_target = build_local_tfsk_energy(dbqs_largeS, J, h, g_vector=None)
    local_annealing = build_local_tfsk_energy(
        dbqs_largeS, jnp.zeros_like(J), jnp.zeros(n), g_vector=g
    )
    local_param_h2 = build_local_parametric_hamiltonian(local_target, local_annealing)

    auto_est_big = build_parametric_gradient_estimator(
        dbqs_largeS,
        local_param_h2,
        method="auto",
        diag_shift=1e-3,
        return_aux=False,
    )
    sr_est_big = build_parametric_gradient_estimator(
        dbqs_largeS,
        local_param_h2,
        method="SR",
        diag_shift=1e-3,
        return_aux=False,
    )

    key2 = jax.random.PRNGKey(21)
    samples2 = random_configs(key2, 256, dbqs_largeS.num_units_list)
    couplings2 = jnp.array([1.0, 0.4])

    g_auto_big = auto_est_big(dbqs_largeS.params, samples2, couplings2)
    g_sr_big = sr_est_big(dbqs_largeS.params, samples2, couplings2)
    np.testing.assert_allclose(np.array(g_auto_big), np.array(g_sr_big), rtol=1e-6, atol=1e-8)


def test_tdvp_zero_couplings_gives_zero_grad():
    dbqs, local_param_h, *_ = setup_small_system(num_samples_for_build=64)

    key = jax.random.PRNGKey(3)
    samples = random_configs(key, 128, dbqs.num_units_list)
    couplings = jnp.array([0.0, 0.0])

    for method in ["SR", "minSR"]:
        est = build_parametric_gradient_estimator(
            dbqs,
            local_param_h,
            method=method,
            diag_shift=1e-3,
            return_aux=False,
        )
        g = est(dbqs.params, samples, couplings)
        np.testing.assert_allclose(np.array(g), np.zeros_like(np.array(g)), rtol=0, atol=1e-10)


def test_tdvp_prefactor_scales_gradient_linearly():
    dbqs, local_param_h, *_ = setup_small_system(num_samples_for_build=128)
    key = jax.random.PRNGKey(4)
    samples = random_configs(key, 256, dbqs.num_units_list)
    couplings = jnp.array([0.8, 0.3])

    base = build_parametric_gradient_estimator(
        dbqs, local_param_h, method="SR", prefactor=1.0, diag_shift=1e-3, return_aux=False
    )
    scaled = build_parametric_gradient_estimator(
        dbqs, local_param_h, method="SR", prefactor=2.5, diag_shift=1e-3, return_aux=False
    )

    g_base = base(dbqs.params, samples, couplings)
    g_scaled = scaled(dbqs.params, samples, couplings)
    np.testing.assert_allclose(np.array(g_scaled), 2.5 * np.array(g_base), rtol=1e-6, atol=1e-8)


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
        dtype=jnp.complex128,
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

    for method in ["SR", "minSR"]:
        est = build_parametric_gradient_estimator(
            dbqs, local_param_h, method=method, diag_shift=1e-3, return_aux=False
        )
        g = est(zero_params, samples, couplings)
        # Expect exact zeros within numerical tolerance
        np.testing.assert_allclose(np.array(g), np.zeros_like(np.array(g)), rtol=0, atol=1e-10)

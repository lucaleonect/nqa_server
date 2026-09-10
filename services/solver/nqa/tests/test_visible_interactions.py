"""Exact small-system checks for the Hubbard--Stratonovich DBQS extension."""
import itertools
import os

os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from numpy.polynomial.hermite import hermgauss

from utils.boltzmann_quantum_states import DeepBoltzmannQuantumState
from utils.operators import build_local_tfsk_energy
from utils.tdvp import build_parametric_gradient_estimator
from utils.variational_annealer import VariationalAnnealer


def make_model(hidden=(1, 1), use_bias=True, dtype=jnp.complex128, rank=2, **kwargs):
    settings = dict(num_samples=32, num_chains=4, num_thermalization_steps=4, num_sweep_steps=2)
    settings.update(kwargs)
    return DeepBoltzmannQuantumState(
        3, list(hidden), jax.random.PRNGKey(12), dtype=dtype, use_bias=use_bias,
        visible_interactions=True, visible_rank=rank, **settings,
    )


def spin_configs(n):
    return np.array(list(itertools.product([1, -1], repeat=n)), dtype=np.int64)


def fixed_params(model):
    """Nonzero biases/phases, nonsymmetric K, and signed rows of J."""
    tree = model.unravel_params(model.params)
    tree["J"] = jnp.asarray([[0.26, -0.10], [-0.20, 0.17], [0.11, 0.22]])[:, :model.visible_rank]
    if "K" in tree:
        tree["K"] = jnp.asarray([[0.2, 0.07, -0.09], [-0.01, -0.1, 0.04], [0.03, 0.02, 0.3]])
    rng = np.random.default_rng(4)
    for part in ("real", "imag"):
        if part in tree:
            tree[part] = jax.tree.map(
                lambda leaf: jnp.asarray(rng.normal(0, 0.12, leaf.shape), dtype=leaf.dtype),
                tree[part],
            )
    return jax.flatten_util.ravel_pytree(tree)[0]


def reference_logs(model, params, configs):
    """Independent NumPy expression from the supplied sampling notes."""
    biases, weights, factor, phase = model.unpack_params(params)
    layers = np.split(configs, np.cumsum(model.num_units_list)[:-1], axis=1)
    values = np.zeros(len(configs), dtype=np.complex128)
    for x, b in zip(layers, biases):
        values += x @ np.asarray(b)
    for x, w, y in zip(layers[:-1], weights, layers[1:]):
        values += np.einsum("bi,ij,bj->b", x, np.asarray(w), y)
    C = np.asarray(factor) @ np.asarray(factor).T
    if phase is not None:
        C = C + 1j * np.asarray(phase)
    values += np.einsum("bi,ij,bj->b", layers[0], C, layers[0])
    return values


def hamiltonian(model):
    """Dense visible Hamiltonian including ZZ, Z, X and Y terms."""
    coupling = np.array([[0.0, 0.13, -0.17], [0.13, 0.0, 0.21], [-0.17, 0.21, 0.0]])
    h, g, y = np.array([0.1, -0.07, 0.05]), np.array([-0.6, -0.4, -0.7]), np.array([0.2, -0.1, 0.15])
    visible = spin_configs(model.num_spins)
    H = np.diag(np.einsum("bi,ij,bj->b", visible, coupling, visible) + visible @ h + 0.3).astype(complex)
    for row, x in enumerate(visible):
        for i in range(model.num_spins):
            H[row, row ^ (1 << (model.num_spins - 1 - i))] += g[i] - 1j * x[i] * y[i]
    local_xz = build_local_tfsk_energy(model, jnp.asarray(coupling), jnp.asarray(h), jnp.asarray(g), 0.3)

    def local(params, config):
        return local_xz(params, config) + jnp.dot(jnp.asarray(y), model.local_sigma_ys(params, config))

    return H, local


@pytest.mark.parametrize("hidden,use_bias,dtype", [
    ((), True, jnp.complex128), ((2,), False, jnp.complex128),
    ((1, 1), True, jnp.complex128), ((1, 1, 1), False, jnp.float64),
])
def test_amplitudes_and_all_visible_flips(hidden, use_bias, dtype):
    model = make_model(hidden, use_bias, dtype)
    params = fixed_params(model)
    configs = spin_configs(model.num_units)
    expected = reference_logs(model, params, configs)
    actual = jax.vmap(model.logpsi, in_axes=(None, 0))(params, jnp.asarray(configs))
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    ratios = []
    for i in range(model.num_spins):
        flipped = configs.copy()
        flipped[:, i] *= -1
        ratios.append(np.exp(reference_logs(model, params, flipped) - expected))
    ratios = np.stack(ratios, axis=1)
    np.testing.assert_allclose(
        jax.vmap(model.local_sigma_xs, in_axes=(None, 0))(params, jnp.asarray(configs)), ratios, atol=1e-12,
    )
    np.testing.assert_allclose(
        jax.vmap(model.local_sigma_ys, in_axes=(None, 0))(params, jnp.asarray(configs)),
        -1j * configs[:, :3] * ratios, atol=1e-12,
    )
    # Multi-spin flips must include interactions among flipped sites.
    flipped = configs.copy()
    flipped[:, :2] *= -1
    np.testing.assert_allclose(
        jax.vmap(model.psi_ratio_fn, in_axes=(None, 0, 0))(params, jnp.asarray(flipped), jnp.asarray(configs)),
        np.exp(reference_logs(model, params, flipped) - expected), atol=1e-12,
    )


@pytest.mark.parametrize("hidden", [(), (1,), (1, 1)])
def test_dense_density_matrix_energy(hidden):
    model = make_model(hidden)
    params = fixed_params(model)
    configs = spin_configs(model.num_units)
    amplitudes = np.exp(reference_logs(model, params, configs)).reshape(2**3, -1)
    H, local = hamiltonian(model)
    rho = amplitudes @ amplitudes.conj().T
    rho /= np.trace(rho)
    probabilities = np.abs(amplitudes.ravel())**2
    probabilities /= probabilities.sum()
    locals_ = jax.vmap(local, in_axes=(None, 0))(params, jnp.asarray(configs))
    np.testing.assert_allclose(locals_, (H @ amplitudes / amplitudes).ravel(), atol=1e-12)
    np.testing.assert_allclose(probabilities @ locals_, np.trace(rho @ H), atol=1e-12)


def test_gaussian_conditionals_and_exact_marginalization():
    model = make_model()
    params = fixed_params(model)
    biases, weights, J, _ = model.unpack_params(params)
    units = [jnp.asarray([1, -1, 1]), jnp.asarray([-1]), jnp.asarray([1])]
    z = jnp.asarray([0.7, -1.2])
    evens = model.prob_evens_given_odds(biases, weights, units[1::2], J, z)
    # Compute conditional odds by enumerating the augmented density, without
    # calling logpsi (whose Gaussian fields have already been integrated out).
    def joint_log(layers):
        v = sum(np.asarray(x) @ np.asarray(b).real for x, b in zip(layers, biases))
        v += sum(np.asarray(x) @ np.asarray(w).real @ np.asarray(y)
                 for x, w, y in zip(layers[:-1], weights, layers[1:]))
        return 2*v + np.asarray(layers[0]) @ np.asarray(J) @ z - z @ z / 8

    for layer in range(0, len(units), 2):
        for i in range(len(units[layer])):
            plus, minus = list(units), list(units)
            plus[layer] = plus[layer].at[i].set(1)
            minus[layer] = minus[layer].at[i].set(-1)
            expected = 1 / (1 + np.exp(joint_log(minus) - joint_log(plus)))
            np.testing.assert_allclose(evens[layer // 2][i], expected, atol=1e-12)

    keys = jax.random.split(jax.random.PRNGKey(40), 32768)
    draws = np.asarray(jax.vmap(model.sample_auxiliary_fields, in_axes=(0, None, None))(keys, J, units[0]))
    np.testing.assert_allclose(draws.mean(0), 4 * np.asarray(J).T @ units[0], atol=0.045)
    np.testing.assert_allclose(np.cov(draws.T), 4*np.eye(2), atol=0.10)
    # z = sqrt(8) t converts the Gaussian integral to Hermite quadrature.
    nodes, quadrature_weights = hermgauss(24)
    grids = np.array(list(itertools.product(nodes, repeat=2))) * np.sqrt(8)
    qweights = np.outer(quadrature_weights, quadrature_weights).ravel() / np.pi
    for x in spin_configs(3):
        integral = qweights @ np.exp(grids @ np.asarray(J).T @ x)
        np.testing.assert_allclose(integral, np.exp(2 * np.sum((x @ np.asarray(J))**2)), rtol=1e-12)


@pytest.mark.parametrize("hidden,use_bias", [((), False), ((1, 1), True)])
def test_sampling_and_persistent_updates_match_exact_distribution(hidden, use_bias):
    model = make_model(hidden, use_bias, num_samples=32768, num_chains=128,
                       num_thermalization_steps=64, num_sweep_steps=4)
    params = fixed_params(model)
    configs = spin_configs(model.num_units)
    log_weights = 2 * reference_logs(model, params, configs).real
    probabilities = np.exp(log_weights - log_weights.max())
    probabilities /= probabilities.sum()
    samples, endpoints = jax.jit(model.generate_samples)(jax.random.PRNGKey(2), params)
    updated, new_endpoints = jax.jit(model.update_samples)(jax.random.PRNGKey(3), params, endpoints)
    for draws, ends in ((samples, endpoints), (updated, new_endpoints)):
        assert draws.shape == (model.num_samples, model.num_units)
        assert ends.shape == (model.num_chains, model.num_units)
        assert set(np.unique(draws)) == {-1, 1}
        indices = ((1 - np.asarray(draws)) // 2) @ (2 ** np.arange(model.num_units-1, -1, -1))
        empirical = np.bincount(indices, minlength=len(configs)) / len(draws)
        assert np.abs(empirical - probabilities).sum() / 2 < 0.025
    # Sampling is independent of every phase parameter, including K.
    tree = model.unravel_params(params)
    tree["K"] *= 7
    tree["imag"] = jax.tree.map(lambda x: x + 0.9, tree["imag"])
    phase_params = jax.flatten_util.ravel_pytree(tree)[0]
    phase_samples, _ = model.generate_samples(jax.random.PRNGKey(2), phase_params)
    np.testing.assert_array_equal(samples, phase_samples)
    _, local = hamiltonian(model)
    exact_energy = probabilities @ np.asarray(jax.vmap(local, in_axes=(None, 0))(params, jnp.asarray(configs)))
    measured = np.mean(jax.vmap(local, in_axes=(None, 0))(params, samples))
    assert abs(measured - exact_energy) < 0.025


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
@pytest.mark.parametrize("prefactor", [1.0, 1j])
def test_real_coordinate_sr_matches_finite_difference_reference(dtype, prefactor):
    model = make_model(hidden=(1,), dtype=dtype)
    params = fixed_params(model)
    configs = spin_configs(model.num_units)
    # Uneven repetitions exercise centering and the actual sample count.
    configs = np.concatenate([configs, configs[:7]], axis=0)
    eps = 1e-6
    derivatives = np.stack([
        (reference_logs(model, params.at[i].add(eps), configs)
         - reference_logs(model, params.at[i].add(-eps), configs)) / (2*eps)
        for i in range(params.size)
    ], axis=1)
    derivatives -= derivatives.mean(0)
    _, local = hamiltonian(model)
    energies = np.asarray(jax.vmap(local, in_axes=(None, 0))(params, jnp.asarray(configs)))
    metric = (derivatives.conj().T @ derivatives).real / len(configs)
    force = (derivatives.conj().T @ (prefactor * (energies - energies.mean()))).real / len(configs)
    expected = np.linalg.solve(metric + 0.08*np.eye(params.size), force)
    for method in ("SR", "minSR", "auto"):
        estimator = build_parametric_gradient_estimator(
            model, lambda p, x, c: c[0] * local(p, x),
            method=method, diag_shift=0.08, p_inv_rcond=1e-12, prefactor=prefactor, return_aux=True,
        )
        gradient, mean, variance = estimator(params, jnp.asarray(configs), jnp.array([1.0]))
        assert gradient.dtype == params.dtype
        np.testing.assert_allclose(gradient, expected, atol=2e-9, rtol=1e-8)
        np.testing.assert_allclose(mean, energies.mean(), atol=1e-12)
        np.testing.assert_allclose(variance, energies.var(), atol=1e-12)


def test_exact_energy_gradient_uses_marginalized_amplitude():
    model = make_model(hidden=(1,))
    params = fixed_params(model)
    configs = spin_configs(model.num_units)
    H, _ = hamiltonian(model)

    def energy(p):
        amplitudes = np.exp(reference_logs(model, p, configs)).reshape(8, -1)
        return (np.sum(amplitudes.conj() * (H @ amplitudes)) / np.sum(np.abs(amplitudes)**2)).real

    amplitudes = np.exp(reference_logs(model, params, configs)).reshape(8, -1)
    probabilities = np.abs(amplitudes.ravel())**2
    probabilities /= probabilities.sum()
    local_values = (H @ amplitudes / amplitudes).ravel()
    grad_r = jax.grad(lambda p, x: model.logpsi(p, x).real)
    grad_i = jax.grad(lambda p, x: model.logpsi(p, x).imag)
    derivatives = np.asarray(jax.vmap(lambda x: grad_r(params, x) + 1j*grad_i(params, x))(jnp.asarray(configs)))
    analytic = 2 * (derivatives.conj().T @ (probabilities * (local_values - energy(params)))).real
    eps = 1e-6
    numerical = np.array([(energy(params.at[i].add(eps)) - energy(params.at[i].add(-eps))) / (2*eps)
                          for i in range(params.size)])
    np.testing.assert_allclose(analytic, numerical, atol=1e-9, rtol=1e-7)
    assert np.linalg.norm(analytic) > 0.1


@pytest.mark.parametrize("g", [0.1, 1.5])
def test_visible_jastrow_reaches_entangled_ground_state_below_product_bound(g):
    """H=Z1 Z2+g(X1+X2) has an exactly representable Jastrow ground state.

    This checks the expressivity gain using an analytic separable energy bound,
    independently of any optimizer or stochastic energy estimate.
    """
    model = DeepBoltzmannQuantumState(
        2, [1], jax.random.PRNGKey(21), dtype=jnp.complex128,
        visible_interactions=True, visible_rank=2,
        num_samples=16, num_chains=4,
    )
    tree = jax.tree.map(jnp.zeros_like, model.unravel_params(model.params))
    ground_energy = -np.sqrt(1 + 4*g*g)
    pair_coefficient = 0.5*np.log(2*g/(1-ground_energy))
    factor = np.sqrt(-pair_coefficient/2)
    tree["J"] = jnp.asarray([[factor, 0.], [-factor, 0.]])
    tree["imag"][0][0] = jnp.full((2,), np.pi/2)
    params = jax.flatten_util.ravel_pytree(tree)[0]
    configs = jnp.asarray(spin_configs(model.num_units))
    amplitudes = np.asarray(jax.vmap(model.logpsi, in_axes=(None, 0))(params, configs))
    amplitudes = np.exp(amplitudes).reshape(4, 2)
    X, Z = np.array([[0., 1.], [1., 0.]]), np.diag([1., -1.])
    H = np.kron(Z, Z) + g*(np.kron(X, np.eye(2)) + np.kron(np.eye(2), X))
    np.testing.assert_allclose(H @ amplitudes, ground_energy*amplitudes, atol=1e-12)
    ratios = np.asarray(jax.vmap(model.local_sigma_xs, in_axes=(None, 0))(params, configs))
    local_values = np.asarray(configs[:, 0]*configs[:, 1]) + g*ratios.sum(axis=1)
    np.testing.assert_allclose(local_values, ground_energy, atol=1e-12)
    # The old DBQS is a mixture of product states. Its best possible energy
    # here is the analytic minimum over all pure product states.
    product_bound = -2*g if g >= 1 else -1-g*g
    assert ground_energy < product_bound - 1e-3


@pytest.mark.parametrize("use_bias", [True, False])
def test_zero_visible_terms_recover_legacy(use_bias):
    legacy = DeepBoltzmannQuantumState(3, [1], jax.random.PRNGKey(12), use_bias=use_bias)
    model = make_model(hidden=(1,), use_bias=use_bias)
    original = legacy.unravel_params(legacy.params)
    tree = model.unravel_params(model.params)
    tree.update(real=jax.tree.map(jnp.real, original), imag=jax.tree.map(jnp.imag, original),
                J=jnp.zeros_like(tree["J"]), K=jnp.zeros_like(tree["K"]))
    params = jax.flatten_util.ravel_pytree(tree)[0]
    configs = jnp.asarray(spin_configs(model.num_units))
    for name in ("logpsi", "local_sigma_xs", "local_sigma_ys"):
        np.testing.assert_allclose(
            jax.vmap(getattr(model, name), in_axes=(None, 0))(params, configs),
            jax.vmap(getattr(legacy, name), in_axes=(None, 0))(legacy.params, configs), atol=1e-12,
        )
    for default in (make_model(), make_model(dtype=jnp.float64)):
        assert not default.is_holomorphic
        assert np.issubdtype(default.params.dtype, np.floating)
        assert np.linalg.norm(default.unravel_params(default.params)["J"]) > 0


@pytest.mark.parametrize("persistent", [True, False])
def test_annealer_updates_real_parameters_with_y_catalyst(persistent):
    model = make_model(hidden=(1,))
    _, local = hamiltonian(model)
    estimator = build_parametric_gradient_estimator(
        model, lambda p, x, c: c[0]*local(p, x), method="minSR", diag_shift=0.1, return_aux=True,
    )
    annealer = VariationalAnnealer(
        model, estimator, optax.sgd(0.005), jnp.asarray([[1., 0.], [1., 0.]]),
        num_warmup_steps=1, num_finetuning_steps=1, num_updates_per_step=1,
        persistent_chains=persistent, use_tqdm=False, log_every=1, num_replicas=2, log_params=True,
    )
    data = annealer.run(jax.random.PRNGKey(21))
    final = np.asarray(data["optimized_params"])
    assert final.shape == (2, model.params.size)
    assert np.isfinite(final).all()
    assert np.issubdtype(final.dtype, np.floating)
    assert not np.allclose(data["params"][0], data["params"][-1])


@pytest.mark.parametrize("kwargs", [
    {"visible_interactions": "yes"}, {"visible_interactions": True, "visible_rank": 0},
    {"visible_interactions": True, "visible_rank": -1}, {"visible_interactions": True, "visible_rank": 1.5},
    {"visible_interactions": True, "visible_rank": True}, {"visible_rank": 2},
])
def test_invalid_visible_options(kwargs):
    with pytest.raises(ValueError):
        DeepBoltzmannQuantumState(3, [1], jax.random.PRNGKey(0), **kwargs)


def test_default_rank_and_missing_auxiliary_rejected():
    model = make_model(rank=None)
    assert model.unravel_params(model.params)["J"].shape == (3, 3)
    b, w, J, _ = model.unpack_params(model.params)
    units = model.unravel_config(model.dummy_config)
    with pytest.raises(ValueError):
        model.prob_evens_given_odds(b, w, units[1::2])
    with pytest.raises(ValueError):
        model.prob_evens_given_odds(b, w, units[1::2], J)

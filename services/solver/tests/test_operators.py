import os
import time
import numpy as np
import pytest

# Configure JAX runtime to be CI-friendly
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp

from utils.boltzmann_quantum_states import DeepBoltzmannQuantumState
from utils.operators import (
	build_local_tfsk_energy,
	build_local_parametric_hamiltonian,
	build_measurement_function,
)


def _mk(seed=0):
	return jax.random.PRNGKey(seed if seed is not None else int(time.time()))


def _symmetrize(J):
	return 0.5 * (J + J.T)


def _rand_sk(num_spins, key):
	k1, k2, k3 = jax.random.split(key, 3)
	J = jax.random.normal(k1, (num_spins, num_spins))
	J = _symmetrize(J)
	J = J.at[jnp.diag_indices(num_spins)].set(0.0)
	h = jax.random.normal(k2, (num_spins,)) * 0.1
	g = jax.random.normal(k3, (num_spins,)) * 0.2
	return J, h, g


@pytest.mark.parametrize("use_bias", [True, False])
def test_local_sigma_x_ratio_matches_psi_ratio(use_bias):
	# Small DBQS with N=4
	dbqs = DeepBoltzmannQuantumState(
		num_spins=4,
		hidden_layers=[3],
		prngkey=_mk(1),
		num_samples=8,
		num_thermalization_steps=2,
		num_sweep_steps=2,
		num_chains=2,
		dtype=jnp.complex128,
		use_bias=use_bias,
		initial_params_gain=1e-1,
	)
	# random configuration in {-1, +1} for all units
	cfg = (2 * jax.random.bernoulli(_mk(2), p=0.5, shape=(dbqs.num_units,)) - 1).astype(jnp.int64)

	xs = dbqs.local_sigma_xs(dbqs.params, cfg)
	# For each visible spin i, psi_ratio(s^i, s) should equal local_sigma_xs[i]
	for i in range(dbqs.num_units_list[0]):
		flipped = cfg.at[i].set(-cfg[i])
		ratio = dbqs.psi_ratio_fn(dbqs.params, flipped, cfg)
		np.testing.assert_allclose(np.asarray(ratio), np.asarray(xs[i]), rtol=1e-6, atol=1e-6)


def test_local_sigma_y_relation_to_x():
	dbqs = DeepBoltzmannQuantumState(
		num_spins=4,
		hidden_layers=[2],
		prngkey=_mk(3),
		num_samples=8,
		num_thermalization_steps=2,
		num_sweep_steps=2,
		num_chains=2,
		dtype=jnp.complex128,
		use_bias=True,
		initial_params_gain=1e-1,
	)
	cfg = (2 * jax.random.bernoulli(_mk(4), p=0.5, shape=(dbqs.num_units,)) - 1).astype(jnp.int64)
	xs = dbqs.local_sigma_xs(dbqs.params, cfg)
	ys = dbqs.local_sigma_ys(dbqs.params, cfg)
	vis = dbqs.unravel_config(cfg)[0]
	np.testing.assert_allclose(np.asarray(ys), np.asarray(-1.0j * vis * xs), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("g_mode", ["none", "zeros", "nonzero"])
def test_tfsk_energy_components_and_reductions(g_mode):
	num_spins = 4
	dbqs = DeepBoltzmannQuantumState(
		num_spins=num_spins,
		hidden_layers=[2],
		prngkey=_mk(5),
		num_samples=8,
		num_thermalization_steps=2,
		num_sweep_steps=2,
		num_chains=2,
		dtype=jnp.complex128,
		use_bias=True,
		initial_params_gain=1e-1,
	)

	J, h, g = _rand_sk(num_spins, _mk(6))
	shift = 0.123

	if g_mode == "none":
		g_vec = None
	elif g_mode == "zeros":
		g_vec = jnp.zeros((num_spins,))
	else:
		g_vec = g

	local_E = build_local_tfsk_energy(dbqs, J, h_vector=h, g_vector=g_vec, energy_shift=shift)

	# make a random config for all units
	cfg = (2 * jax.random.bernoulli(_mk(7), p=0.5, shape=(dbqs.num_units,)) - 1).astype(jnp.int64)
	s = cfg[:num_spins]

	# Z energy part
	Ez = s.T @ J @ s + s.T @ h + shift

	if g_mode in ("zeros", "none"):
		E = local_E(dbqs.params, cfg)
		np.testing.assert_allclose(np.asarray(E), np.asarray(Ez), rtol=1e-6, atol=1e-6)
	else:
		# X contribution via local_sigma_xs
		Ex = jnp.sum(g * dbqs.local_sigma_xs(dbqs.params, cfg))
		E = local_E(dbqs.params, cfg)
		np.testing.assert_allclose(np.asarray(E), np.asarray(Ez + Ex), rtol=1e-6, atol=1e-6)


def test_parametric_hamiltonian_combination():
	# Define simple locals that return identifiable values
	f_t = lambda p, c: 2.0
	f_a = lambda p, c: -1.0
	f_c = lambda p, c: 0.5

	H2 = build_local_parametric_hamiltonian(f_t, f_a, None)
	H3 = build_local_parametric_hamiltonian(f_t, f_a, f_c)

	params = jnp.array([0.0])
	cfg = jnp.array([1, -1])

	# two-term coupling
	val2 = H2(params, cfg, jnp.array([3.0, 4.0]))  # 3*2 + 4*(-1) = 2
	np.testing.assert_allclose(np.asarray(val2), np.asarray(2.0))

	# three-term coupling
	val3 = H3(params, cfg, jnp.array([2.0, 1.0, 4.0]))  # 2*2 + 1*(-1) + 4*(0.5) = 5
	np.testing.assert_allclose(np.asarray(val3), np.asarray(5.0))


@pytest.mark.parametrize("return_best", [False, True])
def test_measurement_function_matches_direct_stats(return_best):
	num_spins = 4
	dbqs = DeepBoltzmannQuantumState(
		num_spins=num_spins,
		hidden_layers=[2],
		prngkey=_mk(8),
		num_samples=16,
		num_thermalization_steps=2,
		num_sweep_steps=2,
		num_chains=4,
		dtype=jnp.complex128,
		use_bias=True,
		initial_params_gain=1e-1,
	)

	J, h, g = _rand_sk(num_spins, _mk(9))
	shift = -0.321
	local_E = build_local_tfsk_energy(dbqs, J, h_vector=h, g_vector=None, energy_shift=shift)

	# generate samples using the model
	samples, _ = dbqs.generate_samples(_mk(10), dbqs.params)
	# focus on first K samples to keep it light
	K = min(32, samples.shape[0])
	samples = samples[:K]

	# Direct values
	values = jax.vmap(lambda c: local_E(dbqs.params, c))(samples)

	measure = build_measurement_function(local_E, return_best=return_best)
	out = measure(dbqs.params, samples)

	if not return_best:
		mean_v, var_v = out
		np.testing.assert_allclose(np.asarray(mean_v), np.asarray(jnp.mean(values)), rtol=1e-6, atol=1e-6)
		np.testing.assert_allclose(np.asarray(var_v), np.asarray(jnp.var(values)), rtol=1e-6, atol=1e-6)
	else:
		# Current implementation returns (mean, var, best_value, *best_config)
		mean_v, var_v, best_value, *best_cfg_parts = out
		np.testing.assert_allclose(np.asarray(mean_v), np.asarray(jnp.mean(values)), rtol=1e-6, atol=1e-6)
		np.testing.assert_allclose(np.asarray(var_v), np.asarray(jnp.var(values)), rtol=1e-6, atol=1e-6)

		# reconstruct config and compare
		best_idx = int(np.argmin(np.asarray(values)))
		best_cfg = jnp.array(best_cfg_parts)
		np.testing.assert_equal(np.asarray(best_cfg), np.asarray(samples[best_idx]))
		np.testing.assert_allclose(np.asarray(best_value), np.asarray(values[best_idx]), rtol=1e-6, atol=1e-6)


def test_end_to_end_tfske_n4_measurement_consistency():
	# End-to-end: N=4 TFSK, measure energy statistics vs. direct computation
	num_spins = 4
	dbqs = DeepBoltzmannQuantumState(
		num_spins=num_spins,
		hidden_layers=[3],
		prngkey=_mk(11),
		num_samples=24,
		num_thermalization_steps=3,
		num_sweep_steps=2,
		num_chains=3,
		dtype=jnp.complex128,
		use_bias=True,
		initial_params_gain=1e-1,
	)

	J, h, g = _rand_sk(num_spins, _mk(12))
	shift = 0.0
	local_E = build_local_tfsk_energy(dbqs, J, h_vector=h, g_vector=g, energy_shift=shift)

	samples, _ = dbqs.generate_samples(_mk(13), dbqs.params)
	values = jax.vmap(lambda c: local_E(dbqs.params, c))(samples)

	measure = build_measurement_function(local_E, return_best=False)
	mean_v, var_v = measure(dbqs.params, samples)

	np.testing.assert_allclose(np.asarray(mean_v), np.asarray(jnp.mean(values)), rtol=1e-6, atol=1e-6)
	np.testing.assert_allclose(np.asarray(var_v), np.asarray(jnp.var(values)), rtol=1e-6, atol=1e-6)


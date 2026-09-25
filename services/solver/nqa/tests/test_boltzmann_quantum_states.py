import pytest
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_ENABLE_X64"] = "True"
from utils.boltzmann_quantum_states import DeepBoltzmannQuantumState
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import time


def _mk(prng_seed=0):
    return jax.random.PRNGKey(prng_seed if prng_seed is not None else int(time.time()))


@pytest.mark.parametrize(
    "num_spins,hidden_layers,use_bias",
    [
        (3, [2], True),
        (3, [2, 2], False),
        (4, [3, 1], True),
    ],
)
def test_config_and_params_shapes(num_spins, hidden_layers, use_bias):
    dbqs = DeepBoltzmannQuantumState(
        num_spins=num_spins,
        hidden_layers=hidden_layers,
        prngkey=_mk(42),
        num_samples=16,
        num_thermalization_steps=4,
        num_sweep_steps=3,
        num_chains=4,
        use_bias=use_bias,
        initial_params_gain=1e-1,
    )

    # num_units_list and num_units
    expected_units_list = [num_spins] + hidden_layers
    assert dbqs.num_units_list == expected_units_list
    assert dbqs.num_units == sum(expected_units_list)

    # dummy_config shape
    assert dbqs.dummy_config.shape == (dbqs.num_units,)

    # params is 1D and unravel works as inverse
    flat = dbqs.params
    unflat = dbqs.unravel_params(flat)
    reflat, _ = jax.flatten_util.ravel_pytree(unflat)
    assert reflat.shape == flat.shape
    np.testing.assert_allclose(np.asarray(flat), np.asarray(reflat))

    # samples per chain
    assert dbqs.num_samples_per_chain == dbqs.num_samples // dbqs.num_chains


@pytest.mark.parametrize("use_bias", [True, False])
def test_unravel_params_structure(use_bias):
    num_spins, hidden_layers = 3, [2, 2]
    dbqs = DeepBoltzmannQuantumState(
        num_spins=num_spins,
        hidden_layers=hidden_layers,
        prngkey=_mk(0),
        num_samples=8,
        num_thermalization_steps=2,
        num_sweep_steps=2,
        num_chains=2,
        use_bias=use_bias,
        initial_params_gain=1e-1,
    )

    params_tree = dbqs.unravel_params(dbqs.params)
    if use_bias:
        biases, weights = params_tree
        # biases length equals number of layers
        assert len(biases) == len(dbqs.num_units_list)
        for b, n in zip(biases, dbqs.num_units_list):
            assert b.shape == (2, n)
    else:
        weights = params_tree

    # weights length equals number of connections between consecutive layers
    assert len(weights) == len(dbqs.num_units_list) - 1
    for W, (n_in, n_out) in zip(weights, zip(dbqs.num_units_list[:-1], dbqs.num_units_list[1:])):
        assert W.shape == (2, n_in, n_out)


def test_logpsi_and_ratio_identity():
    dbqs = DeepBoltzmannQuantumState(
        num_spins=3,
        hidden_layers=[2],
        prngkey=_mk(1),
        num_samples=8,
        num_thermalization_steps=2,
        num_sweep_steps=2,
        num_chains=2,
        use_bias=True,
        initial_params_gain=1e-1,
    )

    # sample a random configuration in {-1, +1}
    cfg = (2 * jax.random.bernoulli(_mk(2), p=0.5, shape=(dbqs.num_units,)) - 1).astype(jnp.int64)
    # a slightly different config (flip first unit)
    cfg2 = cfg.at[0].set(-cfg[0])

    val = dbqs.logpsi(dbqs.params, cfg)
    assert np.asarray(val).shape == ()  # scalar
    ratio = dbqs.psi_ratio_fn(dbqs.params, cfg2, cfg)
    # Consistency check with logpsi difference
    expected = jnp.exp(dbqs.logpsi(dbqs.params, cfg2) - dbqs.logpsi(dbqs.params, cfg))
    np.testing.assert_allclose(np.asarray(ratio), np.asarray(expected), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("use_bias", [True, False])
def test_local_energies_consistency(use_bias):
    dbqs = DeepBoltzmannQuantumState(
        num_spins=4,
        hidden_layers=[3],
        prngkey=_mk(3),
        num_samples=8,
        num_thermalization_steps=2,
        num_sweep_steps=2,
        num_chains=2,
        use_bias=use_bias,
        initial_params_gain=1e-1,
    )

    cfg = (2 * jax.random.bernoulli(_mk(4), p=0.5, shape=(dbqs.num_units,)) - 1).astype(jnp.int64)
    xs = dbqs.local_sigma_xs(dbqs.params, cfg)
    ys = dbqs.local_sigma_ys(dbqs.params, cfg)

    # Shapes align with visible spins only (first layer length)
    assert xs.shape == (dbqs.num_units_list[0],)
    assert ys.shape == (dbqs.num_units_list[0],)
    ex = dbqs.local_energy_sigma_x(dbqs.params, cfg)
    ey = dbqs.local_energy_sigma_y(dbqs.params, cfg)
    # Energy equals negative sum of locals by definition in implementation
    np.testing.assert_allclose(np.asarray(ex), -np.asarray(jnp.sum(xs)))
    np.testing.assert_allclose(np.asarray(ey), -np.asarray(jnp.sum(ys)))


def test_prob_functions_range_and_shapes():
    dbqs = DeepBoltzmannQuantumState(
        num_spins=3,
        hidden_layers=[2, 3],
        prngkey=_mk(5),
        num_samples=8,
        num_thermalization_steps=2,
        num_sweep_steps=2,
        num_chains=2,
        use_bias=True,
        initial_params_gain=1e-1,
    )
    # Unravel params to get biases and weights
    biases, weights = dbqs.unravel_params(dbqs.params)

    # Make a random config and split into units
    cfg = (2 * jax.random.bernoulli(_mk(6), p=0.5, shape=(dbqs.num_units,)) - 1).astype(jnp.int64)
    units = dbqs.unravel_config(cfg)

    # odds/even layers from units
    evens = units[::2]
    odds = units[1::2]

    pe = dbqs.prob_evens_given_odds(biases, weights, odds)
    po = dbqs.prob_odds_given_evens(biases, weights, evens)

    # Check lengths and shapes
    assert len(pe) == len(evens)
    assert len(po) == len(odds)
    for p, n in zip(pe, [dbqs.num_units_list[i] for i in range(0, len(dbqs.num_units_list), 2)]):
        assert p.shape == (2, n)
        arr = np.asarray(p)
        assert np.all((arr >= 0) & (arr <= 1))
    for p, n in zip(po, [dbqs.num_units_list[i] for i in range(1, len(dbqs.num_units_list), 2)]):
        assert p.shape == (2, n)
        arr = np.asarray(p)
        assert np.all((arr >= 0) & (arr <= 1))


def test_base_sampler_and_gibbs_step_shapes_and_values():
    dbqs = DeepBoltzmannQuantumState(3, [2], _mk(7), num_samples=8, num_thermalization_steps=2, num_sweep_steps=2, num_chains=2, use_bias=False)
    prng, units = dbqs.base_sampler(_mk(7))
    weights, biases = dbqs.unpack_params(dbqs.params)
    _, updated_units = dbqs.gibbs_step(prng, biases, weights, units)
    assert len(updated_units) == len(dbqs.num_units_list)
    assert all(set(np.unique(np.asarray(u))).issubset({-1, 1}) for u in updated_units)


def test_sampling_shapes_and_rounding():
    dbqs = DeepBoltzmannQuantumState(3, [2], _mk(8), num_samples=10, num_thermalization_steps=2, num_sweep_steps=2, num_chains=4)
    samples, endpoints = dbqs.generate_samples(_mk(9), dbqs.params)
    assert dbqs.num_samples % dbqs.num_chains == 0
    assert samples.shape == (dbqs.num_samples, dbqs.num_units)
    assert endpoints.shape == (dbqs.num_chains, dbqs.num_units)


def test_input_validation_raises():
    with pytest.raises(Exception):
        DeepBoltzmannQuantumState(num_spins=0, hidden_layers=[2], prngkey=_mk(16))
    with pytest.raises(Exception):
        DeepBoltzmannQuantumState(num_spins=2, hidden_layers=[0], prngkey=_mk(16))
    with pytest.raises(Exception):
        DeepBoltzmannQuantumState(num_spins=2, hidden_layers=[2], prngkey=_mk(16), num_samples=0)


def test_initialization_uses_real_parameters():
    dbqs = DeepBoltzmannQuantumState(3, [2, 2], _mk(21), use_bias=True)
    assert jnp.issubdtype(dbqs.params.dtype, jnp.floating)
    assert (dbqs.params == jax.flatten_util.ravel_pytree(dbqs.unravel_params(dbqs.params))[0]).all()

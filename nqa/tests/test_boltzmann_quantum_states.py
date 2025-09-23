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
    "num_spins,hidden_layers,use_bias,dtype",
    [
        (3, [2], True, jnp.float64),
        (3, [2, 2], False, jnp.float64),
        (4, [3, 1], True, jnp.complex128),
    ],
)
def test_config_and_params_shapes(num_spins, hidden_layers, use_bias, dtype):
    dbqs = DeepBoltzmannQuantumState(
        num_spins=num_spins,
        hidden_layers=hidden_layers,
        prngkey=_mk(42),
        num_samples=16,
        num_thermalization_steps=4,
        num_sweep_steps=3,
        num_chains=4,
        dtype=dtype,
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
        dtype=jnp.float64,
        use_bias=use_bias,
        initial_params_gain=1e-1,
    )

    params_tree = dbqs.unravel_params(dbqs.params)
    if use_bias:
        biases, weights = params_tree
        # biases length equals number of layers
        assert len(biases) == len(dbqs.num_units_list)
        for b, n in zip(biases, dbqs.num_units_list):
            assert b.shape == (n,)
    else:
        weights = params_tree

    # weights length equals number of connections between consecutive layers
    assert len(weights) == len(dbqs.num_units_list) - 1
    for W, (n_in, n_out) in zip(weights, zip(dbqs.num_units_list[:-1], dbqs.num_units_list[1:])):
        assert W.shape == (n_in, n_out)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_logpsi_and_ratio_identity(dtype):
    dbqs = DeepBoltzmannQuantumState(
        num_spins=3,
        hidden_layers=[2],
        prngkey=_mk(1),
        num_samples=8,
        num_thermalization_steps=2,
        num_sweep_steps=2,
        num_chains=2,
        dtype=dtype,
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
        dtype=jnp.complex128,
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
        dtype=jnp.float64,
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
        assert p.shape == (n,)
        arr = np.asarray(p)
        assert np.all((arr >= 0) & (arr <= 1))
    for p, n in zip(po, [dbqs.num_units_list[i] for i in range(1, len(dbqs.num_units_list), 2)]):
        assert p.shape == (n,)
        arr = np.asarray(p)
        assert np.all((arr >= 0) & (arr <= 1))


def test_base_sampler_and_gibbs_step_shapes_and_values():
    dbqs = DeepBoltzmannQuantumState(
        num_spins=3,
        hidden_layers=[2],
        prngkey=_mk(7),
        num_samples=8,
        num_thermalization_steps=2,
        num_sweep_steps=2,
        num_chains=2,
        dtype=jnp.float64,
        use_bias=False,
        initial_params_gain=1e-1,
    )

    # base sampler
    prng, units = dbqs.base_sampler(_mk(7))
    assert isinstance(units, list)
    assert len(units) == len(dbqs.num_units_list)
    for u, n in zip(units, dbqs.num_units_list):
        assert u.shape == (n,)
        assert set(np.unique(np.asarray(u))).issubset({-1, 1})

    # one gibbs step produces same structure
    params_tree = dbqs.unravel_params(dbqs.params)
    if dbqs.use_bias:
        biases, weights = params_tree
    else:
        biases, weights = [jnp.zeros((n,)) for n in dbqs.num_units_list], params_tree

    prng2, units2 = dbqs.gibbs_step(prng, biases, weights, units)
    assert len(units2) == len(dbqs.num_units_list)
    for u, n in zip(units2, dbqs.num_units_list):
        assert u.shape == (n,)
        assert set(np.unique(np.asarray(u))).issubset({-1, 1})


def test_gibbs_chain_shapes():
    dbqs = DeepBoltzmannQuantumState(
        num_spins=3,
        hidden_layers=[2],
        prngkey=_mk(8),
        num_samples=16,
        num_thermalization_steps=3,
        num_sweep_steps=2,
        num_chains=4,
        dtype=jnp.float64,
        use_bias=True,
        initial_params_gain=1e-1,
    )
    biases, weights = dbqs.unravel_params(dbqs.params)
    samples_chain = dbqs.gibbs_chain(_mk(9), biases, weights)
    assert samples_chain.shape == (dbqs.num_samples_per_chain, dbqs.num_units)
    # spins are in {-1, 1}
    assert set(np.unique(np.asarray(samples_chain))).issubset({-1, 1})


def test_vmapd_gibbs_chain_shapes():
    dbqs = DeepBoltzmannQuantumState(
        num_spins=3,
        hidden_layers=[2],
        prngkey=_mk(10),
        num_samples=12,
        num_thermalization_steps=2,
        num_sweep_steps=2,
        num_chains=3,
        dtype=jnp.float64,
        use_bias=True,
        initial_params_gain=1e-1,
    )
    biases, weights = dbqs.unravel_params(dbqs.params)
    keys = jax.random.split(_mk(11), dbqs.num_chains)
    chains = dbqs.vmapd_gibbs_chain(keys, biases, weights)
    assert chains.shape == (dbqs.num_chains, dbqs.num_samples_per_chain, dbqs.num_units)


@pytest.mark.parametrize("use_bias", [True, False])
def test_generate_samples_and_update_samples_shapes(use_bias):
    dbqs = DeepBoltzmannQuantumState(
        num_spins=3,
        hidden_layers=[2],
        prngkey=_mk(12),
        num_samples=16,
        num_thermalization_steps=2,
        num_sweep_steps=2,
        num_chains=4,
        dtype=jnp.float64,
        use_bias=use_bias,
        initial_params_gain=1e-1,
    )
    samples, endpoints = dbqs.generate_samples(_mk(13), dbqs.params)
    assert samples.shape == (dbqs.num_samples, dbqs.num_units)
    assert endpoints.shape == (dbqs.num_chains, dbqs.num_units)
    assert set(np.unique(np.asarray(samples))).issubset({-1, 1})
    assert set(np.unique(np.asarray(endpoints))).issubset({-1, 1})

    # Update starting from endpoints
    upd_samples, upd_endpoints = dbqs.update_samples(_mk(14), dbqs.params, endpoints)
    assert upd_samples.shape == (dbqs.num_samples, dbqs.num_units)
    assert upd_endpoints.shape == (dbqs.num_chains, dbqs.num_units)


def test_num_samples_rounding_up_to_chains_multiple():
    # num_samples not divisible by num_chains should round up
    dbqs = DeepBoltzmannQuantumState(
        num_spins=2,
        hidden_layers=[2],
        prngkey=_mk(15),
        num_samples=10,  # not divisible by 6
        num_thermalization_steps=2,
        num_sweep_steps=2,
        num_chains=6,
        dtype=jnp.float64,
        use_bias=True,
        initial_params_gain=1e-1,
    )
    assert dbqs.num_samples % dbqs.num_chains == 0
    assert dbqs.num_samples >= 10


def test_input_validation_raises():
    # invalid num_spins
    with pytest.raises(Exception):
        DeepBoltzmannQuantumState(
            num_spins=0,
            hidden_layers=[2],
            prngkey=_mk(16),
        )

    # invalid hidden layers
    with pytest.raises(Exception):
        DeepBoltzmannQuantumState(
            num_spins=2,
            hidden_layers=[0],
            prngkey=_mk(16),
        )

    # invalid num_samples
    with pytest.raises(Exception):
        DeepBoltzmannQuantumState(
            num_spins=2,
            hidden_layers=[2],
            prngkey=_mk(16),
            num_samples=0,
        )

    # invalid chains
    with pytest.raises(Exception):
        DeepBoltzmannQuantumState(
            num_spins=2,
            hidden_layers=[2],
            prngkey=_mk(16),
            num_chains=0,
        )

    # invalid gain
    with pytest.raises(Exception):
        DeepBoltzmannQuantumState(
            num_spins=2,
            hidden_layers=[2],
            prngkey=_mk(16),
            initial_params_gain=-1.0,
        )


def test_is_holomorphic_flag_matches_dtype():
    dbqs_r = DeepBoltzmannQuantumState(
        num_spins=2,
        hidden_layers=[2],
        prngkey=_mk(21),
        dtype=jnp.float64,
    )
    assert dbqs_r.is_holomorphic is False

    dbqs_c = DeepBoltzmannQuantumState(
        num_spins=2,
        hidden_layers=[2],
        prngkey=_mk(22),
        dtype=jnp.complex128,
    )
    assert dbqs_c.is_holomorphic is True


def test_debug_gibbs_chain_shapes():
    dbqs = DeepBoltzmannQuantumState(
        num_spins=3,
        hidden_layers=[2],
        prngkey=_mk(23),
        num_samples=8,
        num_thermalization_steps=1,
        num_sweep_steps=1,
        num_chains=2,
        dtype=jnp.float64,
        use_bias=True,
        initial_params_gain=1e-1,
    )
    chain = dbqs.debug_gibbs_chain(_mk(24), dbqs.params)
    assert chain.shape == (dbqs.num_samples_per_chain, dbqs.num_units)


@pytest.mark.parametrize(
    "num_spins, hidden_layers, prngkey, num_samples, num_thermalization_steps, num_sweep_steps, num_chains, dtype, use_bias, initial_params_gain",
    [
        (3, [2], jax.random.PRNGKey(int(time.time())), 16, 16, 16, 12, jax.numpy.float64, True, 1),
        (3, [2,2], jax.random.PRNGKey(int(time.time())), 14, 7, 8, 8, jax.numpy.float64, False, 1e-2),
        (3, [3,1], jax.random.PRNGKey(int(time.time())), 32, 32, 4, 32, jax.numpy.complex128, True, 1e-1),
    ]
)
def test_boltzmann_quantum_states_initialization(
    num_spins,
    hidden_layers,
    prngkey,
    num_samples,
    num_thermalization_steps,
    num_sweep_steps,
    num_chains,
    dtype,
    use_bias,
    initial_params_gain,
):
    dbqs = DeepBoltzmannQuantumState(
        num_spins=num_spins,
        hidden_layers=hidden_layers,
        prngkey=prngkey,
        num_samples=num_samples,
        num_thermalization_steps=num_thermalization_steps,
        num_sweep_steps=num_sweep_steps,
        num_chains=num_chains,
        dtype=dtype,
        use_bias=use_bias,
        initial_params_gain=initial_params_gain,
    )
    assert dbqs.num_spins == num_spins
    assert dbqs.hidden_layers == hidden_layers
    assert dbqs.num_samples >= num_samples
    assert dbqs.num_thermalization_steps == num_thermalization_steps
    assert dbqs.num_sweep_steps == num_sweep_steps
    assert dbqs.num_chains == num_chains
    assert (dbqs.num_samples % dbqs.num_chains) == 0
    assert dbqs.use_bias == use_bias
    assert dbqs.initial_params_gain == initial_params_gain

    # Assert that if complex dtype, the is_holomorphic flag is set to True
    if dtype == jax.numpy.complex128:
        assert dbqs.is_holomorphic is True
    else:
        assert dbqs.is_holomorphic is False

    # Assert that the config unravel functions work correctly
    assert (dbqs.params == jax.flatten_util.ravel_pytree(dbqs.unravel_params(dbqs.params))[0]).all()

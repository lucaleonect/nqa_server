import os
import time
import numpy as np
import pytest

# CI-friendly JAX runtime settings
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp
import optax

from utils.boltzmann_quantum_states import DeepBoltzmannQuantumState
from utils.operators import build_local_tfsk_energy, build_local_parametric_hamiltonian, build_measurement_function
from utils.tdvp import build_parametric_gradient_estimator
from utils.variational_annealer import VariationalAnnealer


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


def _setup_dbqs_n4(num_samples=64, chains=8):
    prng = _mk(123)
    dbqs = DeepBoltzmannQuantumState(
        num_spins=4,
        hidden_layers=[4],
        prngkey=prng,
        num_samples=num_samples,
        num_thermalization_steps=2,
        num_sweep_steps=2,
        num_chains=chains,
        dtype=jnp.complex128,
        use_bias=True,
        initial_params_gain=1e-1,
    )
    return dbqs


def _build_hamiltonians(dbqs, key=None):
    if key is None:
        key = _mk(7)
    J, h, g = _rand_sk(dbqs.num_spins, key)
    local_target = build_local_tfsk_energy(dbqs, J, h_vector=h, g_vector=None)
    local_annealing = build_local_tfsk_energy(dbqs, jnp.zeros_like(J), jnp.zeros((dbqs.num_spins,)), g_vector=g)
    return local_target, local_annealing, J, h, g


def _build_estimator(dbqs, local_param_h, method="SR"):
    return build_parametric_gradient_estimator(
        dbqs,
        local_param_h,
        prefactor=1.0,
        p_inv_rcond=1e-12,
        diag_shift=1e-3,
        method=method,
        return_aux=True,
    )


def _simple_schedule(len_steps=6, with_catalyst=False):
    if with_catalyst:
        # linear ramp with constant small catalyst
        t = jnp.linspace(0.0, 1.0, len_steps)
        cat = jnp.full_like(t, 0.25)
        return jnp.stack([1.0 - t, t, cat], axis=1)
    else:
        t = jnp.linspace(0.0, 1.0, len_steps)
        return jnp.stack([1.0 - t, t], axis=1)


def test_variational_annealer_basic_run_no_catalyst():
    dbqs = _setup_dbqs_n4(num_samples=64, chains=8)
    local_target, local_annealing, *_ = _build_hamiltonians(dbqs, _mk(10))
    local_param_h = build_local_parametric_hamiltonian(local_target, local_annealing)

    est = _build_estimator(dbqs, local_param_h, method="SR")
    schedule = _simple_schedule(len_steps=8, with_catalyst=False)

    va = VariationalAnnealer(
        variational_quantum_state=dbqs,
        parametric_gradient_estimator=est,
        optimizer=optax.adam(1e-2),
        annealing_schedule=schedule,
        num_warmup_steps=1,
        num_updates_per_step=1,
        num_finetuning_steps=1,
        persistent_chains=True,
        observables_dict={
            "target_energy": build_measurement_function(local_target),
            "annealing_energy": build_measurement_function(local_annealing),
        },
        use_tqdm=False,
        log_every=2,
        log_params=True,
    )

    data = va.run(_mk(11))
    assert isinstance(data, dict)
    # logged keys present
    for k in ["avg_energy", "energy_var", "magnetizations", "target_energy", "annealing_energy"]:
        assert k in data
        assert len(data[k]) >= 1

    # params updated and returned
    assert "optimized_params" in data
    assert data["optimized_params"].shape == dbqs.params.shape

    # schedule echoed
    assert "couplings" in data
    np.testing.assert_allclose(np.asarray(data["couplings"]), np.asarray(schedule))

    # log_every=2 means roughly half as many logs as steps (ceil), allow off-by-one
    expected_logs_min = max(1, schedule.shape[0] // 2)
    assert len(data["avg_energy"]) >= expected_logs_min


def test_variational_annealer_with_catalyst_and_inst_energy():
    dbqs = _setup_dbqs_n4(num_samples=64, chains=8)
    local_target, local_annealing, *_ = _build_hamiltonians(dbqs, _mk(12))

    # simple catalyst: reuse annealing as catalyst to test 3-term path
    local_catalyst = local_annealing
    local_param_h = build_local_parametric_hamiltonian(local_target, local_annealing, local_catalyst)
    est = _build_estimator(dbqs, local_param_h, method="minSR")
    schedule = _simple_schedule(len_steps=7, with_catalyst=True)

    va = VariationalAnnealer(
        variational_quantum_state=dbqs,
        parametric_gradient_estimator=est,
        optimizer=optax.sgd(5e-3),
        annealing_schedule=schedule,
        num_warmup_steps=1,
        num_updates_per_step=1,
        num_finetuning_steps=1,
        persistent_chains=False,
        observables_dict={
            "target_energy": build_measurement_function(local_target),
            "annealing_energy": build_measurement_function(local_annealing),
            "catalyst_energy": build_measurement_function(local_catalyst),
        },
        use_tqdm=False,
        log_every=3,
        log_params=False,
    )

    data = va.run(_mk(13))
    assert data is not None

    # Validate magnetizations shape per log entry
    mags = data["magnetizations"][0]
    assert mags.shape == (dbqs.num_units,)


def test_variational_annealer_early_stop_runtime_cap():
    dbqs = _setup_dbqs_n4(num_samples=64, chains=8)
    local_target, local_annealing, *_ = _build_hamiltonians(dbqs, _mk(14))
    local_param_h = build_local_parametric_hamiltonian(local_target, local_annealing)
    est = _build_estimator(dbqs, local_param_h, method="SR")

    # long schedule, tiny time cap to trigger early stop path
    schedule = _simple_schedule(len_steps=60, with_catalyst=False)
    va = VariationalAnnealer(
        variational_quantum_state=dbqs,
        parametric_gradient_estimator=est,
        optimizer=optax.adam(1e-3),
        annealing_schedule=schedule,
        num_warmup_steps=1,
        num_updates_per_step=1,
        num_finetuning_steps=1,
        persistent_chains=True,
        observables_dict={},
        use_tqdm=False,
        log_every=10,
        log_params=False,
    )

    out = va.run(_mk(15), max_runtime=1)  # effectively immediate cap
    assert out is None  # Early stopping returns None


def test_variational_annealer_invalid_inputs_raise():
    dbqs = _setup_dbqs_n4()
    local_target, local_annealing, *_ = _build_hamiltonians(dbqs, _mk(16))
    local_param_h = build_local_parametric_hamiltonian(local_target, local_annealing)
    est = _build_estimator(dbqs, local_param_h, method="SR")

    with pytest.raises(Exception):
        VariationalAnnealer(
            variational_quantum_state="not a dbqs",
            parametric_gradient_estimator=est,
            optimizer=optax.adam(1e-2),
            annealing_schedule=_simple_schedule(4, False),
        )

    with pytest.raises(Exception):
        VariationalAnnealer(
            variational_quantum_state=dbqs,
            parametric_gradient_estimator=lambda *args, **kwargs: None,
            optimizer=object(),  # missing init/update
            annealing_schedule=_simple_schedule(4, False),
        )

    with pytest.raises(Exception):
        VariationalAnnealer(
            variational_quantum_state=dbqs,
            parametric_gradient_estimator=est,
            optimizer=optax.adam(1e-2),
            annealing_schedule=jnp.ones((0, 2)),  # invalid T
        )

    # invalid schedule width
    with pytest.raises(Exception):
        VariationalAnnealer(
            variational_quantum_state=dbqs,
            parametric_gradient_estimator=est,
            optimizer=optax.adam(1e-2),
            annealing_schedule=jnp.ones((3, 4)),
        )

    # invalid log_every
    with pytest.raises(Exception):
        VariationalAnnealer(
            variational_quantum_state=dbqs,
            parametric_gradient_estimator=est,
            optimizer=optax.adam(1e-2),
            annealing_schedule=_simple_schedule(4, False),
            log_every=0,
        )


def test_variational_annealer_logs_include_params_when_enabled():
    dbqs = _setup_dbqs_n4(num_samples=64, chains=8)
    local_target, local_annealing, *_ = _build_hamiltonians(dbqs, _mk(17))
    local_param_h = build_local_parametric_hamiltonian(local_target, local_annealing)
    est = _build_estimator(dbqs, local_param_h, method="minSR")

    schedule = _simple_schedule(5, False)
    va = VariationalAnnealer(
        variational_quantum_state=dbqs,
        parametric_gradient_estimator=est,
        optimizer=optax.sgd(1e-2),
        annealing_schedule=schedule,
        num_warmup_steps=1,
        num_updates_per_step=1,
        num_finetuning_steps=1,
        persistent_chains=True,
        observables_dict={},
        use_tqdm=False,
        log_every=1,
        log_params=True,
    )

    data = va.run(_mk(18))
    assert data is not None
    assert "params" in data
    assert len(data["params"]) >= 1
    # Each logged params must match shape
    for p in data["params"]:
        assert p.shape == dbqs.params.shape

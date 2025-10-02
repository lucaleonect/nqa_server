import jax
import jax.flatten_util
import jax.numpy as jnp
from typing import Callable, Optional, Tuple, List
from functools import partial
from utils.sampling.metropolis_hasting_sampler import (
    build_base_sampler,
    build_update_proposer,
    build_mcmc_sampler,
)


class cRBM:
    def __init__(
        self,
        num_spins: int,
        num_hidden: int,
        prngkey: jnp.ndarray,
        num_samples: Optional[int] = None,
        num_thermalization_steps: Optional[int] = None,
        num_sweep_steps: Optional[int] = None,
        num_chains: Optional[int] = None,
        use_bias: Optional[bool] = None,
        dtype: Optional[jnp.dtype] = None,
    ):
        if num_samples is None:
            num_samples = 2**10
        if num_thermalization_steps is None:
            num_thermalization_steps = (2**7) * num_spins
        if num_sweep_steps is None:
            num_sweep_steps = (2**4) * num_spins
        if num_chains is None:
            num_chains = 2**8
        if dtype is None:
            dtype = jnp.complex128
        if use_bias is None:
            use_bias = True
        num_samples_per_chain = num_samples // num_chains + (1 * (num_samples % num_chains != 0))
        num_samples = num_samples_per_chain * num_chains

        self.num_spins = num_spins
        self.num_hidden = num_hidden
        self.num_samples = num_samples
        self.num_thermalization_steps = num_thermalization_steps
        self.num_sweep_steps = num_sweep_steps
        self.num_chains = num_chains
        self.use_bias = use_bias
        self.dtype = dtype
        self.num_samples_per_chain = self.num_samples // self.num_chains
        self.is_holomorphic = True if (dtype == jnp.complex64 or dtype == jnp.complex128) else False
        self.params, self.unravel_params = self.init_params(prngkey)
        self.unravel_params = jax.jit(self.unravel_params)
        self.generate_samples, self.update_samples = build_mcmc_sampler(
            base_sampler=build_base_sampler(num_spins),
            update_proposer=build_update_proposer(num_spins),
            probability_ratio_function=self.prob_ratio_fn,
            num_samples=num_samples,
            num_thermalization_steps=num_thermalization_steps,
            num_sweep_steps=num_sweep_steps,
            num_chains=num_chains,
        )

    def init_params(
        self,
        prngkey: jnp.ndarray,
        scale: float = 0.1,
    ):
        visible_bias = jnp.zeros(self.num_spins, dtype=self.dtype)
        hidden_bias = jnp.zeros(self.num_hidden, dtype=self.dtype)
        weights = (
            jax.random.normal(
                prngkey,
                (self.num_spins, self.num_hidden),
                dtype=self.dtype,
            )
            * scale
        )
        if self.use_bias:
            params, unravel_params = jax.flatten_util.ravel_pytree((visible_bias, hidden_bias, weights))
        else:
            params, unravel_params = jax.flatten_util.ravel_pytree(weights)

        return params, unravel_params

    @partial(jax.jit, static_argnums=(0,))
    def logpsi(
        self,
        params,
        config,
    ):
        if self.use_bias:
            (
                visible_bias,
                hidden_bias,
                weights,
            ) = self.unravel_params(params)
        else:
            weights = self.unravel_params(params)
            visible_bias = jnp.zeros(self.num_spins, dtype=self.dtype)
            hidden_bias = jnp.zeros(self.num_hidden, dtype=self.dtype)

        return jnp.sum(visible_bias * config) + jnp.sum(jnp.log(jnp.cosh(config.T @ weights + hidden_bias)))

    @partial(jax.jit, static_argnums=(0,))
    def psi_ratio_fn(
        self,
        params: jnp.ndarray,
        config_num: jnp.ndarray,
        config_den: jnp.ndarray,
    ):
        return jnp.exp(self.logpsi(params, config_num) - self.logpsi(params, config_den))

    @partial(jax.jit, static_argnums=(0,))
    def prob_ratio_fn(
        self,
        params: jnp.ndarray,
        config_num: jnp.ndarray,
        config_den: jnp.ndarray,
    ):
        return jnp.exp(2 * jnp.real(self.logpsi(params, config_num) - self.logpsi(params, config_den)))

    @partial(jax.jit, static_argnums=(0,))
    def local_sigma_x_i(
        self,
        params: jnp.ndarray,
        config: jnp.ndarray,
        i: int,
    ):
        return self.psi_ratio_fn(params, config.at[i].multiply(-1), config)

    @partial(jax.jit, static_argnums=(0,))
    def local_energy_sigma_x(
        self,
        params: jnp.ndarray,
        config: jnp.ndarray,
    ):
        return -jnp.sum(
            jax.vmap(self.local_sigma_x_i, in_axes=(None, None, 0))(
                params,
                config,
                jnp.arange(self.num_spins),
            )
        )

    @partial(jax.jit, static_argnums=(0,))
    def local_sigma_y_i(
        self,
        params: jnp.ndarray,
        config: jnp.ndarray,
        i: int,
    ):
        return -1.0j * config[i] * self.psi_ratio_fn(params, config.at[i].multiply(-1), config)

    @partial(jax.jit, static_argnums=(0,))
    def local_energy_sigma_y(
        self,
        params: jnp.ndarray,
        config: jnp.ndarray,
    ):
        return -jnp.sum(
            jax.vmap(self.local_sigma_y_i, in_axes=(None, None, 0))(
                params,
                config,
                jnp.arange(self.num_spins),
            )
        )

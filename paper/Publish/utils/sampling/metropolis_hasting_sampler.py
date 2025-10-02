import jax
import jax.numpy as jnp
from typing import Callable, Tuple


def build_base_sampler(
    num_spins: int,
) -> Callable[[jax.random.PRNGKey], Tuple[jax.random.PRNGKey, jnp.array]]:

    def base_sampler(
        prngkey: jax.random.PRNGKey,
    ) -> Tuple[jax.random.PRNGKey, jnp.array]:
        """
        Returns a random configuration of spins sampled from the uniform distribution.

        Args:
            prngkey (jax.random.PRNGKey): Random key to generate random numbers.

        Returns:
            Tuple containing:
                jax.random.PRNGKey: Random key to generate random numbers.
                jnp.array: Random configuration of spins.
        """
        prngkey, tempkey = jax.random.split(prngkey)
        return prngkey, 2 * jax.random.bernoulli(tempkey, 0.5, (num_spins,)) - 1

    return base_sampler


def build_update_proposer(
    num_spins: int,
) -> Callable[[jax.random.PRNGKey, jnp.array], Tuple[jax.random.PRNGKey, jnp.array]]:
    def update_proposer(
        prngkey: jax.random.PRNGKey,
        config: jnp.array,
    ) -> Tuple[jax.random.PRNGKey, jnp.array]:
        prngkey, tempkey = jax.random.split(prngkey)
        i = jax.random.randint(tempkey, (), minval=0, maxval=num_spins)
        return prngkey, config.at[i].multiply(-1)

    return update_proposer


def build_mcmc_update_function(
    probability_ratio_function: Callable[[jnp.array, jnp.array, jnp.array], jnp.array],
    update_proposer: Callable[[jax.random.PRNGKey, jnp.array], jnp.array],
) -> Callable[[jax.random.PRNGKey, jnp.array, jnp.array], Tuple[jax.random.PRNGKey, jnp.array]]:
    def mcmc_update_function(
        prngkey: jax.random.PRNGKey,
        params: jnp.ndarray,
        config: jnp.ndarray,
    ) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
        prngkey, proposed_config = update_proposer(prngkey, config)
        acceptance_prob = probability_ratio_function(params, proposed_config, config)
        acceptance_prob = jnp.minimum(1.0, acceptance_prob)
        prngkey, tempkey = jax.random.split(prngkey)
        accepted = jax.random.bernoulli(tempkey, p=acceptance_prob)
        return prngkey, accepted * proposed_config + (1 - accepted) * config

    return jax.jit(mcmc_update_function)


def build_mcmc_multi_update_function(
    mcmc_update_function: Callable[[jax.random.PRNGKey, jnp.array, jnp.array], Tuple[jax.random.PRNGKey, jnp.array]],
    num_updates: int,
) -> Callable[[jax.random.PRNGKey, jnp.array, jnp.array], Tuple[jax.random.PRNGKey, jnp.array]]:
    def mcmc_multi_update_function(
        prngkey: jax.random.PRNGKey,
        params: jnp.ndarray,
        config: jnp.ndarray,
    ) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
        return jax.lax.fori_loop(
            0,
            num_updates,
            lambda _, args: mcmc_update_function(args[0], params, args[1]),
            (prngkey, config),
        )

    return jax.jit(mcmc_multi_update_function)


def build_mcmc_sampler(
    base_sampler: Callable[[jax.random.PRNGKey], jnp.array],
    update_proposer: Callable[[jax.random.PRNGKey, jnp.array], jnp.array],
    probability_ratio_function: Callable[[jnp.array, jnp.array, jnp.array], jnp.array],
    num_samples: int,
    num_thermalization_steps: int,
    num_sweep_steps: int,
    num_chains: int,
) -> Tuple[
    Callable[[jax.random.PRNGKey, jnp.array], Tuple[jnp.array, jnp.array]],
    Callable[[jax.random.PRNGKey, jnp.array, jnp.array], Tuple[jnp.array, jnp.array]],
]:
    num_samples_per_chain = num_samples // num_chains

    mcmc_update_function = build_mcmc_update_function(probability_ratio_function, update_proposer)
    mcmc_thermalization_function = build_mcmc_multi_update_function(mcmc_update_function, num_thermalization_steps)
    mcmc_sweep_function = build_mcmc_multi_update_function(mcmc_update_function, num_sweep_steps)

    num_spins = base_sampler(jax.random.PRNGKey(0))[1].size

    def markov_chain(
        prngkey: jax.random.PRNGKey,
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        prngkey, starting_point = base_sampler(prngkey)
        prngkey, starting_point = mcmc_thermalization_function(prngkey, params, starting_point)
        chain = jnp.zeros((num_samples_per_chain, *starting_point.shape)).at[0].set(starting_point)

        def body_fn(i, args):
            prngkey, chain = args
            prngkey, next_config = mcmc_sweep_function(prngkey, params, chain[i])
            chain = chain.at[i + 1].set(next_config)
            return prngkey, chain

        _, chain = jax.lax.fori_loop(0, num_samples_per_chain - 1, body_fn, (prngkey, chain))

        return chain

    vmapd_markov_chain = jax.vmap(markov_chain, in_axes=(0, None))

    @jax.jit
    def generate_samples(
        prngkey: jax.random.PRNGKey,
        params: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        tempkeys = jax.random.split(prngkey, num_chains)
        chains = vmapd_markov_chain(tempkeys, params)
        samples = chains.reshape(-1, num_spins, order="C")
        end_points = chains[:, -1]
        return samples, end_points

    def markov_chian_from_starting_point(
        prngkey: jax.random.PRNGKey,
        params: jnp.ndarray,
        starting_point: jnp.ndarray,
    ) -> jnp.ndarray:
        prngkey, starting_point = mcmc_sweep_function(prngkey, params, starting_point)
        chain = jnp.zeros((num_samples_per_chain, *starting_point.shape)).at[0].set(starting_point)

        def body_fn(i, args):
            prngkey, chain = args
            prngkey, next_config = mcmc_sweep_function(prngkey, params, chain[i])
            chain = chain.at[i + 1].set(next_config)
            return prngkey, chain

        _, chain = jax.lax.fori_loop(0, num_samples_per_chain - 1, body_fn, (prngkey, chain))

        return chain

    vmapd_markov_chain_from_starting_point = jax.vmap(markov_chian_from_starting_point, in_axes=(0, None, 0))

    @jax.jit
    def generate_samples_from_starting_points(
        prngkey: jax.random.PRNGKey,
        params: jnp.ndarray,
        starting_points: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        tempkeys = jax.random.split(prngkey, num_chains)
        chains = vmapd_markov_chain_from_starting_point(tempkeys, params, starting_points)
        samples = chains.reshape(-1, num_spins, order="C")
        end_points = chains[:, -1]
        return samples, end_points

    return generate_samples, generate_samples_from_starting_points

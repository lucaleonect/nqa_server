import jax
import jax.flatten_util
import jax.numpy as jnp
from typing import Callable, Optional, Tuple, List
from functools import partial


def validate_deep_boltzmann_quantum_state_arguments(
    num_spins: int,
    layers: List[int],
    prngkey: jnp.ndarray,
    num_samples: int,
    num_thermalization_steps: int,
    num_sweep_steps: int,
    num_chains: int,
    use_bias: bool,
):
    """
    Validate the arguments for the constructor of the DeepBoltzmannQuantumState class.

    Args:
        num_spins (int): Number of physical spins in the system.
        layers (List[int]): List of integers representing the number of units in each hidden layer of the spin network.
        prngkey (jnp.ndarray): PRNGKey used for random number generation.
        num_samples (int): Number of samples to generate.
        num_thermalization_steps (int): Number of thermalization steps in the Gibbs chain.
        num_sweep_steps (int): Number of sweep steps in the Gibbs chain.
        num_chains (int): Number of chains to run in parallel.
        use_bias (bool): Whether to use bias terms in the spin network.

    Raises:
        TypeError: If the number of spins is not an integer.
        TypeError: If the layers is not a list of integers.
        TypeError: If the PRNGKey is not a jnp.ndarray object.
        TypeError: If the number of samples is not an integer.
        TypeError: If the number of thermalization steps is not an integer.
        TypeError: If the number of sweep steps is not an integer.
        TypeError: If the number of chains is not an integer.
        TypeError: If use_bias is not a boolean.
        ValueError: If the number of spins is not a positive integer.
        ValueError: If the number of samples is not a positive integer.
        ValueError: If the number of thermalization steps is not a positive integer.
        ValueError: If the number of sweep steps is not a positive integer.
        ValueError: If the number of chains is not a positive integer.
        ValueError: If the number of samples is not divisible by the number of chains.
    """

    if not isinstance(num_spins, int):
        raise TypeError(f"num_spins is expected to be an integer, got {type(num_spins)} instead.")

    if not all(isinstance(layer, int) for layer in layers):
        raise TypeError(f"layers is expected to be a list of integers, got {type(layers)} instead.")

    if not isinstance(prngkey, jnp.ndarray):
        raise TypeError(f"prngkey is expected to be a PRNGKey object, got {type(prngkey)} instead.")

    if not isinstance(num_samples, int):
        raise TypeError(f"num_samples is expected to be an integer, got {type(num_samples)} instead.")

    if not isinstance(num_thermalization_steps, int):
        raise TypeError(
            f"num_thermalization_steps is expected to be an integer, got {type(num_thermalization_steps)} instead."
        )

    if not isinstance(num_sweep_steps, int):
        raise TypeError(f"num_sweep_steps is expected to be an integer, got {type(num_sweep_steps)} instead.")

    if not isinstance(num_chains, int):
        raise TypeError(f"num_chains is expected to be an integer, got {type(num_chains)} instead.")

    if not isinstance(use_bias, bool):
        raise TypeError(f"use_bias is expected to be a boolean, got {type(use_bias)} instead.")

    if not num_spins >= 1:
        raise ValueError(f"num_spins is expected to be a positive integer, got {num_spins} instead.")

    if not num_samples >= 1:
        raise ValueError(f"num_samples is expected to be a positive integer, got {num_samples} instead.")

    if not num_thermalization_steps >= 1:
        raise ValueError(
            f"num_thermalization_steps is expected to be a positive integer, got {num_thermalization_steps} instead."
        )

    if not num_sweep_steps >= 1:
        raise ValueError(f"num_sweep_steps is expected to be a positive integer, got {num_sweep_steps} instead.")

    if not num_chains >= 1:
        raise ValueError(f"num_chains is expected to be a positive integer, got {num_chains} instead.")

    if not num_samples % num_chains == 0:
        raise ValueError(
            f"number of samples is not divisible by the number of chains, got {num_samples} and {num_chains}"
        )


class DeepBoltzmannQuantumState:
    """
    Deep Boltzmann quantum state class.
    Implements block Gibbs sampling to generate samples from the quantum state.
    Details can be found in README.md.
    """

    def __init__(
        self,
        num_spins: int,
        layers: List[int],
        prngkey: jnp.ndarray,
        num_samples: Optional[int] = None,
        num_thermalization_steps: Optional[int] = None,
        num_sweep_steps: Optional[int] = None,
        num_chains: Optional[int] = None,
        dtype: Optional[jnp.dtype] = None,
        use_bias: Optional[bool] = None,
    ):
        """
        Initialize the Deep Boltzmann quantum state.

        Args:
            num_spins (int): Number of spins in the system.
            layers (List[int]): List of integers representing the number of units in each hidden layer of the spin network.
            prngkey (jnp.ndarray): PRNGKey used for random number generation.
            num_samples (int): Number of samples to generate. Default to 2**10.
            num_thermalization_steps (int): Number of thermalization steps in the Gibbs chain. Default to 2**10.
            num_sweep_steps (int): Number of sweep steps in the Gibbs chain. Default to 2**7.
            num_chains (int): Number of chains to run in parallel. Default to 2**8.
            dtype (jnp.dtype): jax.numpy data type to use. Default to jnp.complex128.
            use_bias (bool): Whether to use bias terms in the spin network. Default to True.
        """
        if num_samples is None:
            num_samples = 2**10
        if num_thermalization_steps is None:
            num_thermalization_steps = 2**10
        if num_sweep_steps is None:
            num_sweep_steps = 2**7
        if num_chains is None:
            num_chains = 2**8
        if dtype is None:
            dtype = jnp.complex128
        if use_bias is None:
            use_bias = True

        if num_samples % num_chains != 0:
            num_samples_per_chain = num_samples // num_chains + (1 * (num_samples % num_chains != 0))
            num_samples = num_samples_per_chain * num_chains
            print("Warning: number of samples is not divisible by the number of chains, rounding up to nearest multiple.")
            print(f"num_samples: {num_samples}, num_chains: {num_chains}, num_samples_per_chain: {num_samples_per_chain}")

        try:
            validate_deep_boltzmann_quantum_state_arguments(
                num_spins=num_spins,
                layers=layers,
                prngkey=prngkey,
                num_samples=num_samples,
                num_thermalization_steps=num_thermalization_steps,
                num_sweep_steps=num_sweep_steps,
                num_chains=num_chains,
                use_bias=use_bias,
            )
        except Exception as e:
            raise e

        self.num_spins = num_spins
        self.layers = layers
        self.prngkey = prngkey
        self.num_samples = num_samples
        self.num_thermalization_steps = num_thermalization_steps
        self.num_sweep_steps = num_sweep_steps
        self.num_chains = num_chains
        self.dtype = dtype
        self.use_bias = use_bias

        self.num_samples_per_chain = self.num_samples // self.num_chains
        self.is_holomorphic = True if (dtype == jnp.complex64 or dtype == jnp.complex128) else False
        self.num_units_list = [num_spins] + layers
        self.num_units = sum(self.num_units_list)
        self.dummy_units = [jnp.ones((n,)) for n in self.num_units_list]
        self.dummy_config, self.unravel_config = jax.flatten_util.ravel_pytree(self.dummy_units)
        self.params, self.unravel_params = self.init_params(prngkey)
        self.unravel_config = jax.jit(self.unravel_config)
        self.unravel_params = jax.jit(self.unravel_params)

    def init_params(
        self,
        prngkey: jnp.ndarray,
        scale: float = 1e-1,
    ) -> Tuple[jnp.ndarray, Callable]:
        """
        Initialize the parameters of the quantum state.

        Args:
            prngkey (jnp.ndarray): PRNGKey used for random number generation.
            scale (float): The scale of the random weights.

        Returns:
            Tuple[jnp.ndarray, Callable]: The initial parameters of the quantum state and the unravel function.
        """
        biases = [jnp.zeros((num_units,)) for num_units in self.num_units_list]
        tempkeys = jax.random.split(prngkey, len(self.num_units_list) - 1)
        weights = [
            scale
            * jax.random.normal(
                tempkeys[i],
                (self.num_units_list[i], self.num_units_list[i + 1]),
                dtype=self.dtype,
            )
            / jnp.sqrt(self.num_units_list[i] + self.num_units_list[i + 1])
            for i in range(len(self.num_units_list) - 1)
        ]

        if self.use_bias:
            params, unravel_params = jax.flatten_util.ravel_pytree((biases, weights))
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
            (biases, weights) = self.unravel_params(params)
        else:
            weights = self.unravel_params(params)
            biases = [jnp.zeros((num_units,), dtype=self.dtype) for num_units in self.num_units_list]

        units = self.unravel_config(config)
        bias_terms = jnp.array([unit.T @ bias for unit, bias in zip(units, biases)])
        interaction_terms = jnp.array(
            [unit_a.T @ weight @ unit_b for unit_a, weight, unit_b in zip(units[:-1], weights, units[1:])]
        )
        return jnp.sum(bias_terms) + jnp.sum(interaction_terms)

    # This can be computed more efficiently
    @partial(jax.jit, static_argnums=(0,))
    def psi_ratio_fn(
        self,
        params: jnp.ndarray,
        config_num: jnp.ndarray,
        config_den: jnp.ndarray,
    ):
        return jnp.exp(self.logpsi(params, config_num) - self.logpsi(params, config_den))

    @partial(jax.jit, static_argnums=(0,))
    def local_sigma_xs(
        self,
        params: jnp.ndarray,
        config: jnp.ndarray,
    ):
        # More efficient way to compute the local energy for the sum of the sigma_x operators on the visible spins
        # The computation is much simpler than the general case as we do not need to compute the deeper layers
        # as only the state of the first hidden layers
        # determines the ratio and as the visible units are conditionally independent
        # when given the first layer units
        # We can compute in parallel all the ratios
        if self.use_bias:
            (biases, weights) = self.unravel_params(params)
        else:
            weights = self.unravel_params(params)
            biases = [jnp.zeros((num_units,), dtype=self.dtype) for num_units in self.num_units_list]

        units = self.unravel_config(config)

        visible_biases = biases[0]
        effective_visible_biases = visible_biases + (weights[0] @ units[1])
        visible_spins = units[0]
        return jnp.exp(-2 * visible_spins * effective_visible_biases)

    @partial(jax.jit, static_argnums=(0,))
    def local_sigma_ys(
        self,
        params: jnp.ndarray,
        config: jnp.ndarray,
    ):
        if self.use_bias:
            (biases, weights) = self.unravel_params(params)
        else:
            weights = self.unravel_params(params)
            biases = [jnp.zeros((num_units,), dtype=self.dtype) for num_units in self.num_units_list]

        units = self.unravel_config(config)

        visible_biases = biases[0]
        effective_visible_biases = visible_biases + (weights[0] @ units[1])
        visible_spins = units[0]
        return -1.0j * visible_spins * jnp.exp(-2 * visible_spins * effective_visible_biases)

    @partial(jax.jit, static_argnums=(0,))
    def local_energy_sigma_x(
        self,
        params: jnp.ndarray,
        config: jnp.ndarray,
    ):
        return -jnp.sum(self.local_sigma_xs(params, config))

    @partial(jax.jit, static_argnums=(0,))
    def local_energy_sigma_y(
        self,
        params: jnp.ndarray,
        config: jnp.ndarray,
    ):
        return -jnp.sum(self.local_sigma_ys(params, config))

    @partial(jax.jit, static_argnums=(0,))
    def prob_evens_given_odds(
        self,
        biases,
        weights,
        odd_units,
    ):
        even_biases = biases[::2]
        even_weights, odd_weights = weights[::2], weights[1::2]

        left_interactions = [jnp.zeros(shape=biases[0].shape)] + [u.T @ W for u, W in zip(odd_units, odd_weights)]
        right_interactions = [W @ u for u, W in zip(odd_units, even_weights)]

        if len(self.num_units_list) % 2 == 1:
            right_interactions += [jnp.zeros(shape=biases[-1].shape)]

        preactivations = [
            4 * jnp.real(bias + left + right)
            for bias, left, right in zip(even_biases, left_interactions, right_interactions)
        ]

        return [jax.nn.sigmoid(x) for x in preactivations]

    @partial(jax.jit, static_argnums=(0,))
    def prob_odds_given_evens(
        self,
        biases,
        weights,
        even_units,
    ):
        odd_biases = biases[1::2]
        even_weights, odd_weights = weights[::2], weights[1::2]

        left_interactions = [u.T @ W for u, W in zip(even_units, even_weights)]
        right_interactions = [W @ u for u, W in zip(even_units[1:], odd_weights)]

        if len(self.num_units_list) % 2 == 0:
            right_interactions += [jnp.zeros(shape=biases[-1].shape)]

        preactivations = [
            4 * jnp.real(bias + left + right)
            for bias, left, right in zip(odd_biases, left_interactions, right_interactions)
        ]

        return [jax.nn.sigmoid(x) for x in preactivations]

    def base_sampler(
        self,
        prngkey,
    ):
        prngkey, *tempkeys = jax.random.split(prngkey, len(self.num_units_list) + 1)
        return prngkey, [
            2 * jax.random.bernoulli(tempkeys[i], shape=(self.num_units_list[i],)) - 1
            for i in range(len(self.num_units_list))
        ]

    def gibbs_step(
        self,
        prngkey,
        biases,
        weights,
        units,
    ):
        prngkey, *tempkeys = jax.random.split(prngkey, len(self.num_units_list) + 1)
        p_odds = self.prob_odds_given_evens(biases, weights, units[::2])
        units[1::2] = [2 * jax.random.bernoulli(tempkeys[i], p=p) - 1 for i, p in enumerate(p_odds)]
        p_evens = self.prob_evens_given_odds(biases, weights, units[1::2])
        units[::2] = [2 * jax.random.bernoulli(tempkeys[i], p=p) - 1 for i, p in enumerate(p_evens)]
        return prngkey, units

    def thermalization_fn(
        self,
        prngkey,
        biases,
        weights,
        units,
    ):
        prngkey, units = jax.lax.fori_loop(
            0,
            self.num_thermalization_steps,
            lambda i, args: self.gibbs_step(args[0], biases, weights, args[1]),
            (prngkey, units),
        )
        return prngkey, units

    def sweep_fn(
        self,
        prngkey,
        biases,
        weights,
        units,
    ):
        prngkey, units = jax.lax.fori_loop(
            0,
            self.num_sweep_steps,
            lambda i, args: self.gibbs_step(args[0], biases, weights, args[1]),
            (prngkey, units),
        )
        return prngkey, units

    def gibbs_chain(
        self,
        prngkey,
        biases,
        weights,
    ):
        prngkey, units = self.base_sampler(prngkey)
        prngkey, starting_units = self.thermalization_fn(prngkey, biases, weights, units)
        chain = [
            jnp.zeros(shape=(self.num_samples_per_chain, self.num_units_list[n]))
            .at[0]
            .set(starting_units[n])
            .astype(jnp.int64)
            for n in range(len(self.num_units_list))
        ]

        def fori_func(i, args):
            prngkey, chain = args
            last_units = [_c[i] for _c in chain]
            prngkey, next_units = self.sweep_fn(prngkey, biases, weights, last_units)
            chain = [chain[j].at[i + 1].set(next_units[j]) for j in range(len(chain))]
            return prngkey, chain

        prngkey, chain = jax.lax.fori_loop(0, self.num_samples_per_chain - 1, fori_func, (prngkey, chain))

        return jnp.concatenate(chain, axis=1)

    def gibbs_chain_from_starting_point(
        self,
        prngkey,
        biases,
        weights,
        starting_point,
    ):
        starting_units = self.unravel_config(starting_point)
        prngkey, starting_units = self.sweep_fn(prngkey, biases, weights, starting_units)
        chain = [
            jnp.zeros(shape=(self.num_samples_per_chain, self.num_units_list[n]))
            .at[0]
            .set(starting_units[n])
            .astype(jnp.int64)
            for n in range(len(self.num_units_list))
        ]

        def fori_func(i, args):
            prngkey, chain = args
            last_units = [_c[i] for _c in chain]
            prngkey, next_units = self.sweep_fn(prngkey, biases, weights, last_units)
            chain = [chain[j].at[i + 1].set(next_units[j]) for j in range(len(chain))]
            return prngkey, chain

        prngkey, chain = jax.lax.fori_loop(0, self.num_samples_per_chain - 1, fori_func, (prngkey, chain))

        return jnp.concatenate(chain, axis=1)

    def vmapd_gibbs_chain(
        self,
        prngkeys,
        biases,
        weights,
    ):
        return jax.vmap(self.gibbs_chain, in_axes=(0, None, None))(prngkeys, biases, weights)

    def vmapd_gibbs_chain_from_starting_points(
        self,
        prngkeys,
        biases,
        weights,
        starting_points,
    ):
        return jax.vmap(self.gibbs_chain_from_starting_point, in_axes=(0, None, None, 0))(
            prngkeys, biases, weights, starting_points
        )

    def generate_samples(
        self,
        prngkey,
        params,
    ):
        tempkeys = jax.random.split(prngkey, self.num_chains)
        if self.use_bias:
            (biases, weights) = self.unravel_params(params)
        else:
            weights = self.unravel_params(params)
            biases = [jnp.zeros((num_units,), dtype=self.dtype) for num_units in self.num_units_list]
        chains = self.vmapd_gibbs_chain(tempkeys, biases, weights)
        samples = jnp.reshape(chains, (-1, self.num_units))
        endpoints = chains[:, -1]
        return samples, endpoints

    def update_samples(
        self,
        prngkey,
        params,
        starting_points,
    ):
        tempkeys = jax.random.split(prngkey, self.num_chains)
        if self.use_bias:
            (biases, weights) = self.unravel_params(params)
        else:
            weights = self.unravel_params(params)
            biases = [jnp.zeros((num_units,), dtype=self.dtype) for num_units in self.num_units_list]
        chains = self.vmapd_gibbs_chain_from_starting_points(tempkeys, biases, weights, starting_points)
        samples = jnp.reshape(chains, (-1, self.num_units))
        endpoints = chains[:, -1]
        return samples, endpoints

    def debug_gibbs_chain(
        self,
        prngkey,
        params,
    ):
        """
        Runs a Gibbs chain with no sweeps or thermalization
        """
        tempkeys = jax.random.split(prngkey, self.num_chains)
        if self.use_bias:
            (biases, weights) = self.unravel_params(params)
        else:
            weights = self.unravel_params(params)
            biases = [jnp.zeros((num_units,), dtype=self.dtype) for num_units in self.num_units_list]
        prngkey, starting_units = self.base_sampler(prngkey)
        chain = [
            jnp.zeros(shape=(self.num_samples_per_chain, self.num_units_list[n]))
            .at[0]
            .set(starting_units[n])
            .astype(jnp.int64)
            for n in range(len(self.num_units_list))
        ]

        def fori_func(i, args):
            prngkey, chain = args
            last_units = [_c[i] for _c in chain]
            prngkey, next_units = self.gibbs_step(prngkey, biases, weights, last_units)
            chain = [chain[j].at[i + 1].set(next_units[j]) for j in range(len(chain))]
            return prngkey, chain

        prngkey, chain = jax.lax.fori_loop(0, self.num_samples_per_chain - 1, fori_func, (prngkey, chain))

        return jnp.concatenate(chain, axis=1)

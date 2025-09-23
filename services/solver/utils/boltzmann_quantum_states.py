import jax
import jax.flatten_util
import jax.numpy as jnp
from typing import Callable, Optional, Tuple, List
from functools import partial


class DeepBoltzmannQuantumState:
    """
    Deep Boltzmann quantum state class.

    This class implements a quantum state representation using a deep Boltzmann machine. 
    It provides methods for block Gibbs sampling to generate samples from the quantum state. 

    Attributes:
        num_spins (int): Number of spins in the system.
        hidden_layers (List[int]): List of integers representing the number of units in each hidden layer of the spin network.
        prngkey (jnp.ndarray): PRNGKey used for random number generation.
        num_samples (int): Number of samples to generate.
        num_thermalization_steps (int): Number of thermalization steps in the Gibbs chain.
        num_sweep_steps (int): Number of sweep steps in the Gibbs chain.
        num_chains (int): Number of chains to run in parallel.
        dtype (jnp.dtype): JAX numpy data type to use.
        use_bias (bool): Whether to use bias terms in the spin network.
        num_samples_per_chain (int): Number of samples per chain.
        is_holomorphic (bool): Indicates if the dtype is complex.
        num_units_list (List[int]): List of the number of units in each layer, including the input layer.
        num_units (int): Total number of units across all layers.
        dummy_units (List[jnp.ndarray]): Dummy units for initializing the network.
        dummy_config (jnp.ndarray): Flattened dummy configuration.
        unravel_config (Callable): Function to unflatten configurations.
        params (jnp.ndarray): Flattened parameters of the quantum state.
        unravel_params (Callable): Function to unflatten parameters.
    
    Methods:
        __init__: Initializes the quantum state with the given parameters.
        init_params: Initializes the parameters of the quantum state.
        logpsi: Computes the logarithm of the wavefunction amplitude.
        psi_ratio_fn: Computes the ratio of wavefunction amplitudes for two configurations.
        local_sigma_xs: Computes the local energy for sigma_x operators.
        local_sigma_ys: Computes the local energy for sigma_y operators.
        local_energy_sigma_x: Computes the total local energy for sigma_x operators.
        local_energy_sigma_y: Computes the total local energy for sigma_y operators.
        prob_evens_given_odds: Computes the conditional probabilities of even units given odd units.
        prob_odds_given_evens: Computes the conditional probabilities of odd units given even units.
        base_sampler: Generates initial random samples for the network.
        gibbs_step: Performs a single Gibbs sampling step.
        thermalization_fn: Performs thermalization steps for the Gibbs chain.
        sweep_fn: Performs sweep steps for the Gibbs chain.
        gibbs_chain: Runs a full Gibbs sampling chain.
        gibbs_chain_from_starting_point: Runs a Gibbs chain starting from a given configuration.
        vmapd_gibbs_chain: Runs multiple Gibbs chains in parallel.
        vmapd_gibbs_chain_from_starting_points: Runs multiple Gibbs chains in parallel from given starting points.
        generate_samples: Generates samples from the quantum state.
        update_samples: Updates samples starting from given configurations.
        debug_gibbs_chain: Runs a Gibbs chain with no sweeps or thermalization for debugging purposes.
    """

    def __init__(
        self,
        num_spins: int,
        hidden_layers: List[int],
        prngkey: jnp.ndarray,
        num_samples: Optional[int] = None,
        num_thermalization_steps: Optional[int] = None,
        num_sweep_steps: Optional[int] = None,
        num_chains: Optional[int] = None,
        dtype: Optional[jnp.dtype] = None,
        use_bias: Optional[bool] = None,
        initial_params_gain: Optional[float] = None,
    ):
        """
        Initialize the Deep Boltzmann quantum state.

        Args:
            num_spins (int): Number of spins in the system.
            hidden_layers (List[int]): List of integers representing the number of units in each hidden layer of the spin network.
            prngkey (jnp.ndarray): PRNGKey used for random number generation.
            num_samples (Optional[int]): Number of samples to generate. Defaults to 2**10.
            num_thermalization_steps (Optional[int]): Number of thermalization steps in the Gibbs chain. Defaults to 2**10.
            num_sweep_steps (Optional[int]): Number of sweep steps in the Gibbs chain. Defaults to 2**7.
            num_chains (Optional[int]): Number of chains to run in parallel. Defaults to 2**8.
            dtype (Optional[jnp.dtype]): JAX numpy data type to use. Defaults to jnp.complex128.
            use_bias (Optional[bool]): Whether to use bias terms in the spin network. Defaults to True.
            initial_params_gain (Optional[float]): Scaling factor for initializing the parameters. Defaults to 1e-1.
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
        if initial_params_gain is None:
            initial_params_gain = 1e-1

        if num_samples % num_chains != 0:
            num_samples_per_chain = num_samples // num_chains + (1 * (num_samples % num_chains != 0))
            num_samples = num_samples_per_chain * num_chains
            print("Warning: number of samples is not divisible by the number of chains, rounding up to nearest multiple.")
            print(f"num_samples: {num_samples}, num_chains: {num_chains}, num_samples_per_chain: {num_samples_per_chain}")

        self.num_spins = num_spins
        self.hidden_layers = hidden_layers
        self.prngkey = prngkey
        self.num_samples = num_samples
        self.num_thermalization_steps = num_thermalization_steps
        self.num_sweep_steps = num_sweep_steps
        self.num_chains = num_chains
        self.dtype = dtype
        self.use_bias = use_bias
        self.initial_params_gain = initial_params_gain

        self.num_samples_per_chain = self.num_samples // self.num_chains
        self.is_holomorphic = True if (dtype == jnp.complex64 or dtype == jnp.complex128) else False
        self.num_units_list = [num_spins] + hidden_layers
        self.num_units = sum(self.num_units_list)
        self.dummy_units = [jnp.ones((n,)) for n in self.num_units_list]
        self.dummy_config, self.unravel_config = jax.flatten_util.ravel_pytree(self.dummy_units)
        self.params, self.unravel_params = self.init_params(prngkey)
        self.unravel_config = jax.jit(self.unravel_config)
        self.unravel_params = jax.jit(self.unravel_params)

        self.validate_inputs()  # Validate inputs during initialization

    def init_params(
        self,
        prngkey: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Callable]:
        """
        Initialize the parameters of the quantum state.

        Args:
            prngkey (jnp.ndarray): PRNGKey used for random number generation.
            scale (float): The scale of the random weights. Defaults to 1e-1.

        Returns:
            Tuple[jnp.ndarray, Callable]: The initial parameters of the quantum state and the unravel function.
        """
        biases = [jnp.zeros((num_units,), dtype=self.dtype) for num_units in self.num_units_list]
        tempkeys = jax.random.split(prngkey, len(self.num_units_list) - 1)
        weights = [
            self.initial_params_gain
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
        params: jnp.ndarray,
        config: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the logarithm of the wavefunction amplitude for a given configuration.

        Args:
            params (jnp.ndarray): Parameters of the quantum state.
            config (jnp.ndarray): Configuration of the quantum state.

        Returns:
            jnp.ndarray: Logarithm of the wavefunction amplitude.
        """
        if self.use_bias:
            (biases, weights) = self.unravel_params(params)
        else:
            weights = self.unravel_params(params)
            biases = [jnp.zeros((numUnits,), dtype=self.dtype) for numUnits in self.num_units_list]

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
    ) -> jnp.ndarray:
        """
        Compute the ratio of wavefunction amplitudes for two configurations.

        Args:
            params (jnp.ndarray): Parameters of the quantum state.
            config_num (jnp.ndarray): Numerator configuration.
            config_den (jnp.ndarray): Denominator configuration.

        Returns:
            jnp.ndarray: Ratio of wavefunction amplitudes.
        """
        return jnp.exp(self.logpsi(params, config_num) - self.logpsi(params, config_den))

    @partial(jax.jit, static_argnums=(0,))
    def local_sigma_xs(
        self,
        params: jnp.ndarray,
        config: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the local energy for sigma_x operators on the visible spins.

        Args:
            params (jnp.ndarray): Parameters of the quantum state.
            config (jnp.ndarray): Configuration of the quantum state.

        Returns:
            jnp.ndarray: Local energy for sigma_x operators.
        """
        # More efficient way to compute the local energy for the sum of the sigma_x operators on the visible spins
        # The computation is much simpler than the general case as we do not need to compute the deeper hidden_layers
        # as only the state of the first hidden hidden_layers
        # determines the ratio and as the visible units are conditionally independent
        # when given the first layer units
        # We can compute in parallel all the ratios
        if self.use_bias:
            (biases, weights) = self.unravel_params(params)
        else:
            weights = self.unravel_params(params)
            biases = [jnp.zeros((numUnits,), dtype=self.dtype) for numUnits in self.num_units_list]

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
    ) -> jnp.ndarray:
        """
        Compute the local energy for sigma_y operators on the visible spins.

        Args:
            params (jnp.ndarray): Parameters of the quantum state.
            config (jnp.ndarray): Configuration of the quantum state.

        Returns:
            jnp.ndarray: Local energy for sigma_y operators.
        """
        if self.use_bias:
            (biases, weights) = self.unravel_params(params)
        else:
            weights = self.unravel_params(params)
            biases = [jnp.zeros((numUnits,), dtype=self.dtype) for numUnits in self.num_units_list]

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
    ) -> jnp.ndarray:
        """
        Compute the total local energy for sigma_x operators.

        Args:
            params (jnp.ndarray): Parameters of the quantum state.
            config (jnp.ndarray): Configuration of the quantum state.

        Returns:
            jnp.ndarray: Total local energy for sigma_x operators.
        """
        return -jnp.sum(self.local_sigma_xs(params, config))

    @partial(jax.jit, static_argnums=(0,))
    def local_energy_sigma_y(
        self,
        params: jnp.ndarray,
        config: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the total local energy for sigma_y operators.

        Args:
            params (jnp.ndarray): Parameters of the quantum state.
            config (jnp.ndarray): Configuration of the quantum state.

        Returns:
            jnp.ndarray: Total local energy for sigma_y operators.
        """
        return -jnp.sum(self.local_sigma_ys(params, config))

    @partial(jax.jit, static_argnums=(0,))
    def prob_evens_given_odds(
        self,
        biases: List[jnp.ndarray],
        weights: List[jnp.ndarray],
        odd_units: List[jnp.ndarray],
    ) -> List[jnp.ndarray]:
        """
        Compute the conditional probabilities of even units given odd units.

        Args:
            biases (List[jnp.ndarray]): Biases for each layer.
            weights (List[jnp.ndarray]): Weights between layers.
            odd_units (List[jnp.ndarray]): Odd-layer units.

        Returns:
            List[jnp.ndarray]: Probabilities of even units given odd units.
        """
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
        biases: List[jnp.ndarray],
        weights: List[jnp.ndarray],
        even_units: List[jnp.ndarray],
    ) -> List[jnp.ndarray]:
        """
        Compute the conditional probabilities of odd units given even units.

        Args:
            biases (List[jnp.ndarray]): Biases for each layer.
            weights (List[jnp.ndarray]): Weights between layers.
            even_units (List[jnp.ndarray]): Even-layer units.

        Returns:
            List[jnp.ndarray]: Probabilities of odd units given even units.
        """
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
        prngkey: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """
        Generate initial random samples for the network.

        Args:
            prngkey (jnp.ndarray): PRNGKey used for random number generation.

        Returns:
            Tuple[jnp.ndarray, List[jnp.ndarray]]: Updated PRNGKey and initial samples.
        """
        prngkey, *tempkeys = jax.random.split(prngkey, len(self.num_units_list) + 1)
        return prngkey, [
            2 * jax.random.bernoulli(tempkeys[i], shape=(self.num_units_list[i],)) - 1
            for i in range(len(self.num_units_list))
        ]

    def gibbs_step(
        self,
        prngkey: jnp.ndarray,
        biases: List[jnp.ndarray],
        weights: List[jnp.ndarray],
        units: List[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """
        Perform a single Gibbs sampling step.

        Args:
            prngkey (jnp.ndarray): PRNGKey used for random number generation.
            biases (List[jnp.ndarray]): Biases for each layer.
            weights (List[jnp.ndarray]): Weights between layers.
            units (List[jnp.ndarray]): Current state of the units.

        Returns:
            Tuple[jnp.ndarray, List[jnp.ndarray]]: Updated PRNGKey and new state of the units.
        """
        prngkey, *tempkeys = jax.random.split(prngkey, len(self.num_units_list) + 1)
        p_odds = self.prob_odds_given_evens(biases, weights, units[::2])
        units[1::2] = [2 * jax.random.bernoulli(tempkeys[i], p=p) - 1 for i, p in enumerate(p_odds)]
        p_evens = self.prob_evens_given_odds(biases, weights, units[1::2])
        units[::2] = [2 * jax.random.bernoulli(tempkeys[i], p=p) - 1 for i, p in enumerate(p_evens)]
        return prngkey, units

    def thermalization_fn(
        self,
        prngkey: jnp.ndarray,
        biases: List[jnp.ndarray],
        weights: List[jnp.ndarray],
        units: List[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """
        Perform thermalization steps for the Gibbs chain.

        Args:
            prngkey (jnp.ndarray): PRNGKey used for random number generation.
            biases (List[jnp.ndarray]): Biases for each layer.
            weights (List[jnp.ndarray]): Weights between layers.
            units (List[jnp.ndarray]): Current state of the units.

        Returns:
            Tuple[jnp.ndarray, List[jnp.ndarray]]: Updated PRNGKey and thermalized state of the units.
        """
        prngkey, units = jax.lax.fori_loop(
            0,
            self.num_thermalization_steps,
            lambda i, args: self.gibbs_step(args[0], biases, weights, args[1]),
            (prngkey, units),
        )
        return prngkey, units

    def sweep_fn(
        self,
        prngkey: jnp.ndarray,
        biases: List[jnp.ndarray],
        weights: List[jnp.ndarray],
        units: List[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """
        Perform sweep steps for the Gibbs chain.

        Args:
            prngkey (jnp.ndarray): PRNGKey used for random number generation.
            biases (List[jnp.ndarray]): Biases for each layer.
            weights (List[jnp.ndarray]): Weights between layers.
            units (List[jnp.ndarray]): Current state of the units.

        Returns:
            Tuple[jnp.ndarray, List[jnp.ndarray]]: Updated PRNGKey and new state of the units after sweeps.
        """
        prngkey, units = jax.lax.fori_loop(
            0,
            self.num_sweep_steps,
            lambda i, args: self.gibbs_step(args[0], biases, weights, args[1]),
            (prngkey, units),
        )
        return prngkey, units

    def gibbs_chain(
        self,
        prngkey: jnp.ndarray,
        biases: List[jnp.ndarray],
        weights: List[jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Run a full Gibbs sampling chain.

        Args:
            prngkey (jnp.ndarray): PRNGKey used for random number generation.
            biases (List[jnp.ndarray]): Biases for each layer.
            weights (List[jnp.ndarray]): Weights between layers.

        Returns:
            jnp.ndarray: Generated samples from the Gibbs chain.
        """
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
        prngkey: jnp.ndarray,
        biases: List[jnp.ndarray],
        weights: List[jnp.ndarray],
        starting_point: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Run a Gibbs chain starting from a given configuration.

        Args:
            prngkey (jnp.ndarray): PRNGKey used for random number generation.
            biases (List[jnp.ndarray]): Biases for each layer.
            weights (List[jnp.ndarray]): Weights between layers.
            starting_point (jnp.ndarray): Initial configuration for the Gibbs chain.

        Returns:
            jnp.ndarray: Generated samples from the Gibbs chain.
        """
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
        prngkeys: jnp.ndarray,
        biases: List[jnp.ndarray],
        weights: List[jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Run multiple Gibbs chains in parallel.

        Args:
            prngkeys (jnp.ndarray): Array of PRNGKeys for parallel chains.
            biases (List[jnp.ndarray]): Biases for each layer.
            weights (List[jnp.ndarray]): Weights between layers.

        Returns:
            jnp.ndarray: Generated samples from the parallel Gibbs chains.
        """
        return jax.vmap(self.gibbs_chain, in_axes=(0, None, None))(prngkeys, biases, weights)

    def vmapd_gibbs_chain_from_starting_points(
        self,
        prngkeys: jnp.ndarray,
        biases: List[jnp.ndarray],
        weights: List[jnp.ndarray],
        starting_points: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Run multiple Gibbs chains in parallel from given starting points.

        Args:
            prngkeys (jnp.ndarray): Array of PRNGKeys for parallel chains.
            biases (List[jnp.ndarray]): Biases for each layer.
            weights (List[jnp.ndarray]): Weights between layers.
            starting_points (jnp.ndarray): Initial configurations for the parallel Gibbs chains.

        Returns:
            jnp.ndarray: Generated samples from the parallel Gibbs chains.
        """
        return jax.vmap(self.gibbs_chain_from_starting_point, in_axes=(0, None, None, 0))(
            prngkeys, biases, weights, starting_points
        )

    def generate_samples(
        self,
        prngkey: jnp.ndarray,
        params: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate samples from the quantum state.

        Args:
            prngkey (jnp.ndarray): PRNGKey used for random number generation.
            params (jnp.ndarray): Parameters of the quantum state.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Generated samples and their endpoints.
        """
        tempkeys = jax.random.split(prngkey, self.num_chains)
        if self.use_bias:
            (biases, weights) = self.unravel_params(params)
        else:
            weights = self.unravel_params(params)
            biases = [jnp.zeros((numUnits,), dtype=self.dtype) for numUnits in self.num_units_list]
        chains = self.vmapd_gibbs_chain(tempkeys, biases, weights)
        samples = jnp.reshape(chains, (-1, self.num_units))
        endpoints = chains[:, -1]
        return samples, endpoints

    def update_samples(
        self,
        prngkey: jnp.ndarray,
        params: jnp.ndarray,
        starting_points: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Update samples starting from given configurations.

        Args:
            prngkey (jnp.ndarray): PRNGKey used for random number generation.
            params (jnp.ndarray): Parameters of the quantum state.
            starting_points (jnp.ndarray): Initial configurations for the samples.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Updated samples and their endpoints.
        """
        tempkeys = jax.random.split(prngkey, self.num_chains)
        if self.use_bias:
            (biases, weights) = self.unravel_params(params)
        else:
            weights = self.unravel_params(params)
            biases = [jnp.zeros((numUnits,), dtype=self.dtype) for numUnits in self.num_units_list]
        chains = self.vmapd_gibbs_chain_from_starting_points(tempkeys, biases, weights, starting_points)
        samples = jnp.reshape(chains, (-1, self.num_units))
        endpoints = chains[:, -1]
        return samples, endpoints

    def debug_gibbs_chain(
        self,
        prngkey: jnp.ndarray,
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Run a Gibbs chain with no sweeps or thermalization for debugging purposes.

        Args:
            prngkey (jnp.ndarray): PRNGKey used for random number generation.
            params (jnp.ndarray): Parameters of the quantum state.

        Returns:
            jnp.ndarray: Generated samples from the debug Gibbs chain.
        """
        """
        Runs a Gibbs chain with no sweeps or thermalization
        """
        tempkeys = jax.random.split(prngkey, self.num_chains)
        if self.use_bias:
            (biases, weights) = self.unravel_params(params)
        else:
            weights = self.unravel_params(params)
            biases = [jnp.zeros((numUnits,), dtype=self.dtype) for numUnits in self.num_units_list]
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

    def validate_inputs(self):
        """
        Validate the inputs provided to the DeepBoltzmannQuantumState class.

        Raises:
            ValueError: If any of the inputs are invalid.
        """
        if not isinstance(self.num_spins, int) or self.num_spins <= 0:
            raise ValueError("num_spins must be a positive integer.")

        if not isinstance(self.hidden_layers, list) or not all(isinstance(x, int) and x > 0 for x in self.hidden_layers):
            raise ValueError("hidden_layers must be a list of positive integers.")

        if not isinstance(self.prngkey, jnp.ndarray):
            raise ValueError("prngkey must be a JAX ndarray.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")

        if not isinstance(self.num_thermalization_steps, int) or self.num_thermalization_steps <= 0:
            raise ValueError("num_thermalization_steps must be a positive integer.")

        if not isinstance(self.num_sweep_steps, int) or self.num_sweep_steps <= 0:
            raise ValueError("num_sweep_steps must be a positive integer.")

        if not isinstance(self.num_chains, int) or self.num_chains <= 0:
            raise ValueError("num_chains must be a positive integer.")

        if self.dtype not in (jnp.float64, jnp.complex128):
            raise ValueError("dtype must be one of jnp.float64 or jnp.complex128.")

        if not isinstance(self.use_bias, bool):
            raise ValueError("use_bias must be a boolean value.")

        if not isinstance(self.initial_params_gain, (float, int)) or self.initial_params_gain <= 0:
            raise ValueError("initial_params_gain must be a positive float or integer.")

        if self.num_samples % self.num_chains != 0:
            raise ValueError("num_samples must be divisible by num_chains.")

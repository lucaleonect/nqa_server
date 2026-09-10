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
        is_holomorphic (bool): Whether logpsi is holomorphic in its flattened parameters.
        complex_output (bool): Whether amplitudes include trainable phases.
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
        visible_interactions: bool = False,
        visible_rank: Optional[int] = None,
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
            visible_interactions (bool): Add x.T @ (J @ J.T + 1j*K) @ x on the
                visible layer. All trainable coordinates are real in this mode;
                complex dtype enables separate real/imaginary network parts and K.
            visible_rank (Optional[int]): Number of Gaussian auxiliary fields
                (columns of J). Defaults to num_spins when interactions are enabled.
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

        if not isinstance(visible_interactions, bool):
            raise ValueError("visible_interactions must be a boolean.")
        if visible_rank is not None:
            if not visible_interactions:
                raise ValueError("visible_rank requires visible_interactions=True.")
            if isinstance(visible_rank, bool) or not isinstance(visible_rank, int) or visible_rank <= 0:
                raise ValueError("visible_rank must be a positive integer.")

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
        self.visible_interactions = visible_interactions
        self.visible_rank = (num_spins if visible_rank is None else visible_rank) if visible_interactions else 0
        self.complex_output = jnp.issubdtype(dtype, jnp.complexfloating)

        self.num_samples_per_chain = self.num_samples // self.num_chains
        self.is_holomorphic = bool(self.complex_output and not visible_interactions)
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

        network = (biases, weights) if self.use_bias else weights
        if self.visible_interactions:
            real_dtype = jnp.real(jnp.zeros((), dtype=self.dtype)).dtype
            # J=0 is a stationary point of J J.T. Use a nonzero initialization
            # so SR can learn amplitude interactions from the first update.
            factor_key = jax.random.fold_in(prngkey, len(self.num_units_list))
            tree = {
                "real": jax.tree.map(jnp.real, network),
                "J": self.initial_params_gain * jax.random.normal(
                    factor_key, (self.num_spins, self.visible_rank), dtype=real_dtype
                ) / jnp.sqrt(self.num_spins + self.visible_rank),
            }
            if self.complex_output:
                tree["imag"] = jax.tree.map(jnp.imag, network)
                tree["K"] = jnp.zeros((self.num_spins, self.num_spins), dtype=real_dtype)
            network = tree
        params, unravel_params = jax.flatten_util.ravel_pytree(network)

        return params, unravel_params

    def unpack_params(self, params):
        """Return (biases, weights, J, K), with absent visible terms set to None.

        The legacy parameter tree is unchanged. With visible interactions, the
        tree has real leaves: J, real, and (for complex output) K and imag.
        The real/imag network trees retain the legacy bias/weight structure.
        """
        network = self.unravel_params(params)
        factor = phase = None
        if self.visible_interactions:
            factor = network["J"]
            if self.complex_output:
                phase = network["K"]
                network = jax.tree.map(lambda r, i: r + 1j * i, network["real"], network["imag"])
            else:
                network = network["real"]
        if self.use_bias:
            biases, weights = network
        else:
            weights = network
            biases = [jnp.zeros((n,), dtype=self.dtype) for n in self.num_units_list]
        return biases, weights, factor, phase

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
        biases, weights, factor, phase = self.unpack_params(params)

        units = self.unravel_config(config)
        bias_terms = jnp.array([unit.T @ bias for unit, bias in zip(units, biases)])
        interaction_terms = jnp.array(
            [unit_a.T @ weight @ unit_b for unit_a, weight, unit_b in zip(units[:-1], weights, units[1:])]
        )
        value = jnp.sum(bias_terms) + jnp.sum(interaction_terms)
        if factor is not None:
            projected = units[0] @ factor
            value += projected @ projected
        if phase is not None:
            value += 1j * (units[0] @ phase @ units[0])
        return value

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
        biases, weights, factor, phase = self.unpack_params(params)
        units = self.unravel_config(config)
        effective_visible_biases = biases[0]
        if weights:
            effective_visible_biases = effective_visible_biases + weights[0] @ units[1]
        visible_spins = units[0]
        log_ratios = -2 * visible_spins * effective_visible_biases
        if factor is not None:
            # The diagonal of J J.T contributes only a constant since x_i^2=1.
            log_ratios += -4 * visible_spins * (factor @ (factor.T @ visible_spins))
            log_ratios += 4 * jnp.sum(factor * factor, axis=1)
        if phase is not None:
            # Valid for nonsymmetric K too; its antisymmetric part cancels.
            log_ratios += -2j * visible_spins * ((phase + phase.T) @ visible_spins)
            log_ratios += 4j * jnp.diag(phase)
        return jnp.exp(log_ratios)

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
        visible_spins = config[:self.num_spins]
        return -1.0j * visible_spins * self.local_sigma_xs(params, config)

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
        visible_factor: Optional[jnp.ndarray] = None,
        auxiliary_fields: Optional[jnp.ndarray] = None,
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
        if self.visible_interactions and visible_factor is None:
            raise ValueError("Visible interactions require visible_factor in Gibbs conditionals.")
        if visible_factor is not None:
            if auxiliary_fields is None:
                raise ValueError("Visible Gibbs probabilities require auxiliary_fields.")
            # The existing preactivation multiplies by four; J z enters the
            # augmented log density once, giving a logit contribution 2 J z.
            left_interactions[0] = 0.5 * (visible_factor @ auxiliary_fields)
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

    @partial(jax.jit, static_argnums=(0,))
    def sample_auxiliary_fields(self, prngkey, visible_factor, visible_spins):
        """Draw z | x ~ N(4 J.T x, 4 I); z is a sampling variable only."""
        mean = 4 * (visible_factor.T @ visible_spins)
        return mean + 2 * jax.random.normal(prngkey, mean.shape, dtype=visible_factor.dtype)

    @partial(jax.jit, static_argnums=(0,))
    def gibbs_step(
        self,
        prngkey: jnp.ndarray,
        biases: List[jnp.ndarray],
        weights: List[jnp.ndarray],
        units: List[jnp.ndarray],
        visible_factor: Optional[jnp.ndarray] = None,
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
        new_units = units[::]
        auxiliary_fields = None
        if visible_factor is not None:
            prngkey, auxiliary_key = jax.random.split(prngkey)
            auxiliary_fields = self.sample_auxiliary_fields(auxiliary_key, visible_factor, units[0])
        prngkey, *tempkeys = jax.random.split(prngkey, len(self.num_units_list) + 1)
        p_odds = self.prob_odds_given_evens(biases, weights, new_units[::2])
        new_odds = [2 * jax.random.bernoulli(tempkeys[i], p=p) - 1 for i, p in enumerate(p_odds)]
        new_units[1::2] = new_odds

        prngkey, *tempkeys = jax.random.split(prngkey, len(self.num_units_list) + 1)
        p_evens = self.prob_evens_given_odds(
            biases, weights, new_units[1::2], visible_factor, auxiliary_fields
        )
        new_evens = [2 * jax.random.bernoulli(tempkeys[i], p=p) - 1 for i, p in enumerate(p_evens)]
        new_units[::2] = new_evens
        return prngkey, new_units

    def thermalization_fn(
        self,
        prngkey: jnp.ndarray,
        biases: List[jnp.ndarray],
        weights: List[jnp.ndarray],
        units: List[jnp.ndarray],
        visible_factor: Optional[jnp.ndarray] = None,
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
            lambda i, args: self.gibbs_step(args[0], biases, weights, args[1], visible_factor),
            (prngkey, units),
        )
        return prngkey, units

    def sweep_fn(
        self,
        prngkey: jnp.ndarray,
        biases: List[jnp.ndarray],
        weights: List[jnp.ndarray],
        units: List[jnp.ndarray],
        visible_factor: Optional[jnp.ndarray] = None,
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
            lambda i, args: self.gibbs_step(args[0], biases, weights, args[1], visible_factor),
            (prngkey, units),
        )
        return prngkey, units

    def gibbs_chain(
        self,
        prngkey: jnp.ndarray,
        biases: List[jnp.ndarray],
        weights: List[jnp.ndarray],
        visible_factor: Optional[jnp.ndarray] = None,
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
        prngkey, starting_units = self.thermalization_fn(prngkey, biases, weights, units, visible_factor)
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
            prngkey, next_units = self.sweep_fn(prngkey, biases, weights, last_units, visible_factor)
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
        visible_factor: Optional[jnp.ndarray] = None,
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
        prngkey, starting_units = self.sweep_fn(prngkey, biases, weights, starting_units, visible_factor)
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
            prngkey, next_units = self.sweep_fn(prngkey, biases, weights, last_units, visible_factor)
            chain = [chain[j].at[i + 1].set(next_units[j]) for j in range(len(chain))]
            return prngkey, chain

        prngkey, chain = jax.lax.fori_loop(0, self.num_samples_per_chain - 1, fori_func, (prngkey, chain))

        return jnp.concatenate(chain, axis=1)

    def vmapd_gibbs_chain(
        self,
        prngkeys: jnp.ndarray,
        biases: List[jnp.ndarray],
        weights: List[jnp.ndarray],
        visible_factor: Optional[jnp.ndarray] = None,
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
        return jax.vmap(self.gibbs_chain, in_axes=(0, None, None, None))(
            prngkeys, biases, weights, visible_factor
        )

    def vmapd_gibbs_chain_from_starting_points(
        self,
        prngkeys: jnp.ndarray,
        biases: List[jnp.ndarray],
        weights: List[jnp.ndarray],
        starting_points: jnp.ndarray,
        visible_factor: Optional[jnp.ndarray] = None,
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
        return jax.vmap(self.gibbs_chain_from_starting_point, in_axes=(0, None, None, 0, None))(
            prngkeys, biases, weights, starting_points, visible_factor
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

        Gaussian fields are redrawn before each visible update and then
        discarded. Samples and endpoints contain only visible/hidden spins;
        logpsi and physical estimators must use the marginalized amplitude.
        """
        tempkeys = jax.random.split(prngkey, self.num_chains)
        biases, weights, visible_factor, _ = self.unpack_params(params)
        chains = self.vmapd_gibbs_chain(tempkeys, biases, weights, visible_factor)
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
        biases, weights, visible_factor, _ = self.unpack_params(params)
        chains = self.vmapd_gibbs_chain_from_starting_points(
            tempkeys, biases, weights, starting_points, visible_factor
        )
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
        biases, weights, visible_factor, _ = self.unpack_params(params)
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
            prngkey, next_units = self.gibbs_step(prngkey, biases, weights, last_units, visible_factor)
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

        if not isinstance(self.dtype, type(jnp.float64)) and not isinstance(self.dtype, type(jnp.complex128)):
            raise ValueError("dtype must be a supported JAX dtype: jnp.float64 or jnp.complex128.")

        if not isinstance(self.use_bias, bool):
            raise ValueError("use_bias must be a boolean value.")

        if not isinstance(self.initial_params_gain, (float, int)) or self.initial_params_gain <= 0:
            raise ValueError("initial_params_gain must be a positive float or integer.")

        if self.num_samples % self.num_chains != 0:
            raise ValueError("num_samples must be divisible by num_chains.")

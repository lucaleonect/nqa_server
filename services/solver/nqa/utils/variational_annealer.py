import jax
import jax.numpy as jnp
import optax
import time
from tqdm import tqdm
from functools import partial
from typing import Callable, Optional, Mapping, Tuple, Dict, Any
from utils.boltzmann_quantum_states import DeepBoltzmannQuantumState


class VariationalAnnealer:
    """Optimization loop for variational quantum states with a parametric schedule.

    Parameters:
        variational_quantum_state: The sampler/state object that provides MCMC sampling
            via `generate_samples` and `update_samples` and holds the current `params`.
        parametric_gradient_estimator: A callable implementing the parametric TDVP
            estimator. It must accept `(params, mcmc_samples, couplings)` and return
            `(grads, avg_energy, energy_var)` where `grads` has the same shape as
            `params`.
        optimizer: An Optax optimizer transformation providing `.init` and `.update`.
        annealing_schedule: Array of couplings over iterations. Shape `(T, 2)` for
            target+annealing or `(T, 3)` if a catalyst is present.
        num_warmup_steps: Number of initial repetitions of the first schedule point.
        num_updates_per_step: Number of repeats for interior schedule points.
        num_finetuning_steps: Number of final repetitions of the last schedule point.
        persistent_chains: If True, uses endpoints of previous MCMC as starting points.
        observables_dict: Mapping from names to callables `f(params, samples) -> Any`.
        use_tqdm: Whether to wrap the schedule with a tqdm progress bar.
        log_every: Log cadence (iterations). If None, a default proportional to total
            iterations is used; it is always clamped to at least 1.
        log_params: If True, logs model parameters at the chosen cadence.

    Raises:
        ValueError: If provided inputs are of invalid type, shape, or contain NaNs.
    """

    def __init__(
        self,
        variational_quantum_state: DeepBoltzmannQuantumState,
        parametric_gradient_estimator: Callable[
            [jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ],
        optimizer: optax.GradientTransformation,
        annealing_schedule: jnp.ndarray,
        num_warmup_steps: Optional[int] = None,
        num_updates_per_step: Optional[int] = None,
        num_finetuning_steps: Optional[int] = None,
        persistent_chains: Optional[bool] = None,
        observables_dict: Optional[Mapping[str, Callable[[jnp.ndarray, jnp.ndarray], Any]]] = None,
        use_tqdm: Optional[bool] = None,
        log_every: Optional[int] = None,
        log_params: Optional[bool] = None,
        num_replicas: Optional[int] = None,
    ) -> None:
        if persistent_chains is None:
            persistent_chains = True
        if observables_dict is None:
            observables_dict = {}
        if num_warmup_steps is None:
            num_warmup_steps = 1
        if num_updates_per_step is None:
            num_updates_per_step = 1
        if num_finetuning_steps is None:
            num_finetuning_steps = 1
        if use_tqdm is None:
            use_tqdm = False
        if log_params is None:
            log_params = False
        if num_replicas is None:
            num_replicas = 1

        # Validate basic inputs before materializing the schedule
        self._validate_constructor_inputs(
            variational_quantum_state,
            parametric_gradient_estimator,
            optimizer,
            annealing_schedule,
            num_warmup_steps,
            num_updates_per_step,
            num_finetuning_steps,
            persistent_chains,
            observables_dict,
            use_tqdm,
            log_params,
            num_replicas,
        )

        # Build expanded schedule with warmup/updates/finetuning repeats
        annealing_schedule = jnp.asarray(annealing_schedule)
        expanded_schedule = jnp.concatenate(
            (
                jnp.repeat(annealing_schedule[:1], num_warmup_steps, axis=0),
                jnp.repeat(annealing_schedule[1:-1], num_updates_per_step, axis=0),
                jnp.repeat(annealing_schedule[-1:], num_finetuning_steps, axis=0),
            ),
            axis=0,
        )

        if log_every is None:
            # Default to roughly ~100 logged entries over the whole run (at least 1)
            log_every = max(int(len(expanded_schedule) // 100) or 1, 1)
        else:
            if not isinstance(log_every, int) or log_every <= 0:
                raise ValueError("log_every must be a positive integer")

        self.variational_quantum_state = variational_quantum_state
        self.parametric_gradient_estimator = parametric_gradient_estimator
        self.optimizer = optimizer
        self.annealing_schedule = expanded_schedule
        self.persistent_chains = persistent_chains
        self.observables_dict = observables_dict
        self.num_warmup_steps = num_warmup_steps
        self.num_updates_per_step = num_updates_per_step
        self.num_finetuning_steps = num_finetuning_steps
        self.use_tqdm = use_tqdm
        self.log_every = log_every
        self.log_params = log_params
        self.num_replicas = num_replicas

        # Catalyst flag inferred from schedule width (2 or 3 columns)
        if self.annealing_schedule.ndim != 2 or self.annealing_schedule.shape[1] not in (2, 3):
            raise ValueError("annealing_schedule must have shape (T, 2) or (T, 3)")
        self.uses_catalyst = self.annealing_schedule.shape[1] == 3

    def _validate_constructor_inputs(
        self,
        vqs: DeepBoltzmannQuantumState,
        estimator: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
        optimizer: optax.GradientTransformation,
        schedule: jnp.ndarray,
        warmup: int,
        updates_per: int,
        finetune: int,
        persistent: bool,
        observables: Mapping[str, Callable[[jnp.ndarray, jnp.ndarray], Any]],
        use_tqdm_flag: bool,
        log_params_flag: bool,
        num_replicas: int = 1,
    ) -> None:
        """Validate constructor inputs for type/shape consistency.

        Raises ValueError if any check fails.
        """
        if not isinstance(vqs, DeepBoltzmannQuantumState):
            raise ValueError("variational_quantum_state must be a DeepBoltzmannQuantumState instance")
        if not callable(estimator):
            raise ValueError("parametric_gradient_estimator must be callable")
        if not hasattr(optimizer, "init") or not hasattr(optimizer, "update"):
            raise ValueError("optimizer must be an Optax GradientTransformation with init and update")

        schedule = jnp.asarray(schedule)
        if schedule.ndim != 2 or schedule.shape[0] <= 0 or schedule.shape[1] not in (2, 3):
            raise ValueError("annealing_schedule must be a 2D array with shape (T, 2) or (T, 3) and T>0")
        if not jnp.all(jnp.isfinite(schedule)):
            raise ValueError("annealing_schedule contains NaN or Inf values")

        for name, val in {
            "num_warmup_steps": warmup,
            "num_updates_per_step": updates_per,
            "num_finetuning_steps": finetune,
        }.items():
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        if not isinstance(persistent, bool):
            raise ValueError("persistent_chains must be a boolean")
        if not isinstance(use_tqdm_flag, bool):
            raise ValueError("use_tqdm must be a boolean")
        if not isinstance(log_params_flag, bool):
            raise ValueError("log_params must be a boolean")

        if not isinstance(observables, Mapping):
            raise ValueError("observables_dict must be a mapping of name -> callable")
        for key, val in observables.items():
            if not isinstance(key, str) or not callable(val):
                raise ValueError("observables_dict must map strings to callables")

        if isinstance(num_replicas, bool) or not isinstance(num_replicas, int) or num_replicas < 1:
            raise ValueError("num_replicas must be a positive integer")

    @partial(jax.jit, static_argnums=(0,))
    def opt_step(
        self,
        prngkey: jnp.ndarray,
        params: jnp.ndarray,
        opt_state: optax.OptState,
        mcmc_samples: jnp.ndarray,
        mcmc_endpoints: jnp.ndarray,
        couplings: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, optax.OptState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Perform one optimization step.

        - Optionally updates MCMC samples (persistent or fresh).
        - Evaluates the gradient estimator and applies optimizer updates.
        - Computes and returns step diagnostics.

        Parameters:
            prngkey: PRNGKey for sampling randomness.
            params: Current variational parameters.
            opt_state: Optimizer state.
            mcmc_samples: Current set of MCMC samples used for gradient estimation.
            mcmc_endpoints: Last states of chains for persistent sampling updates.
            couplings: Current schedule couplings, shape `(2,)` or `(3,)`.

        Returns:
            Tuple of updated `(prngkey, params, opt_state, mcmc_samples, mcmc_endpoints, step_data)`.
        """
        prngkey, tempkey = jax.random.split(prngkey)
        if self.persistent_chains:
            mcmc_samples, mcmc_endpoints = self.variational_quantum_state.update_samples(
                prngkey=tempkey,
                params=params,
                starting_points=mcmc_endpoints,
            )
        else:
            mcmc_samples, mcmc_endpoints = self.variational_quantum_state.generate_samples(
                prngkey=tempkey,
                params=params,
            )
        grads, avg_energy, energy_var = self.parametric_gradient_estimator(params, mcmc_samples, couplings)
        updates, opt_state = self.optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        step_data = {}
        step_data["avg_energy"] = avg_energy
        step_data["energy_var"] = energy_var
        step_data["magnetizations"] = mcmc_samples.mean(axis=0)
        if self.log_params:
            step_data["params"] = params
        # step_data["params"] = params
        for key, observable in self.observables_dict.items():
            step_data[key] = observable(params, mcmc_samples)

        return prngkey, params, opt_state, mcmc_samples, mcmc_endpoints, step_data
    
    @partial(jax.jit, static_argnums=(0,))
    def vmapd_opt_step(
            self,
            prngkeys,
            params_arrays,
            opt_states,
            mcmc_samples_arrays,
            mcmc_endpoints_arrays,
            couplings,
        ):
            return jax.vmap(self.opt_step, in_axes=(0, 0, 0, 0, 0, None))(
                prngkeys,
                params_arrays,
                opt_states,
                mcmc_samples_arrays,
                mcmc_endpoints_arrays,
                couplings,
            )

    def run(
        self,
        prngkey: jnp.ndarray,
        max_runtime: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute the full annealing schedule.

        Parameters:
            prngkey: PRNGKey used to initialize sampling and steps.
            max_runtime: Optional runtime cap in seconds. If provided and the
                projected remaining time would exceed this cap, the run stops
                early and returns `None`.

        Returns:
            A dictionary with logged arrays of metrics, final parameters and
            runtime statistics, or `None` if early-stopped due to `max_runtime`.
        """
        data = {}
        data["avg_energy"] = []
        data["energy_var"] = []
        data["magnetizations"] = []
        if self.log_params:
            data["params"] = []
        for key in self.observables_dict.keys():
            data[key] = []

        replica_keys = []
        params_array = []
        opt_states=[]
        mcmc_samples_arrays=[]
        mcmc_endpoints_arrays=[]


        for r in range(self.num_replicas):
            prngkey, tempkey_a, tempkey_b = jax.random.split(prngkey, 3)
            params, _ = self.variational_quantum_state.init_params(tempkey_a)
            params_array.append(params)
            replica_keys.append(tempkey_b)

            optimizer_state = self.optimizer.init(params)
            opt_states.append(optimizer_state)

            prngkey, tempkey = jax.random.split(prngkey)
            mcmc_samples, mcmc_endpoints = self.variational_quantum_state.generate_samples(
                prngkey=tempkey,
                params=params,
            )
            mcmc_samples_arrays.append(mcmc_samples)
            mcmc_endpoints_arrays.append(mcmc_endpoints)

        replica_keys = jnp.stack(replica_keys)
        params_array = jnp.stack(params_array)
        optimizer_states = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *opt_states)
        mcmc_samples_arrays = jnp.stack(mcmc_samples_arrays)
        mcmc_endpoints_arrays = jnp.stack(mcmc_endpoints_arrays)
                                                                           

        pbar = self.annealing_schedule
        if self.use_tqdm:
            pbar = tqdm(pbar)

        replica_keys, params_array, optimizer_states, mcmc_samples_arrays, mcmc_endpoints_arrays, step_data = self.vmapd_opt_step(
            replica_keys,
            params_array,
            optimizer_states,
            mcmc_samples_arrays,
            mcmc_endpoints_arrays,
            self.annealing_schedule[0],
        )

        jax.block_until_ready(step_data["avg_energy"])

        iterations_counter = 0
        iters_since_log = 0

        start_time = time.time()
        for couplings in pbar:
            iterations_counter += 1
            replica_keys, params_array, optimizer_states, mcmc_samples_arrays, mcmc_endpoints_arrays, step_data = self.vmapd_opt_step(
                replica_keys,
                params_array,
                optimizer_states,
                mcmc_samples_arrays,
                mcmc_endpoints_arrays,
                couplings,
            )
            jax.block_until_ready(params_array)

            iters_since_log += 1
            if iters_since_log == self.log_every:
                iters_since_log = 0
                for key, value in step_data.items():
                    data[key].append(value)

            if max_runtime is not None:
                elapsed_time = time.time() - start_time
                remaining_time = (
                    elapsed_time * (len(self.annealing_schedule) - iterations_counter) / iterations_counter
                )

                if (elapsed_time > max_runtime / 10) and (remaining_time > max_runtime):
                    print("Early stopping due to runtime limit")
                    return None

        if iters_since_log != 0:
            for key, value in step_data.items():
                data[key].append(value)

        for key, value in data.items():
            data[key] = jnp.array(value)

        runtime = time.time() - start_time
        data["couplings"] = self.annealing_schedule
        data["optimized_params"] = params_array
        data["runtime"] = runtime
        data["num_spins"] = self.variational_quantum_state.num_spins
        data["num_params"] = self.variational_quantum_state.params.size
        best_replica = jnp.argmin(jnp.real(data["avg_energy"][-1]))
        data["best_replica_index"] = int(best_replica)

        return data

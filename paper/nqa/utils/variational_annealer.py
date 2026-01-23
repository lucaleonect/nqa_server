import jax
import jax.numpy as jnp
import optax
import time
from tqdm import tqdm
from functools import partial
from typing import Callable, Optional


class VariationalAnnealer:
    def __init__(
        self,
        variational_quantum_state,
        parametric_gradient_estimator: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        optimizer: optax.GradientTransformation,
        annealing_schedule: jnp.ndarray,
        num_warmup_steps: Optional[int] = None,
        num_updates_per_step: Optional[int] = None,
        num_finetuning_steps: Optional[int] = None,
        persistent_chains: Optional[bool] = None,
        observables_dict: Optional[dict] = None,
        use_tqdm: Optional[bool] = None,
        log_every: Optional[int] = None,
        log_params: Optional[bool] = None,
    ):
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

        annealing_schedule = jnp.concatenate(
            (
                jnp.repeat(annealing_schedule[:1], num_warmup_steps, axis=0),
                jnp.repeat(annealing_schedule[1:-1], num_updates_per_step, axis=0),
                jnp.repeat(annealing_schedule[-1:], num_finetuning_steps, axis=0),
            ),
            axis=0,
        )
        
        if log_every is None:
            # Default to logging 100 times in total (total number of iterations is len(annealing_schedule))
            log_every = len(annealing_schedule) // 100

        self.variational_quantum_state = variational_quantum_state
        self.parametric_gradient_estimator = parametric_gradient_estimator
        self.optimizer = optimizer
        self.annealing_schedule = annealing_schedule
        self.persistent_chains = persistent_chains
        self.observables_dict = observables_dict
        self.num_warmup_steps = num_warmup_steps
        self.num_updates_per_step = num_updates_per_step
        self.num_finetuning_steps = num_finetuning_steps
        self.use_tqdm = use_tqdm
        self.log_every = log_every
        self.log_params = log_params
        
        if self.annealing_schedule.shape[1] == 3:
            self.uses_catalyst = True
        else:
            self.uses_catalyst = False

    @partial(jax.jit, static_argnums=(0,))
    def opt_step(
        self,
        prngkey,
        params,
        opt_state,
        mcmc_samples,
        mcmc_endpoints,
        couplings,
    ):
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

        if self.uses_catalyst:
            step_data["inst_energy"] = (
                couplings[0] * step_data["target_energy"][0]
                + couplings[1] * step_data["annealing_energy"][0]
                + couplings[2] * step_data["catalyst_energy"][0]
            )
        else:
            step_data["inst_energy"] = (
                couplings[0] * step_data["target_energy"][0] + couplings[1] * step_data["annealing_energy"][0]
            )
            

        return prngkey, params, opt_state, mcmc_samples, mcmc_endpoints, step_data

    def run(
        self,
        prngkey,
        max_runtime=None,
    ):
        data = {}
        data["avg_energy"] = []
        data["energy_var"] = []
        data["magnetizations"] = []
        data["inst_energy"] = []
        # data["params"] = []
        if self.log_params:
            data["params"] = []
        for key in self.observables_dict.keys():
            data[key] = []

        params = self.variational_quantum_state.params
        optimizer_state = self.optimizer.init(params)
        prngkey, tempkey = jax.random.split(prngkey)
        mcmc_samples, mcmc_endpoints = self.variational_quantum_state.generate_samples(
            prngkey=tempkey,
            params=params,
        )

        pbar = self.annealing_schedule
        if self.use_tqdm:
            pbar = tqdm(pbar)

        prngkey, params, optimizer_state, mcmc_samples, mcmc_endpoints, step_data = self.opt_step(
            prngkey,
            params,
            optimizer_state,
            mcmc_samples,
            mcmc_endpoints,
            self.annealing_schedule[0],
        )

        jax.block_until_ready(step_data["inst_energy"])

        iterations_counter = 0
        iters_since_log = 0

        start_time = time.time()
        for couplings in pbar:
            iterations_counter += 1
            prngkey, params, optimizer_state, mcmc_samples, mcmc_endpoints, step_data = self.opt_step(
                prngkey,
                params,
                optimizer_state,
                mcmc_samples,
                mcmc_endpoints,
                couplings,
            )
            jax.block_until_ready(params)


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

        runtime = time.time() - start_time
        data["couplings"] = self.annealing_schedule
        data["optimized_params"] = params
        data["runtime"] = runtime
        data["num_spins"] = self.variational_quantum_state.num_spins
        data["num_params"] = self.variational_quantum_state.params.size

        return data

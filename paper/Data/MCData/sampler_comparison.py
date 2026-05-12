import argparse

parser = argparse.ArgumentParser(description="Run MHC vs bGC comparison.")

parser.add_argument("--cuda_device", type=int, default=None)
parser.add_argument("--bqs_num_layers", type=int, default=3)
parser.add_argument("--params_scale", type=float, default=1)
parser.add_argument("--bqs_unit_density_per_layer", type=float, default=1.0)
parser.add_argument("--bqs_mcmc_num_samples", type=int, default=100000)

args = parser.parse_args()

import os

os.environ["JAX_ENABLE_X64"] = "True"  # Use double precision

if args.cuda_device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

import jax
import jax.numpy as jnp
import optax
import numpy as np
import time
import matplotlib.pyplot as plt
from utils.sampling.metropolis_hasting_sampler import build_base_sampler, build_update_proposer, build_mcmc_sampler
from utils.sampling.markov_chain_analysis_tools import running_means, autocorrelation, autocorrelation_time
from utils.variational_quantum_states.boltzmann_quantum_states import DeepBoltzmannQuantumState


def main():
    data = {}

    for L in [5, 10, 15, 20, 25, 30, 50]:
        data[str(L)] = {}
        data[str(L)]["mh"] = {}
        data[str(L)]["mh"]["actime"] = []
        data[str(L)]["gibbs"] = {}
        data[str(L)]["gibbs"]["actime"] = []

        for trial in range(10):
            start_time = time.time()
            prngseed = int(start_time)
            prngkey = jax.random.PRNGKey(prngseed)

            num_spins = L

            prngkey, tempkey = jax.random.split(prngkey)
            vqs = DeepBoltzmannQuantumState(
                num_spins=num_spins,
                layers=[int(num_spins * args.bqs_unit_density_per_layer)] * args.bqs_num_layers,
                prngkey=tempkey,
                num_samples=args.bqs_mcmc_num_samples,
                num_thermalization_steps=1,
                num_sweep_steps=1,
                num_chains=1,
                dtype=jnp.complex128,
            )
            vqs.params = vqs.params * args.params_scale

            prngkey, tempkey = jax.random.split(prngkey)
            gibbs_chain = vqs.debug_gibbs_chain(tempkey, vqs.params)

            gibbs_running_means = running_means(gibbs_chain)
            gibbs_autocorr = autocorrelation(gibbs_chain)
            gibbs_autocorr_time = autocorrelation_time(gibbs_chain)
            print("gibbs a_corr: ", gibbs_autocorr_time.max())

            p_ratio_fn = jax.jit(
                lambda params, s, sp: jnp.exp(2 * np.real(vqs.logpsi(params, s) - vqs.logpsi(params, sp)))
            )
            mh_sampler, _ = build_mcmc_sampler(
                base_sampler=build_base_sampler(vqs.num_units),
                update_proposer=build_update_proposer(vqs.num_units),
                probability_ratio_function=p_ratio_fn,
                num_samples=args.bqs_mcmc_num_samples,
                num_thermalization_steps=1,
                num_sweep_steps=1,
                num_chains=1,
            )

            prngkey, tempkey = jax.random.split(prngkey)
            mh_chain, _ = mh_sampler(tempkey, vqs.params)
            mh_running_means = running_means(mh_chain)
            mh_autocorr = autocorrelation(mh_chain)
            mh_autocorr_time = autocorrelation_time(mh_chain)
            print("mh a_corr: ", mh_autocorr_time.max())

            data[str(L)]["gibbs"]["actime"] += [gibbs_autocorr_time]
            data[str(L)]["mh"]["actime"] += [mh_autocorr_time]

    np.savez("mcdata", **data)
    print(data)


if __name__ == "__main__":
    main()





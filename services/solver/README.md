# Neural Quantum Annealing (NQA)

Neural Quantum Annealing (NQA) is a variational framework that optimizes a Deep Boltzmann Quantum State (DBQS) with (stochastic) natural gradients under a time‑dependent Hamiltonian schedule. This implementation targets classical and quantum Ising problems such as the Transverse‑Field Sherrington–Kirkpatrick model (TFSK).

Given a coupling matrix `J` (and optional fields `h`, `g`), NQA minimizes the expectation of the problem Hamiltonian using a parametric annealing schedule with optional catalyst, persistent MCMC sampling, and the Time‑Dependent Variational Principle (TDVP).


**Key Ideas**
- **Variational State**: Deep Boltzmann Quantum State (DBQS) over visible spins and hidden layers.
- **TDVP**: Natural or minimal SR step computed from Monte Carlo samples of the state.
- **Annealing Schedule**: Interpolates between target and driver (and optional catalyst) Hamiltonians.
- **Sampling**: Block‑Gibbs with optional persistent chains for efficiency.
- **HPO**: Optional Optuna search over annealing/optimizer state via `nqa/master.py`.


**Repository Layout**
- `nqa/worker.py`: Single‑run CLI; executes one annealing run and writes artifacts.
- `nqa/master.py`: Hyperparameter search with Optuna; orchestrates many runs.
- `nqa/utils/boltzmann_quantum_states.py`: DBQS model and sampling.
- `nqa/utils/operators.py`: Local energies and parametric Hamiltonian builders.
- `nqa/utils/tdvp.py`: TDVP gradient estimators (SR, minSR, auto).
- `nqa/utils/serialization.py`: Result serialization and basic plots.
- `nqa/tests/…`: Unit tests for operators, TDVP, annealer, and state.
- `paper.pdf`: Reference manuscript for the method/details.


## Installation

NQA is implemented in Python with JAX. A working Python 3.10+ (3.11 recommended) is required. GPU acceleration is optional but recommended for larger problems.
Recommendend to use the docker included
```bash
sudo docker build -t nqa-jax .
sudo docker run --rm -it --gpus all nqa-jax bash
```


## Quick Start (Single Run)

Prepare inputs for a problem of size `N`:

- `J.npy`: `(N, N)` symmetric real matrix (NumPy `.npy`).
- `h.npy` (optional): `(N,)` real vector for longitudinal Z field.
- `g.npy` (optional): `(N,)` real vector for transverse X field. If omitted, the target is purely classical SK.

Example: generate a small random instance and run NQA on CPU:

```bash
python nqa/worker.py \
	--J_matrix_path J.npy \
	--h_vector_path h.npy \
	--g_vector_path g.npy \
	--save_path runs/sk32_demo \
```

Artifacts appear under `runs/sk32_demo`:

- `results.txt`: Final metrics and summary.
- `data.npz`: All logged arrays and final params.
- `plots/`: Basic diagnostic plots (`energies.png`, `magnetizations.png`, `couplings.png`).


## Inputs & Formats

- `J_matrix_path` (`.npy`): `(N, N)` real symmetric matrix. Diagonal is treated as zero.
- `h_vector_path` or `--h_value`: either a `(N,)` `.npy` file or a constant to build a uniform field.
- `g_vector_path` or `--g_value`: either a `(N,)` `.npy` file or a constant to build a uniform X field. If omitted, the target is classical only.
- `energy_shift`: optional scalar added to the energy (for baseline shifts).

Mutually exclusive options:
- Do not pass both `--h_value` and `--h_vector_path`.
- Do not pass both `--g_value` and `--g_vector_path`.


## Important CLI Flags (Single Run)

- `--cuda_device`: GPU index to use (sets `CUDA_VISIBLE_DEVICES`).
- `--max_runtime`: Soft cap; the run halts when projected duration exceeds this (default 6 hours).
- `--save_path`: Destination directory for logs, arrays, and plots; created if missing.
- `--vqa_num_annealing_steps`, `--vqa_num_warmup_steps`, `--vqa_num_updates_per_step`, `--vqa_num_finetuning_steps`: Control how often the schedule points repeat.
- `--vqa_annealing_field_scale`, `--vqa_catalyst_field_scale`, `--vqa_no_catalyst`: Scale or disable the driver/catalyst terms in the Hamiltonian.
- `--sgd_learning_rate`, `--sgd_momentum`: Optimizer hyperparameters (Optax SGD).
- `--sr_method`, `--sr_diagonal_shift`, `--sr_prefactor`: Natural-gradient solver choices and regularisation.
- `--dbqs_num_hidden_layers`, `--dbqs_unit_density_per_layer`, `--dbqs_param_dtype`, `--dbqs_use_bias`: DBQS architecture and parameter dtype/bias toggles.
- `--mcmc_num_samples`, `--mcmc_num_chains`, `--mcmc_num_sweep_steps`, `--mcmc_num_thermalization_steps`, `--mcmc_disable_persistent_markov_chains`: Sampler budget and persistence controls.

Run `python -m nqa.worker -h` to see the full list and defaults.


## Hyperparameter Search (Optuna)

`nqa/master.py` runs parallel Optuna trials and tracks the best configuration per study. Note: run it from inside the `nqa/` directory so the spawned `worker.py` resolves correctly.

Example:

```bash
# From inside the nqa/ directory:
cd nqa
python master.py \
	--J_matrix_path ../J.npy \
	--study_name sk_demo \
```

Notes:
- A study is created under `studies/sk_demo/` with an Optuna SQLite DB and per‑trial output folders `test_<trial_id>`.
- `master.py` spawns `worker.py` processes with sampled hyperparameters; results are polled until `finished.txt` appears.
- Objective is the final `target_energy` average.

You can resume a study by rerunning with the same `--study_name`.


## Outputs & Programmatic Access

The `data.npz` bundle contains arrays logged at the chosen cadence (see `log_every` inside `VariationalAnnealer`):

- `avg_energy`, `energy_var`: Instantaneous energy estimate and variance.
- `target_energy`, `annealing_energy`, `[catalyst_energy]`: Mean/var for each operator; for classical targets also best value/config along the run.
- `magnetizations`: Sample means of visible spins.
- `couplings`: The annealing schedule used.
- `optimized_params`: Final DBQS parameters.
- `runtime`, `num_spins`, `num_params`.

Example to read and inspect:

```python
import numpy as np
d = np.load('runs/sk32_demo/data.npz')
print('Final target energy:', d['target_energy'][-1,0])
print('Best-so-far (if classical):', d.get('best_energy_so_far'))
```


## Tips & Performance

- **Persistent Chains**: Keeping `--mcmc_persistent_markov_chains` on usually reduces variance and burn‑in.
- **Budget**: Increase `--mcmc_num_samples` and `--mcmc_num_sweep_steps` to stabilize gradients at the cost of runtime.
- **Conditioning**: Tune `--sr_diagonal_shift` to keep inverses stable; `auto` method selects minSR when samples ≤ params.
- **Catalyst**: Try enabling `--vqa_use_catalyst` with moderate `--vqa_catalyst_field_scale` for hard instances.
- **Annealing**: More schedule steps and some warmup/finetuning repeats often help.


## Reproducibility & Environment

- `--prng_seed`: Fix for deterministic JAX PRNG seeding of sampling and initialization.
- `JAX_ENABLE_X64` is set to `True` by `worker.py` for double precision.
- `--cuda_device` sets `CUDA_VISIBLE_DEVICES`; leave unset to run on CPU.


## Development

- Run tests (requires `pytest` installed); run them inside `nqa/` so imports resolve:

```bash
pip install pytest
cd nqa
pytest -q
```

- Code entry points and main classes:
	- `nqa/worker.py`: CLI; see `main()` for argument parsing and flow.
	- `nqa/utils/variational_annealer.py`: `VariationalAnnealer` run loop and logging.
	- `nqa/utils/boltzmann_quantum_states.py`: `DeepBoltzmannQuantumState` model and MCMC.
	- `nqa/utils/tdvp.py`: SR/minSR gradient estimators.
	- `nqa/utils/operators.py`: Energy operators and schedule assembly.


## Citing

If you use this code, please cite the accompanying manuscript (`paper.pdf`). A BibTeX stub you can adapt:

```bibtex
@misc{nqa2025,
	title        = {Neural Quantum Annealing},
	author       = {Authors},
	year         = {2025},
	howpublished = {arXiv preprint},
	note         = {See included paper.pdf}
}
```


## License

No license file is included.

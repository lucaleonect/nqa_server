# Visible interactions in DBQS

This implements the Hubbard–Stratonovich extension in the supplied **BQS
Sampling.md**, on the repository's `nqa-server` branch at
`24a13139f40617bd0be1e54d51edacb3a9a073eb`.

## Enable it

For an N-spin system, use `visible_interactions=True` in Python. The default
factor has shape `(N, N)`, as in the notes. `visible_rank=r` optionally uses an
`(N, r)` factor. Omitting the option preserves the existing model and parameter
layout.

```python
import jax
import jax.numpy as jnp
from utils.boltzmann_quantum_states import DeepBoltzmannQuantumState

jax.config.update("jax_enable_x64", True)
N = 8
state = DeepBoltzmannQuantumState(
    num_spins=N,
    hidden_layers=[8, 4],
    prngkey=jax.random.PRNGKey(0),
    dtype=jnp.complex128,
    visible_interactions=True,
    num_samples=1024,
    num_chains=64,
    num_thermalization_steps=128,
    num_sweep_steps=16,
)
samples, endpoints = state.generate_samples(jax.random.PRNGKey(1), state.params)
samples, endpoints = state.update_samples(
    jax.random.PRNGKey(2), state.params, endpoints
)
biases, weights, J, K = state.unpack_params(state.params)
```

Run from `services/solver/nqa`, or add that directory to `PYTHONPATH`.
Use the existing operator builders, `build_parametric_gradient_estimator`, and
`VariationalAnnealer` with this state.

Both `master.py` and `worker.py` accept `--dbqs_visible_rank r`. Its default,
`0`, disables visible interactions; a positive value enables them. For the
square factor in the notes, supply the number of visible spins:

```bash
cd services/solver/nqa
python master.py --J_matrix_path /absolute/path/to/J.npy \
    --dbqs_visible_rank 8
```

Here the problem has eight spins. The Hamiltonian's `J.npy` and the variational
factor `J` are different objects.

The web form includes **DBQS visible interaction rank (0 disables)**. The
HTTP service accepts the same value under
`"study_args": {"dbqs_visible_rank": 8}`. The scheduler forwards it to the solver;
the master records it in study attributes and passes it to every worker.
It is a fixed study setting, not an Optuna search range.

## Amplitudes and parameters

Let \(s=(x,h^{(1)},h^{(2)},\ldots)\) contain only binary spins, with \(x_i=\pm1\).
The log amplitude is

\[
\ell_\theta(s)=x^T(JJ^T+iK)x
+\sum_l x^{(l)T}(W_R^{(l)}+iW_I^{(l)})x^{(l+1)}
+\sum_l(b_R^{(l)}+ib_I^{(l)})^Tx^{(l)}.
\]

There is no implicit factor of one half in the quadratic form. This follows the
notes and the existing DBQS convention.

With visible interactions enabled, every element of `state.params` is real,
even for `dtype=jnp.complex128`. In that case, `dtype` specifies the amplitude
type and `state.params.dtype` is its real counterpart. `is_holomorphic` is false.

The parameter tree returned by `unravel_params` contains:

| Key | Contents |
|---|---|
| `J` | Real `(N, r)` amplitude factor |
| `K` | Real `(N, N)` phase matrix, present for complex amplitudes |
| `real` | Real network weights and, if enabled, biases |
| `imag` | Imaginary network weights/biases, present for complex amplitudes |

The `real` and `imag` subtrees have the legacy layout: `(biases, weights)` when
`use_bias=True`, and just `weights` otherwise. Reassemble a modified tree using
`jax.flatten_util.ravel_pytree(tree)[0]`. `unpack_params` reconstructs complex
network parameters for convenient inspection.

`dtype=jnp.float64` enables a purely real log amplitude: no `K` or `imag`
parameters are allocated. Empty hidden-layer lists and `use_bias=False` work.

`J` starts with small, nonzero random values. Setting it to zero exactly makes
\(\partial_J(x^TJJ^Tx)=2xx^TJ\) vanish, preventing first-order optimization from
learning amplitude interactions from that point. `K` starts at zero.

## Positive block Gibbs sampling

The spin sampling distribution is \(p_\theta(s)\propto
\exp(2\operatorname{Re}\ell_\theta(s))\). Introduce a real auxiliary vector
\(z\in\mathbb R^r\):

\[
q_\theta(s,z)\propto
\exp\left[-z^Tz/8+x^TJz+
2\sum_l x^{(l)T}W_R^{(l)}x^{(l+1)}
+2\sum_l b_R^{(l)T}x^{(l)}\right].
\]

Integrating over \(z\) gives
\((8\pi)^{r/2}\exp(2x^TJJ^Tx)\), times the original network weight. The factor
is independent of the trainable parameters and cancels on normalization.
Strictly, the final expression called a “Born probability” in the notes is this
**augmented density**; its Gaussian marginal is the Born distribution over spins.

The new conditionals are

\[
z\mid x\sim\mathcal N(4J^Tx,4I),\qquad
q(x_i=+1\mid z,h^{(1)})
=\sigma\!\left(2(Jz)_i+
4(b_R^{(0)}+W_R^{(0)}h^{(1)})_i\right).
\]

All other binary layer conditionals are unchanged. A sweep draws \(z\) and all
odd hidden layers conditional on the old even layers, then draws all even
layers using the new odd layers and \(z\). These are the two color classes of the
augmented graph, with the Gaussian layer belonging to the odd class.

The Gaussian fields are discarded after each sweep and drawn afresh at the
next sweep. This gives a valid Markov kernel on binary configurations after
marginalizing \(z\). It keeps the public sample and endpoint shapes unchanged:
`(num_samples, num_units)` and `(num_chains, num_units)`. Neither includes
Gaussian fields, and `num_units` still counts visible and hidden binary spins.
Persistent sampling therefore needs no additional checkpoint state.

For direct low-level sampler use, pass the optional `visible_factor=J` argument
to the existing chain methods. `prob_evens_given_odds` additionally requires the
current `auxiliary_fields=z`. `generate_samples`, `update_samples`, and
`debug_gibbs_chain` handle these arguments automatically.

## Physical estimators and optimization

The Gaussian fields are not additional quantum degrees of freedom. Wavefunction
ratios and log derivatives use \(\ell_\theta(s)\), with the Gaussian fields
already integrated out, not the augmented density or its square root.

Write \(C=JJ^T+iK\) and \(a=b^{(0)}+W^{(0)}h^{(1)}\), with the hidden term absent
when there are no hidden layers. Flipping visible spin \(i\) gives

\[
\Delta_i\ell=-2x_i a_i-2x_i[(C+C^T)x]_i+4C_{ii}.
\]

Thus the local Pauli estimators are
\((X_i)_{\rm loc}=e^{\Delta_i\ell}\) and
\((Y_i)_{\rm loc}=-ix_i e^{\Delta_i\ell}\). The diagonal correction is necessary
because \(x_i^2=1\). The implementation evaluates the real contribution through
`J @ (J.T @ x)` without materializing `J @ J.T`; it also handles a nonsymmetric
`K`. Generic multiple-spin ratios remain available through `psi_ratio_fn`.

For the DBQS visible density matrix, these estimators evaluate
\(\operatorname{Tr}(\rho H)\), where
\(\rho_{x,x'}\propto\sum_h\Psi(x,h)\overline{\Psi(x',h)}\).
The tests compare this expression to an explicitly constructed density matrix.

The extension uses real-coordinate SR. For \(M\) samples, let \(O\) be the
centered complex log derivatives and \(e\) the centered local energies. It solves

\[
(S+\lambda I)\,\delta\theta=f,\qquad
S=\operatorname{Re}(O^\dagger O)/M,\quad
f=\operatorname{Re}(O^\dagger c\,e)/M,
\]

where \(c\) is the existing TDVP `prefactor`. For complex output, both the real
and imaginary parts of the log amplitude are differentiated. The factor derivative
is \(2x(x^TJ)\), not an auxiliary-field score such as \(xz^T\).

SR and minSR use the same normalization and diagonal shift in this mode.
minSR uses a real sample-space system formed by stacking the real and imaginary
rows of \(O/\sqrt M\); `auto` accounts for its \(2M\) rows. Updates remain real
with a Y catalyst and with a complex TDVP prefactor. The legacy model retains its
previous optimization path.

## Scope and costs

- The extra Gaussian draw and amplitude contractions cost \(O(Nr)\) per
  configuration. A dense `K` adds \(O(N^2)\) work to phase estimators.
- Full square factors can encode any real symmetric pair coupling up to a
  diagonal shift: choose \(d\) so \(A+dI\) is positive semidefinite and factor it.
  The added \(dN\) in the log amplitude cancels on normalization. Smaller factors
  impose a rank restriction.
- The diagonal and antisymmetric parts of `K` are physically redundant; the
  full matrix is retained to follow the notes. `J` also has factorization
  redundancies, which can make the metric singular; regularization remains useful.
- Tractable conditional draws do not guarantee fast Markov-chain mixing.
  Strong couplings, including large diagonal shifts in a Gram representation,
  can slow mixing. Check autocorrelations and convergence for research runs.
- This implements pairwise visible interactions. It does not establish
  universality, efficient mixing at arbitrary parameters, or improved
  ground-state energies on large benchmarks.

## Verification

Install the repository requirements plus a suitable JAX and Optax installation.
For a CPU environment, for example:

```bash
python -m pip install -r services/solver/requirements.txt "jax[cpu]" optax
PYTHONPATH=services/solver/nqa JAX_ENABLE_X64=True \
    python -m pytest -q services/solver/nqa/tests/test_visible_interactions.py
```

The 28 new cases cover amplitudes, all single-spin flips and multiple-spin
ratios, exact density-matrix energies, Gaussian conditionals and Hermite
quadrature, sampled distributions and persistent endpoints, phase-independent
sampling, finite-difference gradients, SR/minSR agreement, real-valued updates
with a Y term and two replicas, legacy reduction, and invalid options. Two
analytic entanglement checks construct the exact ground state of
`Z1 Z2 + g (X1 + X2)` at `g=0.1` and `g=1.5`, verifying that the implemented
visible interactions lower the energy below the best possible product state.

Validated on CPU with JAX/JAXlib 0.11.1, Optax 0.2.8, NumPy 2.3.5, SciPy 1.17.0,
and pytest 8.4.2, with JAX 64-bit arithmetic enabled. Before the two additional
entanglement checks, the full suite reported
**66 passed, 3 failed**. The same three failures occur on the unchanged base
revision: `test_variational_annealer_basic_run_no_catalyst`,
`test_variational_annealer_with_catalyst_and_inst_energy`, and
`test_variational_annealer_logs_include_params_when_enabled`. Their assertions
omit the replica axis already returned by the annealer.

A direct worker run also completed with two replicas, two hidden layers, rank
three, an X-field target and a Y catalyst, saving finite float64 parameter
arrays. A one-trial Optuna master run completed for a classical target with the
visible-interaction option enabled.

The mathematical checks validate the implementation on small systems. No GPU
performance or large-system mixing claim follows from these tests.

## QSK optimization and transverse-field phase

For the repository convention `H(g) = x.T @ J_file @ x + g sum(X)`, uniform
positive `g` is unitarily equivalent to negative `g` through `G = product(Z)`.
An amplitude with visible bias `i*pi/2` at every site carries the corresponding
configuration phase, proportional to `product(x)`. The ground state of the
negative-field Hamiltonian can be chosen positive. This gives a useful
initialization for direct SR on the positive-field target; it does not change
the target Hamiltonian or require fixing the phase parameters during training.

The usual catalyzed path `s H(g) - (1-s) sum(X) - s(1-s) sum(Y)` has an X
coefficient `s*g-(1-s)`. At `g=1.5`, this crosses zero at `s=0.4`, where the
remaining Y field has magnitude `0.24`. Relative to the Ising coefficient
`s=0.4`, the effective transverse field is only `0.6`. This avoidable low-field
part of the path can make a fixed-budget benchmark a poor measure of the
extension's expressivity. Exact final energy evaluation does not establish
that the optimization converged.

An N=16 check on all ten repository instances mapped independently optimized
real Jastrow states into the native `J J.T` parameterization by adding a
diagonal shift before factorization. Their native energies agreed with the
independent evaluator within `1.1e-12`, with mean relative error `0.115%` at
`g=1.5`. The hidden-layer couplings were zero for this representability check;
these are feasible states, not a proof of the globally best Jastrow energy.
This is a useful control before attributing poor training results to the
ansatz or its Gaussian sampler.

A matched direct-SR rerun at `N=16, g=1.5` used the same ten instances, two
seeds, approximately 950 real trainable coordinates per model, 128 samples
per update, four sweeps, learning rate `0.05`, and normalized SR shift `0.01`.
It initialized the known transverse-field phase, started network weights
real, and spent all 1,510 updates on the unchanged positive-field target.
All final energies were evaluated exactly. Mean relative errors (SEM across
ten instance averages) were:

| Ansatz | Relative energy error (%) |
|---|---:|
| Old DBQS, hidden [8,36] | 4.7577 ± 0.5149 |
| Visible-interaction DBQS, hidden [8,8], rank 16 | 0.1608 ± 0.0425 |
| cRBM, 27 hidden units | 0.0132 ± 0.0009 |

This is a fixed-budget optimization comparison, not a global optimum or
equal-wall-time result. The original catalyzed annealing protocol gave
`4.9540%` for the visible-interaction DBQS under the same total update budget.
Changing the protocol recovers the expected advantage without altering the
ansatz, Gaussian conditionals, or physical spin-flip formulas.

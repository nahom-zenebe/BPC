# Bayesian Predictive Coding

This repository contains a JAX implementation of the paper **Bayesian
Predictive Coding** (arXiv:2503.24016v1), organized as a reusable `bpc/`
package with thin experiment entry points for two moons and MNIST.

## Layout

- `bpc/` contains reusable BPC implementation code:
  - `config.py`: defaults, presets, logging config, JAX dtype/setup.
  - `distributions/`, `posterior/`, `priors/`: Matrix-Normal Wishart natural
    parameters, prior/posterior initialization, and moment conversions.
  - `inference/`: hidden-state gradients, energy diagnostics, and Adam latent
    inference.
  - `updates/`: sufficient statistics, kappa/scale schedules, and posterior
    updates.
  - `training/`: JIT step factories, trainers, evaluation, and metrics.
  - `data.py`: MNIST/two-moons loading and batching.
  - `utils/logging.py`: run manifests, CSV/JSONL diagnostics, anomaly logs, and
    optional plots.
- `vbpc/` is a top-level package containing the VBPC module (see below); it builds
  on the shared `bpc/` utilities.
- `experiments/` contains only the two thin entry points (plus the VBPC ones).
- `configs/` contains YAML configs for the entry points.
- `tests/` contains minimal behavior tests.

## Run Experiments

Install the runtime dependencies in your environment first: `jax`, `jaxlib`,
`numpy`, and experiment-specific packages such as `scikit-learn`, `tensorflow`,
`matplotlib`, `pyyaml`, and `pytest` as needed.

Run two moons:

```bash
python -m experiments.two_moons --config configs/two_moons.yaml
```

Run MNIST:

```bash
python -m experiments.mnist --config configs/mnist.yaml
```

The YAML files override logging output to `runs/`. Algorithm defaults and
presets are inherited from `bpc.config`.

## Variational Bayesian Predictive Coding (VBPC)

`vbpc/` is a dedicated VBPC module that reuses the shared BPC utilities
(`config`, `data`, `activations`, tensor ops, reproducibility helpers, logging
conventions) . It replaces the weight posterior with a factorized Gaussian and adds reparameterized Gaussian
latent states:

```
q(W_l) = N(mu_W,l, diag(sigma_W,l^2))   W_l = mu_W,l + sigma_W,l * eps
q(z_l|x) = N(mu_l, diag(sigma_l^2))     z_l = mu_l + sigma_l * eps
L_PC   = 0.5 * sum_l e_l^T Sigma_l^{-1} e_l        (Sigma_l = error_variance * I)
L_KL   = 0.5 * sum(mu_W^2 + sigma_W^2 - 1 - log sigma_W^2)   (standard-normal prior)
L_total = L_PC + beta * L_KL
```

Layer numbering follows the rest of the BPC package (bottom-up: `z_0 = x`,
`z_L = y` clamped), so the proposal's residual `e_l = z_l - W_l f(z_{l+1})` is
written as `e_l = z_{l+1} - W_l f(z_l)`: the residual attached to weight `W_l`
predicts the layer above it. Latent inference optimizes `(mu_l, log_sigma_l)`
with Adam on **`L_PC` only**; the weight/variational parameters `(mu_W, log_sigma_W)`
take an Adam step on **`L_total`**, with the inferred latents detached and the
same reparameterized weight sample reused (single-sample reparameterized
gradient estimate). Optimizers come from Optax (`vbpc/optim.py`, with a
built-in Adam fallback if Optax is unavailable).

Layout:

- `vbpc/config.py`: `VBPCConfig`, presets, the `beta` sweep list, validation.
- `vbpc/posterior/weight_posterior.py`: factorized Gaussian `q(W)`, reparameterization, KL.
- `vbpc/inference/objectives.py`: `q(z)` states, `e_l`, `L_PC`, `L_total`.
- `vbpc/inference/latent_inference.py`: Adam latent inference on `L_PC`.
- `vbpc/layers/vbpc_network.py`: mean-weight forward passes.
- `vbpc/training/`: train step, metrics (accuracy/NLL/ECE/weight variance/memory),
  run logger and the MNIST trainer with `beta` sweep.

Run the VBPC MNIST experiment (sweep over `beta = [0, 0.001, 0.01, 0.1, 1.0]`):

```bash
python -m experiments.vbpc_mnist --config configs/vbpc_mnist.yaml
python -m experiments.vbpc_mnist --config configs/vbpc_mnist.yaml --mode single --beta 0.01
python -m experiments.vbpc_mnist --betas 0,0.01,0.1
```

Each epoch logs accuracy, NLL, ECE, runtime, memory usage, `beta` and the mean
weight variance to `runs/vbpc_mnist/<config>/<timestamp>/epoch.csv`, with the
per-step objectives in `batch.csv`/`verbose.jsonl`, a `manifest.json` and plots.
The sweep writes a combined summary (`vbpc_beta_sweep.json`) and comparison
figure. All runs reuse `cfg.seed`, so initialization and batch order are
identical across betas.

Run the VBPC two-moons experiment (same `beta` sweep on the two-moons dataset
from `configs/vbpc_two_moons.yaml`, whose `data:` block sets
`n_train`/`n_test`/`noise`):

```bash
python -m experiments.vbpc_two_moons --config configs/vbpc_two_moons.yaml
python -m experiments.vbpc_two_moons --config configs/vbpc_two_moons.yaml --mode single --beta 0.01
python -m experiments.vbpc_two_moons --betas 0,0.01,0.1
```

Two-moons runs are namespaced separately from MNIST: logs go to
`runs/vbpc_two_moons/<config>/<timestamp>/`, the legacy CSV is
`runs/vbpc_two_moons_experiment_log.csv` and the sweep summary is
`runs/vbpc_two_moons_beta_sweep.json` with a `vbpc_two_moons_beta_sweep.png`
comparison figure.


## Test

```bash
pytest
```

The tests cover core math, inference, update schedules, MNIST NPZ
preprocessing, and the VBPC equations (`tests/test_vbpc.py`: Gaussian
reparameterization, analytic KL, PC energy, latent inference, the VBPC training
step and shape/gradient/JIT correctness). If JAX or NumPy are not installed, the
tests skip rather than downloading dependencies.

"""Variational Bayesian Predictive Coding (VBPC).

Dedicated module implementing VBPC on top of the shared BPC utilities.  VBPC
replaces the Matrix-Normal Wishart weight posterior of the BPC package with a
factorized Gaussian ``q(W_l) = N(mu_W,l, diag(sigma_W,l^2))`` and adds
reparameterized Gaussian latent states ``z_l = mu_l + sigma_l * eps``, trained by

    L_total = L_PC + beta * L_KL,

where ``L_PC = 0.5 * sum_l e_l^T Sigma_l^{-1} e_l`` drives latent inference (the
KL term is excluded there) and ``L_KL`` is the analytic Gaussian KL against a
standard-normal prior.

Layout (mirrors the existing ``inference/``, ``posterior/``, ``layers/`` and
``training/`` packages of the BPC implementation):

- ``config.py``: :class:`VBPCConfig`, presets, ``beta`` sweep and validation.
- ``posterior/weight_posterior.py``: factorized Gaussian ``q(W)``.
- ``inference/objectives.py``: ``q(z)`` states, ``e_l``, ``L_PC`` and ``L_total``.
- ``inference/latent_inference.py``: Adam latent inference on ``L_PC``.
- ``layers/vbpc_network.py``: mean-weight forward passes.
- ``optim.py``: Optax optimizers (built-in Adam fallback).
- ``training/``: train steps, metrics (accuracy/NLL/ECE/variance/memory) and
  the MNIST trainer with beta sweep.
"""

from vbpc.config import VBPC_BETA_SWEEP, VBPCConfig, make_vbpc_presets, validate_vbpc_config
from vbpc.inference.latent_inference import (
    VBPCLatentDiagnostics,
    deterministic_pc_energy,
    init_vbpc_latents,
    latent_pc_energy,
    vbpc_latent_inference,
)
from vbpc.inference.objectives import (
    VBPCStates,
    clamped_states,
    mean_states,
    pc_energy,
    pc_energy_per_layer,
    pc_errors,
    reparameterize_states,
    sample_states,
    state_sigma,
    total_vbpc_objective,
    total_vbpc_objective_for_batch,
)
from vbpc.layers.vbpc_network import forward_hidden, forward_logits, forward_top
from vbpc.optim import OPTAX_AVAILABLE, VBPCOptimizer, make_optimizer, optimizer_backend
from vbpc.posterior.weight_posterior import (
    VBPCWeightParams,
    gaussian_kl_elementwise,
    init_vbpc_weight_params,
    mean_weight_variance,
    reparameterize_weight,
    reparameterize_weight_matrices,
    sample_weight_epsilons,
    sample_weight_matrices,
    weight_diagnostics,
    weight_kl,
    weight_shapes,
    weight_sigma,
)
from vbpc.training.logger import VBPCRunLogger, plot_beta_sweep
from vbpc.training.metrics import (
    ece_from_logits,
    evaluate,
    expected_calibration_error,
    memory_usage_mb,
    negative_log_likelihood,
    nll_from_logits,
    vbpc_accuracy,
    weight_variance_metric,
)
from vbpc.training.train_step import (
    VBPCDiagnostics,
    make_vbpc_train_step,
    vbpc_loss_and_grads,
    vbpc_train_step,
)
from vbpc.training.trainer import (
    default_vbpc_config,
    default_vbpc_two_moons_config,
    run_vbpc_beta_sweep,
    train_vbpc_mnist,
    train_vbpc_mnist_dataset,
    train_vbpc_two_moons_dataset,
)

__all__ = [
    "OPTAX_AVAILABLE",
    "VBPC_BETA_SWEEP",
    "VBPCDiagnostics",
    "VBPCLatentDiagnostics",
    "VBPCOptimizer",
    "VBPCRunLogger",
    "VBPCConfig",
    "VBPCStates",
    "VBPCWeightParams",
    "clamped_states",
    "default_vbpc_config",
    "default_vbpc_two_moons_config",
    "deterministic_pc_energy",
    "ece_from_logits",
    "evaluate",
    "expected_calibration_error",
    "forward_hidden",
    "forward_logits",
    "forward_top",
    "gaussian_kl_elementwise",
    "init_vbpc_latents",
    "init_vbpc_weight_params",
    "latent_pc_energy",
    "make_optimizer",
    "make_vbpc_presets",
    "make_vbpc_train_step",
    "mean_states",
    "mean_weight_variance",
    "memory_usage_mb",
    "negative_log_likelihood",
    "nll_from_logits",
    "optimizer_backend",
    "pc_energy",
    "pc_energy_per_layer",
    "pc_errors",
    "plot_beta_sweep",
    "reparameterize_states",
    "reparameterize_weight",
    "reparameterize_weight_matrices",
    "run_vbpc_beta_sweep",
    "sample_states",
    "sample_weight_epsilons",
    "sample_weight_matrices",
    "state_sigma",
    "total_vbpc_objective",
    "total_vbpc_objective_for_batch",
    "train_vbpc_mnist",
    "train_vbpc_mnist_dataset",
    "train_vbpc_two_moons_dataset",
    "validate_vbpc_config",
    "vbpc_accuracy",
    "vbpc_latent_inference",
    "vbpc_loss_and_grads",
    "vbpc_train_step",
    "weight_diagnostics",
    "weight_kl",
    "weight_shapes",
    "weight_sigma",
    "weight_variance_metric",
]

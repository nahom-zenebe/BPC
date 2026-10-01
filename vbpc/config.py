"""Configuration for the Variational Bayesian Predictive Coding (VBPC) module.

VBPC keeps the predictive-coding architecture and energy of the BPC package but
replaces the Matrix-Normal Wishart weight posterior with a *factorized Gaussian*

    q(W_l) = N(mu_W,l, diag(sigma_W,l^2))    (reparameterized: W = mu + sigma * eps)

and adds reparameterized Gaussian latent states

    q(z_l | x) = N(mu_l, diag(sigma_l^2))    (reparameterized: z = mu + sigma * eps)

trained with the objective

    L_total = L_PC + beta * L_KL.

Normalization conventions
-------------------------
The proposal writes the two objective terms as unnormalized sums.
``pc_reduction="sum"`` and ``kl_reduction="sum"`` reproduce those formulas
literally (``L_PC = 0.5 * sum_l e_l^T Sigma_l^{-1} e_l`` and
``L_KL = 0.5 * sum(mu_W^2 + sigma_W^2 - 1 - log sigma_W^2)``).

``"mean"`` divides ``L_PC`` by ``batch x features`` and ``L_KL`` by the number of
weight entries, so both terms become *batch-size independent* averages and
``beta`` is a genuine unitless trade-off between the data term and the prior.

Why ``"mean"`` is the default, and what is *not* a valid argument for it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The loss *values* under the two reductions are not comparable, but the
*gradient norms* are what decide how much ``beta`` matters.  Measured at
initialization on the two-moons preset (502 weights, ``n_train=1000``,
8-sample average of the real training step):

    reduction  batch   L_PC     L_KL     |grad_mu KL|/|grad_mu PC|
                                               |grad_logsig KL|/|grad_logsig PC|
    mean/mean  100     0.045    5.084    0.110                    8.40
    mean/mean  1000    0.045    5.084    0.111                    8.37
    sum/sum    100     64.8     1280.0   0.029                    1.83
    sum/sum    1000    634.1    1280.0   0.003                    0.19

Two consequences, both measured:

* ``sum``/``sum`` only brings the two *losses* within ~2x at ``batch_size=1000``.
  Because ``L_PC`` under ``"sum"`` grows linearly with the batch while ``L_KL``
  does not, shrinking the batch (which is required to get more than one
  optimizer step per epoch) makes ``sum``/``sum`` *worse*: at ``batch_size=100``
  the losses are 19.8x apart and ``beta`` moves the weight gradient 4x less than
  it does under ``"mean"``.  ``sum``/``sum`` also ties the effective learning
  rate to the batch size.
* Under ``"mean"`` the ``L_PC``/``L_KL`` *loss* ratio is ~110x, which looks
  alarming, but the corresponding *gradient* ratio on the weight means is only
  0.110 per unit ``beta``.  A loss ratio is therefore not evidence that
  ``beta`` is inert: ``L_KL`` carries a large additive constant
  (``-log sigma_W^2``, ~96% of its value at initialization) that contributes
  almost no gradient.

The previous version of this docstring claimed ``"mean"`` "places ``beta`` on
the interval swept by the proposal, ``beta in [0, 0.001, 0.01, 0.1, 1.0]``".
That claim does not follow from a loss ratio.  What does decide whether the
sweep probes anything is the gradient ratio, so the run logs
``pc_grad_norm`` / ``kl_grad_norm`` (see :mod:`vbpc.training.train_step`) and
those should be read directly instead.  The identity
``L_total = L_PC + beta * L_KL`` holds for every reduction choice.

Weight prior
------------
The prior is ``p(W_l) = N(0, tau^2 I)`` with ``tau = exp(prior_weight_log_sigma)``
(``prior_weight_log_sigma = 0`` gives the proposal's standard normal
``N(0, I)``).  It must be chosen consistently with ``init_weight_log_sigma``:
a unit prior against a thin initialization makes
``dKL/dlog_sigma = sigma^2 - tau^2 ~ -1``, which drives ``sigma_W`` up towards
``1`` -- a 20x noise inflation relative to the ``sigma = exp(-3)`` start -- and
once the run has more than a handful of steps (see ``batch_size`` below) that
inflates the sampled weights enough to destroy the learned mean network.  The
presets therefore set ``prior_weight_log_sigma = init_weight_log_sigma`` so the
prior is exactly the initialization distribution and that gradient is zero at
initialization.  Note the trade-off this implies: a tight prior is strong L2
shrinkage on ``mu_W`` (its ``grad_mu`` ratio is ~45x larger than the unit
prior's per unit ``beta``), so with a matched prior ``beta`` above ~0.1
legitimately collapses the weights.  The sweep is expected to show that.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import jax.numpy as jnp

from bpc.config import BATCH_SIZE, DTYPE, HIDDEN, LATENT_LR, LATENT_STEPS, MNIST_SOURCE, SEED


#: ``beta`` values required by the VBPC proposal.
VBPC_BETA_SWEEP = (0.0, 0.001, 0.01, 0.1, 1.0)

VBPC_MNIST_EPOCHS = 100
VBPC_TWO_MOONS_EPOCHS = 100
VBPC_TWO_MOONS_HIDDEN = 100

#: Two moons trains on ``n_train=1000`` points (see ``configs/vbpc_two_moons.yaml``).
#: ``batch_size`` must be *strictly smaller* than ``n_train``: with a single batch
#: per epoch ``batch_iterator`` yields exactly one optimizer step per epoch, so a
#: 100-epoch run performs only 100 Adam steps.  At ``weight_lr=1e-3`` that caps
#: the total movement of any weight at ~0.1, which is the same order as the
#: ``U(-1/sqrt(d_in), 1/sqrt(d_in))`` initialization, so the mean network never
#: leaves its initialization.  100 gives 10 steps/epoch and 1000 steps/run.
VBPC_TWO_MOONS_BATCH_SIZE = 100

#: ``tau = exp(VBPC_INIT_WEIGHT_LOG_SIGMA)`` is the standard deviation of the
#: weight prior.  Matching the prior to the initialization is what keeps
#: ``dKL/dlog_sigma = sigma^2 - tau^2`` at zero instead of ~-1 (see the module
#: docstring); it is applied to every preset below.
VBPC_INIT_WEIGHT_LOG_SIGMA = -3.0

_PC_REDUCTIONS = ("mean", "sum")
_KL_REDUCTIONS = ("mean", "sum")


@dataclass(frozen=True)
class VBPCConfig:
    """Configuration of one VBPC run.

    The field names shared with :class:`bpc.config.BPCConfig` (``seed``,
    ``epochs``, ``batch_size``, ``hidden``, ``hidden_layers``, ``normalize``,
    ``mnist_source``, ``mnist_npz``, ``latent_steps``, ``latent_lr``,
    ``hidden_init``, ``input_activation``, ``eval_batch_size``) keep the same
    meaning so that the existing dataset, logging and reproducibility helpers
    can be reused unchanged.
    """

    name: str
    seed: int = SEED
    epochs: int = VBPC_MNIST_EPOCHS
    batch_size: int = BATCH_SIZE
    hidden: int = HIDDEN
    hidden_layers: int = 2

    normalize: str = "zero_one"
    mnist_source: str = MNIST_SOURCE
    mnist_npz: Optional[str] = None

    # Objective: L_total = L_PC + beta * L_KL
    beta: float = 0.01
    pc_reduction: str = "mean"
    kl_reduction: str = "mean"

    # Weight variational parameter learning (q(W) = N(mu_W, diag(sigma_W^2))).
    weight_lr: float = 1e-3
    weight_grad_clip: Optional[float] = None
    init_weight_log_sigma: float = VBPC_INIT_WEIGHT_LOG_SIGMA
    #: log standard deviation of the Gaussian weight prior, p(W) = N(0, tau^2 I).
    #: ``0.0`` is the proposal's standard normal ``N(0, I)``.  Set it equal to
    #: ``init_weight_log_sigma`` so the prior matches the initialization and
    #: ``dKL/dlog_sigma = sigma_W^2 - tau^2`` starts at zero instead of ~-1
    #: (a -1 there inflates the sampled weight noise ~20x over a full run).
    prior_weight_log_sigma: float = VBPC_INIT_WEIGHT_LOG_SIGMA

    # Latent variational parameter inference (q(z_l) = N(mu_l, diag(sigma_l^2))).
    latent_steps: int = LATENT_STEPS
    latent_lr: float = LATENT_LR * 5.0
    init_latent_log_sigma: float = -3.0
    hidden_init: str = "feedforward"
    input_activation: str = "identity"

    # Per-layer error covariance Sigma_l = error_variance * I used by L_PC.
    error_variance: float = 1.0

    # Numerical guards for sigma = exp(log_sigma).
    weight_sigma_min: float = 1e-8
    weight_sigma_max: float = 1e3
    state_sigma_min: float = 1e-8
    state_sigma_max: float = 1e3

    eval_batch_size: int = 4096
    keep_best: bool = True


def validate_vbpc_config(cfg: VBPCConfig) -> None:
    """Reject unsupported VBPC configurations early."""

    if cfg.pc_reduction not in _PC_REDUCTIONS:
        raise ValueError(f"Unknown pc_reduction={cfg.pc_reduction}")
    if cfg.kl_reduction not in _KL_REDUCTIONS:
        raise ValueError(f"Unknown kl_reduction={cfg.kl_reduction}")
    if cfg.hidden_init not in _HIDDEN_INITS:
        raise ValueError(f"Unknown hidden_init={cfg.hidden_init}")
    if cfg.input_activation not in _INPUT_ACTIVATIONS:
        raise ValueError(f"Unknown input_activation={cfg.input_activation}")
    if cfg.normalize not in _NORMALIZATIONS:
        raise ValueError(f"Unknown normalize={cfg.normalize}")
    if cfg.beta < 0.0:
        raise ValueError(f"beta must be non-negative, got {cfg.beta}")
    if not jnp.isfinite(jnp.asarray(cfg.prior_weight_log_sigma, dtype=DTYPE)):
        raise ValueError(f"prior_weight_log_sigma must be finite, got {cfg.prior_weight_log_sigma}")
    if cfg.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {cfg.batch_size}")
    if cfg.error_variance <= 0.0:
        raise ValueError(f"error_variance must be positive, got {cfg.error_variance}")
    if cfg.weight_sigma_min <= 0.0 or cfg.weight_sigma_max <= cfg.weight_sigma_min:
        raise ValueError("weight_sigma_min/max must be positive and increasing.")
    if cfg.state_sigma_min <= 0.0 or cfg.state_sigma_max <= cfg.state_sigma_min:
        raise ValueError("state_sigma_min/max must be positive and increasing.")

_HIDDEN_INITS = ("feedforward", "zeros")
_INPUT_ACTIVATIONS = ("identity", "relu")
_NORMALIZATIONS = ("zero_one", "standardize_pixel", "standardize_scalar", "centered_m11")



def make_vbpc_presets() -> Dict[str, VBPCConfig]:
    """Named VBPC presets, mirroring :func:`bpc.config.make_presets`."""

    base = VBPCConfig(name="vbpc_base")
    return {
        "vbpc_mnist_default": replace(
            base,
            name="vbpc_mnist_default",
            epochs=VBPC_MNIST_EPOCHS,
            hidden=HIDDEN,
            hidden_layers=2,
            normalize="zero_one",
            beta=0.01,
        ),
        "vbpc_mnist_beta0": replace(
            base,
            name="vbpc_mnist_beta0",
            epochs=VBPC_MNIST_EPOCHS,
            hidden=HIDDEN,
            hidden_layers=2,
            normalize="zero_one",
            beta=0.0,
        ),
        "vbpc_mnist_standardized": replace(
            base,
            name="vbpc_mnist_standardized",
            epochs=VBPC_MNIST_EPOCHS,
            hidden=HIDDEN,
            hidden_layers=2,
            normalize="standardize_pixel",
            beta=0.01,
        ),
        "vbpc_two_moons": replace(
            base,
            name="vbpc_two_moons",
            epochs=VBPC_TWO_MOONS_EPOCHS,
            hidden=VBPC_TWO_MOONS_HIDDEN,
            hidden_layers=1,
            batch_size=VBPC_TWO_MOONS_BATCH_SIZE,
            beta=0.01,
            prior_weight_log_sigma=VBPC_INIT_WEIGHT_LOG_SIGMA,
        ),
    }

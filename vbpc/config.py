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

Because a raw weight sum scales with the number of weights (~1e5 for MNIST)
while the raw PC energy scales with the number of examples/features, the two
terms are not on a comparable scale and ``beta`` would only be meaningful over
many orders of magnitude.  The defaults therefore use ``"mean"`` reductions
(mean over batch and features for ``L_PC``, mean over weight entries for
``L_KL``), which places ``beta`` on the interval swept by the proposal,
``beta in [0, 0.001, 0.01, 0.1, 1.0]``.  The identity
``L_total = L_PC + beta * L_KL`` holds for every reduction choice.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

from bpc.config import BATCH_SIZE, HIDDEN, LATENT_LR, LATENT_STEPS, MNIST_SOURCE, SEED


#: ``beta`` values required by the VBPC proposal.
VBPC_BETA_SWEEP = (0.0, 0.001, 0.01, 0.1, 1.0)

VBPC_MNIST_EPOCHS = 100
VBPC_TWO_MOONS_EPOCHS = 100
VBPC_TWO_MOONS_HIDDEN = 100

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
    init_weight_log_sigma: float = -3.0

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
            batch_size=1000,
            beta=0.01,
        ),
    }

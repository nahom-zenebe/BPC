"""Variational objectives of VBPC: PC energy, analytic KL and ``L_total``.

Local predictive-coding energy (proposal notation)

    L_PC = 0.5 * sum_l e_l^T Sigma_l^{-1} e_l,   e_l = z_l - W_l f(z_{l+1})

with ``Sigma_l = error_variance * I`` (the fixed per-layer error covariance).
The predictor that produces ``z_l`` from its parent ``z_{l+1}`` is the same
weight matrix that the rest of the BPC package stores per layer; numbering
layers bottom-up (``z_0 = x`` input, ``z_L = y`` target) as the BPC package
does, the residual attached to weight ``W_l`` is written

    e_l = z_{l+1} - W_l f(z_l),

which is the proposal formula after the equivalent top-down relabelling of the
layer indices.  Activations, bias augmentation and the ReLU used between hidden
layers are imported from the shared BPC modules.

Analytic KL against a standard-normal prior:

    L_KL = 0.5 * sum(mu_W^2 + sigma_W^2 - 1 - log(sigma_W^2))

and the trained objective is ``L_total = L_PC + beta * L_KL``.  See
:mod:`vbpc.config` for the reduction conventions.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from bpc.activations import activation
from bpc.config import DTYPE, Array
from bpc.utils.tensor_ops import augment
from vbpc.config import VBPCConfig
from vbpc.posterior.weight_posterior import VBPCWeightParams, weight_kl


class VBPCStates(NamedTuple):
    """Variational parameters of the reparameterized Gaussian latent states.

    ``mu[l]`` and ``log_sigma[l]`` describe ``q(z_{l+1} | x) = N(mu, diag(sigma^2))``
    for hidden layer ``l + 1`` of ``layer_dims`` (i.e. layers ``1 .. L-1``).
    The input ``z_0 = x`` and the target ``z_L = y`` are clamped, not variational.
    """

    mu: Tuple[Array, ...]
    log_sigma: Tuple[Array, ...]


def state_sigma(log_sigma: Array, sigma_min: float = 1e-8, sigma_max: float = 1e3) -> Array:
    """Map unconstrained ``log_sigma`` to a strictly positive standard deviation."""

    return jnp.clip(jnp.exp(log_sigma), sigma_min, sigma_max)


def sample_state_epsilons(states: VBPCStates, key: Array) -> Tuple[Array, ...]:
    """Draw one ``eps ~ N(0, I)`` per hidden layer."""

    keys = jax.random.split(key, max(len(states.mu), 1))
    return tuple(jax.random.normal(keys[l], m.shape, dtype=m.dtype) for l, m in enumerate(states.mu))



def reparameterize_states(
    states: VBPCStates,
    epsilons: Tuple[Array, ...],
    sigma_min: float = 1e-8,
    sigma_max: float = 1e3,
) -> Tuple[Array, ...]:
    """``z_l = mu_l + sigma_l * eps_l`` for every hidden layer."""

    return tuple(
        m + state_sigma(ls, sigma_min, sigma_max) * eps
        for m, ls, eps in zip(states.mu, states.log_sigma, epsilons)
    )


def sample_states(
    states: VBPCStates,
    key: Array,
    sigma_min: float = 1e-8,
    sigma_max: float = 1e3,
) -> Tuple[Array, ...]:
    """Sample the hidden states through the reparameterization trick."""

    return reparameterize_states(states, sample_state_epsilons(states, key), sigma_min, sigma_max)


def mean_states(states: VBPCStates) -> Tuple[Array, ...]:
    """Variational means (the deterministic part of the reparameterization)."""

    return tuple(states.mu)


def clamped_states(hidden: Tuple[Array, ...], x: Array, y: Array) -> Tuple[Array, ...]:
    """Full state chain ``(z_0, z_1, ..., z_L)`` with ``z_0 = x`` and ``z_L = y``."""

    return (x,) + tuple(hidden) + (y,)


def pc_errors(
    z_states: Tuple[Array, ...],
    weights: Tuple[Array, ...],
    cfg: VBPCConfig,
) -> Tuple[Array, ...]:
    """Per-layer prediction residuals ``e_l = z_{l+1} - W_l f(z_l)``.

    ``z_states`` must be the full chain returned by :func:`clamped_states`.
    """

    if len(z_states) != len(weights) + 1:
        raise ValueError(
            f"Expected {len(weights) + 1} states for {len(weights)} weights, got {len(z_states)}."
        )
    errors = []
    for l, W in enumerate(weights):
        f_in = augment(activation(z_states[l], l, cfg.input_activation))
        errors.append(z_states[l + 1] - f_in @ W.T)
    return tuple(errors)


def pc_energy_per_layer(
    z_states: Tuple[Array, ...],
    weights: Tuple[Array, ...],
    cfg: VBPCConfig,
    reduction: Optional[str] = None,
) -> Tuple[Array, ...]:
    """``0.5 * e_l^T Sigma_l^{-1} e_l`` for every layer, with ``Sigma_l = error_variance * I``."""

    mode = reduction if reduction is not None else cfg.pc_reduction
    if mode not in ("mean", "sum"):
        raise ValueError(f"Unknown reduction={mode}")
    scale = 0.5 / jnp.asarray(cfg.error_variance, dtype=DTYPE)
    energies = []
    for e in pc_errors(z_states, weights, cfg):
        quadratic = jnp.sum(e * e) if mode == "sum" else jnp.mean(e * e)
        energies.append(scale * quadratic)
    return tuple(energies)


def pc_energy(
    z_states: Tuple[Array, ...],
    weights: Tuple[Array, ...],
    cfg: VBPCConfig,
    reduction: Optional[str] = None,
) -> Array:
    """Local predictive-coding energy ``L_PC`` summed over layers."""

    per_layer = pc_energy_per_layer(z_states, weights, cfg, reduction)
    return sum(per_layer, jnp.asarray(0.0, dtype=DTYPE))


def total_vbpc_objective(
    z_states: Tuple[Array, ...],
    weights: Tuple[Array, ...],
    weight_params: Tuple[VBPCWeightParams, ...],
    cfg: VBPCConfig,
) -> Tuple[Array, Tuple[Array, Array]]:
    """``L_total = L_PC + beta * L_KL`` together with the individual terms."""

    l_pc = pc_energy(z_states, weights, cfg)
    l_kl = weight_kl(
        weight_params,
        cfg.kl_reduction,
        cfg.weight_sigma_min,
        cfg.weight_sigma_max,
        cfg.prior_weight_log_sigma,
    )
    beta = jnp.asarray(cfg.beta, dtype=DTYPE)
    return l_pc + beta * l_kl, (l_pc, l_kl)


def total_vbpc_objective_for_batch(
    hidden: Tuple[Array, ...],
    x: Array,
    y: Array,
    weights: Tuple[Array, ...],
    weight_params: Tuple[VBPCWeightParams, ...],
    cfg: VBPCConfig,
) -> Tuple[Array, Tuple[Array, Array]]:
    """Same as :func:`total_vbpc_objective` but starting from hidden states and ``(x, y)``."""

    return total_vbpc_objective(clamped_states(hidden, x, y), weights, weight_params, cfg)

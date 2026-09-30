"""Reparameterized Gaussian latent-state inference for VBPC.

The latent variational distribution ``q(z_l | x) = N(mu_l, diag(sigma_l^2))`` is
represented by ``(mu_l, log_sigma_l)`` and sampled as ``z_l = mu_l + sigma_l * eps``.
Following the proposal, latent inference optimizes the *local predictive-coding
energy only*,

    L_PC = 0.5 * sum_l e_l^T Sigma_l^{-1} e_l,

i.e. the weight KL term is deliberately excluded here; it only enters the
weight/variational-parameter objective ``L_total = L_PC + beta * L_KL``.

Because no KL regularizes the latent scale, ``L_PC`` alone pushes the variational
latent scales ``sigma_l`` downwards; the diagnostics expose
``sigma_mean_per_step`` so the effect is observable in the logs.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from bpc.config import DTYPE, Array
from vbpc.config import VBPCConfig
from vbpc.inference.objectives import (
    VBPCStates,
    clamped_states,
    mean_states,
    pc_energy,
    sample_states,
    state_sigma,
)
from vbpc.layers.vbpc_network import forward_hidden
from vbpc.optim import make_optimizer
from vbpc.posterior.weight_posterior import VBPCWeightParams


class VBPCLatentDiagnostics(NamedTuple):
    """Trace of one latent-inference run (``T`` = ``cfg.latent_steps``)."""

    energy_per_step: Array
    mean_energy_per_step: Array
    grad_norm_per_step: Array
    log_sigma_grad_norm_per_step: Array
    state_norm_per_step: Array
    sigma_mean_per_step: Array
    init_grad_norm: Array
    final_grad_norm: Array
    init_energy: Array
    final_energy: Array


def init_vbpc_latents(
    weight_params: Tuple[VBPCWeightParams, ...],
    layer_dims: Tuple[int, ...],
    x: Array,
    cfg: VBPCConfig,
) -> VBPCStates:
    """Initialize ``q(z_l | x)`` from the mean-weight feedforward pass."""

    n_hidden = max(len(layer_dims) - 2, 0)
    if cfg.hidden_init == "feedforward":
        mu = forward_hidden(weight_params, layer_dims, x, cfg)
    elif cfg.hidden_init == "zeros":
        mu = tuple(jnp.zeros((x.shape[0], int(layer_dims[l])), dtype=x.dtype) for l in range(1, len(layer_dims) - 1))
    else: 
        raise ValueError(f"Unknown hidden_init={cfg.hidden_init}")
    log_sigma = tuple(
        jnp.full(m.shape, float(cfg.init_latent_log_sigma), dtype=m.dtype) for m in mu
    )
    if len(mu) != n_hidden:  # pragma: no cover - defensive
        raise ValueError(f"Expected {n_hidden} hidden layers, got {len(mu)}.")
    return VBPCStates(mu=mu, log_sigma=log_sigma)


def latent_pc_energy(
    states: VBPCStates,
    weights: Tuple[Array, ...],
    x: Array,
    y: Array,
    cfg: VBPCConfig,
    key: Array,
) -> Array:
    """Single-sample reparameterized estimate of ``L_PC`` used for latent gradients."""

    hidden = sample_states(states, key, cfg.state_sigma_min, cfg.state_sigma_max)
    return pc_energy(clamped_states(hidden, x, y), weights, cfg)


def deterministic_pc_energy(
    states: VBPCStates,
    weights: Tuple[Array, ...],
    x: Array,
    y: Array,
    cfg: VBPCConfig,
) -> Array:
    """``L_PC`` at the variational means (used for convergence diagnostics)."""

    return pc_energy(clamped_states(mean_states(states), x, y), weights, cfg)


def vbpc_latent_inference(
    weight_params: Tuple[VBPCWeightParams, ...],
    layer_dims: Tuple[int, ...],
    weights: Tuple[Array, ...],
    x: Array,
    y: Array,
    cfg: VBPCConfig,
    key: Array,
) -> Tuple[VBPCStates, VBPCLatentDiagnostics]:
    """Adam inference of ``(mu_l, log_sigma_l)`` minimizing ``L_PC`` only."""

    steps = int(cfg.latent_steps)
    if steps < 1:
        raise ValueError(f"latent_steps must be >= 1, got {steps}")
    n_states = max(len(weights) - 1, 0)

    states = init_vbpc_latents(weight_params, layer_dims, x, cfg)
    optimizer = make_optimizer(cfg.latent_lr)
    opt_state = optimizer.init(states)

    e_log = jnp.zeros((steps,), dtype=DTYPE)
    em_log = jnp.zeros((steps,), dtype=DTYPE)
    g_log = jnp.zeros((steps, n_states), dtype=DTYPE)
    gl_log = jnp.zeros((steps, n_states), dtype=DTYPE)
    z_log = jnp.zeros((steps, n_states), dtype=DTYPE)
    s_log = jnp.zeros((steps, n_states), dtype=DTYPE)

    def body(carry, i):
        states, opt_state, key, e_log, em_log, g_log, gl_log, z_log, s_log = carry
        key, sub = jax.random.split(key)
        energy, grads = jax.value_and_grad(latent_pc_energy)(states, weights, x, y, cfg, sub)
        e_log = e_log.at[i].set(energy)
        em_log = em_log.at[i].set(deterministic_pc_energy(states, weights, x, y, cfg))
        g_log = g_log.at[i].set(jnp.stack([jnp.linalg.norm(g) for g in grads.mu]))
        gl_log = gl_log.at[i].set(jnp.stack([jnp.linalg.norm(g) for g in grads.log_sigma]))
        z_log = z_log.at[i].set(jnp.stack([jnp.linalg.norm(m) for m in states.mu]))
        s_log = s_log.at[i].set(
            jnp.stack([jnp.mean(state_sigma(ls, cfg.state_sigma_min, cfg.state_sigma_max)) for ls in states.log_sigma])
        )
        states, opt_state = optimizer.update(states, grads, opt_state)
        return (states, opt_state, key, e_log, em_log, g_log, gl_log, z_log, s_log), None

    carry = (states, opt_state, key, e_log, em_log, g_log, gl_log, z_log, s_log)
    (states, _, _, e_log, em_log, g_log, gl_log, z_log, s_log), _ = jax.lax.scan(
        body, carry, jnp.arange(steps)
    )

    diag = VBPCLatentDiagnostics(
        energy_per_step=e_log,
        mean_energy_per_step=em_log,
        grad_norm_per_step=g_log,
        log_sigma_grad_norm_per_step=gl_log,
        state_norm_per_step=z_log,
        sigma_mean_per_step=s_log,
        init_grad_norm=g_log[0],
        final_grad_norm=g_log[-1],
        init_energy=em_log[0],
        final_energy=em_log[-1],
    )
    return states, diag

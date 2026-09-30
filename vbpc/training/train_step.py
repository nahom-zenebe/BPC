"""JIT-compiled VBPC training steps.

One VBPC step performs the two stages required by the proposal:

1. **Latent inference** -- draw one reparameterized weight sample
   ``W = mu_W + sigma_W * eps`` and optimize ``q(z_l | x)`` with Adam on the
   *local PC energy only* (``L_PC``); the weight KL term is excluded here.
2. **Weight/variational-parameter learning** -- freeze the inferred latents
   (``stop_gradient``), sample ``z`` from the frozen ``q(z_l | x)`` and take an
   Adam step on ``L_total = L_PC + beta * L_KL`` with respect to
   ``(mu_W, log_sigma_W)``, reusing the *same* weight epsilon so that the step is
   a proper single-sample reparameterized gradient estimate of ``E_q[L_PC]``.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from bpc.config import DTYPE, Array
from vbpc.config import VBPCConfig
from vbpc.inference.latent_inference import (
    VBPCLatentDiagnostics,
    vbpc_latent_inference,
)
from vbpc.inference.objectives import (
    VBPCStates,
    clamped_states,
    pc_energy,
    reparameterize_states,
    sample_state_epsilons,
)
from vbpc.optim import VBPCOptimizer, make_optimizer
from vbpc.posterior.weight_posterior import (
    VBPCWeightParams,
    mean_weight_variance,
    reparameterize_weight_matrices,
    sample_weight_epsilons,
    weight_kl,
)


class VBPCDiagnostics(NamedTuple):
    """Per-step VBPC diagnostics written to the batch logs."""

    pc_energy: Array
    weight_kl: Array
    total_loss: Array
    mean_weight_variance: Array
    beta: Array
    eps_norm: Array
    latent_energy_init: Array
    latent_energy_final: Array
    latent_grad_norm_init: Array
    latent_grad_norm_final: Array
    latent_sigma_mean_final: Array


def resolve_beta(cfg: VBPCConfig, beta: Optional[Array]) -> Array:
    """Beta as an array, defaulting to ``cfg.beta``."""

    return jnp.asarray(cfg.beta if beta is None else beta, dtype=DTYPE)


def sample_weights_and_latents(
    weight_params: Tuple[VBPCWeightParams, ...],
    layer_dims: Tuple[int, ...],
    xb: Array,
    yb: Array,
    cfg: VBPCConfig,
    key: Array,
) -> Tuple[Tuple[Array, ...], Tuple[Array, ...], VBPCStates, VBPCLatentDiagnostics]:
    """One weight sample plus latent inference on ``L_PC`` for a batch."""

    key_w, key_latent = jax.random.split(key)
    epsilons = sample_weight_epsilons(weight_params, key_w)
    weights = reparameterize_weight_matrices(
        weight_params, epsilons, cfg.weight_sigma_min, cfg.weight_sigma_max
    )
    latents, latent_diag = vbpc_latent_inference(
        weight_params, layer_dims, weights, xb, yb, cfg, key_latent
    )
    return weights, epsilons, latents, latent_diag


def vbpc_loss_and_grads(
    cfg: VBPCConfig,
    layer_dims: Tuple[int, ...],
    weight_params: Tuple[VBPCWeightParams, ...],
    xb: Array,
    yb: Array,
    key: Array,
    beta: Optional[Array] = None,
) -> Tuple[Array, Tuple[Array, Array], Any, VBPCStates, VBPCLatentDiagnostics, Tuple[Array, ...]]:
    """``L_total``, its two terms, gradients and the inferred latents for one batch."""

    weights, epsilons, latents, latent_diag = sample_weights_and_latents(
        weight_params, layer_dims, xb, yb, cfg, key
    )
    frozen = jax.lax.stop_gradient(latents)
    eps_states = sample_state_epsilons(frozen, jax.random.fold_in(key, 1))
    beta_value = resolve_beta(cfg, beta)

    def loss_fn(params):
        sampled_weights = reparameterize_weight_matrices(
            params, epsilons, cfg.weight_sigma_min, cfg.weight_sigma_max
        )
        hidden = reparameterize_states(
            frozen, eps_states, cfg.state_sigma_min, cfg.state_sigma_max
        )
        l_pc = pc_energy(clamped_states(hidden, xb, yb), sampled_weights, cfg)
        l_kl = weight_kl(params, cfg.kl_reduction, cfg.weight_sigma_min, cfg.weight_sigma_max)
        return l_pc + beta_value * l_kl, (l_pc, l_kl)

    (total, (l_pc, l_kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(weight_params)
    return total, (l_pc, l_kl), grads, latents, latent_diag, epsilons


def build_step_diagnostics(
    cfg: VBPCConfig,
    new_params: Tuple[VBPCWeightParams, ...],
    terms: Tuple[Array, Array],
    total: Array,
    latent_diag: VBPCLatentDiagnostics,
    epsilons: Tuple[Array, ...],
    beta: Array,
) -> VBPCDiagnostics:
    """Bundle the per-step diagnostics."""

    l_pc, l_kl = terms
    eps_norm = jnp.sqrt(
        sum((jnp.sum(e * e) for e in epsilons), jnp.asarray(0.0, dtype=DTYPE))
    )
    return VBPCDiagnostics(
        pc_energy=l_pc,
        weight_kl=l_kl,
        total_loss=total,
        mean_weight_variance=mean_weight_variance(
            new_params, cfg.weight_sigma_min, cfg.weight_sigma_max
        ),
        beta=beta,
        eps_norm=eps_norm,
        latent_energy_init=latent_diag.init_energy,
        latent_energy_final=latent_diag.final_energy,
        latent_grad_norm_init=latent_diag.init_grad_norm,
        latent_grad_norm_final=latent_diag.final_grad_norm,
        latent_sigma_mean_final=latent_diag.sigma_mean_per_step[-1],
    )


def vbpc_train_step(
    cfg: VBPCConfig,
    layer_dims: Tuple[int, ...],
    weight_params: Tuple[VBPCWeightParams, ...],
    opt_state: Any,
    xb: Array,
    yb: Array,
    key: Array,
    optimizer: VBPCOptimizer,
    beta: Optional[Array] = None,
) -> Tuple[Tuple[VBPCWeightParams, ...], Any, VBPCDiagnostics]:
    """One VBPC step: latent inference on ``L_PC`` then an Adam step on ``L_total``."""

    beta_value = resolve_beta(cfg, beta)
    total, terms, grads, _, latent_diag, epsilons = vbpc_loss_and_grads(
        cfg, layer_dims, weight_params, xb, yb, key, beta_value
    )
    new_params, new_opt_state = optimizer.update(weight_params, grads, opt_state)
    diag = build_step_diagnostics(cfg, new_params, terms, total, latent_diag, epsilons, beta_value)
    return new_params, new_opt_state, diag


def make_vbpc_train_step(
    cfg: VBPCConfig,
    layer_dims: Tuple[int, ...],
    optimizer: Optional[VBPCOptimizer] = None,
):
    """Build the JIT-compiled VBPC step used by the trainer.

    Returns ``(train_step, optimizer)``; the optimizer keeps its Adam moments in
    ``opt_state`` across steps, so the returned instance must be reused for the
    whole run.
    """

    optimizer = optimizer if optimizer is not None else make_optimizer(cfg.weight_lr, cfg.weight_grad_clip)

    @jax.jit
    def train_step(weight_params, opt_state, xb, yb, key, beta):
        return vbpc_train_step(
            cfg, layer_dims, weight_params, opt_state, xb, yb, key, optimizer, beta
        )

    return train_step, optimizer

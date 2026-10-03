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
    make_latent_optimizer,
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
    weight_kl_grads,
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
    #: ``||grad_{mu_W} L_PC||`` -- the data term's contribution, in isolation.
    pc_grad_norm: Array
    #: ``||grad_{mu_W} (beta * L_KL)||`` -- the prior's contribution, in isolation.
    kl_grad_norm: Array
    #: ``kl_grad_norm / (pc_grad_norm + kl_grad_norm)``, the relative magnitude of
    #: the prior's contribution to the weight-mean gradient.  ``0`` means ``beta``
    #: is doing nothing, ``~0.5`` means the two terms are in parity, ``~1`` means
    #: the prior dominates the update (and the weight means will collapse).
    kl_grad_share: Array
    #: Signed mean of ``dL_KL/dlog_sigma_W`` over all weight entries, at the
    #: pre-update parameters.  A large negative value means the prior is actively
    #: inflating the sampled weight noise; it is ~0 when
    #: ``prior_weight_log_sigma == init_weight_log_sigma`` and ~-0.11 (per
    #: ``"mean"`` reduction) under the proposal's unit prior.
    kl_log_sigma_grad: Array


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
    latent_opt_state=None,
) -> Tuple[Tuple[Array, ...], Tuple[Array, ...], VBPCStates, VBPCLatentDiagnostics, Any]:
    """One weight sample plus latent inference on ``L_PC`` for a batch.

    Returns the updated ``latent_opt_state`` so the caller can persist Adam
    moments across batches within an epoch.
    """

    key_w, key_latent = jax.random.split(key)
    epsilons = sample_weight_epsilons(weight_params, key_w)
    weights = reparameterize_weight_matrices(
        weight_params, epsilons, cfg.weight_sigma_min, cfg.weight_sigma_max
    )
    latents, latent_diag, new_latent_opt_state = vbpc_latent_inference(
        weight_params, layer_dims, weights, xb, yb, cfg, key_latent, latent_opt_state
    )
    return weights, epsilons, latents, latent_diag, new_latent_opt_state


def vbpc_loss_and_grads(
    cfg: VBPCConfig,
    layer_dims: Tuple[int, ...],
    weight_params: Tuple[VBPCWeightParams, ...],
    xb: Array,
    yb: Array,
    key: Array,
    beta: Optional[Array] = None,
    latent_opt_state=None,
) -> Tuple[Array, Tuple[Array, Array], Any, VBPCStates, VBPCLatentDiagnostics, Tuple[Array, ...], Any, Any]:
    """``L_total``, its two terms, gradients and the inferred latents for one batch.

    Gradients are averaged over ``cfg.n_weight_samples`` independent weight
    epsilon draws to reduce single-sample variance.  Returns the updated
    ``latent_opt_state`` as the last element.
    """

    weights, epsilons, latents, latent_diag, new_latent_opt_state = sample_weights_and_latents(
        weight_params, layer_dims, xb, yb, cfg, key, latent_opt_state
    )
    frozen = jax.lax.stop_gradient(latents)
    beta_value = resolve_beta(cfg, beta)

    def kl_only(params):
        return weight_kl(
            params, cfg.kl_reduction, cfg.weight_sigma_min, cfg.weight_sigma_max,
            cfg.prior_weight_log_sigma,
        )

    def loss_fn_for_eps(params, eps, eps_state_key):
        """L_PC for one weight epsilon sample with a fresh state epsilon."""
        sampled_weights = reparameterize_weight_matrices(
            params, eps, cfg.weight_sigma_min, cfg.weight_sigma_max
        )
        eps_states = sample_state_epsilons(frozen, eps_state_key)
        hidden = reparameterize_states(
            frozen, eps_states, cfg.state_sigma_min, cfg.state_sigma_max
        )
        return pc_energy(clamped_states(hidden, xb, yb), sampled_weights, cfg)

    def loss_fn(params):
        # Average L_PC gradient over n_weight_samples independent weight epsilons.
        n = int(cfg.n_weight_samples)
        keys = jax.random.split(jax.random.fold_in(key, 2), n)
        eps_keys = jax.random.split(jax.random.fold_in(key, 3), n)
        if n == 1:
            l_pc = loss_fn_for_eps(params, epsilons, eps_keys[0])
        else:
            extra_eps = [
                sample_weight_epsilons(params, keys[i]) for i in range(1, n)
            ]
            all_eps = [epsilons] + extra_eps
            l_pc = sum(loss_fn_for_eps(params, e, eps_keys[i]) for i, e in enumerate(all_eps)) / n
        l_kl = kl_only(params)
        return l_pc + beta_value * l_kl, (l_pc, l_kl)

    (total, (l_pc, l_kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(weight_params)
    kl_grads = weight_kl_grads(
        weight_params, cfg.kl_reduction, cfg.weight_sigma_min, cfg.weight_sigma_max,
        cfg.prior_weight_log_sigma,
    )
    return total, (l_pc, l_kl), grads, latents, latent_diag, epsilons, kl_grads, new_latent_opt_state


def _grad_norm(grads: Any, field: str) -> Array:
    """L2 norm over every leaf of a ``VBPCWeightParams`` gradient tree."""

    return jnp.sqrt(
        sum(
            (jnp.sum(getattr(g, field) ** 2) for g in grads),
            jnp.asarray(0.0, dtype=DTYPE),
        )
    )


def build_step_diagnostics(
    cfg: VBPCConfig,
    new_params: Tuple[VBPCWeightParams, ...],
    terms: Tuple[Array, Array],
    total: Array,
    latent_diag: VBPCLatentDiagnostics,
    epsilons: Tuple[Array, ...],
    beta: Array,
    total_grads: Any,
    kl_grads: Any,
) -> VBPCDiagnostics:
    """Bundle the per-step diagnostics."""

    l_pc, l_kl = terms
    eps_norm = jnp.sqrt(
        sum((jnp.sum(e * e) for e in epsilons), jnp.asarray(0.0, dtype=DTYPE))
    )
    # grad(L_total) = grad(L_PC) + beta * grad(L_KL) exactly, so subtracting the
    # prior's contribution tree-wise recovers grad(L_PC) without a second backward
    # pass through the (expensive) PC graph.  Subtracting norms instead would be
    # wrong: it can go negative when the two gradients largely cancel.
    pc_grads = jax.tree.map(lambda g, k: g - beta * k, total_grads, kl_grads)
    pc_grad_norm = _grad_norm(pc_grads, "mu")
    kl_grad_norm = beta * _grad_norm(kl_grads, "mu")
    # Relative magnitude of the prior's contribution, bounded in [0, 1] and
    # monotone in beta.  (Dividing by ||grad(L_total)|| instead would be the
    # "share of the realised update", but the two gradients can partially cancel,
    # which can push that ratio above 1 and makes it a poor inertness indicator.)
    kl_grad_share = kl_grad_norm / jnp.maximum(
        pc_grad_norm + kl_grad_norm, jnp.asarray(1e-30, dtype=DTYPE)
    )
    # dL_KL/dlog_sigma_W, unweighted by beta, to expose prior/init mismatch.  This
    # is a *signed mean* over all entries, not a norm: the sign is the whole
    # point (negative => the prior is inflating the sampled weight noise,
    # positive => it is shrinking it), and a norm would discard it.
    n_entries = max(sum(int(p.mu.size) for p in new_params), 1)
    kl_log_sigma_grad = sum(
        (jnp.sum(g.log_sigma) for g in kl_grads), jnp.asarray(0.0, dtype=DTYPE)
    ) / jnp.asarray(n_entries, dtype=DTYPE)
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
        pc_grad_norm=pc_grad_norm,
        kl_grad_norm=kl_grad_norm,
        kl_grad_share=kl_grad_share,
        kl_log_sigma_grad=kl_log_sigma_grad,
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
    latent_opt_state=None,
) -> Tuple[Tuple[VBPCWeightParams, ...], Any, VBPCDiagnostics, Any]:
    """One VBPC step: latent inference on ``L_PC`` then an Adam step on ``L_total``.

    Returns ``(new_weight_params, new_opt_state, diag, new_latent_opt_state)``.
    The caller should pass ``new_latent_opt_state`` back on the next call within
    the same epoch and reset it to ``None`` at the start of each new epoch.
    """

    beta_value = resolve_beta(cfg, beta)
    total, terms, grads, _, latent_diag, epsilons, kl_grads, new_latent_opt_state = vbpc_loss_and_grads(
        cfg, layer_dims, weight_params, xb, yb, key, beta_value, latent_opt_state
    )
    new_params, new_opt_state = optimizer.update(weight_params, grads, opt_state)
    diag = build_step_diagnostics(
        cfg, new_params, terms, total, latent_diag, epsilons, beta_value, grads, kl_grads
    )
    return new_params, new_opt_state, diag, new_latent_opt_state


def make_vbpc_train_step(
    cfg: VBPCConfig,
    layer_dims: Tuple[int, ...],
    optimizer: Optional[VBPCOptimizer] = None,
):
    """Build the JIT-compiled VBPC step used by the trainer.

    Returns ``(train_step, optimizer, latent_optimizer)``.
    - ``optimizer`` keeps weight Adam moments in ``opt_state`` across steps.
    - ``latent_optimizer`` is used by the trainer to initialize a fresh
      ``latent_opt_state`` at the start of each epoch; the state is then
      threaded through batches so latent Adam moments persist within an epoch.
    """

    optimizer = optimizer if optimizer is not None else make_optimizer(cfg.weight_lr, cfg.weight_grad_clip)
    latent_optimizer = make_latent_optimizer(cfg)

    @jax.jit
    def train_step(weight_params, opt_state, xb, yb, key, beta, latent_opt_state):
        return vbpc_train_step(
            cfg, layer_dims, weight_params, opt_state, xb, yb, key, optimizer, beta, latent_opt_state
        )

    return train_step, optimizer, latent_optimizer

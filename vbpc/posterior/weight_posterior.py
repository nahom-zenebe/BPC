"""Factorized Gaussian variational posterior over VBPC weights.

VBPC replaces the Matrix-Normal Wishart posterior used by the BPC package with a
diagonal Gaussian per layer,

    q(W_l) = N(mu_W,l, diag(sigma_W,l^2)),

reparameterized as ``W_l = mu_W,l + sigma_W,l * eps`` with ``eps ~ N(0, I)``.
The prior is the standard normal

    p(W_l) = N(0, I),

so the analytic KL term of the objective is

    L_KL = 0.5 * sum(mu_W^2 + sigma_W^2 - 1 - log(sigma_W^2)).

Weights are stored in the same layout as the BPC package: layer ``l`` has shape
``(layer_dims[l + 1], layer_dims[l] + 1)``, the extra column being the bias that
multiplies the constant ``1`` appended by :func:`bpc.utils.tensor_ops.augment`.
Bias entries are Gaussian too, and therefore contribute to ``L_KL`` as well.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp

from bpc.config import DTYPE, Array


class VBPCWeightParams(NamedTuple):
    """Variational parameters of the factorized Gaussian ``q(W_l)``."""

    mu: Array
    log_sigma: Array


def weight_shapes(layer_dims: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    """Weight shape of every layer, including the augmented bias column."""

    return tuple(
        (int(layer_dims[l + 1]), int(layer_dims[l]) + 1)
        for l in range(len(layer_dims) - 1)
    )


def init_vbpc_weight_params(
    layer_dims: Sequence[int],
    key: Array,
    init_log_sigma: float = -6.0,
) -> Tuple[VBPCWeightParams, ...]:
    """Initialize ``q(W)`` with the BPC-style uniform mean and a fixed thin scale.

    The mean uses the same ``1 / sqrt(d_in)`` uniform bound as
    :func:`bpc.priors.weight_prior.init_posterior`, so VBPC and BPC start from an
    identical mean network; only the width of the Gaussian differs.
    """

    keys = jax.random.split(key, max(len(layer_dims) - 1, 1))
    params = []
    for l, (d_out, d_in_aug) in enumerate(weight_shapes(layer_dims)):
        bound = 1.0 / jnp.sqrt(jnp.asarray(d_in_aug - 1, dtype=DTYPE))
        mu = jax.random.uniform(keys[l], (d_out, d_in_aug), minval=-bound, maxval=bound, dtype=DTYPE)
        log_sigma = jnp.full((d_out, d_in_aug), float(init_log_sigma), dtype=DTYPE)
        params.append(VBPCWeightParams(mu=mu, log_sigma=log_sigma))
    return tuple(params)


def weight_sigma(log_sigma: Array, sigma_min: float = 1e-8, sigma_max: float = 1e3) -> Array:
    """Map unconstrained ``log_sigma`` to a strictly positive standard deviation."""

    return jnp.clip(jnp.exp(log_sigma), sigma_min, sigma_max)


def reparameterize_weight(mu: Array, sigma: Array, epsilon: Array) -> Array:
    """Gaussian reparameterization ``W = mu + sigma * eps``."""

    return mu + sigma * epsilon


def sample_weight_epsilons(params: Tuple[VBPCWeightParams, ...], key: Array) -> Tuple[Array, ...]:
    """Draw one ``eps ~ N(0, I)`` per layer, shaped like the corresponding mean."""

    keys = jax.random.split(key, max(len(params), 1))
    return tuple(
        jax.random.normal(keys[l], p.mu.shape, dtype=p.mu.dtype) for l, p in enumerate(params)
    )


def reparameterize_weight_matrices(
    params: Tuple[VBPCWeightParams, ...],
    epsilons: Tuple[Array, ...],
    sigma_min: float = 1e-8,
    sigma_max: float = 1e3,
) -> Tuple[Array, ...]:
    """Turn fixed epsilons into weight matrices ``W_l = mu_l + sigma_l * eps_l``."""

    return tuple(
        reparameterize_weight(p.mu, weight_sigma(p.log_sigma, sigma_min, sigma_max), eps)
        for p, eps in zip(params, epsilons)
    )


def sample_weight_matrices(
    params: Tuple[VBPCWeightParams, ...],
    key: Array,
    sigma_min: float = 1e-8,
    sigma_max: float = 1e3,
) -> Tuple[Array, ...]:
    """Sample one weight matrix per layer through the reparameterization trick."""

    return reparameterize_weight_matrices(
        params, sample_weight_epsilons(params, key), sigma_min, sigma_max
    )


def gaussian_kl_elementwise(mu: Array, sigma: Array, log_sigma: Array) -> Array:
    """``0.5 * (mu^2 + sigma^2 - 1 - log(sigma^2))`` against a standard normal prior."""

    return 0.5 * (mu ** 2 + sigma ** 2 - 1.0 - 2.0 * log_sigma)


def weight_kl(
    params: Tuple[VBPCWeightParams, ...],
    reduction: str = "mean",
    sigma_min: float = 1e-8,
    sigma_max: float = 1e3,
) -> Array:
    """Analytic ``KL(q(W) || N(0, I))`` summed over layers.

    ``reduction="sum"`` is the literal proposal formula (sum over every weight
    entry); ``reduction="mean"`` averages over the entries of each layer before
    summing over layers, which keeps ``L_KL`` on the same scale as the
    batch-averaged ``L_PC`` term (see :mod:`vbpc.config`).
    """

    if reduction not in ("mean", "sum"):
        raise ValueError(f"Unknown reduction={reduction}")
    total = jnp.asarray(0.0, dtype=DTYPE)
    for p in params:
        sigma = weight_sigma(p.log_sigma, sigma_min, sigma_max)
        elementwise = gaussian_kl_elementwise(p.mu, sigma, jnp.log(sigma))
        total = total + (jnp.mean(elementwise) if reduction == "mean" else jnp.sum(elementwise))
    return total


def mean_weight_variance(
    params: Tuple[VBPCWeightParams, ...],
    sigma_min: float = 1e-8,
    sigma_max: float = 1e3,
) -> Array:
    """Mean of ``sigma_W^2`` over every weight entry (the logged weight variance)."""

    if not params:
        return jnp.asarray(0.0, dtype=DTYPE)
    count = sum(int(p.mu.size) for p in params)
    total = sum(
        (jnp.sum(weight_sigma(p.log_sigma, sigma_min, sigma_max) ** 2) for p in params),
        jnp.asarray(0.0, dtype=DTYPE),
    )
    return total / jnp.asarray(count, dtype=DTYPE)


def weight_diagnostics(
    params: Tuple[VBPCWeightParams, ...],
    sigma_min: float = 1e-8,
    sigma_max: float = 1e3,
) -> Dict[str, float]:
    """Per-layer mean/scale diagnostics used in logs and manifests."""

    out: Dict[str, float] = {}
    for l, p in enumerate(params):
        sigma = weight_sigma(p.log_sigma, sigma_min, sigma_max)
        out[f"mu{l}_norm"] = float(jnp.linalg.norm(p.mu))
        out[f"sigma{l}_mean"] = float(jnp.mean(sigma))
        out[f"var{l}_mean"] = float(jnp.mean(sigma ** 2))
        out[f"var{l}_min"] = float(jnp.min(sigma ** 2))
        out[f"var{l}_max"] = float(jnp.max(sigma ** 2))
        out[f"log_sigma{l}_min"] = float(jnp.min(p.log_sigma))
        out[f"log_sigma{l}_max"] = float(jnp.max(p.log_sigma))
    out["mean_weight_variance"] = float(mean_weight_variance(params, sigma_min, sigma_max))
    return out


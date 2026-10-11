"""Evaluation metrics and resource diagnostics for VBPC.

All metrics are computed from the mean-weight forward pass
(:func:`vbpc.layers.vbpc_network.forward_logits`) in the same batched style
as :func:`bpc.training.metrics.classification_accuracy`, and they cover the
quantities the VBPC experiment must log: accuracy, negative log-likelihood,
expected calibration error and the mean weight variance.
"""

from __future__ import annotations

import resource
import sys
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from bpc.activations import activation
from bpc.config import DTYPE, Array
from bpc.utils.tensor_ops import augment
from vbpc.config import VBPCConfig
from vbpc.layers.vbpc_network import forward_logits
from vbpc.posterior.weight_posterior import (
    VBPCWeightParams,
    mean_weight_variance,
    sample_weight_matrices,
    weight_sigma,
)


def softmax(logits: Array) -> Array:
    return jax.nn.softmax(logits, axis=-1)


def collect_logits(
    weight_params: Tuple[VBPCWeightParams, ...],
    x: Array,
    cfg: VBPCConfig,
) -> Array:
    """Batched mean-weight logits for a whole split."""

    pred_fn = jax.jit(lambda p, xb: forward_logits(p, xb, cfg))
    total = int(x.shape[0])
    chunks: List[Array] = []
    for start in range(0, total, cfg.eval_batch_size):
        chunks.append(pred_fn(weight_params, x[start:start + cfg.eval_batch_size]))
    if not chunks:
        return jnp.zeros((0, 0), dtype=DTYPE)
    return jnp.concatenate(chunks, axis=0)


def _forward_logits_raw(
    weights: Tuple[Array, ...],
    x: Array,
    cfg: VBPCConfig,
) -> Array:
    """Forward pass with raw weight matrices (no ``.mu`` indirection)."""
    z = x
    for l, W in enumerate(weights):
        z = augment(activation(z, l, cfg.input_activation)) @ W.T
    return z / jnp.asarray(cfg.error_variance, dtype=x.dtype)


def collect_mc_probs(
    weight_params: Tuple[VBPCWeightParams, ...],
    x: Array,
    key: Array,
    cfg: VBPCConfig,
    n_samples: int = 30,
) -> np.ndarray:
    """Monte Carlo posterior-predictive probabilities averaged over ``n_samples`` weight draws."""
    keys = jax.random.split(key, n_samples)
    pred_fn = jax.jit(lambda w, xb: jax.nn.softmax(_forward_logits_raw(w, xb, cfg), axis=-1))
    sum_probs: np.ndarray | None = None
    for k in keys:
        w = sample_weight_matrices(weight_params, k, cfg.weight_sigma_min, cfg.weight_sigma_max)
        chunks = [
            np.asarray(pred_fn(w, x[s:s + cfg.eval_batch_size]))
            for s in range(0, int(x.shape[0]), cfg.eval_batch_size)
        ]
        probs = np.concatenate(chunks, axis=0)
        sum_probs = probs if sum_probs is None else sum_probs + probs
    return sum_probs / n_samples


def mc_nll_from_probs(probs: np.ndarray, y: Array, eps: float = 1e-12) -> float:
    labels = np.asarray(jnp.argmax(y, axis=-1))
    picked = probs[np.arange(labels.shape[0]), labels]
    return float(-np.mean(np.log(np.maximum(picked, eps))))


def mc_ece_from_probs(probs: np.ndarray, y: Array, n_bins: int = 15) -> float:
    labels = np.asarray(jnp.argmax(y, axis=-1))
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    n = confidence.shape[0]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & (confidence <= hi) if i == 0 else (confidence > lo) & (confidence <= hi)
        count = int(mask.sum())
        if count:
            ece += (count / n) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def predictive_entropy_from_probs(probs: np.ndarray, eps: float = 1e-12) -> float:
    """H[p̄] = -Σ_c p̄_c log p̄_c, averaged over the dataset.

    For mean-weight probs this is the entropy of the point-estimate predictive;
    for MC-averaged probs it is the full posterior-predictive entropy.
    """
    return float(-np.mean(np.sum(probs * np.log(np.maximum(probs, eps)), axis=1)))


def latent_gaussian_entropy(
    weight_params: Tuple[VBPCWeightParams, ...],
    cfg: VBPCConfig,
) -> float:
    """Analytic differential entropy of q(W) = Π_l N(mu_l, diag(sigma_l²)).

    H[q] = Σ_l Σ_ij 0.5 * log(2π e σ_{l,ij}²)

    This is the entropy of the *latent weight distribution*, distinct from the
    classification NLL (cross-entropy of softmax outputs) logged elsewhere.
    """
    total = jnp.asarray(0.0, dtype=DTYPE)
    for p in weight_params:
        sigma = weight_sigma(p.log_sigma, cfg.weight_sigma_min, cfg.weight_sigma_max)
        total = total + jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e * sigma ** 2))
    return float(total)


def mc_evaluate(
    weight_params: Tuple[VBPCWeightParams, ...],
    x: Array,
    y: Array,
    key: Array,
    cfg: VBPCConfig,
    n_samples: int = 30,
    n_bins: int = 15,
) -> Dict[str, float]:
    """MC posterior-predictive NLL, ECE, accuracy and predictive entropy."""
    probs = collect_mc_probs(weight_params, x, key, cfg, n_samples)
    return {
        "mc_nll": mc_nll_from_probs(probs, y),
        "mc_ece": mc_ece_from_probs(probs, y, n_bins=n_bins),
        "mc_acc": float((probs.argmax(axis=1) == np.asarray(jnp.argmax(y, axis=-1))).mean()),
        "mc_pred_entropy": predictive_entropy_from_probs(probs),
    }


def accuracy_from_logits(logits: Array, y: Array) -> float:
    if logits.size == 0:
        return float("nan")
    correct = jnp.sum(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1))
    return float(correct) / max(int(y.shape[0]), 1)


def nll_from_logits(logits: Array, y: Array, eps: float = 1e-12) -> float:
    if logits.size == 0:
        return float("nan")
    probs = np.asarray(softmax(logits))
    labels = np.asarray(jnp.argmax(y, axis=-1))
    picked = probs[np.arange(labels.shape[0]), labels]
    return float(-np.mean(np.log(np.maximum(picked, eps))))


def ece_from_logits(logits: Array, y: Array, n_bins: int = 15) -> float:
    """Expected calibration error over equally spaced confidence bins."""

    if logits.size == 0:
        return float("nan")
    probs = np.asarray(softmax(logits))
    labels = np.asarray(jnp.argmax(y, axis=-1))
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    n = confidence.shape[0]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / n) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def vbpc_accuracy(
    weight_params: Tuple[VBPCWeightParams, ...],
    x: Array,
    y: Array,
    cfg: VBPCConfig,
) -> float:
    """Classification accuracy of the mean-weight network."""

    return accuracy_from_logits(collect_logits(weight_params, x, cfg), y)


def negative_log_likelihood(
    weight_params: Tuple[VBPCWeightParams, ...],
    x: Array,
    y: Array,
    cfg: VBPCConfig,
) -> float:
    """Mean ``-log p(class | x)`` with ``p = softmax(logits)`` (classification NLL)."""

    return nll_from_logits(collect_logits(weight_params, x, cfg), y)


def expected_calibration_error(
    weight_params: Tuple[VBPCWeightParams, ...],
    x: Array,
    y: Array,
    cfg: VBPCConfig,
    n_bins: int = 15,
) -> float:
    """Expected calibration error of the mean-weight network."""

    return ece_from_logits(collect_logits(weight_params, x, cfg), y, n_bins=n_bins)


def weight_variance_metric(
    weight_params: Tuple[VBPCWeightParams, ...],
    cfg: VBPCConfig,
) -> float:
    """Mean of ``sigma_W^2`` over all weight entries (the logged weight variance)."""

    return float(mean_weight_variance(weight_params, cfg.weight_sigma_min, cfg.weight_sigma_max))


def evaluate(
    weight_params: Tuple[VBPCWeightParams, ...],
    x: Array,
    y: Array,
    cfg: VBPCConfig,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Accuracy, classification NLL, ECE, predictive entropy and latent Gaussian entropy.

    ``nll`` is the softmax cross-entropy (classification NLL), distinct from the
    Gaussian likelihood that underlies the ELBO training objective.
    ``latent_entropy`` is the analytic H[q(W)] of the variational weight posterior.
    """

    logits = collect_logits(weight_params, x, cfg)
    probs = np.asarray(softmax(logits))
    return {
        "acc": accuracy_from_logits(logits, y),
        "nll": nll_from_logits(logits, y),
        "ece": ece_from_logits(logits, y, n_bins=n_bins),
        "pred_entropy": predictive_entropy_from_probs(probs),
        "latent_entropy": latent_gaussian_entropy(weight_params, cfg),
    }


def memory_usage_mb() -> Dict[str, float]:
    """Process/JAX memory diagnostics for the run log.

    ``rss_max_mb`` comes from ``resource.getrusage`` (peak resident set size,
    reported in bytes on macOS and kilobytes on Linux); any device statistics
    exposed by the active JAX backend are included as ``jax_*_mb`` entries.
    """

    out: Dict[str, float] = {}
    try:
        usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
        out["rss_max_mb"] = usage / divisor
    except Exception:  # pragma: no cover - platform dependent
        out["rss_max_mb"] = float("nan")
    try:
        stats = jax.local_devices()[0].memory_stats()
        if stats:
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    out[f"jax_{key}_mb"] = float(value) / (1024.0 * 1024.0)
    except Exception:
        pass
    return out

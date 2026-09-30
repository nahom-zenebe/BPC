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

from bpc.config import DTYPE, Array
from vbpc.config import VBPCConfig
from vbpc.layers.vbpc_network import forward_logits
from vbpc.posterior.weight_posterior import VBPCWeightParams, mean_weight_variance


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
    """Mean ``-log p(class | x)`` with ``p = softmax(logits)``."""

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
    """Accuracy, NLL and ECE from a single shared set of logits."""

    logits = collect_logits(weight_params, x, cfg)
    return {
        "acc": accuracy_from_logits(logits, y),
        "nll": nll_from_logits(logits, y),
        "ece": ece_from_logits(logits, y, n_bins=n_bins),
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

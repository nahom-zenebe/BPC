"""Forward passes through the VBPC network.

The VBPC network keeps the BPC architecture: layer ``l`` computes

    z_{l+1} = W_l f(z_l)

with ``f`` the identity on the input layer (configurable) and ReLU elsewhere,
and the bias implemented by augmenting ``f(z_l)`` with a constant ``1``.  This
module uses the *mean* weights ``mu_W`` for prediction/evaluation, exactly as
:func:`bpc.layers.predictive_network.predict` uses ``E[M]`` for BPC.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import jax.numpy as jnp

from bpc.activations import activation
from bpc.config import Array
from bpc.utils.tensor_ops import augment
from vbpc.config import VBPCConfig
from vbpc.posterior.weight_posterior import VBPCWeightParams


def forward_hidden(
    weight_params: Sequence[VBPCWeightParams],
    layer_dims: Sequence[int],
    x: Array,
    cfg: VBPCConfig,
) -> Tuple[Array, ...]:
    """Mean-weight activations of the hidden layers ``1 .. L-1``.

    Mirrors :func:`bpc.priors.hidden_state_prior.feedforward_init`: the returned
    tuple is exactly the ``mu`` initialization of the latent variational params.
    """

    z = x
    hidden: List[Array] = []
    for l in range(len(layer_dims) - 2):
        z = augment(activation(z, l, cfg.input_activation)) @ weight_params[l].mu.T
        hidden.append(z)
    return tuple(hidden)


def forward_top(
    weight_params: Sequence[VBPCWeightParams],
    x: Array,
    cfg: VBPCConfig,
) -> Array:
    """Mean-weight output-layer prediction ``z_L`` (the top of the network)."""

    z = x
    for l, p in enumerate(weight_params):
        z = augment(activation(z, l, cfg.input_activation)) @ p.mu.T
    return z


def forward_logits(
    weight_params: Sequence[VBPCWeightParams],
    x: Array,
    cfg: VBPCConfig,
) -> Array:
    """Class scores used by the accuracy/NLL/ECE metrics.

    The top layer is trained to reconstruct the one-hot target ``y`` under the
    Gaussian error model ``z_L ~ N(W_{L-1} f(z_{L-1}), Sigma_L)`` with
    ``Sigma_L = error_variance * I``.  For one-hot targets the resulting class
    log-posterior is linear in ``z_L``, namely ``softmax(z_L / error_variance)``,
    so the prediction divided by the top-layer error variance is the natural
    logit vector (the default ``error_variance = 1.0`` gives ``softmax(z_L)``).
    """

    return forward_top(weight_params, x, cfg) / jnp.asarray(cfg.error_variance, dtype=x.dtype)


"""Optimizers for VBPC variational parameters.

The primary backend is `Optax <https://github.com/google-deepmind/optax>`_
(added to ``requirements.txt`` for VBPC) and is used for both latent-state
inference and weight variational-parameter learning.  A small built-in Adam is
provided as a fallback so that the module stays importable in environments where
Optax is not installed; both backends expose the same
:class:`VBPCOptimizer` interface and share the same update rule.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

from bpc.config import DTYPE, Array 

try:  
    import optax

    OPTAX_AVAILABLE = True
except Exception:  
    optax = None
    OPTAX_AVAILABLE = False


def _zeros_like_tree(params: Any) -> Any:
    return jax.tree.map(jnp.zeros_like, params)


class _BuiltinAdam:
    """Minimal Adam with Optax-compatible ``init``/``update`` signatures."""

    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

    def init(self, params: Any) -> Tuple[Array, Any, Any]:
        return (
            jnp.asarray(0.0, dtype=DTYPE),
            _zeros_like_tree(params),
            _zeros_like_tree(params),
        )

    def update(self, grads: Any, state: Any, params: Any = None) -> Tuple[Any, Any]:
        count, mu, nu = state
        count = count + 1.0
        mu = jax.tree.map(lambda m, g: self.beta1 * m + (1.0 - self.beta1) * g, mu, grads)
        nu = jax.tree.map(lambda v, g: self.beta2 * v + (1.0 - self.beta2) * g * g, nu, grads)
        bc1 = 1.0 - self.beta1 ** count
        bc2 = 1.0 - self.beta2 ** count
        updates = jax.tree.map(
            lambda m, v: -self.learning_rate * (m / bc1) / (jnp.sqrt(v / bc2) + self.eps),
            mu,
            nu,
        )
        return updates, (count, mu, nu)


class _BuiltinClipByGlobalNorm:
    """Global-norm gradient clipping used when Optax is unavailable."""

    def __init__(self, max_norm: float):
        self.max_norm = float(max_norm)

    def init(self, params: Any) -> None:
        return None

    def update(self, grads: Any, state: Any = None, params: Any = None) -> Tuple[Any, None]:
        leaves = jax.tree.leaves(grads)
        if not leaves:
            return grads, None
        sq = sum((jnp.sum(g * g) for g in leaves), jnp.asarray(0.0, dtype=leaves[0].dtype))
        norm = jnp.sqrt(sq)
        scale = jnp.minimum(1.0, jnp.asarray(self.max_norm, dtype=norm.dtype) / (norm + 1e-12))
        return jax.tree.map(lambda g: g * scale, grads), None


def _chain(*transforms):
    """Chain simple ``init``/``update`` transforms (mirrors ``optax.chain``)."""

    if not transforms:
        raise ValueError("chain requires at least one transform")

    def init(params):
        return tuple(t.init(params) for t in transforms)

    def update(grads, state, params=None):
        states = []
        current = grads
        for t, s in zip(transforms, state):
            current, new_s = t.update(current, s, params)
            states.append(new_s)
        return current, tuple(states)

    class _Chained:
        pass

    chained = _Chained()
    chained.init = init
    chained.update = update
    return chained


class VBPCOptimizer:
    """Thin uniform wrapper around an Optax (or built-in) gradient transform."""

    def __init__(self, learning_rate: float, clip: Optional[float] = None):
        self.learning_rate = float(learning_rate)
        self.clip = None if clip is None else float(clip)
        self.uses_optax = OPTAX_AVAILABLE
        if OPTAX_AVAILABLE:
            transforms = []
            if self.clip is not None:
                transforms.append(optax.clip_by_global_norm(self.clip))
            transforms.append(optax.adam(learning_rate=self.learning_rate))
            self._tx = optax.chain(*transforms) if len(transforms) > 1 else transforms[0]
        else:
            transforms = []
            if self.clip is not None:
                transforms.append(_BuiltinClipByGlobalNorm(self.clip))
            transforms.append(_BuiltinAdam(self.learning_rate))
            self._tx = transforms[0] if len(transforms) == 1 else _chain(*transforms)

    def init(self, params: Any) -> Any:
        return self._tx.init(params)

    def update(self, params: Any, grads: Any, state: Any) -> Tuple[Any, Any]:
        updates, new_state = self._tx.update(grads, state, params)
        if OPTAX_AVAILABLE:
            new_params = optax.apply_updates(params, updates)
        else:
            new_params = jax.tree.map(lambda p, u: p + u, params, updates)
        return new_params, new_state


def make_optimizer(learning_rate: float, clip: Optional[float] = None) -> VBPCOptimizer:
    """Create the optimizer used by VBPC (Optax Adam, optionally global-norm clipped)."""

    return VBPCOptimizer(learning_rate, clip=clip)


def optimizer_backend() -> str:
    """Name of the active backend, logged in the run manifest."""

    return "optax" if OPTAX_AVAILABLE else "builtin_adam"

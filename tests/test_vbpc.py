"""Tests for the Variational Bayesian Predictive Coding (VBPC) implementation.

Covered here, as required by the VBPC proposal:

1. Gaussian reparameterization of the weight posterior,
2. analytic KL calculation,
3. local predictive-coding energy,
4. latent-state inference,
5. the VBPC training step (latent inference on ``L_PC``, weights on ``L_total``),
6. shape/gradient/JIT correctness, plus the VBPC YAML configuration.

The BPC tests in ``tests/test_bpc.py`` continue to cover the original
Matrix-Normal Wishart implementation.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

from bpc.config import DTYPE  
from vbpc.config import VBPCConfig  
from vbpc.inference.objectives import ( 
    clamped_states,
    pc_energy,
    pc_energy_per_layer,
    pc_errors,
)
from vbpc.optim import OPTAX_AVAILABLE  
from vbpc.posterior.weight_posterior import ( 
    VBPCWeightParams,
    gaussian_kl_elementwise,
    init_vbpc_weight_params,
    mean_weight_variance,
    reparameterize_weight,
    reparameterize_weight_matrices,
    sample_weight_epsilons,
    sample_weight_matrices,
    weight_diagnostics,
    weight_kl,
    weight_shapes,
    weight_sigma,
)


def assert_tree_finite(tree):
    for leaf in jax.tree.leaves(tree):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def tiny_cfg(**overrides) -> VBPCConfig:
    base = VBPCConfig(
        name="vbpc_tiny",
        hidden=3,
        hidden_layers=1,
        latent_steps=4,
        latent_lr=0.05,
        batch_size=4,
        beta=0.01,
    )
    return replace(base, **overrides)


def tiny_batch():
    x = jnp.asarray([[0.2, -0.4], [1.0, 0.5], [-0.7, 0.3], [0.0, 0.9]], dtype=DTYPE)
    y = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    return x, y


def test_gaussian_reparameterization_is_exact_deterministic_and_unbiased():
    """1. ``W = mu_W + sigma_W * eps`` with ``eps ~ N(0, I)``."""

    layer_dims = (2, 3, 2)
    key = jax.random.PRNGKey(0)
    params = init_vbpc_weight_params(layer_dims, key, init_log_sigma=-6.0)

    assert weight_shapes(layer_dims) == ((3, 3), (2, 4))
    assert [tuple(p.mu.shape) for p in params] == [(3, 3), (2, 4)]
    for p in params:
        assert p.mu.dtype == DTYPE
        assert p.log_sigma.dtype == DTYPE
        np.testing.assert_allclose(np.asarray(p.log_sigma), -3.0)
        np.testing.assert_allclose(np.asarray(weight_sigma(p.log_sigma)), np.exp(-3.0))

    mu = jnp.asarray([[1.0, -2.0], [0.5, 0.0]], dtype=DTYPE)
    sigma = jnp.asarray([[0.5, 2.0], [0.25, 1.0]], dtype=DTYPE)
    eps = jnp.asarray([[1.0, 1.0], [-1.0, 0.0]], dtype=DTYPE)

    np.testing.assert_allclose(np.asarray(reparameterize_weight(mu, sigma, eps)), np.asarray(mu + sigma * eps))
    np.testing.assert_allclose(
        np.asarray(reparameterize_weight(mu, sigma, jnp.zeros_like(mu))), np.asarray(mu), atol=0, rtol=0
    )

    weights_a = sample_weight_matrices(params, key)
    weights_b = sample_weight_matrices(params, key)
    weights_c = sample_weight_matrices(params, jax.random.PRNGKey(1))
    for wa, wb in zip(weights_a, weights_b):
        np.testing.assert_allclose(np.asarray(wa), np.asarray(wb), atol=0, rtol=0)
    assert any(not np.allclose(np.asarray(a), np.asarray(c)) for a, c in zip(weights_a, weights_c))

    epsilons = sample_weight_epsilons(params, key)
    first = reparameterize_weight_matrices(params, epsilons)
    second = reparameterize_weight_matrices(params, epsilons)
    for a, b in zip(first, second):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=0, rtol=0)
    assert_tree_finite(first)

    single = (VBPCWeightParams(mu=jnp.asarray([[0.7]], dtype=DTYPE), log_sigma=jnp.full((1, 1), float(np.log(0.3)), dtype=DTYPE)),)
    draws = jnp.stack(
        [sample_weight_matrices(single, jax.random.PRNGKey(i))[0][0, 0] for i in range(4000)]
    )
    assert abs(float(jnp.mean(draws)) - 0.7) < 0.03
    assert abs(float(jnp.std(draws)) - 0.3) < 0.03

    collapsed = (VBPCWeightParams(mu=mu, log_sigma=jnp.full(mu.shape, -30.0, dtype=DTYPE)),)
    np.testing.assert_allclose(np.asarray(sample_weight_matrices(collapsed, key)[0]), np.asarray(mu), atol=1e-6)

    diag = weight_diagnostics(params)
    assert abs(diag["mean_weight_variance"] - float(mean_weight_variance(params))) < 1e-15


def test_weight_kl_matches_analytic_formula_and_monte_carlo():
    """2. ``L_KL = 0.5 * sum(mu^2 + sigma^2 - 1 - log(sigma^2))`` for ``N(0, I)``."""

    mu = jnp.asarray([[1.0, -1.0], [0.5, 0.0]], dtype=DTYPE)
    log_sigma = jnp.asarray([[0.0, -1.0], [0.5, -2.0]], dtype=DTYPE)
    params = (VBPCWeightParams(mu=mu, log_sigma=log_sigma),)
    sigma = jnp.exp(log_sigma)

    expected_sum = 0.5 * jnp.sum(mu ** 2 + sigma ** 2 - 1.0 - jnp.log(sigma ** 2))
    np.testing.assert_allclose(
        np.asarray(weight_kl(params, "sum")), np.asarray(expected_sum), rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(weight_kl(params, "mean")), np.asarray(expected_sum / mu.size), rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(gaussian_kl_elementwise(mu, sigma, log_sigma)),
        np.asarray(0.5 * (mu ** 2 + sigma ** 2 - 1.0 - jnp.log(sigma ** 2))),
        rtol=1e-12,
    )

    matching_prior = (VBPCWeightParams(mu=jnp.zeros((3, 4), dtype=DTYPE), log_sigma=jnp.zeros((3, 4), dtype=DTYPE)),)
    assert float(weight_kl(matching_prior, "sum")) == pytest.approx(0.0, abs=1e-24)
    assert float(weight_kl(matching_prior, "mean")) == pytest.approx(0.0, abs=1e-24)

    two_layers = (params[0], VBPCWeightParams(mu=mu / 2.0, log_sigma=log_sigma + 0.2))
    np.testing.assert_allclose(
        np.asarray(weight_kl(two_layers, "sum")),
        np.asarray(weight_kl((two_layers[0],), "sum") + weight_kl((two_layers[1],), "sum")),
        rtol=1e-12,
    )

    single_mu = float(mu[0, 0])
    single_log_sigma = float(log_sigma[0, 0])
    single_sigma = float(sigma[0, 0])
    draws = single_mu + single_sigma * jax.random.normal(jax.random.PRNGKey(3), (400000,), dtype=DTYPE)
    log_q = -0.5 * ((draws - single_mu) / single_sigma) ** 2 - jnp.log(single_sigma) - 0.5 * jnp.log(2 * jnp.pi)
    log_p = -0.5 * draws ** 2 - 0.5 * jnp.log(2 * jnp.pi)
    monte_carlo = float(jnp.mean(log_q - log_p))
    analytic = 0.5 * (single_mu ** 2 + single_sigma ** 2 - 1.0 - 2.0 * single_log_sigma)
    assert monte_carlo == pytest.approx(analytic, abs=0.02)

    with pytest.raises(ValueError):
        weight_kl(params, "median")


def test_pc_energy_matches_hand_computed_residuals():
    """3. ``L_PC = 0.5 * sum_l e_l^T Sigma_l^{-1} e_l`` with ``e_l = z_{l+1} - W_l f(z_l)``."""

    cfg = VBPCConfig(name="energy", input_activation="identity", pc_reduction="sum", error_variance=1.0)
    layer_dims = (2, 2, 2)
    W0 = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE)
    W1 = jnp.asarray([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=DTYPE)
    weights = (W0, W1)
    x = jnp.asarray([[1.0, 2.0], [0.0, 1.0]], dtype=DTYPE)
    hidden = (jnp.asarray([[0.5, -0.5], [1.0, 2.0]], dtype=DTYPE),)
    y = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    states = clamped_states(hidden, x, y)

    errors = pc_errors(states, weights, cfg)
    assert len(errors) == 2
    assert errors[0].shape == (2, 2)

    aug_x = np.concatenate([np.asarray(x), np.ones((2, 1))], axis=1)
    e0 = np.asarray(hidden[0]) - aug_x @ np.asarray(W0).T
    aug_h = np.concatenate([np.maximum(np.asarray(hidden[0]), 0.0), np.ones((2, 1))], axis=1)
    e1 = np.asarray(y) - aug_h @ np.asarray(W1).T
    np.testing.assert_allclose(np.asarray(errors[0]), e0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(errors[1]), e1, rtol=1e-12)

    expected_sum = 0.5 * float(np.sum(e0 ** 2) + np.sum(e1 ** 2))
    assert float(pc_energy(states, weights, cfg)) == pytest.approx(expected_sum, rel=1e-12)

    per_layer = pc_energy_per_layer(states, weights, cfg)
    assert len(per_layer) == 2
    assert float(sum(per_layer)) == pytest.approx(expected_sum, rel=1e-12)
    np.testing.assert_allclose(
        np.asarray(per_layer),
        np.asarray([0.5 * np.sum(e0 ** 2), 0.5 * np.sum(e1 ** 2)]),
        rtol=1e-12,
    )

    cfg_mean = replace(cfg, pc_reduction="mean")
    expected_mean = 0.5 * (float(np.mean(e0 ** 2)) + float(np.mean(e1 ** 2)))
    assert float(pc_energy(states, weights, cfg_mean)) == pytest.approx(expected_mean, rel=1e-12)

    cfg_var = replace(cfg, error_variance=4.0)
    assert float(pc_energy(states, weights, cfg_var)) == pytest.approx(expected_sum / 4.0, rel=1e-12)

    perfect_hidden = (jnp.asarray(aug_x, dtype=DTYPE) @ W0.T,)
    perfect_y = jnp.concatenate([jax.nn.relu(perfect_hidden[0]), jnp.ones((2, 1), dtype=DTYPE)], axis=1) @ W1.T
    perfect_energy = pc_energy(clamped_states(perfect_hidden, x, perfect_y), weights, cfg)
    assert float(perfect_energy) == pytest.approx(0.0, abs=1e-18)

    with pytest.raises(ValueError):
        pc_errors(states[:-1], weights, cfg)


def test_latent_inference_minimizes_pc_energy_only():
    """4. Reparameterized latent states optimized with ``L_PC`` (no weight KL)."""

    from vbpc.inference.latent_inference import (
        deterministic_pc_energy,
        init_vbpc_latents,
        latent_pc_energy,
        vbpc_latent_inference,
    )
    from vbpc.inference.objectives import reparameterize_states
    from vbpc.layers.vbpc_network import forward_hidden

    cfg = tiny_cfg(latent_steps=8)
    layer_dims = (2, 3, 2)
    key = jax.random.PRNGKey(0)
    params = init_vbpc_weight_params(layer_dims, key, cfg.init_weight_log_sigma)
    weights = sample_weight_matrices(params, key)
    x, y = tiny_batch()

    states0 = init_vbpc_latents(params, layer_dims, x, cfg)
    assert len(states0.mu) == len(layer_dims) - 2
    assert states0.mu[0].shape == (4, 3)
    np.testing.assert_allclose(
        np.asarray(states0.mu[0]), np.asarray(forward_hidden(params, layer_dims, x, cfg)[0]), rtol=1e-12
    )
    np.testing.assert_allclose(np.asarray(states0.log_sigma[0]), cfg.init_latent_log_sigma)

    eps = jnp.asarray(
        [[1.0, -1.0, 0.5], [0.25, 0.0, -0.25], [0.5, 0.5, 0.5], [-1.0, 2.0, 0.0]], dtype=DTYPE
    )
    sampled = reparameterize_states(states0, (eps,), cfg.state_sigma_min, cfg.state_sigma_max)
    np.testing.assert_allclose(
        np.asarray(sampled[0]), np.asarray(states0.mu[0] + jnp.exp(states0.log_sigma[0]) * eps), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(
            reparameterize_states(states0, (jnp.zeros_like(eps),), cfg.state_sigma_min, cfg.state_sigma_max)[0]
        ),
        np.asarray(states0.mu[0]),
        rtol=0,
        atol=0,
    )

    assert_tree_finite(latent_pc_energy(states0, weights, x, y, cfg, key))

    states, diag = vbpc_latent_inference(params, layer_dims, weights, x, y, cfg, key)
    assert diag.energy_per_step.shape == (8,)
    assert diag.mean_energy_per_step.shape == (8,)
    assert diag.grad_norm_per_step.shape == (8, 1)
    assert diag.log_sigma_grad_norm_per_step.shape == (8, 1)
    assert diag.sigma_mean_per_step.shape == (8, 1)
    assert_tree_finite(diag)
    assert float(diag.final_energy) < float(diag.init_energy)
    assert float(deterministic_pc_energy(states, weights, x, y, cfg)) < float(
        deterministic_pc_energy(states0, weights, x, y, cfg)
    )
    assert float(jnp.linalg.norm(diag.init_grad_norm)) > 0.0
    assert float(diag.sigma_mean_per_step[-1][0]) <= float(diag.sigma_mean_per_step[0][0])

    cfg_zero = replace(cfg, beta=0.0)
    cfg_huge = replace(cfg, beta=100.0)
    states_zero, diag_zero = vbpc_latent_inference(params, layer_dims, weights, x, y, cfg_zero, key)
    states_huge, diag_huge = vbpc_latent_inference(params, layer_dims, weights, x, y, cfg_huge, key)
    for a, b in zip(states_zero.mu, states_huge.mu):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(diag_zero.mean_energy_per_step), np.asarray(diag_huge.mean_energy_per_step), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(diag_zero.sigma_mean_per_step), np.asarray(diag_huge.sigma_mean_per_step), rtol=0, atol=0
    )

    zero_init_cfg = replace(cfg, hidden_init="zeros")
    zeros_states = init_vbpc_latents(params, layer_dims, x, zero_init_cfg)
    np.testing.assert_allclose(np.asarray(zeros_states.mu[0]), np.zeros((4, 3)), atol=0, rtol=0)


def test_vbpc_training_step_uses_pc_for_latents_and_total_for_weights():
    """5. Latent inference on ``L_PC``; weight update on ``L_total = L_PC + beta L_KL``."""

    from vbpc.training.train_step import (
        make_vbpc_train_step,
        vbpc_loss_and_grads,
        vbpc_train_step,
    )

    cfg = tiny_cfg()
    layer_dims = (2, 3, 2)
    key = jax.random.PRNGKey(2)
    params = init_vbpc_weight_params(layer_dims, key, cfg.init_weight_log_sigma)
    x, y = tiny_batch()
    beta = jnp.asarray(cfg.beta, dtype=DTYPE)

    total, terms, grads, latents, latent_diag, epsilons = vbpc_loss_and_grads(
        cfg, layer_dims, params, x, y, key, beta
    )
    assert float(total) == pytest.approx(float(terms[0]) + cfg.beta * float(terms[1]), rel=1e-12)
    assert float(terms[0]) > 0.0
    assert float(terms[1]) == pytest.approx(float(weight_kl(params, cfg.kl_reduction)), rel=1e-12)
    assert_tree_finite((grads, latents, epsilons))
    assert len(grads) == len(params)
    assert float(latent_diag.final_energy) < float(latent_diag.init_energy)

    step, optimizer = make_vbpc_train_step(cfg, layer_dims)
    opt_state = optimizer.init(params)
    new_params, new_state, diag = step(params, opt_state, x, y, key, beta)

    assert isinstance(new_params, tuple) and isinstance(new_params[0], VBPCWeightParams)
    assert [tuple(p.mu.shape) for p in new_params] == [tuple(p.mu.shape) for p in params]
    assert [tuple(p.log_sigma.shape) for p in new_params] == [tuple(p.log_sigma.shape) for p in params]
    assert_tree_finite((new_params, new_state))

    assert float(diag.pc_energy) == pytest.approx(float(terms[0]), rel=1e-9)
    assert float(diag.weight_kl) == pytest.approx(float(terms[1]), rel=1e-9)
    assert float(diag.total_loss) == pytest.approx(float(total), rel=1e-9)
    assert float(diag.beta) == pytest.approx(cfg.beta)
    assert float(diag.mean_weight_variance) == pytest.approx(
        float(mean_weight_variance(new_params, cfg.weight_sigma_min, cfg.weight_sigma_max)), rel=1e-12
    )

    assert not np.allclose(np.asarray(new_params[0].mu), np.asarray(params[0].mu))
    assert not np.allclose(np.asarray(new_params[0].log_sigma), np.asarray(params[0].log_sigma))
    assert all(float(jnp.linalg.norm(g.mu)) > 0.0 for g in grads)

    _, _, diag_zero = vbpc_train_step(
        replace(cfg, beta=0.0),
        layer_dims,
        params,
        optimizer.init(params),
        x,
        y,
        key,
        optimizer,
        jnp.asarray(0.0),
    )
    assert float(diag_zero.total_loss) == pytest.approx(float(diag_zero.pc_energy), rel=1e-12)

    _, _, diag_big = vbpc_train_step(
        cfg, layer_dims, params, optimizer.init(params), x, y, key, optimizer, jnp.asarray(1.0)
    )
    assert float(diag_big.total_loss) > float(diag_zero.total_loss)
    assert float(diag_big.pc_energy) == pytest.approx(float(diag_zero.pc_energy), rel=1e-9)

    _, terms_zero, grads_zero, latents_zero, _, _ = vbpc_loss_and_grads(
        replace(cfg, beta=0.0), layer_dims, params, x, y, key, jnp.asarray(0.0)
    )
    _, terms_big, grads_big, latents_big, _, _ = vbpc_loss_and_grads(
        cfg, layer_dims, params, x, y, key, jnp.asarray(1.0)
    )
    for a, b in zip(latents_zero.mu, latents_big.mu):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)
    assert float(terms_zero[0]) == pytest.approx(float(terms_big[0]), rel=1e-12)
    assert any(not np.allclose(np.asarray(a.mu), np.asarray(b.mu)) for a, b in zip(grads_zero, grads_big))


def test_shapes_gradients_jit_and_optimizer_backends():
    """6. Shape/gradient/JIT correctness of the VBPC objectives and step."""

    from jax.test_util import check_grads

    from vbpc.inference.objectives import total_vbpc_objective
    from vbpc.optim import _BuiltinAdam, make_optimizer
    from vbpc.training.train_step import make_vbpc_train_step, vbpc_train_step

    cfg = tiny_cfg()
    layer_dims = (2, 3, 2)
    key = jax.random.PRNGKey(5)
    params = init_vbpc_weight_params(layer_dims, key, cfg.init_weight_log_sigma)
    x, y = tiny_batch()
    beta = jnp.asarray(cfg.beta, dtype=DTYPE)

    weights = sample_weight_matrices(params, key)
    hidden = jnp.asarray(
        [[0.4, 0.8, 1.2], [0.5, 0.3, 0.9], [1.1, 0.2, 0.6], [0.7, 1.4, 0.3]], dtype=DTYPE
    )

    def energy_of_hidden(h):
        return pc_energy(clamped_states((h,), x, y), weights, cfg)

    check_grads(energy_of_hidden, (hidden,), order=1)

    mu = jnp.asarray([[0.1, -0.2], [0.3, 0.0]], dtype=DTYPE)
    log_sigma = jnp.asarray([[-1.0, -0.5], [0.0, 0.5]], dtype=DTYPE)

    def kl_of_params(m, ls):
        return weight_kl((VBPCWeightParams(m, ls),), "sum")

    check_grads(kl_of_params, (mu, log_sigma), order=1)

    epsilons = sample_weight_epsilons(params, key)

    def total_of_params(p0, p1):
        sampled = reparameterize_weight_matrices((p0, p1), epsilons, cfg.weight_sigma_min, cfg.weight_sigma_max)
        total, _ = total_vbpc_objective(clamped_states((hidden,), x, y), sampled, (p0, p1), cfg)
        return total

    check_grads(total_of_params, (params[0], params[1]), order=1)

    assert len(jax.tree.leaves(params)) == 2 * len(params)
    grads_of = jax.grad(total_of_params)(params[0], params[1])
    assert_tree_finite(grads_of)
    assert float(jnp.linalg.norm(grads_of.mu)) > 0.0
    assert float(jnp.linalg.norm(grads_of.log_sigma)) > 0.0

    step_jit, optimizer = make_vbpc_train_step(cfg, layer_dims)
    state = optimizer.init(params)
    jit_params, jit_state, jit_diag = step_jit(params, state, x, y, key, beta)
    eager_params, eager_state, eager_diag = vbpc_train_step(
        cfg, layer_dims, params, state, x, y, key, optimizer, beta
    )
    for a, b in zip(jit_params, eager_params):
        np.testing.assert_allclose(np.asarray(a.mu), np.asarray(b.mu), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(a.log_sigma), np.asarray(b.log_sigma), rtol=1e-12)
    for leaf_a, leaf_b in zip(jax.tree.leaves(jit_state), jax.tree.leaves(eager_state)):
        np.testing.assert_allclose(np.asarray(leaf_a), np.asarray(leaf_b), rtol=1e-12)
    assert float(jit_diag.total_loss) == pytest.approx(float(eager_diag.total_loss), rel=1e-12)
    assert float(jit_diag.pc_energy) == pytest.approx(float(eager_diag.pc_energy), rel=1e-12)

    sample_params = (jnp.asarray([1.0, -2.0, 0.5], dtype=DTYPE),)
    sample_grads = (jnp.asarray([0.3, -0.4, 0.2], dtype=DTYPE),)
    from vbpc.optim import VBPCOptimizer

    assert isinstance(make_optimizer(0.1), VBPCOptimizer)
    clipped = make_optimizer(0.1, clip=1.0)
    clipped_params, _ = clipped.update(sample_params, sample_grads, clipped.init(sample_params))
    assert_tree_finite(clipped_params)

    if OPTAX_AVAILABLE:
        import optax

        fallback = _BuiltinAdam(0.1)
        fallback_updates, _ = fallback.update(sample_grads, fallback.init(sample_params), sample_params)
        optax_tx = optax.adam(0.1)
        optax_updates, _ = optax_tx.update(sample_grads, optax_tx.init(sample_params), sample_params)
        np.testing.assert_allclose(
            np.asarray(fallback_updates[0]), np.asarray(optax_updates[0]), rtol=1e-9, atol=1e-12
        )


def test_metrics_accuracy_nll_ece_and_memory_reporting():
    """Accuracy, NLL, ECE and memory usage - the quantities the VBPC log must contain."""

    from vbpc.training.metrics import (
        accuracy_from_logits,
        ece_from_logits,
        memory_usage_mb,
        negative_log_likelihood,
        nll_from_logits,
        vbpc_accuracy,
        weight_variance_metric,
    )

    logits = jnp.asarray([[4.0, 0.0], [0.0, 4.0], [1.0, 3.0]], dtype=DTYPE)
    y = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=DTYPE)
    assert accuracy_from_logits(logits, y) == pytest.approx(2.0 / 3.0)

    probs = np.asarray(jax.nn.softmax(logits))
    labels = np.asarray(jnp.argmax(y, axis=-1))  # [0, 1, 0]
    expected_nll = -float(np.mean(np.log(probs[np.arange(labels.shape[0]), labels])))
    assert nll_from_logits(logits, y) == pytest.approx(expected_nll, rel=1e-12)

    confident = jnp.asarray([[2.0, 0.0]] * 10, dtype=DTYPE)
    confident_y = jnp.asarray([[1.0, 0.0]] * 10, dtype=DTYPE)
    confidence = float(jax.nn.softmax(confident[0])[0])
    assert ece_from_logits(confident, confident_y) == pytest.approx(1.0 - confidence, rel=1e-9)

    cfg = tiny_cfg()
    key = jax.random.PRNGKey(11)
    params = init_vbpc_weight_params((2, 3, 2), key, cfg.init_weight_log_sigma)
    x, y_batch = tiny_batch()
    assert 0.0 <= vbpc_accuracy(params, x, y_batch, cfg) <= 1.0
    assert float(negative_log_likelihood(params, x, y_batch, cfg)) > 0.0
    assert weight_variance_metric(params, cfg) == pytest.approx(float(mean_weight_variance(params)), rel=1e-12)

    memory = memory_usage_mb()
    assert "rss_max_mb" in memory and memory["rss_max_mb"] > 0.0


def test_vbpc_yaml_config_uses_preset_and_beta_sweep():
    """YAML configuration, preset selection and beta-sweep support."""

    from bpc.utils.config_io import load_config_with_presets
    from vbpc.config import VBPC_BETA_SWEEP, make_vbpc_presets, validate_vbpc_config
    from vbpc.training.trainer import default_vbpc_config

    preset = make_vbpc_presets()["vbpc_mnist_default"]
    cfg, lcfg, raw = load_config_with_presets(
        "configs/vbpc_mnist.yaml", default_vbpc_config(), make_vbpc_presets()
    )

    assert cfg.name == "vbpc_mnist"
    assert cfg.beta == preset.beta == 0.01
    assert cfg.epochs == preset.epochs
    assert cfg.hidden == preset.hidden
    assert lcfg.save_dir == "runs"
    assert raw["mode"] == "sweep"
    assert tuple(float(b) for b in raw["betas"]) == VBPC_BETA_SWEEP

    validate_vbpc_config(cfg)
    with pytest.raises(ValueError):
        validate_vbpc_config(replace(cfg, beta=-1.0))
    with pytest.raises(ValueError):
        validate_vbpc_config(replace(cfg, pc_reduction="median"))
    with pytest.raises(ValueError):
        validate_vbpc_config(replace(cfg, hidden_init="magic"))
    with pytest.raises(KeyError):
        load_config_with_presets(
            "configs/vbpc_mnist.yaml", default_vbpc_config(), make_vbpc_presets(), "does_not_exist"
        )


def test_vbpc_two_moons_yaml_config_uses_preset_and_beta_sweep():
    """``configs/vbpc_two_moons.yaml`` selects the two-moons preset, data block and sweep."""

    from bpc.utils.config_io import load_config_with_presets
    from vbpc.config import (
        VBPC_BETA_SWEEP,
        VBPC_TWO_MOONS_EPOCHS,
        VBPC_TWO_MOONS_HIDDEN,
        make_vbpc_presets,
        validate_vbpc_config,
    )
    from vbpc.training.trainer import default_vbpc_two_moons_config

    preset = make_vbpc_presets()["vbpc_two_moons"]
    cfg, lcfg, raw = load_config_with_presets(
        "configs/vbpc_two_moons.yaml", default_vbpc_two_moons_config(), make_vbpc_presets()
    )

    assert cfg.name == "vbpc_two_moons"
    assert cfg == preset
    assert cfg.beta == preset.beta == 0.01
    assert cfg.epochs == VBPC_TWO_MOONS_EPOCHS
    assert cfg.hidden == VBPC_TWO_MOONS_HIDDEN
    assert cfg.hidden_layers == 1
    assert cfg.batch_size == preset.batch_size
    assert lcfg.save_dir == "runs"
    assert raw["mode"] == "sweep"
    assert tuple(float(b) for b in raw["betas"]) == VBPC_BETA_SWEEP
    assert raw["data"]["n_train"] == 1000
    assert raw["data"]["n_test"] == 300
    assert raw["data"]["noise"] == 0.10

    validate_vbpc_config(cfg)
    with pytest.raises(KeyError):
        load_config_with_presets(
            "configs/vbpc_two_moons.yaml", default_vbpc_two_moons_config(), make_vbpc_presets(), "does_not_exist"
        )


def test_train_vbpc_two_moons_dataset_writes_two_moons_artifacts(tmp_path):
    """The two-moons VBPC trainer logs under ``vbpc_two_moons`` (not ``vbpc_mnist``)."""

    from bpc.config import LoggingConfig
    from vbpc.training.trainer import train_vbpc_two_moons_dataset

    rng = np.random.default_rng(0)
    x_train = rng.random((8, 2)).astype(np.float64)
    y_train = np.eye(2)[rng.integers(0, 2, size=8)]
    x_test = rng.random((4, 2)).astype(np.float64)
    y_test = np.eye(2)[rng.integers(0, 2, size=4)]

    cfg = replace(
        tiny_cfg(),
        name="vbpc_two_moons_tiny",
        epochs=2,
        batch_size=4,
        hidden=3,
        hidden_layers=1,
        eval_batch_size=4,
    )
    lcfg = LoggingConfig(save_dir=str(tmp_path), extra_plots=True, per_class_acc=False)

    params, metrics, rows = train_vbpc_two_moons_dataset(
        cfg, (x_train, y_train, x_test, y_test), (2, 3, 2), lcfg
    )

    assert len(params) == 2
    assert len(rows) == 2
    for row in rows:
        for key in ("test_acc", "test_nll", "test_ece", "beta", "mean_weight_variance"):
            assert key in row and np.isfinite(float(row[key]))
    assert metrics["best_acc"] >= 0.0

    run_root = tmp_path / "vbpc_two_moons" / cfg.name
    run_dirs = sorted(p for p in run_root.iterdir() if p.is_dir())
    assert run_dirs, "two-moons runs must be namespaced under vbpc_two_moons"
    run_dir = run_dirs[-1]
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "epoch.csv").exists()
    assert (tmp_path / "vbpc_two_moons_experiment_log.csv").exists()
    assert not (tmp_path / "vbpc_mnist" / cfg.name).exists()


def test_vbpc_two_moons_entry_point_single_mode(tmp_path, monkeypatch):
    """The thin entry point runs a tiny single-mode run from a YAML config."""

    import sys

    from experiments.vbpc_two_moons import main

    config_path = tmp_path / "vbpc_two_moons_smoke.yaml"
    config_path.write_text(
        "\n".join(
            [
                "preset: vbpc_two_moons",
                "config:",
                "  name: vbpc_two_moons_smoke",
                "  epochs: 1",
                "  hidden: 8",
                "  batch_size: 16",
                "  latent_steps: 2",
                "  eval_batch_size: 64",
                "logging:",
                f"  save_dir: {tmp_path}",
                "mode: single",
                "data:",
                "  n_train: 16",
                "  n_test: 8",
                "  noise: 0.10",
                "",
            ]
        )
    )

    monkeypatch.setattr(
        sys, "argv", ["vbpc_two_moons", "--config", str(config_path), "--mode", "single", "--beta", "0.01"]
    )
    main()

    run_root = tmp_path / "vbpc_two_moons" / "vbpc_two_moons_smoke"
    run_dirs = sorted(p for p in run_root.iterdir() if p.is_dir())
    assert run_dirs
    assert (run_dirs[-1] / "epoch.csv").exists()
    assert (tmp_path / "vbpc_two_moons_experiment_log.csv").exists()


def test_trainer_and_logger_end_to_end_on_synthetic_mnist_shaped_data(tmp_path):
    """End-to-end trainer + logger: every required quantity reaches the epoch log."""

    import csv as csv_module

    from bpc.config import LoggingConfig
    from vbpc.training.trainer import train_vbpc_mnist_dataset

    rng = np.random.default_rng(0)
    x_train = rng.random((8, 4)).astype(np.float64)
    y_train = np.eye(2)[rng.integers(0, 2, size=8)]
    x_test = rng.random((4, 4)).astype(np.float64)
    y_test = np.eye(2)[rng.integers(0, 2, size=4)]
    layer_dims = (4, 3, 2)

    cfg = replace(
        tiny_cfg(),
        name="vbpc_synthetic",
        epochs=2,
        batch_size=4,
        hidden=3,
        hidden_layers=1,
        latent_steps=2,
        eval_batch_size=4,
    )
    lcfg = LoggingConfig(save_dir=str(tmp_path), extra_plots=True, per_class_acc=False)

    params, metrics, rows = train_vbpc_mnist_dataset(
        cfg, (x_train, y_train, x_test, y_test), layer_dims, lcfg
    )

    assert len(params) == 2
    assert len(rows) == 2
    for row in rows:
        for key in ("test_acc", "test_nll", "test_ece", "time_sec", "runtime_sec", "beta", "mean_weight_variance"):
            assert key in row and np.isfinite(float(row[key]))
        assert any(key.startswith("mem_") for key in row)
        assert float(row["beta"]) == pytest.approx(cfg.beta)
    assert metrics["best_acc"] >= 0.0
    assert np.isfinite(metrics["final_nll"]) and np.isfinite(metrics["final_ece"])
    assert metrics["mean_weight_variance"] > 0.0

    run_root = tmp_path / "vbpc_mnist" / cfg.name
    run_dirs = sorted(p for p in run_root.iterdir() if p.is_dir())
    assert run_dirs, "logger should create a timestamped run directory"
    run_dir = run_dirs[-1]
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "batch.csv").exists()
    assert (run_dir / "verbose.jsonl").exists()
    assert (run_dir / "plots" / "accuracy.png").exists()

    with open(run_dir / "epoch.csv") as handle:
        header = next(csv_module.reader(handle))
    for key in ("beta", "test_acc", "test_nll", "test_ece", "mean_weight_variance", "runtime_sec"):
        assert key in header
    assert any(key.startswith("mem_") for key in header)
    assert (tmp_path / "vbpc_mnist_experiment_log.csv").exists()

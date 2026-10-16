"""Regression tests for the latent-prior ablation study.

Background
----------
The ablation commit (5340cee) only *adds* code, yet all three ablation
conditions stalled at 20-28% MNIST accuracy with final NLL ~= ln(10) = 2.303
(uniform predictions).  The actual root cause was the weight-sampling change
in 9c8ce1c: ``vbpc_loss_and_grads`` split the step key with ``n + 1`` instead
of the original ``n + 2``, which silently dropped the second ``L_PC`` sample
of the Monte-Carlo estimator.  With a single sample the data-term gradient is
too weak/noisy relative to ``beta * dL_KL/dmu`` (the matched prior still pulls
``mu`` toward zero with strength ``beta * mu / tau^2``, ``tau = exp(-3)``), so
the weight means collapse to zero, predictions go uniform and NLL stalls at
``ln(10)``.  The original preset fails through the exact same path -- the
failure is shared, the ablation merely exposed it.

These tests pin the invariants of the study:

1. ``ablation_no_latent_kl`` is field-for-field identical to the original
   ``vbpc_mnist_default`` config (except ``name``); the other two conditions
   differ only in their intended latent-KL fields, with matched ``lambda`` and
   effective prior variance ``tau^2``.
2. Under an equivalent config and the same random key, the ``no_latent_kl``
   training step matches a reference implementation of the *original* VBPC
   objective (the original key derivation ``split(key, n + 2)`` with its
   ``n + 1``-sample ``L_PC`` estimator plus ``beta * L_KL``).  This test fails
   if the weight-sampling regression is reintroduced in any form.
3. The latent inference objective of each condition contains exactly its
   intended term: none for ``no_latent_kl``, ``KL[q(z) || N(0, I)]`` for
   ``standard_normal``, ``KL[q(z) || N(h_l^ff, tau^2 I)]`` for ``ff_centered``
   -- with the configured reduction and the intended gradients.
4. Full training with ``no_latent_kl`` behaves identically to training with
   the original preset under a matched seed and random keys.
"""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

from bpc.config import DTYPE, LoggingConfig
from vbpc.config import (
    ABLATION_LATENT_KL_WEIGHT,
    ABLATION_LATENT_TAU,
    VBPCConfig,
    make_vbpc_presets,
    validate_vbpc_config,
)
from vbpc.inference.latent_inference import latent_pc_energy, vbpc_latent_inference
from vbpc.inference.objectives import (
    VBPCStates,
    clamped_states,
    feedforward_latent_kl,
    latent_kl,
    pc_energy,
    reparameterize_states,
    sample_state_epsilons,
    sample_states,
)
from vbpc.layers.vbpc_network import forward_hidden
from vbpc.posterior.weight_posterior import (
    init_vbpc_weight_params,
    reparameterize_weight_matrices,
    sample_weight_epsilons,
    weight_kl,
    weight_kl_grads,
)
from vbpc.training.train_step import vbpc_loss_and_grads


ABLATION_CONDITIONS = ("no_latent_kl", "standard_normal", "ff_centered")
PRESET_BY_CONDITION = {
    "no_latent_kl": "ablation_no_latent_kl",
    "standard_normal": "ablation_standard_normal",
    "ff_centered": "ablation_ff_centered",
}


def ablation_presets():
    presets = make_vbpc_presets()
    return {c: presets[PRESET_BY_CONDITION[c]] for c in ABLATION_CONDITIONS}


def config_differences(a: VBPCConfig, b: VBPCConfig) -> set:
    da, db = asdict(a), asdict(b)
    return {k for k in da if da[k] != db[k]}


def tiny_step_cfg(**overrides) -> VBPCConfig:
    """A tiny config whose objective/latent settings can be overridden per test."""

    base = VBPCConfig(
        name="tiny_step",
        hidden=3,
        hidden_layers=1,
        latent_steps=4,
        latent_lr=0.05,
        batch_size=4,
        beta=0.01,
    )
    return replace(base, **overrides)


def tiny_batch():
    x = jnp.asarray(
        [[0.2, -0.4, 0.7, 0.1], [1.0, 0.5, -0.2, 0.9],
         [-0.7, 0.3, 0.4, -0.5], [0.0, 0.9, 0.6, 0.2]],
        dtype=DTYPE,
    )
    y = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    return x, y


# ---------------------------------------------------------------------------
# 1. Config equivalence: no_latent_kl IS the original VBPC configuration.
# ---------------------------------------------------------------------------


def test_no_latent_kl_preset_is_the_original_mnist_config_with_latent_kl_disabled():
    """``ablation_no_latent_kl`` must equal ``vbpc_mnist_default`` except ``name``.

    This is the "no unintended behavioral changes" invariant: any future edit
    that makes the ablation's baseline diverge from the working VBPC config
    (beta, prior, optimizer, architecture, epochs, ...) fails here.
    """

    presets = make_vbpc_presets()
    original = presets["vbpc_mnist_default"]
    baseline = presets["ablation_no_latent_kl"]

    assert config_differences(original, baseline) == {"name"}
    assert baseline.latent_kl_weight == 0.0
    assert baseline.latent_prior_tau == 0.0
    validate_vbpc_config(baseline)


def test_ablation_conditions_differ_only_in_their_intended_latent_kl_fields():
    """``standard_normal`` / ``ff_centered`` may change only the latent-KL fields.

    The KL coefficient ``lambda`` must be identical between the two prior
    variants and the effective prior variance ``tau^2`` must be matched
    (``standard_normal`` uses the ``N(0, I)`` fallback, i.e. ``tau^2 = 1``;
    ``ff_centered`` sets ``latent_prior_tau = 1.0``).
    """

    presets = ablation_presets()
    baseline = presets["no_latent_kl"]
    allowed = {"name", "latent_kl_weight", "latent_prior_tau", "latent_prior_stop_grad"}

    for condition in ("standard_normal", "ff_centered"):
        assert config_differences(baseline, presets[condition]) <= allowed
        validate_vbpc_config(presets[condition])

    standard = presets["standard_normal"]
    ff = presets["ff_centered"]

    # Matched KL coefficient.
    assert standard.latent_kl_weight == ABLATION_LATENT_KL_WEIGHT
    assert ff.latent_kl_weight == ABLATION_LATENT_KL_WEIGHT
    assert standard.latent_kl_weight == ff.latent_kl_weight

    # Matched effective prior variance tau^2: standard_normal goes through the
    # unit-normal fallback (tau^2 = 1) and ff_centered sets tau = ABLATION_TAU.
    assert standard.latent_prior_tau == 0.0  # -> N(0, I) fallback, tau^2 = 1
    assert ff.latent_prior_tau == ABLATION_LATENT_TAU
    assert ABLATION_LATENT_TAU ** 2 == pytest.approx(1.0)

    # ff_centered keeps the feedforward reference detached from the weights;
    # the other two conditions never build it at all.
    assert ff.latent_prior_stop_grad is True


# ---------------------------------------------------------------------------
# 2. Latent objective: each condition adds exactly its intended term.
# ---------------------------------------------------------------------------


def test_latent_objective_contains_only_the_terms_of_each_condition():
    """``latent_pc_energy`` == ``L_PC`` (+ the condition's KL term only)."""

    presets = ablation_presets()
    layer_dims = (4, 3, 2)
    x, y = tiny_batch()
    key = jax.random.PRNGKey(3)
    params = init_vbpc_weight_params(layer_dims, jax.random.PRNGKey(4), -3.0)
    eps = sample_weight_epsilons(params, key)
    weights = reparameterize_weight_matrices(params, eps, 1e-8, 1e3)

    states = VBPCStates(
        mu=(
            jnp.asarray(
                [[0.4, -0.6, 0.1], [0.3, -0.2, 0.5], [0.7, -0.4, 0.9], [0.1, -0.3, 0.2]],
                dtype=DTYPE,
            ),
        ),
        log_sigma=(
            jnp.asarray(
                [[-3.0, -2.5, -3.2], [-2.8, -3.1, -3.0], [-2.7, -3.3, -2.9], [-3.4, -3.0, -2.6]],
                dtype=DTYPE,
            ),
        ),
    )
    hidden = sample_states(states, key, 1e-8, 1e3)
    l_pc = pc_energy(clamped_states(hidden, x, y), weights, presets["no_latent_kl"])

    # --- no_latent_kl: pure L_PC, nothing else. ---
    cfg = presets["no_latent_kl"]
    assert cfg.latent_kl_weight == 0.0 and cfg.latent_prior_tau == 0.0
    value = latent_pc_energy(states, weights, x, y, cfg, key, ())
    np.testing.assert_allclose(np.asarray(value), np.asarray(l_pc), rtol=1e-6, atol=1e-9)

    # --- standard_normal: L_PC + lambda * KL[q(z) || N(0, I)]. ---
    cfg = presets["standard_normal"]
    value = latent_pc_energy(states, weights, x, y, cfg, key, ())
    expected = l_pc + jnp.asarray(cfg.latent_kl_weight, dtype=DTYPE) * latent_kl(states, cfg)
    np.testing.assert_allclose(np.asarray(value), np.asarray(expected), rtol=1e-6, atol=1e-9)

    # The "mean" reduction is a per-layer elementwise mean, summed over layers.
    manual_kl = jnp.zeros((), dtype=DTYPE)
    for mu_l, ls_l in zip(states.mu, states.log_sigma):
        sigma = jnp.clip(jnp.exp(ls_l), 1e-8, 1e3)
        manual_kl = manual_kl + jnp.mean(0.5 * (mu_l ** 2 + sigma ** 2 - 1.0 - jnp.log(sigma ** 2)))
    np.testing.assert_allclose(np.asarray(latent_kl(states, cfg)), np.asarray(manual_kl), rtol=1e-6, atol=1e-12)

    # --- ff_centered: L_PC + lambda * KL[q(z) || N(h_l^ff, tau^2 I)]. ---
    cfg = presets["ff_centered"]
    ff_hidden = forward_hidden(params, layer_dims, x, cfg)
    value = latent_pc_energy(states, weights, x, y, cfg, key, ff_hidden)
    expected = l_pc + jnp.asarray(cfg.latent_kl_weight, dtype=DTYPE) * feedforward_latent_kl(
        states, ff_hidden, cfg
    )
    np.testing.assert_allclose(np.asarray(value), np.asarray(expected), rtol=1e-6, atol=1e-9)

    # --- matched tau^2: with h = 0 both KL formulas coincide (same width). ---
    zero_hidden = tuple(jnp.zeros_like(h) for h in ff_hidden)
    np.testing.assert_allclose(
        np.asarray(feedforward_latent_kl(states, zero_hidden, cfg)),
        np.asarray(latent_kl(states, replace(cfg, latent_prior_tau=0.0))),
        rtol=1e-5,
        atol=1e-7,
    )

    # --- the two KL conditions differ ONLY by the prior center. ---
    delta_standard = (
        latent_pc_energy(states, weights, x, y, presets["standard_normal"], key, ()) - l_pc
    ) / presets["standard_normal"].latent_kl_weight
    delta_ff_zero_center = (
        latent_pc_energy(states, weights, x, y, presets["ff_centered"], key, zero_hidden) - l_pc
    ) / presets["ff_centered"].latent_kl_weight
    np.testing.assert_allclose(
        np.asarray(delta_standard), np.asarray(delta_ff_zero_center), rtol=1e-5, atol=1e-7
    )


def test_ff_centered_stop_grad_severs_the_weight_coupling_of_the_latent_kl():
    """``latent_prior_stop_grad=True`` must keep ``h_l^ff`` out of the weight graph."""

    presets = ablation_presets()
    layer_dims = (4, 3, 2)
    x, _ = tiny_batch()
    params = init_vbpc_weight_params(layer_dims, jax.random.PRNGKey(7), -3.0)
    states = VBPCStates(
        mu=(
            jnp.asarray(
                [[0.3, -0.1, 0.2], [0.4, 0.5, -0.3], [0.1, 0.0, 0.2], [0.6, -0.5, 0.3]],
                dtype=DTYPE,
            ),
        ),
        log_sigma=(jnp.full((4, 3), -3.0, dtype=DTYPE),),
    )
    cfg = presets["ff_centered"]

    def kl_grads_through(mode_ff):
        def fn(p):
            ff = forward_hidden(p, layer_dims, x, cfg)
            if mode_ff == "detached":
                ff = tuple(jax.lax.stop_gradient(h) for h in ff)
            elif mode_ff == "zero":
                ff = tuple(jnp.zeros_like(h) for h in ff)
            return feedforward_latent_kl(states, ff, cfg)
        return jax.grad(fn)(params)

    grads_detached = kl_grads_through("detached")
    grads_raw = kl_grads_through("raw")
    grads_zero = kl_grads_through("zero")

    # With h detached (or absent) the KL has no path into the weight means.
    for g_det, g_zero in zip(grads_detached, grads_zero):
        np.testing.assert_allclose(np.asarray(g_det.mu), np.asarray(g_zero.mu), atol=0, rtol=0)
        np.testing.assert_allclose(np.asarray(g_det.log_sigma), np.asarray(g_zero.log_sigma), atol=0, rtol=0)
    # With the raw feedforward reference the coupling exists -- that is exactly
    # what ``latent_prior_stop_grad`` switches off for the ff_centered run.
    assert any(
        not np.allclose(np.asarray(a.mu), np.asarray(b.mu)) for a, b in zip(grads_raw, grads_detached)
    )


# ---------------------------------------------------------------------------
# 3. The core regression test: no_latent_kl step == original VBPC objective.
# ---------------------------------------------------------------------------


def original_reference_loss_and_grads(cfg, layer_dims, weight_params, xb, yb, key, beta=None):
    """Reference implementation of the *original* (verified-good) VBPC step.

    Reproduces the original estimator at the structural level:
    ``key_w0, key_latent, *keys_extra = jax.random.split(key, n + 2)`` with
    ``len(keys_extra) == n + 1`` ``L_PC`` samples averaged at scale ``1/n``
    plus a single ``beta * L_KL`` term.  Written independently of
    :mod:`vbpc.training.train_step` so that any change to the weight-sampling
    path (including "correcting" the split count) makes the production step
    diverge from this reference and fails the test below.
    """

    n = int(cfg.n_weight_samples)
    beta_value = jnp.asarray(cfg.beta if beta is None else beta, dtype=DTYPE)

    key_w0, key_latent, *keys_extra = jax.random.split(key, n + 2)
    eps0 = sample_weight_epsilons(weight_params, key_w0)
    weights = reparameterize_weight_matrices(
        weight_params, eps0, cfg.weight_sigma_min, cfg.weight_sigma_max
    )
    latents, _ = vbpc_latent_inference(weight_params, layer_dims, weights, xb, yb, cfg, key_latent)
    frozen = jax.lax.stop_gradient(latents)
    eps_states = sample_state_epsilons(frozen, jax.random.fold_in(key, 1))

    def pc_loss_fn(params, eps_w):
        sampled_weights = reparameterize_weight_matrices(
            params, eps_w, cfg.weight_sigma_min, cfg.weight_sigma_max
        )
        hidden = reparameterize_states(frozen, eps_states, cfg.state_sigma_min, cfg.state_sigma_max)
        return pc_energy(clamped_states(hidden, xb, yb), sampled_weights, cfg)

    extra_eps = tuple(sample_weight_epsilons(weight_params, k) for k in keys_extra)
    l_pc = None
    pc_grads = None
    for sample_eps in (eps0, *extra_eps):
        value, grads = jax.value_and_grad(pc_loss_fn)(weight_params, sample_eps)
        l_pc = value if l_pc is None else l_pc + value
        pc_grads = grads if pc_grads is None else jax.tree.map(lambda a, b: a + b, pc_grads, grads)

    scale = jnp.asarray(1.0 / n, dtype=DTYPE)
    l_pc = l_pc * scale
    pc_grads = jax.tree.map(lambda g: g * scale, pc_grads)

    l_kl = weight_kl(
        weight_params, cfg.kl_reduction, cfg.weight_sigma_min, cfg.weight_sigma_max,
        cfg.prior_weight_log_sigma,
    )
    kl_grads = weight_kl_grads(
        weight_params, cfg.kl_reduction, cfg.weight_sigma_min, cfg.weight_sigma_max,
        cfg.prior_weight_log_sigma,
    )
    total = l_pc + beta_value * l_kl
    grads = jax.tree.map(lambda p, k: p + beta_value * k, pc_grads, kl_grads)
    return total, (l_pc, l_kl), grads


@pytest.mark.parametrize("n_weight_samples", [1, 2])
def test_no_latent_kl_step_matches_the_original_reference_estimator(n_weight_samples):
    """The production step must reproduce the original VBPC objective exactly.

    ``no_latent_kl`` runs the original VBPC path: same estimator, same key
    derivation (``split(key, n + 2)``), same ``L_PC`` / ``beta * L_KL`` split.
    This is the test that fails if the 9c8ce1c weight-sampling regression
    (``split(key, n + 1)``, dropping the extra ``L_PC`` sample) is reintroduced.
    """

    presets = ablation_presets()
    layer_dims = (4, 3, 2)
    x, y = tiny_batch()
    key = jax.random.PRNGKey(0)
    params = init_vbpc_weight_params(layer_dims, jax.random.PRNGKey(1), -3.0)

    original_cfg = replace(tiny_step_cfg(n_weight_samples=n_weight_samples), name="original_ref")
    ablation_cfg = replace(
        tiny_step_cfg(
            n_weight_samples=n_weight_samples,
            beta=presets["no_latent_kl"].beta,
            prior_weight_log_sigma=presets["no_latent_kl"].prior_weight_log_sigma,
            pc_reduction=presets["no_latent_kl"].pc_reduction,
            kl_reduction=presets["no_latent_kl"].kl_reduction,
            latent_kl_weight=presets["no_latent_kl"].latent_kl_weight,
            latent_prior_tau=presets["no_latent_kl"].latent_prior_tau,
        ),
        name="ablation_ref",
    )
    # The ablation baseline must carry the original's objective settings.
    assert original_cfg.beta == ablation_cfg.beta
    assert original_cfg.prior_weight_log_sigma == ablation_cfg.prior_weight_log_sigma
    assert ablation_cfg.latent_kl_weight == 0.0
    assert ablation_cfg.latent_prior_tau == 0.0

    total, terms, grads, *_ = vbpc_loss_and_grads(ablation_cfg, layer_dims, params, x, y, key)
    ref_total, ref_terms, ref_grads = original_reference_loss_and_grads(
        original_cfg, layer_dims, params, x, y, key
    )

    np.testing.assert_allclose(np.asarray(total), np.asarray(ref_total), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(np.asarray(terms[0]), np.asarray(ref_terms[0]), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(np.asarray(terms[1]), np.asarray(ref_terms[1]), rtol=1e-6, atol=1e-9)
    for got, want in zip(grads, ref_grads):
        np.testing.assert_allclose(np.asarray(got.mu), np.asarray(want.mu), rtol=1e-5, atol=1e-9)
        np.testing.assert_allclose(
            np.asarray(got.log_sigma), np.asarray(want.log_sigma), rtol=1e-5, atol=1e-9
        )

    # Original config and ablation config are interchangeable: identical fields
    # (except name) under the same key -> bit-identical step outputs.
    total_o, terms_o, grads_o, *_ = vbpc_loss_and_grads(
        original_cfg, layer_dims, params, x, y, key
    )
    np.testing.assert_allclose(np.asarray(total), np.asarray(total_o), rtol=0, atol=0)
    for a, b in zip(terms, terms_o):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)
    for got, want in zip(grads, grads_o):
        np.testing.assert_allclose(np.asarray(got.mu), np.asarray(want.mu), rtol=0, atol=0)
        np.testing.assert_allclose(
            np.asarray(got.log_sigma), np.asarray(want.log_sigma), rtol=0, atol=0
        )


def test_reference_estimator_is_wired_to_the_original_split_count():
    """Guard the reference itself: it must encode ``split(key, n + 2)``.

    A changed split count changes the derived sub-keys, hence every sampled
    epsilon, hence the objective value -- keeping this assertion honest keeps
    the regression test above meaningful.
    """

    cfg = tiny_step_cfg(n_weight_samples=1)
    layer_dims = (4, 3, 2)
    x, y = tiny_batch()
    key = jax.random.PRNGKey(5)
    params = init_vbpc_weight_params(layer_dims, jax.random.PRNGKey(6), -3.0)

    total_ref, _, _ = original_reference_loss_and_grads(cfg, layer_dims, params, x, y, key)

    # Recompute the step with the WRONG (n + 1) split; it must differ.
    n = int(cfg.n_weight_samples)
    key_w0, key_latent, *keys_extra = jax.random.split(key, n + 1)
    eps0 = sample_weight_epsilons(params, key_w0)
    weights = reparameterize_weight_matrices(
        params, eps0, cfg.weight_sigma_min, cfg.weight_sigma_max
    )
    latents, _ = vbpc_latent_inference(params, layer_dims, weights, x, y, cfg, key_latent)
    frozen = jax.lax.stop_gradient(latents)
    eps_states = sample_state_epsilons(frozen, jax.random.fold_in(key, 1))

    def pc_loss_fn(p, eps_w):
        sw = reparameterize_weight_matrices(p, eps_w, cfg.weight_sigma_min, cfg.weight_sigma_max)
        hidden = reparameterize_states(frozen, eps_states, cfg.state_sigma_min, cfg.state_sigma_max)
        return pc_energy(clamped_states(hidden, x, y), sw, cfg)

    total_wrong, _ = jax.value_and_grad(pc_loss_fn)(params, eps0)
    total_wrong = total_wrong + cfg.beta * weight_kl(
        params, cfg.kl_reduction, cfg.weight_sigma_min, cfg.weight_sigma_max,
        cfg.prior_weight_log_sigma,
    )
    assert not np.allclose(np.asarray(total_ref), np.asarray(total_wrong), rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# 4. Training behavior: ablation baseline trains identically to the original.
# ---------------------------------------------------------------------------


def test_no_latent_kl_training_matches_original_under_matched_seed(tmp_path):
    """Two full trainings (original vs ``no_latent_kl``) must be indistinguishable.

    Same config fields, same seed => same init, same batches, same updates:
    identical final parameters and identical metrics (apart from wall-clock
    and naming fields).  Small synthetic data + a tiny net keep this fast.
    """

    from vbpc.training.trainer import train_vbpc_mnist_dataset

    presets = make_vbpc_presets()
    rng = np.random.default_rng(0)
    x_train = rng.random((8, 4))
    y_train = np.eye(2)[rng.integers(0, 2, size=8)]
    x_test = rng.random((4, 4))
    y_test = np.eye(2)[rng.integers(0, 2, size=4)]
    data = (x_train, y_train, x_test, y_test)
    layer_dims = (4, 3, 2)

    def tiny(name: str, preset_name: str):
        return replace(
            presets[preset_name],
            name=name,
            seed=7,
            epochs=2,
            batch_size=4,
            hidden=3,
            hidden_layers=1,
            latent_steps=2,
            eval_batch_size=4,
        )

    lcfg = LoggingConfig(save_dir=str(tmp_path), extra_plots=False, per_class_acc=False)

    params_a, metrics_a, rows_a = train_vbpc_mnist_dataset(
        tiny("ablation_baseline", "ablation_no_latent_kl"), data, layer_dims, lcfg
    )
    params_b, metrics_b, rows_b = train_vbpc_mnist_dataset(
        tiny("original_baseline", "vbpc_mnist_default"), data, layer_dims, lcfg
    )

    for pa, pb in zip(params_a, params_b):
        np.testing.assert_array_equal(np.asarray(pa.mu), np.asarray(pb.mu))
        np.testing.assert_array_equal(np.asarray(pa.log_sigma), np.asarray(pb.log_sigma))

    volatile = {"runtime_sec", "beta"}
    volatile |= {k for k in metrics_a if k.startswith("mem_")}
    for key in set(metrics_a) | set(metrics_b):
        if key in volatile:
            continue
        assert metrics_a[key] == metrics_b[key], key

    row_volatile = {"config", "time_sec", "runtime_sec"} | {
        k for k in rows_a[0] if k.startswith("mem_")
    }
    assert len(rows_a) == len(rows_b)
    for ra, rb in zip(rows_a, rows_b):
        for key in set(ra) | set(rb):
            if key in row_volatile:
                continue
            assert ra[key] == rb[key], key


# ---------------------------------------------------------------------------
# 5. Runner wiring: each condition runs exactly its preset, per seed.
# ---------------------------------------------------------------------------


def test_run_latent_prior_ablation_runs_exactly_the_condition_presets(tmp_path, monkeypatch):
    """The runner must train ``preset + seed`` verbatim for every (condition, seed)."""

    import vbpc.training.trainer as trainer_module
    from vbpc.training.trainer import LATENT_PRIOR_CONDITIONS, run_latent_prior_ablation

    presets = make_vbpc_presets()
    captured = []

    def fake_train(cfg, data, layer_dims, lcfg, mode="vbpc_mnist"):
        captured.append(cfg)
        metrics = {
            "best_acc": 0.5,
            "final_acc": 0.5,
            "final_nll": 1.0,
            "final_ece": 0.1,
            "mean_weight_variance": 1e-3,
            "runtime_sec": 0.0,
        }
        return (), metrics, []

    monkeypatch.setattr(trainer_module, "train_vbpc_mnist_dataset", fake_train)

    layer_dims = (4, 3, 2)
    data = (np.zeros((4, 4)), np.zeros((4, 2)), np.zeros((2, 4)), np.zeros((2, 2)))
    lcfg = LoggingConfig(save_dir=str(tmp_path))
    seeds = (0, 1)

    summary = run_latent_prior_ablation(
        data=data, layer_dims=layer_dims, lcfg=lcfg, seeds=seeds
    )

    assert tuple(LATENT_PRIOR_CONDITIONS) == ("no_latent_kl", "standard_normal", "ff_centered")
    assert len(captured) == len(seeds) * len(LATENT_PRIOR_CONDITIONS)
    assert len(summary) == len(captured)

    expected_names = [
        f"{presets[PRESET_BY_CONDITION[c]].name}_seed{s}"
        for c in LATENT_PRIOR_CONDITIONS
        for s in seeds
    ]
    assert [cfg.name for cfg in captured] == expected_names

    for cfg, entry in zip(captured, summary):
        condition = entry["condition"]
        preset = presets[PRESET_BY_CONDITION[condition]]
        # The trained config is the preset with only seed/name overridden.
        assert config_differences(cfg, preset) <= {"name", "seed"}
        assert cfg.seed == int(entry["seed"])
        assert cfg.name == entry["name"]
        assert cfg.latent_kl_weight == entry["latent_kl_weight"]
        assert cfg.latent_prior_tau == entry["latent_prior_tau"]

    # The runner itself must not add or strip any latent KL terms.
    by_condition = {cfg.name.rsplit("_seed", 1)[0]: cfg for cfg in captured}
    assert by_condition["ablation_no_latent_kl"].latent_kl_weight == 0.0
    assert by_condition["ablation_no_latent_kl"].latent_prior_tau == 0.0
    assert by_condition["ablation_standard_normal"].latent_kl_weight == ABLATION_LATENT_KL_WEIGHT
    assert by_condition["ablation_ff_centered"].latent_kl_weight == ABLATION_LATENT_KL_WEIGHT
    assert by_condition["ablation_ff_centered"].latent_prior_tau == ABLATION_LATENT_TAU

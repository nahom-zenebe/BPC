"""High-level VBPC training loops.

``train_vbpc_mnist_dataset`` mirrors :func:`bpc.training.trainer.train_mnist_dataset`
(seeded initialization, batch iteration, keep-best selection, legacy CSV log) but
drives the VBPC step: latent inference on ``L_PC`` followed by an Adam step on
``L_total = L_PC + beta * L_KL`` for the factorized Gaussian weight posterior.

Every epoch logs accuracy, NLL, ECE, runtime, memory usage, ``beta`` and the mean
weight variance, and ``run_vbpc_beta_sweep`` repeats the experiment for
``beta in [0, 0.001, 0.01, 0.1, 1.0]`` while preserving the seed/config system.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, replace
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from bpc.config import (
    DTYPE,
    LOGGING,
    REPORT_TRAIN_SUBSET_ACC,
    SAVE_DIR,
    SEED,
    STOP_ON_NAN,
    TRAIN_SUBSET_SIZE,
    LoggingConfig,
)
from bpc.data import batch_iterator, build_mnist_dims, compute_data_stats, load_mnist
from bpc.training.metrics import has_bad_values
from bpc.utils.reproducibility import make_jax_key, make_numpy_rng
from bpc.utils.validation import validate_layer_dims
from vbpc.config import VBPC_BETA_SWEEP, VBPCConfig, make_vbpc_presets, validate_vbpc_config
from vbpc.optim import optimizer_backend
from vbpc.posterior.weight_posterior import (
    VBPCWeightParams,
    init_vbpc_weight_params,
    mean_weight_variance,
    weight_diagnostics,
)
from vbpc.training.logger import VBPCRunLogger, plot_beta_sweep
from vbpc.training.metrics import (
    accuracy_from_logits,
    collect_logits,
    evaluate,
    memory_usage_mb,
)
from vbpc.training.train_step import make_vbpc_train_step


MNDataset = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def default_vbpc_config(seed: int = SEED) -> VBPCConfig:
    """VBPC default configuration (MNIST, ``beta = 0.01``)."""

    return replace(make_vbpc_presets()["vbpc_mnist_default"], seed=seed)


def default_vbpc_two_moons_config(seed: int = SEED) -> VBPCConfig:
    """VBPC default configuration (two moons, ``beta = 0.01``)."""

    return replace(make_vbpc_presets()["vbpc_two_moons"], seed=seed)


def _flatten_memory(memory: Dict[str, float]) -> Dict[str, float]:
    return {f"mem_{key}": float(value) for key, value in memory.items()}


def train_vbpc_mnist_dataset(
    cfg: VBPCConfig,
    data: MNDataset,
    layer_dims: Tuple[int, ...],
    lcfg: Optional[LoggingConfig] = None,
    mode: str = "vbpc_mnist",
) -> Tuple[Tuple[VBPCWeightParams, ...], Dict[str, float], List[Dict[str, object]]]:
    """Train VBPC on a labeled dataset and return ``(params, metrics, epoch rows)``.

    ``mode`` namespaces the artifacts of the run: the logger writes
    ``<save_dir>/<mode>/<config name>/<timestamp>/`` and the legacy CSV log is
    named ``<mode>_experiment_log.csv``.  The MNIST entry point keeps the
    default ``"vbpc_mnist"`` while the two-moons entry point passes
    ``"vbpc_two_moons"``.
    """

    validate_vbpc_config(cfg)
    validate_layer_dims(layer_dims)
    lcfg = lcfg if lcfg is not None else LOGGING
    save_dir = lcfg.save_dir or SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "=" * 100)
    print(f"{mode} VBPC config: {cfg.name} (beta={cfg.beta}, optimizer={optimizer_backend()})")
    print("=" * 100)
    print("JAX devices:", jax.devices())
    print(json.dumps(asdict(cfg), indent=2))

    x_train_np, y_train_np, x_test_np, y_test_np = data
    key = make_jax_key(cfg.seed)
    rng = make_numpy_rng(cfg.seed)
    key, sub = jax.random.split(key)
    weight_params = init_vbpc_weight_params(layer_dims, sub, cfg.init_weight_log_sigma)

    train_step, optimizer = make_vbpc_train_step(cfg, layer_dims)
    opt_state = optimizer.init(weight_params)
    beta_value = jnp.asarray(cfg.beta, dtype=DTYPE)

    logger = VBPCRunLogger(cfg, lcfg, layer_dims, mode=mode)
    data_stats = None
    if lcfg.log_data_stats:
        try:
            data_stats = compute_data_stats(x_train_np, y_train_np)
        except Exception as e:
            print(f"data stats skipped: {e}")
    logger.write_manifest(
        jax.devices(),
        data_stats,
        weight_params,
        extra={"beta": float(cfg.beta), "optimizer_backend": optimizer_backend()},
    )

    x_test = jnp.asarray(x_test_np, dtype=DTYPE)
    y_test = jnp.asarray(y_test_np, dtype=DTYPE)
    x_train_eval = jnp.asarray(x_train_np[:TRAIN_SUBSET_SIZE], dtype=DTYPE) if REPORT_TRAIN_SUBSET_ACC else None
    y_train_eval = jnp.asarray(y_train_np[:TRAIN_SUBSET_SIZE], dtype=DTYPE) if REPORT_TRAIN_SUBSET_ACC else None

    best_params = weight_params
    best_acc = -1.0
    best_epoch = 0
    rows: List[Dict[str, object]] = []
    t0 = time.time()
    global_step = 1
    for epoch in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        batch_in_epoch = 0
        epoch_pc = 0.0
        epoch_kl = 0.0
        epoch_total = 0.0
        for xb_np, yb_np in batch_iterator(x_train_np, y_train_np, cfg.batch_size, rng, shuffle=True):
            xb = jnp.asarray(xb_np, dtype=DTYPE)
            yb = jnp.asarray(yb_np, dtype=DTYPE)
            key, sub = jax.random.split(key)
            weight_params, opt_state, diag = train_step(weight_params, opt_state, xb, yb, sub, beta_value)
            logger.log_batch(global_step, epoch, batch_in_epoch, diag, cfg.beta)
            epoch_pc += float(diag.pc_energy)
            epoch_kl += float(diag.weight_kl)
            epoch_total += float(diag.total_loss)
            global_step += 1
            batch_in_epoch += 1
            if STOP_ON_NAN and has_bad_values(weight_params):
                print(f"Stopped early: NaN/Inf detected after update {global_step} in epoch {epoch}.")
                break

        n_batches = max(batch_in_epoch, 1)
        test_metrics = evaluate(weight_params, x_test, y_test, cfg)
        train_acc = float("nan")
        if x_train_eval is not None and y_train_eval is not None:
            train_acc = accuracy_from_logits(collect_logits(weight_params, x_train_eval, cfg), y_train_eval)
        memory = _flatten_memory(memory_usage_mb())
        mean_var = float(mean_weight_variance(weight_params, cfg.weight_sigma_min, cfg.weight_sigma_max))

        if test_metrics["acc"] > best_acc and cfg.keep_best:
            best_acc = test_metrics["acc"]
            best_epoch = epoch
            best_params = weight_params
        elif not cfg.keep_best:
            best_acc = test_metrics["acc"]
            best_epoch = epoch
            best_params = weight_params

        row: Dict[str, object] = {
            "config": cfg.name,
            "beta": float(cfg.beta),
            "epoch": epoch,
            "global_step": global_step,
            "train10k_acc": train_acc,
            "test_acc": test_metrics["acc"],
            "test_nll": test_metrics["nll"],
            "test_ece": test_metrics["ece"],
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "mean_weight_variance": mean_var,
            "mean_pc_energy": epoch_pc / n_batches,
            "mean_weight_kl": epoch_kl / n_batches,
            "mean_total_loss": epoch_total / n_batches,
            "time_sec": time.time() - ep_t0,
            "runtime_sec": time.time() - t0,
        }
        row.update(memory)
        rows.append(row)
        logger.log_epoch(row)
        print(
            f"epoch {epoch:3d}/{cfg.epochs}: train10k_acc={train_acc * 100:6.2f}% "
            f"test_acc={test_metrics['acc'] * 100:6.2f}% nll={test_metrics['nll']:.4f} "
            f"ece={test_metrics['ece']:.4f} beta={cfg.beta:g} var_W={mean_var:.3e} "
            f"time={row['time_sec']:.1f}s rss_max={memory.get('mem_rss_max_mb', float('nan')):.1f}MB"
        )
        if STOP_ON_NAN and has_bad_values(weight_params):
            break

    final_row: Dict[str, object] = rows[-1] if rows else {}
    final_acc = float(final_row.get("test_acc", float("nan")))
    selected = best_params if cfg.keep_best else weight_params
    diag = weight_diagnostics(selected, cfg.weight_sigma_min, cfg.weight_sigma_max)
    print(f"final test_acc={final_acc * 100:.2f}% best_acc={best_acc * 100:.2f}% at epoch {best_epoch}")
    print(f"total_time={time.time() - t0:.1f}s")

    csv_path = os.path.join(save_dir, f"{mode}_experiment_log.csv")
    write_header = not os.path.exists(csv_path)
    try:
        with open(csv_path, "a", newline="") as f:
            fieldnames = (list(rows[0].keys()) if rows else []) + list(asdict(cfg).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for r in rows:
                full = dict(r)
                full.update(asdict(cfg))
                writer.writerow(full)
        print("VBPC CSV log:", csv_path)
    except Exception as e:  # pragma: no cover - IO guard
        print(f"VBPC CSV skipped: {e}")

    try:
        logger.write_plots()
    except Exception as e:  # pragma: no cover - plotting guard
        print(f"plots skipped: {e}")
    logger.close()

    metrics: Dict[str, float] = {
        "beta": float(cfg.beta),
        "final_acc": final_acc,
        "best_acc": float(best_acc),
        "best_epoch": float(best_epoch),
        "final_nll": float(final_row.get("test_nll", float("nan"))),
        "final_ece": float(final_row.get("test_ece", float("nan"))),
        "mean_weight_variance": float(diag["mean_weight_variance"]),
        "runtime_sec": time.time() - t0,
    }
    metrics.update({k: float(v) for k, v in final_row.items() if k.startswith("mem_")})
    return selected, metrics, rows


def train_vbpc_mnist(
    cfg: Optional[VBPCConfig] = None,
    lcfg: Optional[LoggingConfig] = None,
) -> Tuple[Tuple[VBPCWeightParams, ...], Dict[str, float], List[Dict[str, object]]]:
    """Load MNIST through the shared data module and run VBPC."""

    cfg = cfg if cfg is not None else default_vbpc_config()
    data = load_mnist(cfg)
    layer_dims = build_mnist_dims(cfg)
    return train_vbpc_mnist_dataset(cfg, data, layer_dims, lcfg)


def run_vbpc_beta_sweep(
    cfg: Optional[VBPCConfig] = None,
    data: Optional[MNDataset] = None,
    layer_dims: Optional[Tuple[int, ...]] = None,
    lcfg: Optional[LoggingConfig] = None,
    betas: Tuple[float, ...] = VBPC_BETA_SWEEP,
    mode: str = "vbpc_mnist",
    summary_name: str = "vbpc_beta_sweep",
) -> List[Dict[str, object]]:
    """Run VBPC for every ``beta`` in the proposal sweep and summarize the results.

    Each run reuses the same seed, so data ordering and weight initialization are
    identical across betas and the comparison isolates the effect of ``beta``.

    ``mode`` namespaces the per-run directories and legacy CSV log (passed through
    to :func:`train_vbpc_mnist_dataset`), while ``summary_name`` names the combined
    ``<summary_name>.json`` summary so different datasets do not overwrite each
    other in the shared ``save_dir``.
    """

    cfg = cfg if cfg is not None else default_vbpc_config()
    data = data if data is not None else load_mnist(cfg)
    layer_dims = tuple(layer_dims) if layer_dims is not None else build_mnist_dims(cfg)
    active_lcfg = lcfg if lcfg is not None else LOGGING
    save_dir = active_lcfg.save_dir or SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    summary: List[Dict[str, object]] = []
    for beta in betas:
        run_cfg = replace(cfg, beta=float(beta), name=f"{cfg.name}_beta{beta:g}")
        _, metrics, _ = train_vbpc_mnist_dataset(run_cfg, data, layer_dims, active_lcfg, mode=mode)
        summary.append({
            "beta": float(beta),
            "name": run_cfg.name,
            "seed": int(cfg.seed),
            "best_acc": metrics["best_acc"],
            "best_epoch": metrics["best_epoch"],
            "final_acc": metrics["final_acc"],
            "final_nll": metrics["final_nll"],
            "final_ece": metrics["final_ece"],
            "mean_weight_variance": metrics["mean_weight_variance"],
            "runtime_sec": metrics["runtime_sec"],
        })
        print(
            f"[beta sweep] beta={beta:g}: best_acc={metrics['best_acc'] * 100:.2f}% "
            f"nll={metrics['final_nll']:.4f} ece={metrics['final_ece']:.4f} "
            f"var_W={metrics['mean_weight_variance']:.3e}"
        )

    json_path = os.path.join(save_dir, f"{summary_name}.json")
    try:
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print("VBPC beta sweep summary:", json_path)
    except Exception as e:  # pragma: no cover - IO guard
        print(f"beta sweep summary write failed: {e}")
    plot_beta_sweep(summary, save_dir, dpi=active_lcfg.plot_dpi, name=f"{cfg.name}_beta_sweep")
    return summary


def train_vbpc_two_moons_dataset(
    cfg: VBPCConfig,
    data: MNDataset,
    layer_dims: Tuple[int, ...],
    lcfg: Optional[LoggingConfig] = None,
) -> Tuple[Tuple[VBPCWeightParams, ...], Dict[str, float], List[Dict[str, object]]]:
    """Train VBPC on a two-moons dataset (same loop, ``vbpc_two_moons`` namespace)."""

    return train_vbpc_mnist_dataset(cfg, data, layer_dims, lcfg, mode="vbpc_two_moons")

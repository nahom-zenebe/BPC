"""Run logging for VBPC experiments.

``VBPCRunLogger`` follows the conventions of :class:`bpc.utils.logging.RunLogger`
(the same ``<save_dir>/<mode>/<config name>/<timestamp>`` layout, plus
``manifest.json``, ``epoch.csv``, ``batch.csv``, ``verbose.jsonl`` and a ``plots``
folder) and reuses :func:`bpc.utils.logging.diag_to_np` to serialize the VBPC
diagnostics.  Compared to the BPC logger it drops the Matrix-Normal Wishart
specific fields and instead records the VBPC quantities: accuracy, NLL, ECE,
runtime, memory usage, ``beta`` and the mean weight variance.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from bpc.config import DTYPE, ENABLE_X64, SAVE_DIR, LoggingConfig
from bpc.utils.logging import diag_to_np
from vbpc.config import VBPCConfig
from vbpc.posterior.weight_posterior import weight_diagnostics
from vbpc.training.train_step import VBPCDiagnostics


class VBPCRunLogger:
    """Pure-Python logger for one VBPC run."""

    def __init__(
        self,
        cfg: VBPCConfig,
        lcfg: LoggingConfig,
        layer_dims: Tuple[int, ...],
        mode: str = "vbpc_mnist",
    ):
        self.cfg = cfg
        self.lcfg = lcfg
        self.layer_dims = tuple(int(d) for d in layer_dims)
        self.mode = mode
        self.L_eta = len(self.layer_dims) - 1
        self.L_state = max(0, len(self.layer_dims) - 2)

        self._batch_records: List[Dict[str, float]] = []
        self._epoch_records: List[Dict[str, float]] = []

        self.run_dir: Optional[str] = None
        self.run_id: Optional[str] = None
        self._batch_csv_fh = None
        self._batch_csv_writer = None
        self._verbose_fh = None
        self._epoch_csv_fh = None
        self._epoch_csv_writer = None

        if not lcfg.enabled:
            return

        base = lcfg.save_dir or SAVE_DIR
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{lcfg.run_id_prefix}{ts}_{cfg.name}" if lcfg.run_id_prefix else f"{ts}_{cfg.name}"
        self.run_dir = os.path.join(base, mode, cfg.name, ts)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "plots"), exist_ok=True)

        if lcfg.batch_csv:
            self._batch_csv_fh = open(os.path.join(self.run_dir, "batch.csv"), "w", newline="")
        if lcfg.verbose_jsonl:
            self._verbose_fh = open(os.path.join(self.run_dir, "verbose.jsonl"), "w")
        self._epoch_csv_fh = open(os.path.join(self.run_dir, "epoch.csv"), "w", newline="")
        print(f"[VBPCRunLogger] writing to {self.run_dir}")

    def write_manifest(
        self,
        jax_devices: Sequence[object],
        data_stats: Optional[Dict[str, object]] = None,
        init_weight_params: Optional[Sequence[object]] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self.lcfg.enabled or not self.lcfg.write_manifest or self.run_dir is None:
            return
        manifest: Dict[str, object] = {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mode": self.mode,
            "config": asdict(self.cfg),
            "logging_config": asdict(self.lcfg),
            "jax_devices": [str(d) for d in jax_devices],
            "dtype": str(DTYPE),
            "x64_enabled": ENABLE_X64,
            "layer_dims": list(self.layer_dims),
            "L_eta": self.L_eta,
            "L_state": self.L_state,
        }
        if data_stats is not None and self.lcfg.log_data_stats:
            manifest["data_stats"] = data_stats
        if init_weight_params is not None and self.lcfg.log_init_posterior:
            manifest["initial_weight_posterior"] = weight_diagnostics(
                tuple(init_weight_params), self.cfg.weight_sigma_min, self.cfg.weight_sigma_max
            )
        if extra:
            manifest.update(extra)
        try:
            with open(os.path.join(self.run_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2, default=str)
        except Exception as e:  # pragma: no cover - IO guard
            print(f"[VBPCRunLogger] manifest write failed: {e}")

    def log_batch(
        self,
        global_step: int,
        epoch: int,
        batch_in_epoch: int,
        diag: VBPCDiagnostics,
        beta: float,
    ) -> None:
        """Record one optimizer step (``batch.csv`` and ``verbose.jsonl``)."""

        if not self.lcfg.enabled or self.run_dir is None:
            return
        values = diag_to_np(diag)
        row: Dict[str, object] = {
            "global_step": int(global_step),
            "epoch": int(epoch),
            "batch_in_epoch": int(batch_in_epoch),
            "beta": float(beta),
        }
        for key, value in values.items():
            array = np.asarray(value)
            row[key] = float(array) if array.ndim == 0 else array.tolist()
        if self.lcfg.batch_csv and self._batch_csv_fh is not None:
            flat_row = {
                k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()
            }
            if self._batch_csv_writer is None:
                self._batch_csv_writer = csv.DictWriter(self._batch_csv_fh, fieldnames=list(flat_row.keys()))
                self._batch_csv_writer.writeheader()
            self._batch_csv_writer.writerow(flat_row)
            self._batch_csv_fh.flush()
        self._batch_records.append(row)
        if self.lcfg.verbose_jsonl and self._verbose_fh is not None:
            self._verbose_fh.write(json.dumps(row, default=str) + "\n")

    def log_epoch(self, row_dict: Dict[str, object]) -> None:
        """Record one epoch of metrics (``epoch.csv``)."""

        self._epoch_records.append(dict(row_dict))
        if not self.lcfg.enabled or self._epoch_csv_fh is None:
            return
        if self._epoch_csv_writer is None:
            self._epoch_csv_writer = csv.DictWriter(self._epoch_csv_fh, fieldnames=list(row_dict.keys()))
            self._epoch_csv_writer.writeheader()
        try:
            self._epoch_csv_writer.writerow(row_dict)
        except ValueError:  # pragma: no cover - schema guard
            pass
        self._epoch_csv_fh.flush()

    def write_plots(self) -> None:
        """Optional matplotlib diagnostics (guarded, like the BPC logger)."""

        if not self.lcfg.enabled or not self.lcfg.extra_plots or self.run_dir is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover - optional dependency
            print(f"[VBPCRunLogger] plot import failed: {e}")
            return

        plot_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        dpi = self.lcfg.plot_dpi
        eps = self._epoch_records
        bts = self._batch_records

        def _series(records, key):
            return [float(r.get(key, float("nan"))) for r in records]

        if eps:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(_series(eps, "epoch"), [100 * v for v in _series(eps, "test_acc")], label="test")
                if "train10k_acc" in eps[0]:
                    ax.plot(_series(eps, "epoch"), [100 * v for v in _series(eps, "train10k_acc")], label="train10k")
                ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)")
                ax.set_title(f"{self.cfg.name} (beta={self.cfg.beta})")
                ax.legend(); ax.grid(True, alpha=0.3)
                fig.savefig(os.path.join(plot_dir, "accuracy.png"), dpi=dpi, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:  # pragma: no cover - plotting guard
                print(f"[VBPCRunLogger] accuracy plot failed: {e}")

            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(_series(eps, "epoch"), _series(eps, "test_nll"), label="test NLL", color="tab:red")
                ax2 = ax.twinx()
                ax2.plot(_series(eps, "epoch"), _series(eps, "test_ece"), label="test ECE", color="tab:green")
                ax.set_xlabel("epoch"); ax.set_ylabel("NLL"); ax2.set_ylabel("ECE")
                ax.set_title(f"{self.cfg.name}: NLL and ECE")
                ax.grid(True, alpha=0.3)
                fig.savefig(os.path.join(plot_dir, "nll_ece.png"), dpi=dpi, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:  # pragma: no cover - plotting guard
                print(f"[VBPCRunLogger] NLL/ECE plot failed: {e}")

            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(_series(eps, "epoch"), _series(eps, "mean_weight_variance"), color="tab:purple")
                ax.set_xlabel("epoch"); ax.set_ylabel("mean sigma_W^2")
                ax.set_title(f"{self.cfg.name}: mean weight variance (beta={self.cfg.beta})")
                ax.grid(True, alpha=0.3)
                fig.savefig(os.path.join(plot_dir, "weight_variance.png"), dpi=dpi, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:  # pragma: no cover - plotting guard
                print(f"[VBPCRunLogger] weight variance plot failed: {e}")

        if bts:
            try:
                fig, ax = plt.subplots(figsize=(9, 5))
                steps = _series(bts, "global_step")
                for key, label in (("pc_energy", "L_PC"), ("weight_kl", "L_KL"), ("total_loss", "L_total")):
                    ax.plot(steps, _series(bts, key), label=label)
                ax.set_yscale("symlog", linthresh=1e-8)
                ax.set_xlabel("global_step"); ax.set_ylabel("objective")
                ax.set_title(f"{self.cfg.name}: VBPC objective (beta={self.cfg.beta})")
                ax.legend(); ax.grid(True, alpha=0.3)
                fig.savefig(os.path.join(plot_dir, "objectives.png"), dpi=dpi, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:  # pragma: no cover - plotting guard
                print(f"[VBPCRunLogger] objective plot failed: {e}")

            try:
                fig, ax = plt.subplots(figsize=(9, 5))
                steps = _series(bts, "global_step")
                ax.plot(steps, _series(bts, "latent_energy_init"), label="L_PC before latent inference", linestyle="--")
                ax.plot(steps, _series(bts, "latent_energy_final"), label="L_PC after latent inference")
                ax.set_yscale("symlog", linthresh=1e-8)
                ax.set_xlabel("global_step"); ax.set_ylabel("L_PC")
                ax.set_title(f"{self.cfg.name}: latent inference convergence")
                ax.legend(); ax.grid(True, alpha=0.3)
                fig.savefig(os.path.join(plot_dir, "latent_inference.png"), dpi=dpi, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:  # pragma: no cover - plotting guard
                print(f"[VBPCRunLogger] latent inference plot failed: {e}")

        plt.close("all")

    def close(self) -> None:
        if not self.lcfg.enabled:
            return
        for fh_name in ("_batch_csv_fh", "_verbose_fh", "_epoch_csv_fh"):
            fh = getattr(self, fh_name, None)
            if fh is not None:
                try:
                    fh.flush()
                    fh.close()
                except Exception:  # pragma: no cover - IO guard
                    pass


def plot_beta_sweep(
    summary: List[Dict[str, object]],
    out_dir: str,
    dpi: int = 160,
    name: str = "vbpc_beta_sweep",
) -> Optional[str]:
    """Bar chart of best accuracy, final NLL and mean weight variance against beta."""

    if not summary:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - optional dependency
        print(f"[VBPCRunLogger] beta sweep plot import failed: {e}")
        return None
    try:
        os.makedirs(out_dir, exist_ok=True)
        betas = [float(r["beta"]) for r in summary]
        labels = [f"{b:g}" for b in betas]
        x = np.arange(len(betas))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].bar(x, [100 * float(r["best_acc"]) for r in summary]); axes[0].set_ylabel("best accuracy (%)")
        axes[1].bar(x, [float(r["final_nll"]) for r in summary]); axes[1].set_ylabel("final NLL")
        axes[2].bar(x, [float(r["mean_weight_variance"]) for r in summary]); axes[2].set_ylabel("mean sigma_W^2")
        for ax in axes:
            ax.set_xticks(x); ax.set_xticklabels(labels)
            ax.set_xlabel("beta"); ax.grid(True, alpha=0.3)
        fig.suptitle(name)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:  # pragma: no cover - plotting guard
        print(f"[VBPCRunLogger] beta sweep plot failed: {e}")
        return None

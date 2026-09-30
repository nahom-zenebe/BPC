"""Run logging, diagnostic JSON/CSV output, and optional plots."""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from bpc.config import BPCConfig, DTYPE, ENABLE_X64, LoggingConfig, SAVE_DIR
from bpc.posterior.natural_params import Eta
from bpc.posterior.posterior_state import LatentDiagnostics, MStepDiagnostics, StatsDiagnostics
from bpc.training.metrics import parameter_diagnostics


def _diag_to_np(diag) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for f in diag._fields:
        v = getattr(diag, f)
        out[f] = np.asarray(v)
    return out


#: Public alias used by the VBPC logger to serialize any diagnostics namedtuple.
diag_to_np = _diag_to_np


class RunLogger:
    """Pure-Python logger for one training run."""

    def __init__(
        self,
        cfg: BPCConfig,
        lcfg: LoggingConfig,
        layer_dims: Tuple[int, ...],
        mode: str = "mnist",
    ):
        self.cfg = cfg
        self.lcfg = lcfg
        self.layer_dims = tuple(int(d) for d in layer_dims)
        self.mode = mode
        self.L_eta = len(self.layer_dims) - 1
        self.L_state = max(0, len(self.layer_dims) - 2)

        self._batch_records: List[Dict[str, float]] = []
        self._epoch_records: List[Dict[str, float]] = []
        self._anomaly_count = 0

        self.run_dir: Optional[str] = None
        self.run_id: Optional[str] = None

        self._batch_csv_fh = None
        self._batch_csv_writer = None
        self._verbose_fh = None
        self._adam_fh = None
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
        if lcfg.adam_trace_jsonl:
            self._adam_fh = open(os.path.join(self.run_dir, "adam_trace.jsonl"), "w")
        self._epoch_csv_fh = open(os.path.join(self.run_dir, "epoch.csv"), "w", newline="")

        print(f"[RunLogger] writing to {self.run_dir}")

    def write_manifest(
        self,
        jax_devices,
        prior: Tuple[Eta, ...],
        init_params: Tuple[Eta, ...],
        data_stats: Optional[Dict[str, object]] = None,
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
            "prior_v0": float(self.cfg.v0),
            "prior_psi0": float(self.cfg.psi0),
            "prior_nu0_extra": float(self.cfg.nu0_extra),
            "L_eta": self.L_eta,
            "L_state": self.L_state,
        }
        if data_stats is not None and self.lcfg.log_data_stats:
            manifest["data_stats"] = data_stats
        if init_params is not None and self.lcfg.log_init_posterior:
            manifest["initial_posterior"] = parameter_diagnostics(init_params, self.layer_dims)
        if prior is not None:
            manifest["prior_summary"] = parameter_diagnostics(prior, self.layer_dims)
        try:
            with open(os.path.join(self.run_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2, default=str)
        except Exception as e:
            print(f"[RunLogger] manifest write failed: {e}")

    def log_batch(
        self,
        global_step: int,
        epoch: int,
        batch_in_epoch: int,
        kappa: Optional[float],
        scale: Optional[float],
        stats_diag: StatsDiagnostics,
        latent_diag: LatentDiagnostics,
        mstep_diag: Optional[MStepDiagnostics],
    ) -> None:
        if not self.lcfg.enabled or self.run_dir is None:
            return

        sd = _diag_to_np(stats_diag)
        ld = _diag_to_np(latent_diag)
        md = _diag_to_np(mstep_diag) if mstep_diag is not None else None

        anomaly_reasons: List[str] = []
        if float(sd["T1_norm"].max()) > self.lcfg.t_norm_alarm:
            anomaly_reasons.append("T1_norm_high")
        if float(ld["psi_inv_raw_min"].min()) < self.lcfg.psi_alarm:
            anomaly_reasons.append("psi_inv_min_low")
        if md is not None and float(md["eta_delta_Vinv"].max()) > self.lcfg.eta_jump_alarm:
            anomaly_reasons.append("eta_jump")
        if float(ld["final_grad_norm"].max()) > self.lcfg.grad_alarm:
            anomaly_reasons.append("grad_alarm")
        anomaly = len(anomaly_reasons) > 0
        if anomaly:
            self._anomaly_count += 1

        if self.lcfg.batch_csv and self._batch_csv_fh is not None:
            row = self._build_batch_row(global_step, epoch, batch_in_epoch, kappa, scale, sd, ld, md, anomaly)
            if self._batch_csv_writer is None:
                self._batch_csv_writer = csv.DictWriter(self._batch_csv_fh, fieldnames=list(row.keys()))
                self._batch_csv_writer.writeheader()
            self._batch_csv_writer.writerow(row)
            self._batch_records.append(row)
            if int(global_step) % 100 == 0:
                self._batch_csv_fh.flush()

        if self.lcfg.verbose_jsonl and self._verbose_fh is not None:
            verbose_record = {
                "global_step": int(global_step),
                "epoch": int(epoch),
                "batch_in_epoch": int(batch_in_epoch),
                "kappa": float(kappa) if kappa is not None else None,
                "scale": float(scale) if scale is not None else None,
                "anomaly": anomaly,
                "anomaly_reasons": anomaly_reasons,
                "stats": {k: v.tolist() for k, v in sd.items()},
                "latent": {k: v.tolist() for k, v in ld.items()},
            }
            if md is not None:
                verbose_record["mstep"] = {k: v.tolist() for k, v in md.items()}
            self._verbose_fh.write(json.dumps(verbose_record, default=str) + "\n")

        if self.lcfg.adam_trace_jsonl and self._adam_fh is not None:
            grad_steps = ld["grad_norm_per_step"]
            latent_steps = ld["latent_norm_per_step"]
            energy_steps = ld["energy_per_step"]
            T = grad_steps.shape[0]
            for t in range(T):
                rec = {
                    "global_step": int(global_step),
                    "epoch": int(epoch),
                    "batch_in_epoch": int(batch_in_epoch),
                    "adam_step": t,
                    "grad_norm": grad_steps[t].tolist(),
                    "latent_norm": latent_steps[t].tolist(),
                    "energy": energy_steps[t].tolist(),
                }
                self._adam_fh.write(json.dumps(rec, default=str) + "\n")

    def _build_batch_row(
        self,
        global_step: int,
        epoch: int,
        batch_in_epoch: int,
        kappa: Optional[float],
        scale: Optional[float],
        sd: Dict[str, np.ndarray],
        ld: Dict[str, np.ndarray],
        md: Optional[Dict[str, np.ndarray]],
        anomaly: bool,
    ) -> Dict[str, float]:
        row: Dict[str, float] = {
            "global_step": int(global_step),
            "epoch": int(epoch),
            "batch_in_epoch": int(batch_in_epoch),
            "kappa": float(kappa) if kappa is not None else float("nan"),
            "scale": float(scale) if scale is not None else float("nan"),
            "anomaly": int(anomaly),
        }
        for l in range(self.L_eta):
            row[f"T1_norm_l{l}"] = float(sd["T1_norm"][l])
            row[f"T2_norm_l{l}"] = float(sd["T2_norm"][l])
            row[f"T3_norm_l{l}"] = float(sd["T3_norm"][l])
            row[f"T4_l{l}"] = float(sd["T4"][l])
            row[f"M_norm_l{l}"] = float(ld["M_norm"][l])
            row[f"Vinv_fro_l{l}"] = float(ld["Vinv_fro"][l])
            row[f"psi_inv_raw_min_l{l}"] = float(ld["psi_inv_raw_min"][l])
            row[f"E_prec_diag_mean_l{l}"] = float(ld["E_prec_diag_mean"][l])
            if md is not None:
                row[f"eta_delta_Vinv_l{l}"] = float(md["eta_delta_Vinv"][l])
                row[f"eta_delta_MV_l{l}"] = float(md["eta_delta_MV"][l])
                row[f"eta_delta_S_l{l}"] = float(md["eta_delta_S"][l])
                row[f"eta_delta_nu_l{l}"] = float(md["eta_delta_nu"][l])
                row[f"new_M_norm_l{l}"] = float(md["new_M_norm"][l])
                row[f"new_psi_inv_raw_min_l{l}"] = float(md["new_psi_inv_raw_min"][l])
        for l in range(self.L_state):
            row[f"init_grad_norm_l{l}"] = float(ld["init_grad_norm"][l])
            row[f"final_grad_norm_l{l}"] = float(ld["final_grad_norm"][l])
        row["total_energy"] = float(np.sum(ld["energy_per_step"][-1])) if ld["energy_per_step"].size else 0.0
        return row

    def log_epoch(self, row_dict: Dict[str, object]) -> None:
        if not self.lcfg.enabled or self._epoch_csv_fh is None:
            self._epoch_records.append(dict(row_dict))
            return
        if self._epoch_csv_writer is None:
            self._epoch_csv_writer = csv.DictWriter(self._epoch_csv_fh, fieldnames=list(row_dict.keys()))
            self._epoch_csv_writer.writeheader()
        try:
            self._epoch_csv_writer.writerow(row_dict)
        except ValueError:
            pass
        self._epoch_csv_fh.flush()
        self._epoch_records.append(dict(row_dict))

    def log_nan_event(self, global_step: int, epoch: int, params: Tuple[Eta, ...]) -> None:
        if not self.lcfg.enabled or self.run_dir is None:
            return
        path = os.path.join(self.run_dir, f"nan_event_step{global_step}_ep{epoch}.npz")
        try:
            arrays: Dict[str, np.ndarray] = {}
            for l, eta in enumerate(params):
                arrays[f"V_inv_l{l}"] = np.asarray(eta.V_inv)
                arrays[f"MV_l{l}"] = np.asarray(eta.MV)
                arrays[f"S_l{l}"] = np.asarray(eta.S)
                arrays[f"nu_shift_l{l}"] = np.asarray(eta.nu_shift)
            np.savez(path, **arrays)
            print(f"[RunLogger] dumped NaN params to {path}")
        except Exception as e:
            print(f"[RunLogger] NaN dump failed: {e}")

    def write_plots(self, x_train_np: Optional[np.ndarray] = None) -> None:
        if not self.lcfg.enabled or not self.lcfg.extra_plots or self.run_dir is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[RunLogger] plot import failed: {e}")
            return

        plot_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        dpi = self.lcfg.plot_dpi

        eps = self._epoch_records
        bts = self._batch_records

        if eps:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot([r["epoch"] for r in eps], [100 * float(r.get("test_acc", float("nan"))) for r in eps], label="test")
                if "train10k_acc" in eps[0]:
                    ax.plot([r["epoch"] for r in eps], [100 * float(r.get("train10k_acc", float("nan"))) for r in eps], label="train10k")
                ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)"); ax.set_title(self.cfg.name)
                ax.legend(); ax.grid(True, alpha=0.3)
                fig.savefig(os.path.join(plot_dir, "accuracy.png"), dpi=dpi, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"[RunLogger] accuracy plot failed: {e}")

        if eps and any(k.startswith("acc_class_") for k in eps[0].keys()):
            try:
                cls_keys = sorted([k for k in eps[0].keys() if k.startswith("acc_class_")],
                                  key=lambda k: int(k.split("_")[-1]))
                fig, ax = plt.subplots(figsize=(8, 5))
                for k in cls_keys:
                    ax.plot([r["epoch"] for r in eps], [100 * float(r.get(k, float("nan"))) for r in eps], label=k)
                ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)")
                ax.set_title(f"{self.cfg.name} per-class")
                ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
                fig.savefig(os.path.join(plot_dir, "per_class_acc.png"), dpi=dpi, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"[RunLogger] per-class plot failed: {e}")

        if not bts:
            return

        steps = np.array([r["global_step"] for r in bts])

        def _plot_per_step(yname_template: str, ylabel: str, fname: str, log: bool = False, count: Optional[int] = None) -> None:
            try:
                n = count if count is not None else self.L_eta
                fig, ax = plt.subplots(figsize=(9, 5))
                for l in range(n):
                    key = yname_template.format(l=l)
                    if key not in bts[0]:
                        continue
                    ax.plot(steps, [float(r.get(key, float("nan"))) for r in bts], label=f"layer {l}")
                if log:
                    ax.set_yscale("symlog", linthresh=1e-12)
                ax.set_xlabel("global_step"); ax.set_ylabel(ylabel)
                ax.set_title(f"{self.cfg.name}: {ylabel}")
                ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
                fig.savefig(os.path.join(plot_dir, fname), dpi=dpi, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"[RunLogger] plot {fname} failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(steps, [float(r.get("kappa", float("nan"))) for r in bts], color="tab:blue")
            ax.set_xlabel("global_step"); ax.set_ylabel("kappa")
            ax.set_title(f"{self.cfg.name}: kappa schedule"); ax.grid(True, alpha=0.3)
            fig.savefig(os.path.join(plot_dir, "kappa_schedule.png"), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[RunLogger] kappa plot failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(steps, [float(r.get("scale", float("nan"))) for r in bts], color="tab:orange")
            ax.set_xlabel("global_step"); ax.set_ylabel("scale")
            ax.set_title(f"{self.cfg.name}: scale schedule"); ax.grid(True, alpha=0.3)
            fig.savefig(os.path.join(plot_dir, "scale_schedule.png"), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[RunLogger] scale plot failed: {e}")

        try:
            fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
            channels = ["eta_delta_Vinv_l{l}", "eta_delta_MV_l{l}", "eta_delta_S_l{l}", "eta_delta_nu_l{l}"]
            titles = ["||Δ V_inv||", "||Δ MV||", "||Δ S||", "|Δ ν_shift|"]
            for ax, tmpl, ttl in zip(axes.flat, channels, titles):
                for l in range(self.L_eta):
                    key = tmpl.format(l=l)
                    if key not in bts[0]:
                        continue
                    vals = np.array([float(r.get(key, float("nan"))) for r in bts])
                    ax.plot(steps, np.maximum(vals, 1e-30), label=f"layer {l}")
                ax.set_yscale("log"); ax.set_title(ttl)
                ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
            axes[1, 0].set_xlabel("global_step"); axes[1, 1].set_xlabel("global_step")
            fig.suptitle(f"{self.cfg.name}: M-step parameter deltas (log scale)")
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, "eta_delta.png"), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[RunLogger] eta_delta plot failed: {e}")

        _plot_per_step("M_norm_l{l}", "||M||", "M_norm.png")
        _plot_per_step("psi_inv_raw_min_l{l}", "min eig psi_inv_raw (symlog)", "psi_inv_min.png", log=True)
        _plot_per_step("init_grad_norm_l{l}", "||g_init||", "latent_grad_norm_init.png", count=self.L_state)
        _plot_per_step("final_grad_norm_l{l}", "||g_final||", "latent_grad_norm_final.png", count=self.L_state)
        _plot_per_step("T1_norm_l{l}", "||T1||", "T1_norm.png")
        _plot_per_step("T3_norm_l{l}", "||T3||", "T3_norm.png")
        _plot_per_step("E_prec_diag_mean_l{l}", "mean diag E[Σ⁻¹]", "E_prec_diag_mean.png")

        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            for l in range(self.L_state):
                init_vals = np.array([float(r.get(f"init_grad_norm_l{l}", np.nan)) for r in bts])
                final_vals = np.array([float(r.get(f"final_grad_norm_l{l}", np.nan)) for r in bts])
                ax.plot(steps, init_vals, label=f"init l{l}", linestyle="--")
                ax.plot(steps, final_vals, label=f"final l{l}")
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.set_xlabel("global_step"); ax.set_ylabel("||g||")
            ax.set_title(f"{self.cfg.name}: Adam init vs final grad norm")
            ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
            fig.savefig(os.path.join(plot_dir, "latent_grad_norm_init_vs_final.png"), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[RunLogger] init vs final plot failed: {e}")

        try:
            if self.lcfg.adam_trace_jsonl:
                self._plot_adam_trajectory(plot_dir, dpi)
        except Exception as e:
            print(f"[RunLogger] adam_trajectory plot failed: {e}")

        if x_train_np is not None:
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(x_train_np.ravel(), bins=80, color="tab:purple")
                ax.set_xlabel("input value"); ax.set_ylabel("count")
                ax.set_title(f"{self.cfg.name}: input histogram (normalize={self.cfg.normalize})")
                fig.savefig(os.path.join(plot_dir, "data_histogram.png"), dpi=dpi, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"[RunLogger] data histogram plot failed: {e}")

        try:
            mpath = os.path.join(self.run_dir, "manifest.json")
            if os.path.exists(mpath):
                with open(mpath) as f:
                    manifest = json.load(f)
                ip = manifest.get("initial_posterior")
                if ip:
                    metrics = ["M{l}_norm", "Vinv{l}_eig_min", "PsiInvRaw{l}_eig_min", "Psi{l}_diag_max"]
                    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
                    for ax, key_tmpl in zip(axes, metrics):
                        vals = [ip.get(key_tmpl.format(l=l), float("nan")) for l in range(self.L_eta)]
                        ax.bar(range(self.L_eta), vals)
                        ax.set_xticks(range(self.L_eta))
                        ax.set_title(key_tmpl.replace("{l}", "l")); ax.grid(True, alpha=0.3)
                    fig.suptitle(f"{self.cfg.name}: initial posterior")
                    fig.tight_layout()
                    fig.savefig(os.path.join(plot_dir, "init_diagnostics.png"), dpi=dpi, bbox_inches="tight")
                    plt.close(fig)
        except Exception as e:
            print(f"[RunLogger] init diag plot failed: {e}")

        plt.close("all")

    def _plot_adam_trajectory(self, plot_dir: str, dpi: int) -> None:
        import matplotlib.pyplot as plt

        path = os.path.join(self.run_dir, "adam_trace.jsonl") if self.run_dir else None
        if path is None or not os.path.exists(path):
            return
        records: Dict[int, List[dict]] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records.setdefault(rec["global_step"], []).append(rec)
        steps = sorted(records.keys())
        if not steps:
            return
        n = len(steps)
        sample_idxs = sorted(set([0, 1, 2,
                                   max(0, n // 2 - 1), n // 2, min(n - 1, n // 2 + 1),
                                   max(0, n - 3), max(0, n - 2), n - 1]))
        sample_steps = [steps[i] for i in sample_idxs if i < n]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, key, ttl in zip(axes, ["grad_norm", "latent_norm", "energy"],
                                 ["||grad|| per step", "||z|| per step", "energy per step"]):
            for s in sample_steps:
                recs = sorted(records[s], key=lambda r: r["adam_step"])
                ts = [r["adam_step"] for r in recs]
                arr = np.array([r[key] for r in recs])
                ax.plot(ts, arr.mean(axis=1), label=f"step {s}", alpha=0.7)
            ax.set_xlabel("adam step"); ax.set_title(ttl)
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
        fig.suptitle(f"{self.cfg.name}: Adam latent inference trajectory (sampled batches)")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "adam_trajectory.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        last_step = sample_steps[-1]
        recs = sorted(records[last_step], key=lambda r: r["adam_step"])
        arr = np.array([r["energy"] for r in recs])
        for l in range(arr.shape[1]):
            ax.plot([r["adam_step"] for r in recs], arr[:, l], label=f"layer {l}")
        ax.set_xlabel("adam step"); ax.set_ylabel("energy (per layer)")
        ax.set_title(f"{self.cfg.name}: per-layer energy at global_step={last_step}")
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(plot_dir, "energy_curves.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def close(self) -> None:
        if not self.lcfg.enabled:
            return
        if getattr(self, "_batch_csv_fh", None) is not None:
            try:
                self._batch_csv_fh.flush(); self._batch_csv_fh.close()
            except Exception:
                pass
        if getattr(self, "_verbose_fh", None) is not None:
            try:
                self._verbose_fh.flush(); self._verbose_fh.close()
            except Exception:
                pass
        if getattr(self, "_adam_fh", None) is not None:
            try:
                self._adam_fh.flush(); self._adam_fh.close()
            except Exception:
                pass
        if getattr(self, "_epoch_csv_fh", None) is not None:
            try:
                self._epoch_csv_fh.flush(); self._epoch_csv_fh.close()
            except Exception:
                pass
        if self._anomaly_count > 0:
            print(f"[RunLogger] {self._anomaly_count} batches flagged as anomalies.")

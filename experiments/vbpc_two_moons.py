"""Thin two-moons VBPC experiment entry point.

Example commands::

    python -m experiments.vbpc_two_moons --config configs/vbpc_two_moons.yaml
    python -m experiments.vbpc_two_moons --config configs/vbpc_two_moons.yaml --mode single --beta 0.01
    python -m experiments.vbpc_two_moons --betas 0,0.01,0.1

The YAML file selects a VBPC preset, optional config overrides, the run mode
(``single`` or ``sweep``), the ``beta`` list of the sweep and the two-moons
``data`` block (``n_train``/``n_test``/``noise``).
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional, Tuple

from bpc.config import TWO_MOONS_NOISE
from bpc.data import generate_two_moons
from bpc.utils.config_io import load_config_with_presets
from vbpc.config import VBPC_BETA_SWEEP, make_vbpc_presets
from vbpc.training.trainer import (
    default_vbpc_two_moons_config,
    run_vbpc_beta_sweep,
    train_vbpc_two_moons_dataset,
)


def parse_betas(values: Optional[str]) -> Optional[Tuple[float, ...]]:
    if not values:
        return None
    return tuple(float(v) for v in values.split(",") if v.strip() != "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VBPC two-moons experiment.")
    parser.add_argument("--config", default="configs/vbpc_two_moons.yaml", help="Path to the YAML experiment config.")
    parser.add_argument("--preset", default=None, help="Optional VBPC preset override.")
    parser.add_argument("--mode", default=None, choices=["single", "sweep"], help="Override the YAML mode.")
    parser.add_argument("--beta", type=float, default=None, help="Beta used in single mode.")
    parser.add_argument("--betas", default=None, help="Comma separated beta list used in sweep mode.")
    args = parser.parse_args()

    cfg, lcfg, raw = load_config_with_presets(
        args.config, default_vbpc_two_moons_config(), make_vbpc_presets(), args.preset
    )
    if args.beta is not None:
        from dataclasses import replace

        cfg = replace(cfg, beta=float(args.beta))

    mode = args.mode or str(raw.get("mode", "single"))
    data_cfg = raw.get("data", {})
    n_train = int(data_cfg.get("n_train", 1000))
    n_test = int(data_cfg.get("n_test", 300))
    noise = float(data_cfg.get("noise", TWO_MOONS_NOISE))
    data = generate_two_moons(n_train=n_train, n_test=n_test, noise=noise, seed=cfg.seed)
    layer_dims = (2, cfg.hidden, 2)

    if mode == "sweep":
        betas: Optional[Tuple[float, ...]] = parse_betas(args.betas) or parse_betas(
            ",".join(str(b) for b in raw.get("betas", [])) or None
        )
        summary = run_vbpc_beta_sweep(
            cfg,
            data,
            layer_dims,
            lcfg,
            betas or VBPC_BETA_SWEEP,
            mode="vbpc_two_moons",
            summary_name="vbpc_two_moons_beta_sweep",
        )
        print(json.dumps({"beta_sweep": summary}, indent=2, default=str))
        return

    _, metrics, _ = train_vbpc_two_moons_dataset(cfg, data, layer_dims, lcfg)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()

"""Thin MNIST VBPC experiment entry point """

from __future__ import annotations

import argparse
import json
from typing import List, Optional, Tuple

from bpc.data import build_mnist_dims, load_mnist
from bpc.utils.config_io import load_config_with_presets
from vbpc.config import VBPC_BETA_SWEEP, make_vbpc_presets
from vbpc.training.trainer import (
    default_vbpc_config,
    run_vbpc_beta_sweep,
    train_vbpc_mnist_dataset,
)


def parse_betas(values: Optional[str]) -> Optional[Tuple[float, ...]]:
    if not values:
        return None
    return tuple(float(v) for v in values.split(",") if v.strip() != "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VBPC MNIST experiment.")
    parser.add_argument("--config", default="configs/vbpc_mnist.yaml", help="Path to the YAML experiment config.")
    parser.add_argument("--preset", default=None, help="Optional VBPC preset override.")
    parser.add_argument("--mode", default=None, choices=["single", "sweep"], help="Override the YAML mode.")
    parser.add_argument("--beta", type=float, default=None, help="Beta used in single mode.")
    parser.add_argument("--betas", default=None, help="Comma separated beta list used in sweep mode.")
    args = parser.parse_args()

    cfg, lcfg, raw = load_config_with_presets(
        args.config, default_vbpc_config(), make_vbpc_presets(), args.preset
    )
    if args.beta is not None:
        from dataclasses import replace

        cfg = replace(cfg, beta=float(args.beta))

    mode = args.mode or str(raw.get("mode", "single"))
    data = load_mnist(cfg)
    layer_dims = build_mnist_dims(cfg)

    if mode == "sweep":
        betas: Optional[Tuple[float, ...]] = parse_betas(args.betas) or parse_betas(
            ",".join(str(b) for b in raw.get("betas", [])) or None
        )
        summary = run_vbpc_beta_sweep(cfg, data, layer_dims, lcfg, betas or VBPC_BETA_SWEEP)
        print(json.dumps({"beta_sweep": summary}, indent=2, default=str))
        return

    _, metrics, _ = train_vbpc_mnist_dataset(cfg, data, layer_dims, lcfg)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()

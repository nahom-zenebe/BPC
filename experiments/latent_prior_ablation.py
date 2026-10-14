"""Latent-prior ablation study entry point.

Compares three latent-KL conditions across multiple seeds on MNIST:

  no_latent_kl   -- no KL on latent states (original VBPC behaviour)
  standard_normal -- KL[q(z_l) || N(0, tau^2 I)]
  ff_centered     -- KL[q(z_l) || N(h_l^ff(x), tau^2 I)]

The KL coefficient and prior variance are matched between the two prior
variants; all other training settings are fixed across conditions and seeds.
Reports accuracy, NLL and entropy per condition with mean ± std over seeds.

Example commands::

    python -m experiments.latent_prior_ablation
    python -m experiments.latent_prior_ablation --config configs/latent_prior_ablation.yaml
    python -m experiments.latent_prior_ablation --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Tuple

from bpc.data import build_mnist_dims, load_mnist
from bpc.utils.config_io import load_config_with_presets
from vbpc.config import make_vbpc_presets
from vbpc.training.trainer import default_vbpc_config, run_latent_prior_ablation


def _parse_seeds(value: Optional[str]) -> Optional[Tuple[int, ...]]:
    if not value:
        return None
    return tuple(int(s) for s in value.split(",") if s.strip() != "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the latent-prior ablation study.")
    parser.add_argument(
        "--config",
        default="configs/latent_prior_ablation.yaml",
        help="Path to the YAML experiment config.",
    )
    parser.add_argument("--preset", default=None, help="Optional VBPC preset override.")
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed list, e.g. '0,1,2,3,4'. Overrides the YAML seeds.",
    )
    args = parser.parse_args()

    cfg, lcfg, raw = load_config_with_presets(
        args.config, default_vbpc_config(), make_vbpc_presets(), args.preset
    )

    seeds: Tuple[int, ...]
    if args.seeds is not None:
        seeds = _parse_seeds(args.seeds) or (0, 1, 2)
    else:
        yaml_seeds = raw.get("seeds", [])
        seeds = tuple(int(s) for s in yaml_seeds) if yaml_seeds else (0, 1, 2)

    data = load_mnist(cfg)
    layer_dims = build_mnist_dims(cfg)

    summary = run_latent_prior_ablation(
        cfg=cfg,
        data=data,
        layer_dims=layer_dims,
        lcfg=lcfg,
        seeds=seeds,
        summary_name="latent_prior_ablation",
    )
    print(json.dumps({"ablation": summary}, indent=2, default=str))


if __name__ == "__main__":
    main()

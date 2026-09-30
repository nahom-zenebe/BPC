"""Factorized Gaussian variational posterior over VBPC weights."""

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

__all__ = [
    "VBPCWeightParams",
    "gaussian_kl_elementwise",
    "init_vbpc_weight_params",
    "mean_weight_variance",
    "reparameterize_weight",
    "reparameterize_weight_matrices",
    "sample_weight_epsilons",
    "sample_weight_matrices",
    "weight_diagnostics",
    "weight_kl",
    "weight_shapes",
    "weight_sigma",
]

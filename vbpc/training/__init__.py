"""Training, evaluation and logging for VBPC."""

from vbpc.training.logger import VBPCRunLogger, plot_beta_sweep
from vbpc.training.metrics import (
    accuracy_from_logits,
    collect_logits,
    ece_from_logits,
    evaluate,
    expected_calibration_error,
    memory_usage_mb,
    negative_log_likelihood,
    nll_from_logits,
    softmax,
    vbpc_accuracy,
    weight_variance_metric,
)
from vbpc.training.train_step import (
    VBPCDiagnostics,
    build_step_diagnostics,
    make_vbpc_train_step,
    sample_weights_and_latents,
    vbpc_loss_and_grads,
    vbpc_train_step,
)
from vbpc.training.trainer import (
    default_vbpc_config,
    default_vbpc_two_moons_config,
    run_vbpc_beta_sweep,
    train_vbpc_mnist,
    train_vbpc_mnist_dataset,
    train_vbpc_two_moons_dataset,
)

__all__ = [
    "VBPCDiagnostics",
    "VBPCRunLogger",
    "accuracy_from_logits",
    "build_step_diagnostics",
    "collect_logits",
    "default_vbpc_config",
    "default_vbpc_two_moons_config",
    "ece_from_logits",
    "evaluate",
    "expected_calibration_error",
    "make_vbpc_train_step",
    "memory_usage_mb",
    "negative_log_likelihood",
    "nll_from_logits",
    "plot_beta_sweep",
    "run_vbpc_beta_sweep",
    "sample_weights_and_latents",
    "softmax",
    "train_vbpc_mnist",
    "train_vbpc_mnist_dataset",
    "train_vbpc_two_moons_dataset",
    "vbpc_accuracy",
    "vbpc_loss_and_grads",
    "vbpc_train_step",
    "weight_variance_metric",
]

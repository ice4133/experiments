"""测试集指标、闭环指标和结果绘图。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import DatasetSplit
from .model import MLPController
from .simulation import Rollout
from .trainer import TrainingHistory


@dataclass(frozen=True)
class RegressionMetrics:
    mse: float
    mae: float
    max_absolute_error: float
    r_squared: float


def evaluate_regression(
    model: MLPController,
    test_data: DatasetSplit,
    device: torch.device,
) -> RegressionMetrics:
    model.eval()
    with torch.inference_mode():
        states = test_data.states.to(device)
        targets = test_data.torques.to(device)
        predictions = model(states)
        errors = predictions - targets
        mse = torch.mean(errors.square())
        mae = torch.mean(errors.abs())
        max_error = torch.max(errors.abs())
        target_variance = torch.sum((targets - targets.mean()).square())
        r_squared = 1.0 - torch.sum(errors.square()) / target_variance

    return RegressionMetrics(
        mse=mse.item(),
        mae=mae.item(),
        max_absolute_error=max_error.item(),
        r_squared=r_squared.item(),
    )


def save_training_plot(history: TrainingHistory, path: Path) -> None:
    epochs = range(1, len(history.train_loss) + 1)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.semilogy(epochs, history.train_loss, label="train")
    axis.semilogy(epochs, history.validation_loss, label="validation")
    axis.axvline(history.best_epoch, color="gray", linestyle="--", label="best epoch")
    axis.set(xlabel="Epoch", ylabel="Normalized MSE", title="Training history")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_rollout_plot(pd_rollout: Rollout, mlp_rollout: Rollout, path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    labels = ("Angle (rad)", "Angular velocity (rad/s)", "Torque (N m)")
    pd_values = (pd_rollout.angle, pd_rollout.angular_velocity, pd_rollout.torque)
    mlp_values = (mlp_rollout.angle, mlp_rollout.angular_velocity, mlp_rollout.torque)

    for axis, label, pd_value, mlp_value in zip(
        axes, labels, pd_values, mlp_values, strict=True
    ):
        axis.plot(pd_rollout.time, pd_value, label="PD expert", linewidth=2)
        axis.plot(mlp_rollout.time, mlp_value, "--", label="MLP imitation")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel("Time (s)")
    figure.suptitle("Inverted pendulum closed-loop comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_metrics(metrics: RegressionMetrics, path: Path) -> None:
    path.write_text(
        json.dumps(asdict(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


"""串起数据生成、训练、测试和闭环仿真的完整实验流水线。"""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .controllers import NeuralController, PDController
from .data import (
    compute_normalization,
    create_data_loaders,
    generate_dataset,
    save_dataset,
)
from .evaluation import (
    RegressionMetrics,
    evaluate_regression,
    save_metrics,
    save_rollout_plot,
    save_training_plot,
)
from .model import MLPController
from .simulation import InvertedPendulum
from .trainer import resolve_device, train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(
    config: ExperimentConfig,
    output_directory: Path,
) -> RegressionMetrics:
    """执行第一阶段的全部步骤，并把可复现实验产物保存到磁盘。"""
    output_directory.mkdir(parents=True, exist_ok=True)
    set_seed(config.data.seed)
    device = resolve_device(config.training.device)
    print(f"使用设备：{device}")

    pd_controller = PDController(config.pd)
    splits = generate_dataset(config.data, pd_controller)
    save_dataset(splits, str(output_directory / "dataset.pt"))
    print(
        "数据集："
        f"train={len(splits.train.states)}, "
        f"val={len(splits.validation.states)}, "
        f"test={len(splits.test.states)}"
    )

    normalization = compute_normalization(splits.train)
    model = MLPController(config.model, normalization)
    train_loader, validation_loader = create_data_loaders(
        splits, config.training, config.data.seed
    )
    history = train_model(
        model, train_loader, validation_loader, config.training, device
    )

    metrics = evaluate_regression(model, splits.test, device)
    print(
        f"测试集：MSE={metrics.mse:.6f} | MAE={metrics.mae:.6f} N·m | "
        f"R²={metrics.r_squared:.6f}"
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "best_epoch": history.best_epoch,
        "metrics": asdict(metrics),
    }
    torch.save(checkpoint, output_directory / "mlp_controller.pt")
    save_metrics(metrics, output_directory / "metrics.json")
    save_training_plot(history, output_directory / "training_curve.png")

    pendulum = InvertedPendulum(config.simulation)
    neural_controller = NeuralController(model, device, config.pd.max_torque)
    pd_rollout = pendulum.rollout(pd_controller)
    mlp_rollout = pendulum.rollout(neural_controller)
    save_rollout_plot(pd_rollout, mlp_rollout, output_directory / "rollout.png")

    print(
        f"闭环末端角度：PD={pd_rollout.angle[-1]:.6f} rad | "
        f"MLP={mlp_rollout.angle[-1]:.6f} rad"
    )
    print(f"实验产物已保存到：{output_directory.resolve()}")
    return metrics


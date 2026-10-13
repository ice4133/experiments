"""训练循环：Forward -> Loss -> Backward -> Adam。"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .model import MLPController


@dataclass(frozen=True)
class TrainingHistory:
    train_loss: list[float]
    validation_loss: list[float]
    best_epoch: int


def resolve_device(requested_device: str) -> torch.device:
    """auto 优先 CUDA，也允许命令行强制使用 cpu/cuda。"""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无法使用 CUDA")
    return device


def _run_epoch(
    model: MLPController,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for states, target_torques in loader:
            states = states.to(device)
            target_torques = target_torques.to(device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            predicted_normalized = model.forward_normalized(states)
            target_normalized = model.normalize_torques(target_torques)
            loss = loss_function(predicted_normalized, target_normalized)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = states.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def train_model(
    model: MLPController,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> TrainingHistory:
    """训练并用验证集 early stopping，最终恢复验证损失最小的参数。"""
    model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        validation_loss = _run_epoch(
            model, validation_loader, loss_function, device, optimizer=None
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{config.epochs} | "
                f"train={train_loss:.6f} | val={validation_loss:.6f}"
            )

        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping：验证集连续 {config.early_stopping_patience} 轮未改善")
            break

    model.load_state_dict(best_state)
    return TrainingHistory(train_losses, validation_losses, best_epoch)


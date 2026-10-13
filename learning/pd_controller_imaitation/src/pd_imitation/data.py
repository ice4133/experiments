"""生成、划分并装载 PD 模仿数据。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .config import DataConfig, TrainingConfig
from .controllers import PDController


@dataclass(frozen=True)
class DatasetSplit:
    states: Tensor
    torques: Tensor

    def as_dataset(self) -> TensorDataset:
        return TensorDataset(self.states, self.torques)


@dataclass(frozen=True)
class DatasetSplits:
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit


@dataclass(frozen=True)
class NormalizationStats:
    state_mean: Tensor
    state_std: Tensor
    torque_mean: Tensor
    torque_std: Tensor


def generate_dataset(config: DataConfig, controller: PDController) -> DatasetSplits:
    """均匀采样状态，用 PD 专家打标签，再随机划分数据集。"""
    if config.num_samples < 10:
        raise ValueError("num_samples 至少为 10")
    if config.train_ratio + config.validation_ratio >= 1.0:
        raise ValueError("训练集比例与验证集比例之和必须小于 1")

    generator = torch.Generator().manual_seed(config.seed)
    theta = torch.empty(config.num_samples, 1).uniform_(
        config.theta_min, config.theta_max, generator=generator
    )
    angular_velocity = torch.empty(config.num_samples, 1).uniform_(
        config.angular_velocity_min,
        config.angular_velocity_max,
        generator=generator,
    )
    states = torch.cat((theta, angular_velocity), dim=1)
    torques = controller.batch_torque(states)

    permutation = torch.randperm(config.num_samples, generator=generator)
    states = states[permutation]
    torques = torques[permutation]

    train_end = int(config.num_samples * config.train_ratio)
    validation_end = train_end + int(
        config.num_samples * config.validation_ratio
    )

    return DatasetSplits(
        train=DatasetSplit(states[:train_end], torques[:train_end]),
        validation=DatasetSplit(
            states[train_end:validation_end],
            torques[train_end:validation_end],
        ),
        test=DatasetSplit(states[validation_end:], torques[validation_end:]),
    )


def compute_normalization(train: DatasetSplit) -> NormalizationStats:
    """只使用训练集统计量，防止验证/测试信息泄漏。"""
    epsilon = 1e-6
    return NormalizationStats(
        state_mean=train.states.mean(dim=0),
        state_std=train.states.std(dim=0).clamp_min(epsilon),
        torque_mean=train.torques.mean(dim=0),
        torque_std=train.torques.std(dim=0).clamp_min(epsilon),
    )


def create_data_loaders(
    splits: DatasetSplits,
    config: TrainingConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """训练集打乱；验证集保持固定顺序。"""
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        splits.train.as_dataset(),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        splits.validation.as_dataset(),
        batch_size=config.batch_size,
        shuffle=False,
    )
    return train_loader, validation_loader


def save_dataset(splits: DatasetSplits, path: str) -> None:
    """保存原始物理量，方便直接观察和后续重复实验。"""
    torch.save(
        {
            "train_states": splits.train.states,
            "train_torques": splits.train.torques,
            "validation_states": splits.validation.states,
            "validation_torques": splits.validation.torques,
            "test_states": splits.test.states,
            "test_torques": splits.test.torques,
        },
        path,
    )


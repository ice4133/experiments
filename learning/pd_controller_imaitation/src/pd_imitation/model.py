"""用于模仿 PD 控制器的两层 MLP。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .data import NormalizationStats


class MLPController(nn.Module):
    """输入物理状态，输出物理力矩。

    标准化被封装在模型内部，所以训练和部署时不会忘记使用相同的统计量。
    """

    def __init__(
        self,
        config: ModelConfig,
        normalization: NormalizationStats,
    ) -> None:
        super().__init__()
        hidden_1, hidden_2 = config.hidden_dims
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, config.output_dim),
        )

        # buffer 会随模型迁移到 CPU/GPU，也会进入 state_dict，但不会被优化器更新。
        self.register_buffer("state_mean", normalization.state_mean.clone())
        self.register_buffer("state_std", normalization.state_std.clone())
        self.register_buffer("torque_mean", normalization.torque_mean.clone())
        self.register_buffer("torque_std", normalization.torque_std.clone())

    def forward_normalized(self, states: Tensor) -> Tensor:
        """返回标准化空间中的网络输出，训练计算 loss 时使用。"""
        normalized_states = (states - self.state_mean) / self.state_std
        return self.network(normalized_states)

    def normalize_torques(self, torques: Tensor) -> Tensor:
        return (torques - self.torque_mean) / self.torque_std

    def forward(self, states: Tensor) -> Tensor:
        """返回 N·m 单位的力矩，部署和评估时使用。"""
        normalized_torque = self.forward_normalized(states)
        return normalized_torque * self.torque_std + self.torque_mean


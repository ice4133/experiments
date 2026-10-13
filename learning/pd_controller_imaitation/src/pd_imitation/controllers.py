"""控制器定义：生成标签的 PD 专家，以及部署训练后 MLP 的适配器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .config import PDConfig


@dataclass
class PDController:
    """经典 PD 控制器。

    tau = Kp * (theta_target - theta) - Kd * theta_dot
    """

    config: PDConfig

    def batch_torque(self, states: Tensor) -> Tensor:
        """根据一批状态生成标签，输入形状为 [batch, 2]。"""
        theta = states[:, 0:1]
        angular_velocity = states[:, 1:2]
        torque = (
            self.config.kp * (self.config.target_angle - theta)
            - self.config.kd * angular_velocity
        )
        return torch.clamp(
            torque,
            min=-self.config.max_torque,
            max=self.config.max_torque,
        )

    def __call__(self, state: np.ndarray) -> float:
        """仿真时根据单个 NumPy 状态计算力矩。"""
        theta, angular_velocity = state
        torque = (
            self.config.kp * (self.config.target_angle - theta)
            - self.config.kd * angular_velocity
        )
        return float(np.clip(torque, -self.config.max_torque, self.config.max_torque))


class NeuralController:
    """把 PyTorch 模型包装成仿真器可调用的控制器。"""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        max_torque: float,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.max_torque = max_torque

    def __call__(self, state: np.ndarray) -> float:
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.inference_mode():
            torque = self.model(state_tensor).squeeze().item()
        return float(np.clip(torque, -self.max_torque, self.max_torque))


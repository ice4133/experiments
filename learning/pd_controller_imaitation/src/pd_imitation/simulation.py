"""简化倒立摆动力学与闭环仿真。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig


@dataclass(frozen=True)
class Rollout:
    time: np.ndarray
    angle: np.ndarray
    angular_velocity: np.ndarray
    torque: np.ndarray


class InvertedPendulum:
    """单杆倒立摆；theta=0 表示竖直向上，是开环不稳定平衡点。"""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def dynamics(self, state: np.ndarray, torque: float) -> np.ndarray:
        theta, angular_velocity = state
        moment_of_inertia = self.config.mass * self.config.length**2
        angular_acceleration = (
            self.config.gravity / self.config.length * np.sin(theta)
            + torque / moment_of_inertia
            - self.config.damping / moment_of_inertia * angular_velocity
        )
        return np.array([angular_velocity, angular_acceleration], dtype=np.float64)

    def rk4_step(self, state: np.ndarray, torque: float) -> np.ndarray:
        """四阶 Runge-Kutta 积分，比简单欧拉法更稳定。"""
        dt = self.config.time_step
        k1 = self.dynamics(state, torque)
        k2 = self.dynamics(state + 0.5 * dt * k1, torque)
        k3 = self.dynamics(state + 0.5 * dt * k2, torque)
        k4 = self.dynamics(state + dt * k3, torque)
        return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def rollout(self, controller: Callable[[np.ndarray], float]) -> Rollout:
        steps = int(self.config.duration / self.config.time_step) + 1
        time = np.arange(steps, dtype=np.float64) * self.config.time_step
        states = np.empty((steps, 2), dtype=np.float64)
        torques = np.empty(steps, dtype=np.float64)
        states[0] = (
            self.config.initial_angle,
            self.config.initial_angular_velocity,
        )

        for index in range(steps - 1):
            torques[index] = controller(states[index])
            states[index + 1] = self.rk4_step(states[index], torques[index])
        torques[-1] = controller(states[-1])

        return Rollout(
            time=time,
            angle=states[:, 0],
            angular_velocity=states[:, 1],
            torque=torques,
        )


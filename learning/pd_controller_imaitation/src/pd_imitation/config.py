"""实验的全部超参数。

把参数集中在一个文件里，避免在数据、训练和仿真代码中散落“魔法数字”。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PDConfig:
    """专家 PD 控制器参数。角度单位是 rad，力矩单位是 N·m。"""

    kp: float = 20.0
    kd: float = 5.0
    target_angle: float = 0.0
    max_torque: float = 20.0


@dataclass(frozen=True)
class DataConfig:
    """监督学习数据的采样范围与划分比例。"""

    num_samples: int = 20_000
    theta_min: float = -1.0
    theta_max: float = 1.0
    angular_velocity_min: float = -5.0
    angular_velocity_max: float = 5.0
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """MLP: [theta, theta_dot] -> 64 -> 64 -> torque。"""

    input_dim: int = 2
    hidden_dims: tuple[int, int] = (64, 64)
    output_dim: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    """训练超参数。device=auto 会优先使用 CUDA。"""

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 15
    device: str = "auto"


@dataclass(frozen=True)
class SimulationConfig:
    """以竖直向上为 theta=0 的简化倒立摆参数。"""

    mass: float = 1.0
    length: float = 1.0
    gravity: float = 9.81
    damping: float = 0.1
    time_step: float = 0.01
    duration: float = 5.0
    initial_angle: float = 0.7
    initial_angular_velocity: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    """一次完整实验所需的配置。"""

    pd: PDConfig = field(default_factory=PDConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


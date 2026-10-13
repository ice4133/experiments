import numpy as np
import torch

from pd_imitation.config import DataConfig, ModelConfig, PDConfig, SimulationConfig
from pd_imitation.controllers import PDController
from pd_imitation.data import compute_normalization, generate_dataset
from pd_imitation.model import MLPController
from pd_imitation.simulation import InvertedPendulum


def test_pd_formula_and_torque_limit() -> None:
    controller = PDController(PDConfig(kp=20.0, kd=5.0, max_torque=20.0))
    states = torch.tensor([[0.1, -0.2], [2.0, 3.0]])
    torques = controller.batch_torque(states)

    assert torch.allclose(torques[0], torch.tensor([-1.0]))
    assert torch.allclose(torques[1], torch.tensor([-20.0]))


def test_dataset_shapes_and_split() -> None:
    config = DataConfig(num_samples=100, seed=7)
    splits = generate_dataset(config, PDController(PDConfig()))

    assert splits.train.states.shape == (80, 2)
    assert splits.validation.states.shape == (10, 2)
    assert splits.test.states.shape == (10, 2)
    assert splits.train.torques.shape == (80, 1)


def test_model_input_output_shape() -> None:
    splits = generate_dataset(DataConfig(num_samples=100), PDController(PDConfig()))
    model = MLPController(ModelConfig(), compute_normalization(splits.train))

    assert model(torch.zeros(8, 2)).shape == (8, 1)


def test_pd_controller_stabilizes_pendulum() -> None:
    controller = PDController(PDConfig())
    pendulum = InvertedPendulum(
        SimulationConfig(duration=5.0, initial_angle=0.7)
    )
    rollout = pendulum.rollout(controller)

    assert np.abs(rollout.angle[-1]) < 0.01
    assert np.abs(rollout.angular_velocity[-1]) < 0.01


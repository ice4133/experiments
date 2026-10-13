"""项目入口：在项目根目录执行 `python train.py`。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pd_imitation.config import ExperimentConfig  # noqa: E402
from pd_imitation.pipeline import run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 MLP 模仿倒立摆 PD 控制器")
    parser.add_argument("--epochs", type=int, default=100, help="最大训练轮数")
    parser.add_argument("--samples", type=int, default=20_000, help="采样数量")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="训练设备；auto 优先使用 CUDA",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="模型、数据和图片的输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig()
    config = replace(
        config,
        data=replace(config.data, num_samples=args.samples, seed=args.seed),
        training=replace(
            config.training,
            epochs=args.epochs,
            device=args.device,
        ),
    )
    run_experiment(config, args.output_dir)


if __name__ == "__main__":
    main()


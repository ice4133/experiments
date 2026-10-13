# 用 PyTorch 让 MLP 模仿倒立摆 PD 控制器

这是“深度学习 + 强化学习”路线的第一步：先不使用强化学习，而是做一次完整的**监督学习/行为克隆**实验。PD 控制器充当专家，MLP 学习从状态到控制力矩的映射。

## 项目结构

```text
pd_controller_imaitation/
├── train.py                       # 命令行入口，串起完整实验
├── requirements.txt               # 除 PyTorch 外的依赖
├── src/pd_imitation/
│   ├── config.py                  # 全部超参数
│   ├── controllers.py             # PD 专家和神经网络控制器适配器
│   ├── data.py                    # 采样、标签、数据划分、标准化统计
│   ├── model.py                   # 2 -> 64 -> 64 -> 1 MLP
│   ├── trainer.py                 # Forward、Loss、BP、Adam
│   ├── simulation.py              # 倒立摆动力学和 RK4 仿真
│   ├── evaluation.py              # 测试指标和绘图
│   └── pipeline.py                # 完整数据/训练/仿真流水线
├── tests/test_pipeline.py         # 关键单元测试
└── artifacts/                     # 训练后自动生成的结果
```

## 在 learning 环境运行

```bash
conda activate learning
cd /home/ice/experiments/learning/pd_controller_imaitation
python train.py
```

程序会自动优先使用 CUDA。也可以显式指定：

```bash
python train.py --device cuda --epochs 100 --samples 20000
```

运行测试：

```bash
python -m pytest -q
```

## 数据链路

```text
随机采样物理状态 x=[theta, theta_dot]
                  │
                  ▼
PD 专家生成标签 tau=Kp(theta_target-theta)-Kd*theta_dot
                  │  再限制到 [-20, 20] N·m
                  ▼
随机划分 train / validation / test = 80% / 10% / 10%
                  │
                  ▼
仅用 train 计算均值和标准差 ──► 标准化 x 和 tau
                  │
                  ▼
MLP: [batch,2] -> Linear(2,64) -> ReLU
              -> Linear(64,64) -> ReLU -> Linear(64,1)
                  │
                  ▼
MSE(predicted_tau, expert_tau)
                  │
                  ▼
loss.backward() 计算梯度 -> Adam 更新权重
                  │
                  ▼
test 集评估误差 -> MLP 接入倒立摆 -> 闭环仿真
```

### 每个张量是什么

| 名称 | 形状 | 含义 |
|---|---:|---|
| `states` | `[batch, 2]` | 每行是 `[角度 theta, 角速度 theta_dot]` |
| `target_torques` | `[batch, 1]` | PD 专家给出的正确力矩 |
| `predicted_torques` | `[batch, 1]` | MLP 预测力矩 |
| `loss` | `[]` | 一个标量，预测和标签之间的均方误差 |

标准化统计只从训练集计算，防止测试集信息泄漏。均值和标准差作为模型 `buffer` 保存，因此模型切换 CPU/GPU 或保存 checkpoint 时不会丢失。

## 控制链路和学习链路的区别

训练时的数据链路是：

```text
(theta, theta_dot) -> MLP -> tau_pred -> 与 tau_PD 比较 -> 更新网络
```

部署到倒立摆后的闭环链路是：

```text
当前状态 -> MLP 力矩 -> 倒立摆动力学 -> 下一状态 -> 再交给 MLP
     ▲                                               │
     └──────────────────── feedback ─────────────────┘
```

闭环验证很重要：测试集误差小只说明“单个状态预测得像”，闭环仿真才说明连续使用预测力矩后，误差会不会累积并导致倒立摆失稳。

## 训练产物

运行后 `artifacts/` 中会得到：

- `dataset.pt`：原始 train/validation/test 状态与 PD 标签。
- `mlp_controller.pt`：最佳模型参数、配置、指标。
- `metrics.json`：MSE、MAE、最大误差、R²。
- `training_curve.png`：训练集和验证集损失。
- `rollout.png`：PD 与 MLP 的闭环角度、角速度和力矩对比。

这一步还不是强化学习，因为动作标签来自 PD 专家；后续可以让策略不再抄专家，而是根据环境奖励，用 PPO 自己探索动作。

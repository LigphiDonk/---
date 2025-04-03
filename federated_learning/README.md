# 联邦学习框架

这是一个基于PyTorch的联邦学习框架，支持服务器和客户端模型训练，以及梯度的传输和聚合。

## 特性

- 服务器和客户端都能进行模型训练
- 客户端能将训练后的梯度发送给服务器
- 服务器能对客户端梯度和自身训练结果进行聚合
- 支持通过WebSocket进行通信
- 提供本地模拟功能，方便开发和测试
- **新增：支持多服务器架构和基于PBFT的共识机制**
- **新增：支持随机选择主服务器进行联邦学习**

## 安装

1. 克隆仓库:
```bash
git clone <repository_url>
cd federated_learning
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

## 目录结构

```
federated_learning/
├── server/           # 服务器端代码
├── client/           # 客户端代码
├── common/           # 通用模块
│   ├── pbft.py       # PBFT共识实现
├── logs/             # 日志目录
├── saved_models/     # 保存的模型
├── results/          # 结果和可视化
├── config.py         # 全局配置
├── simulation.py     # 单服务器模拟脚本
└── multi_server_simulation.py  # 多服务器模拟脚本
```

## 运行方式

### 本地单服务器模拟

要在本地模拟单服务器联邦学习过程，无需启动实际的服务器和客户端，可以使用:

```bash
python simulation.py --clients 3 --rounds 5 --epochs 2
```

参数说明:
- `--clients`: 客户端数量 (默认: 3)
- `--rounds`: 联邦学习轮数 (默认: 5)
- `--epochs`: 每轮本地训练的epoch数 (默认: 2)
- `--batch_size`: 批量大小 (默认: 64)
- `--lr`: 学习率 (默认: 0.01)
- `--seed`: 随机种子 (默认: 42)


本地模拟多服务器多客户端的pbft共识选举联邦学习机制
python federated_learning/simple_multi_server_pbft_simulation.py --servers 3 --clients 3 --rounds 5
```

参数说明:
- `--servers`: 服务器数量 (默认: 3)
- `--clients`: 客户端数量 (默认: 3)
- `--rounds`: 联邦学习轮数 (默认: 5)
- `--epochs`: 每轮本地训练的epoch数 (默认: 2)
- `--batch_size`: 批量大小 (默认: 64)
- `--lr`: 学习率 (默认: 0.01)
- `--seed`: 随机种子 (默认: 42)



### 分布式多服务器运行

1. 启动多个服务器:
```bash
python -m server.server_cluster --id 1
python -m server.server_cluster --id 2
python -m server.server_cluster --id 3
```

2. 启动客户端:
```bash
python -m client.client --id 1
```

服务器会通过PBFT共识随机选择一个主服务器协调联邦学习过程。客户端会自动连接到主服务器。

## 配置

可以在 `config.py` 文件中修改全局配置，包括:

- 模型参数
- 训练参数
- 联邦学习参数
- 通信参数
- 路径配置
- **多服务器配置**
- **PBFT共识配置**

## 可视化

训练完成后，会生成以下可视化结果:

- 训练历史图表 (`results/federated_training_history.png`)
- 混淆矩阵 (`confusion_matrix.png`)
- 预测可视化 (`prediction_visualization.png`)

## 示例

本框架基于MNIST数据集演示，但可以适配其他数据集和模型。 
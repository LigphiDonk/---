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
- **新增：支持PVSS梯度掩码保护客户端梯度隐私**
- **新增：支持二次掩码机制，客户端自生成掩码种子并通过Shamir秘密分享分发**
- **新增：支持比较不同掩码方法的效果**
- **新增：支持防投毒功能，检测并过滤恶意客户端的梯度**
- **新增：丰富的可视化工具，包括3D PCA、ROC曲线、层次聚类等**

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
│   ├── pvss.py       # 可公开验证的秘密共享实现
│   ├── masked_gradient.py # 梯度掩码机制实现
│   ├── secondary_mask.py # 二次掩码机制实现
│   ├── poison_detector.py # 梯度投毒检测器实现
│   ├── utils.py      # 工具函数
├── logs/             # 日志目录
├── saved_models/     # 保存的模型
├── results/          # 结果和可视化
├── config.py         # 全局配置
├── simulation.py     # 单服务器模拟脚本
├── simple_multi_server_pbft_simulation.py   # 多服务器PBFT共识模拟脚本
├── compare_mask_effects.py   # 掩码效果比较工具
├── poison_detection_test.py   # 防投毒功能测试脚本
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

### 本地多服务器PBFT共识模拟

在本地模拟多服务器多客户端的PBFT共识选举联邦学习机制:

```bash
python -m federated_learning.simple_multi_server_pbft_simulation --servers 3 --clients 3 --rounds 5
```

参数说明:
- `--servers`: 服务器数量 (默认: 3)
- `--clients`: 客户端数量 (默认: 3)
- `--rounds`: 联邦学习轮数 (默认: 5)
- `--epochs`: 每轮本地训练的epoch数 (默认: 2)
- `--batch_size`: 批量大小 (默认: 64)
- `--lr`: 学习率 (默认: 0.01)
- `--seed`: 随机种子 (默认: 42)
- `--use_masked`: 使用PVSS梯度掩码保护隐私 (默认: 不启用)
- `--use_secondary_mask`: 使用二次掩码 (默认: 不启用)
- `--secondary_mask_clients`: 二次掩码需要的客户端数量 (默认: 4)
- `--secondary_mask_threshold`: 重建二次掩码所需的最小服务器数量 (默认: 服务器总数的2/3)

### 启用PVSS梯度掩码保护

要在PBFT共识基础上添加PVSS梯度掩码保护，只需添加`--use_masked`参数:

```bash
python simple_multi_server_pbft_simulation --servers 3 --clients 3 --rounds 5 --use_masked
```

PVSS梯度掩码机制会在客户端对梯度添加随机掩码，不同服务器接收到的掩码互相抵消，保证原始梯度在聚合时能被恢复，同时保护了客户端的隐私。

### 启用二次掩码机制

要在PVSS掩码基础上添加二次掩码机制，需要同时添加`--use_masked`和`--use_secondary_mask`参数:

```bash

```python simple_multi_server_pbft_simulation.py --servers 3 --clients 3 --rounds 5 --use_masked --use_secondary_mask --secondary_mask_threshold 2

可以使用`--secondary_mask_threshold`参数指定重建掩码所需的最小服务器数量，默认为服务器总数的2/3。

我们还提供了一个便捷脚本来运行带有新二次掩码功能的模拟:

```bash
python examples/run_simulation_with_new_secondary_mask.py --servers 3 --clients 3 --rounds 5 --threshold 2
```

### 比较不同掩码方法

使用`compare_mask_effects.py`脚本可以运行三种不同的联邦学习模式并比较它们的性能:

```bash
# 运行所有三种模式并比较
python federated_learning/compare_mask_effects.py --clients 4 --rounds 5 --mode all

# 只运行特定模式
python federated_learning/compare_mask_effects.py --clients 4 --rounds 5 --mode no_mask
python federated_learning/compare_mask_effects.py --clients 4 --rounds 5 --mode pvss
python federated_learning/compare_mask_effects.py --clients 4 --rounds 5 --mode dual
```

参数说明:
- `--clients`: 客户端数量 (默认: 4)
- `--servers`: 服务器数量 (默认: 3)
- `--rounds`: 训练轮数 (默认: 5)
- `--mode`: 运行模式，可选值: 'all', 'no_mask', 'pvss', 'dual' (默认: 'all')

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
- 多服务器配置
- PBFT共识配置

## 可视化

训练完成后，会生成以下可视化结果:

- 训练历史图表 (`results/federated_training_history.png`)
- 混淆矩阵 (`results/aggregated_confusion_matrix.png`)
- 预测可视化 (`results/aggregated_prediction_visualization.png`)
- 使用PVSS梯度掩码时生成 (`results/pvss_masked_federated_training_history.png`)
- 使用二次掩码时生成 (`results/dual_masked_federated_training_history.png`)

使用`compare_mask_effects.py`脚本还会生成以下比较图表:

- 不同掩码方法的准确率和损失比较 (`results/mask_methods_comparison.png`)
- 不同掩码方法的最终准确率比较 (`results/mask_methods_final_accuracy.png`)

使用`poison_detection_test.py`脚本会生成以下防投毒相关的可视化图表:

- 余弦相似度矩阵热力图 (`results/[poison_type]_scale[value]/cosine_similarity_matrix.png`)
- 投毒检测结果可视化 (`results/[poison_type]_scale[value]/poison_detection_results.png`)
- 梯度的3D PCA可视化 (`results/[poison_type]_scale[value]/gradients_pca_3d.png`)
- 正常与投毒梯度的分布对比 (`results/[poison_type]_scale[value]/gradients_distribution.png`)
- ROC和PR曲线 (`results/[poison_type]_scale[value]/roc_pr_curves.png`)
- 层次聚类分析 (`results/[poison_type]_scale[value]/hierarchical_clustering.png`)

## 掩码机制原理

### PVSS梯度掩码原理

PVSS (Publicly Verifiable Secret Sharing) 梯度掩码机制的主要流程:

1. 客户端训练本地模型，计算梯度
2. 为每个服务器生成随机掩码，一半服务器使用+1系数，一半使用-1系数
3. 将加掩码的梯度分别发送给各服务器
4. 主服务器收集所有服务器接收到的掩码梯度
5. 由于掩码设计为相互抵消，聚合时掩码被消除，恢复原始梯度
6. 主服务器使用恢复的原始梯度更新全局模型

这种机制保证了客户端梯度在传输过程中的隐私性，同时不影响最终的模型聚合效果。

### 二次掩码原理

新的二次掩码机制在PVSS掩码之后添加了一层额外的保护，其工作原理如下:

1. 在PVSS掩码应用后，每个客户端生成一个随机掩码种子
2. 客户端使用Shamir秘密分享将掩码种子分享给多个服务器
3. 客户端使用掩码种子生成随机掩码并应用到梯度
4. 服务器收集到足够数量的份额后，可以重建掩码种子
5. 使用重建的掩码种子生成相同的掩码，并从梯度中去除
6. 只有当服务器收集到足够数量的份额时，才能去除掩码

这种新的二次掩码机制提供了以下优势:

1. **客户端自主性增强**: 客户端自己生成掩码种子，不再依赖于预定义的掩码生成方式
2. **安全性提升**: 通过Shamir秘密分享机制，确保只有当足够数量的服务器协作时，才能重建掩码种子
3. **灵活的阈值设置**: 可以设置重建掩码所需的最小服务器数量，平衡安全性和容错性
4. **容错性增强**: 即使部分服务器不可用，只要有足够数量的服务器协作，仍然可以重建掩码种子
5. **与原有PVSS掩码兼容**: 可以无缝集成到现有系统中

## 示例

### 二次掩码示例

我们提供了一个简单的示例脚本，展示如何使用新的二次掩码机制:

```bash
python examples/secondary_mask_example.py
```

这个示例展示了如下功能:
1. 客户端生成掩码种子并通过Shamir秘密分享分发
2. 应用掩码到梯度
3. 收集份额并重建掩码种子
4. 去除掩码并恢复原始梯度

### 运行带有新二次掩码功能的模拟

```bash
python examples/run_simulation_with_new_secondary_mask.py --servers 3 --clients 3 --rounds 5 --threshold 2
```

参数说明:
- `--servers`: 服务器数量 (默认: 3)
- `--clients`: 客户端数量 (默认: 3)
- `--rounds`: 训练轮数 (默认: 5)
- `--threshold`: 重建掩码所需的最小服务器数量 (默认: 服务器总数的2/3)

### 防投毒功能

联邦学习中的投毒攻击是指恶意客户端提交的经过特意操纵的梯度，目的是破坏全局模型的性能。我们的防投毒功能可以检测并过滤这些恶意梯度。

### 防投毒原理

我们的防投毒机制基于以下原理：

1. **余弦相似度分析**：正常客户端的梯度应该具有相似的方向，而投毒梯度通常与其他客户端的梯度差异较大
2. **统计异常检测**：将每个客户端的梯度与平均梯度进行比较，识别出异常值
3. **多维分析**：使用PCA降维和聚类分析等技术识别投毒模式

### 投毒类型

我们的测试脚本支持以下几种投毒类型：

1. **随机投毒 (random)**：向梯度中添加随机噪声
2. **反转投毒 (invert)**：将梯度方向反转
3. **常数投毒 (constant)**：将梯度替换为固定值

### 运行防投毒测试

要运行防投毒功能测试，可以使用以下命令：

```bash
python -m federated_learning.poison_detection_test
```

这将生成多种可视化图表，包括余弦相似度矩阵、3D PCA可视化、ROC曲线等，帮助您直观地理解投毒梯度的特征和检测效果。

### 在联邦学习中集成防投毒功能

要在实际的联邦学习过程中使用防投毒功能，可以在服务器聚合梯度前添加以下代码：

```python
from federated_learning.common.poison_detector import GradientPoisonDetector

# 创建投毒检测器
detector = GradientPoisonDetector()

# 检测并过滤投毒梯度
filtered_gradients = detector.detect_and_filter_poisoned_gradients(
    all_client_gradients,
    similarity_threshold=0.85  # 可以根据需要调整阈值
)

# 使用过滤后的梯度进行聚合
```

## 数据集

本框架基于MNIST数据集演示，但可以适配其他数据集和模型。

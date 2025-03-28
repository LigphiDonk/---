# PBFT分布式联邦学习系统

本系统是在原有联邦学习框架的基础上，引入实用拜占庭容错(PBFT)共识算法和基于声誉的选举机制，将原有中心化的联邦学习系统转变为去中心化的分布式系统。

## 系统特点

1. **PBFT共识**：采用PBFT共识机制确保在存在恶意节点的环境中系统依然能够正常运行
2. **基于声誉的选举机制**：通过节点历史表现动态选择验证节点，保证系统安全性
3. **P2P网络通信**：节点间直接通信，消除单点故障风险
4. **兼容原有系统**：保留原有中心化模式的同时支持分布式模式

## 系统架构

- **网络层**：基于TCP的P2P通信网络
- **共识层**：PBFT共识算法实现
- **选举层**：基于声誉的验证节点选举机制
- **联邦学习层**：模型训练、聚合和分发

## 安装和依赖

系统依赖与原联邦学习系统相同，主要包括：

```
pytorch
torchvision
numpy
tqdm
```

## 使用方法

### 1. 启动引导节点

首先需要启动一个引导节点来初始化PBFT网络和分发初始模型：

```bash
python pbft_startup.py --model simple-cnn --dataset mnist --n_parties 5 --comm_round 10
```

参数说明：
- `--model`: 模型类型 (默认: mlp)
- `--dataset`: 数据集 (默认: mnist)
- `--n_parties`: 参与节点数量 (默认: 2)
- `--comm_round`: 通信轮次 (默认: 5)
- `--bootstrap_id`: 引导节点ID (默认: 0)
- `--f`: 容错数量，系统最多容忍f个恶意节点 (默认: 1)

### 2. 启动客户端节点

引导节点启动后，可以启动其他客户端节点加入网络：

```bash
python experiments_client.py --client_id 1 --comm_round 10 --use_pbft --bootstrap=0,127.0.0.1,12000
```

参数说明：
- `--client_id`: 客户端节点ID (必需)
- `--comm_round`: 训练轮次 (必需)
- `--use_pbft`: 启用PBFT分布式模式
- `--bootstrap`: 引导节点信息，格式为"ID,IP,端口"
- `--host`: 本节点IP地址 (默认: 127.0.0.1)
- `--port`: 本节点通信端口 (默认: 12000+client_id)
- `--mask_seed`: 掩码种子值 (默认: 0，不使用掩码)

### 3. 运行示例

启动一个包含5个节点的PBFT联邦学习网络示例：

1. 启动引导节点（节点0）
```bash
python pbft_startup.py --model simple-cnn --dataset mnist --n_parties 5 --comm_round 10
```

2. 启动4个客户端节点
```bash
# 在不同的终端或服务器上运行
python experiments_client.py --client_id 1 --comm_round 10 --use_pbft --bootstrap=0,127.0.0.1,12000
python experiments_client.py --client_id 2 --comm_round 10 --use_pbft --bootstrap=0,127.0.0.1,12000
python experiments_client.py --client_id 3 --comm_round 10 --use_pbft --bootstrap=0,127.0.0.1,12000
python experiments_client.py --client_id 4 --comm_round 10 --use_pbft --bootstrap=0,127.0.0.1,12000
```

## 系统流程

1. 引导节点初始化网络和模型
2. 客户端节点连接到网络
3. 系统进行验证节点选举，选出一组验证节点并确定主节点
4. 每轮训练流程：
   - 各节点进行本地训练
   - 各节点向网络提交模型更新提案
   - 主节点收集提案并发起聚合请求
   - 验证节点通过PBFT共识确认聚合结果
   - 全局模型更新并分发到所有节点
5. 定期进行验证节点重新选举

## 系统监控

系统运行过程中会输出详细的日志信息，包括：
- 节点状态变化
- 选举过程和结果
- 共识流程和决策
- 模型训练和聚合信息

## 扩展和定制

系统设计模块化，可以方便地进行扩展和定制：
- `network.py`: 修改网络通信机制
- `consensus.py`: 调整PBFT共识参数或替换为其他共识算法
- `election.py`: 定制选举机制和信任评估方法
- `pbft_client.py`: 修改联邦学习客户端行为

## 与中心化模式的比较

PBFT分布式模式与原中心化模式对比：

| 特性 | PBFT分布式模式 | 中心化模式 |
|-----|--------------|----------|
| 容错性 | 可容忍f个恶意节点 | 单点故障 |
| 去中心化 | 完全去中心化 | 中心化服务器 |
| 通信开销 | 较高 | 较低 |
| 聚合安全性 | 高 | 中 |
| 部署复杂度 | 较高 | 低 |

## 常见问题

**Q: 如何确定合适的容错参数f？**

A: 容错参数f取决于网络规模和安全需求。通常，对于n个节点的系统，设置f=⌊(n-1)/3⌋，即可容忍最多三分之一的节点出现故障或恶意行为。

**Q: 系统支持的最小节点数量是多少？**

A: PBFT共识需要至少3f+1个节点才能容忍f个故障节点。因此，最小节点数取决于容错需求，通常至少需要4个节点(f=1)。

**Q: 如何处理网络分区问题？**

A: 当前版本不处理网络分区问题。在严重的网络分区情况下，分区的各部分可能无法达成共识。建议在网络稳定的环境中运行。

## 技术文档

- [network.py](network.py): P2P网络通信实现
- [consensus.py](consensus.py): PBFT共识算法实现
- [election.py](election.py): 基于声誉的选举机制
- [pbft_client.py](pbft_client.py): PBFT联邦学习客户端
- [pbft_startup.py](pbft_startup.py): 引导节点启动脚本 
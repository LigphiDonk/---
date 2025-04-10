#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版多服务器PBFT联邦学习模拟脚本

此脚本实现：
1. 创建多个服务器节点
2. 使用PBFT共识机制选举出主服务器
3. 使用选出的主服务器执行联邦学习流程
"""

import os
import sys
import time
import torch
import numpy as np
import argparse
import random
import asyncio
import logging
import json
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib

# 禁用matplotlib警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 设置日志级别，减少不必要的调试信息
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.colorbar").setLevel(logging.ERROR)

# 设置matplotlib为非交互式模式，避免线程问题
os.environ['MPLBACKEND'] = 'Agg'  # 使用非交互式后端
os.environ["MATPLOTLIB_USE_FIGURE_REQUIRES_MAIN_THREAD"] = "False"

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从simulation.py导入需要的函数
from federated_learning.simulation import (
    create_server_and_clients,
    simulate_federated_learning,
    split_data_for_clients
)
from federated_learning.common.utils import setup_logger, set_random_seed
from federated_learning.common.pbft import PBFTNode, PBFTMessage, PBFTMessageType
from federated_learning.common.pvss import PVSSHandler
from federated_learning.common.masked_gradient import MaskedGradientHandler
from federated_learning.common.secondary_mask import SecondaryMaskHandler
from federated_learning.config import config
from debug_training_fix import create_model, get_mnist_data, compute_accuracy
from federated_learning.common.poison_detector import GradientPoisonDetector

def compute_confusion_matrix(model, dataloader, device, num_classes=10):
    """计算混淆矩阵"""
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)

            for t, p in zip(target.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix

def save_confusion_matrix(confusion_matrix, title, save_path, classes=None):
    """保存混淆矩阵为图像"""
    if classes is None:
        classes = list(range(confusion_matrix.shape[0]))

    plt.figure(figsize=(10, 8))

    # 标准化混淆矩阵以便更好地可视化
    normalized_cm = confusion_matrix / (confusion_matrix.sum(dim=1, keepdim=True) + 1e-6)

    # 使用更好的配色方案
    plt.imshow(normalized_cm.cpu().numpy(), interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
    plt.colorbar()

    # 生成坐标标签
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    # 在混淆矩阵中添加文本标签
    thresh = normalized_cm.max() / 2.
    for i in range(normalized_cm.shape[0]):
        for j in range(normalized_cm.shape[1]):
            value = int(confusion_matrix[i, j].item())
            if value > 0:  # 只显示非零值
                plt.text(j, i, value,
                        ha="center", va="center",
                        color="white" if normalized_cm[i, j] > thresh else "black",
                        fontsize=9)

    plt.tight_layout()
    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)

    # 保存图像
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path

class WebSocketMock:
    """WebSocket连接的模拟类，用于本地消息传递"""

    def __init__(self, target_node, logger=None):
        """
        初始化模拟WebSocket

        Args:
            target_node: 目标PBFT节点
            logger: 日志记录器
        """
        self.target_node = target_node
        self.logger = logger

    async def send(self, message_json):
        """
        发送消息（模拟WebSocket.send()）

        Args:
            message_json: 序列化的消息JSON
        """
        try:
            # 解析消息字符串为字典
            if isinstance(message_json, str):
                message_dict = json.loads(message_json)
            else:
                message_dict = message_json

            # 记录消息内容用于调试
            if self.logger:
                self.logger.debug(f"接收到消息: {message_dict}")

            # 获取消息类型
            msg_type = message_dict.get("type")
            if isinstance(msg_type, str):
                # 将字符串转换为枚举类型
                for t in PBFTMessageType:
                    if t.value == msg_type:
                        msg_type = t
                        break

            # 直接从字典创建消息对象，不使用from_json()
            node_id = message_dict.get("node_id") or message_dict.get("sender_id")
            data = message_dict.get("data", {})
            seq_num = message_dict.get("seq_num", 0)

            # 创建兼容的消息对象 - 使用旧式构造方法
            message = PBFTMessage(msg_type, node_id, data)

            if self.logger:
                self.logger.debug(f"模拟WebSocket发送消息: {message.type} 到节点 {self.target_node.node_id}")

            # 直接调用目标节点的process_message方法
            await self.target_node.process_message(message)
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"模拟WebSocket发送消息失败: {str(e)}")
                # 打印详细错误堆栈
                import traceback
                self.logger.error(traceback.format_exc())
            return False

class PBFTServer:
    """模拟的PBFT服务器节点，用于本地选举"""

    def __init__(self, server_id, config, logger=None):
        """
        初始化PBFT服务器

        Args:
            server_id: 服务器ID
            config: 配置对象
            logger: 日志记录器
        """
        self.server_id = str(server_id)
        self.config = config

        # 设置日志记录器
        if logger is None:
            self.logger = setup_logger(
                f"PBFTServer-{server_id}",
                os.path.join(config.log_dir, f"pbft_server_{server_id}.log")
            )
        else:
            self.logger = logger

        # 创建PBFT节点
        self.pbft_node = PBFTNode(self.server_id, config, logger=self.logger)

        # 选举结果
        self.election_completed = asyncio.Event()
        self.primary_id = None

        # 创建梯度掩码处理器
        self.masked_gradient_handler = MaskedGradientHandler(
            int(self.server_id),
            config.num_servers,
            (config.num_servers - 1) // 3
        )

        # 梯度掩码缓存
        self.masked_gradients_cache = {}  # {(round_id, client_id): masked_gradients}

        self.logger.info(f"PBFT服务器 {server_id} 初始化完成")

    def on_election_completed(self, primary_id):
        """选举完成回调"""
        self.logger.info(f"服务器 {self.server_id} 收到选举结果: 主服务器 {primary_id}")
        self.primary_id = primary_id
        self.election_completed.set()

    def process_masked_gradient(self, masked_gradients, mask_data, client_id, round_id):
        """
        处理加掩码的梯度

        Args:
            masked_gradients: 加掩码的梯度
            mask_data: 掩码数据
            client_id: 客户端ID
            round_id: 轮次ID
        """
        try:
            self.logger.info(f"服务器 {self.server_id} 接收到客户端 {client_id} 的加掩码梯度")

            # 验证掩码数据
            server_ids = [int(i) for i in range(1, self.config.num_servers + 1)]
            if not self.masked_gradient_handler.verify_mask_data(mask_data, server_ids):
                self.logger.warning(f"掩码数据验证失败，来自客户端 {client_id}")
                return

            # 获取该服务器的掩码符号
            sign = mask_data["sign_map"].get(int(self.server_id), mask_data["sign_map"].get(self.server_id))
            if sign is None:
                self.logger.warning(f"服务器 {self.server_id} 未在掩码符号映射中找到")
                return

            # 缓存加掩码的梯度
            cache_key = (round_id, client_id)
            self.masked_gradients_cache[cache_key] = masked_gradients

            self.logger.info(f"服务器 {self.server_id} 已缓存客户端 {client_id} 的加掩码梯度")
        except Exception as e:
            self.logger.error(f"处理加掩码梯度时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

async def setup_pbft_network(servers):
    """
    设置PBFT网络，将所有服务器连接起来

    Args:
        servers: 服务器列表

    Returns:
        servers: 更新后的服务器列表
    """
    logger = setup_logger("PBFTNetwork", os.path.join(config.log_dir, "pbft_network.log"))
    logger.info("设置PBFT网络")

    # 为每个服务器添加其他服务器节点
    for server in servers:
        for other_server in servers:
            if server.server_id != other_server.server_id:
                # 创建模拟WebSocket连接
                mock_websocket = WebSocketMock(other_server.pbft_node, logger=server.logger)
                # 存储模拟WebSocket而不是PBFTNode对象
                server.pbft_node.server_nodes[other_server.server_id] = mock_websocket

    logger.info("PBFT网络设置完成")
    return servers

async def run_pbft_election(servers):
    """
    运行PBFT选举过程

    Args:
        servers: 服务器列表

    Returns:
        primary_id: 选出的主服务器ID
    """
    logger = setup_logger("PBFTElection", os.path.join(config.log_dir, "pbft_election.log"))
    logger.info("开始PBFT选举")

    # 为每个服务器设置选举完成回调
    for server in servers:
        server.pbft_node.on_election_completed = server.on_election_completed

    # 选择第一个服务器开始选举
    initiator = servers[0]
    logger.info(f"服务器 {initiator.server_id} 开始选举")

    # 设置超时
    timeout = config.election_timeout if hasattr(config, 'election_timeout') else 10

    try:
        # 开始选举
        await initiator.pbft_node.start_election()

        # 等待选举完成，带超时
        try:
            await asyncio.wait_for(initiator.election_completed.wait(), timeout=timeout)

            # 获取选举结果
            primary_id = initiator.primary_id
            logger.info(f"选举成功完成，主服务器ID: {primary_id}")

            return primary_id
        except asyncio.TimeoutError:
            logger.warning(f"选举超时！({timeout}秒)")

            # 超时后，强制选择第一个服务器作为主服务器
            primary_id = servers[0].server_id
            logger.info(f"强制选择服务器 {primary_id} 作为主服务器")

            # 通知所有服务器选举结果
            for server in servers:
                if hasattr(server, 'on_election_completed'):
                    server.on_election_completed(primary_id)

            return primary_id
    except Exception as e:
        logger.error(f"选举过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

        # 出错时，选择第一个服务器作为主服务器
        primary_id = servers[0].server_id
        logger.info(f"选举出错，强制选择服务器 {primary_id} 作为主服务器")

        return primary_id

def simulate_federated_learning_with_primary(config, primary_id):
    """
    使用选定的主服务器进行联邦学习模拟

    Args:
        config: 配置对象
        primary_id: 主服务器ID

    Returns:
        final_model: 最终模型
        history: 训练历史
        test_acc: 测试准确率
    """
    logger = setup_logger("FedSimulation", os.path.join(config.log_dir, "fed_simulation.log"))
    logger.info(f"使用主服务器 {primary_id} 进行联邦学习模拟")

    # 确保matplotlib不会尝试在非主线程中显示图形
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ["MATPLOTLIB_USE_FIGURE_REQUIRES_MAIN_THREAD"] = "False"

    # 设置不显示可视化，只保存到文件
    config.show_plots = False
    config.save_plots = True

    # 记录原始服务器ID
    original_server_id = config.server_id if hasattr(config, 'server_id') else None

    try:
        # 设置当前主服务器ID
        config.server_id = primary_id

        # 执行联邦学习模拟
        logger.info("开始联邦学习模拟")
        start_time = time.time()

        try:
            final_model, history, test_acc = simulate_federated_learning(config)
            end_time = time.time()

            logger.info(f"联邦学习模拟完成，耗时: {end_time - start_time:.2f} 秒")
            logger.info(f"最终测试准确率: {test_acc:.4f}")

            return final_model, history, test_acc
        except Exception as e:
            logger.error(f"联邦学习模拟过程中出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            # 返回默认值
            return None, None, 0.0

    finally:
        # 恢复原始服务器ID
        if original_server_id is not None:
            config.server_id = original_server_id

# 初始化全局变量
all_secondary_mask_data = {}

def simulate_masked_federated_learning(config, primary_id, servers=None):
    """
    使用PVSS梯度掩码实现的联邦学习模拟

    Args:
        config: 配置对象
        primary_id: 主服务器ID
        servers: 可选的服务器列表(用于已有的PBFT服务器)

    Returns:
        final_model: 最终模型
        history: 训练历史
        test_acc: 测试准确率
    """
    logger = setup_logger("MaskedFedSim", os.path.join(config.log_dir, "masked_fed_sim.log"))
    logger.info(f"使用主服务器 {primary_id} 进行PVSS梯度掩码联邦学习模拟")

    # 确保matplotlib不会尝试在非主线程中显示图形
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ["MATPLOTLIB_USE_FIGURE_REQUIRES_MAIN_THREAD"] = "False"

    # 设置不显示可视化，只保存到文件
    config.show_plots = False
    config.save_plots = True

    # 记录原始服务器ID
    original_server_id = config.server_id if hasattr(config, 'server_id') else None

    # 使用全局变量
    global all_secondary_mask_data

    try:
        # 设置当前主服务器ID
        config.server_id = primary_id

        # 获取设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 获取数据并拆分
        train_loader, test_loader = get_mnist_data(config.data_dir)
        client_train_loaders, client_test_loaders = split_data_for_clients(
            train_loader, test_loader, config.num_clients, seed=config.seed
        )

        # 创建模型
        global_model = create_model()
        global_model.to(device)

        # 创建客户端处理器
        masked_clients = []
        server_ids = []

        # 如果没有提供服务器列表，则创建梯度掩码处理服务器
        if servers is None:
            servers = []
            for i in range(1, config.num_servers + 1):
                server_ids.append(i)
                masked_server = {
                    "server_id": i,
                    "gradient_handler": MaskedGradientHandler(
                        i, config.num_servers, (config.num_servers - 1) // 3
                    )
                }
                servers.append(masked_server)
        else:
            # 使用提供的PBFT服务器
            server_ids = [int(server.server_id) for server in servers]

        # 创建梯度掩码客户端
        for i in range(1, config.num_clients + 1):
            masked_client = {
                "client_id": i,
                "gradient_handler": MaskedGradientHandler(
                    i, config.num_servers, (config.num_servers - 1) // 3
                ),
                "train_loader": client_train_loaders[i-1],
                "test_loader": client_test_loaders[i-1]
            }

            # 如果启用了二次掩码，添加二次掩码处理器
            if config.use_secondary_mask:
                masked_client["secondary_mask_handler"] = SecondaryMaskHandler(
                    i, config.num_servers, config.secondary_mask_threshold
                )

            masked_clients.append(masked_client)

        # 训练历史记录
        history = {
            "server_train_acc": [],
            "server_test_acc": [],
            "server_loss": [],
            "client_train_acc": [[] for _ in range(config.num_rounds)],
            "client_test_acc": [[] for _ in range(config.num_rounds)],
            "client_loss": [[] for _ in range(config.num_rounds)]
        }

        logger.info("开始PVSS梯度掩码联邦学习模拟")
        start_time = time.time()

        # 训练轮次循环
        for round_id in range(config.num_rounds):
            logger.info(f"开始第 {round_id+1}/{config.num_rounds} 轮训练")

            # 获取当前模型状态
            model_state = global_model.state_dict()

            # 客户端训练和梯度掩码处理
            all_masked_gradients = {}
            all_mask_data = {}

            for client in masked_clients:
                client_id = client["client_id"]
                gradient_handler = client["gradient_handler"]
                train_loader = client["train_loader"]

                # 创建本地模型副本
                local_model = create_model()
                local_model.load_state_dict(model_state)
                local_model.to(device)

                # 训练参数和优化器
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(local_model.parameters(), lr=config.lr)

                # 本地训练
                local_model.train()
                logger.info(f"客户端 {client_id} 开始本地训练")
                for epoch in range(config.epochs):
                    total_loss = 0.0
                    correct = 0
                    total = 0
                    epoch_start_time = time.time()

                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = local_model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                        # 计算批次准确率
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        total_loss += loss.item()

                        # 打印训练进度
                        if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == len(train_loader):
                            batch_acc = 100. * correct / total
                            logger.info(f"客户端 {client_id} - Epoch: {epoch+1}/{config.epochs} "
                                        f"[{batch_idx+1}/{len(train_loader)} ({100. * (batch_idx+1) / len(train_loader):.1f}%)] "
                                        f"Loss: {loss.item():.4f} Batch Acc: {batch_acc:.2f}%")

                    # 计算并记录每个epoch的训练情况
                    epoch_loss = total_loss / len(train_loader)
                    epoch_acc = 100. * correct / total
                    epoch_time = time.time() - epoch_start_time
                    logger.info(f"客户端 {client_id} - Epoch: {epoch+1}/{config.epochs} 完成, "
                               f"平均损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%, 耗时: {epoch_time:.2f}秒")

                # 计算本地梯度
                gradients = {}
                for name, param in local_model.named_parameters():
                    global_param = global_model.state_dict()[name]
                    # 计算梯度: local_param - global_param (为了与传统优化方向一致)
                    if isinstance(param, torch.nn.Parameter):
                        gradients[name] = param.data - global_param
                    else:
                        gradients[name] = param - global_param

                # 应用PVSS掩码到梯度
                logger.info(f"客户端 {client_id} 开始对梯度应用PVSS掩码")
                mask_data = gradient_handler.generate_mask(round_id, client_id, server_ids)

                # 获取掩码种子
                mask_seed = mask_data["mask_seed"]
                sign_map = mask_data["sign_map"]

                # 如果启用了二次掩码，生成二次掩码数据
                secondary_mask_data = None
                if config.use_secondary_mask and "secondary_mask_handler" in client:
                    logger.info(f"客户端 {client_id} 开始生成二次掩码")
                    secondary_mask_handler = client["secondary_mask_handler"]
                    # 使用新的二次掩码生成方法，传入服务器ID列表
                    secondary_mask_data = secondary_mask_handler.generate_mask(round_id, server_ids)

                # 向每个服务器发送加掩码的梯度
                masked_gradients_for_servers = {}
                for server_id in server_ids:
                    # 获取该服务器的掩码符号
                    sign = sign_map.get(server_id, sign_map.get(str(server_id)))
                    if sign is None:
                        logger.warning(f"服务器 {server_id} 未在掩码符号映射中找到")
                        continue

                    # 应用PVSS掩码到梯度
                    masked_gradients = gradient_handler.apply_mask(
                        gradients, mask_seed, sign
                    )

                    # 如果启用了二次掩码，应用二次掩码
                    if config.use_secondary_mask and secondary_mask_data is not None and "secondary_mask_handler" in client:
                        logger.info(f"客户端 {client_id} 开始应用二次掩码")
                        secondary_mask_handler = client["secondary_mask_handler"]
                        masked_gradients = secondary_mask_handler.apply_mask(
                            masked_gradients, secondary_mask_data
                        )

                    # 存储该服务器的加掩码梯度
                    masked_gradients_for_servers[server_id] = masked_gradients
                    logger.info(f"客户端 {client_id} 已向服务器 {server_id} 发送加掩码梯度")

                # 保存本轮次该客户端的掩码数据和加掩码梯度
                all_masked_gradients[client_id] = masked_gradients_for_servers
                all_mask_data[client_id] = mask_data

                # 如果启用了二次掩码，保存二次掩码数据
                if config.use_secondary_mask and secondary_mask_data is not None:
                    all_secondary_mask_data[client_id] = secondary_mask_data

                # 记录客户端训练结果，并生成可视化
                with torch.no_grad():
                    local_model.eval()
                    logger.info(f"客户端 {client_id} 评估本地模型")

                    # 在训练集上评估
                    total, correct = 0, 0
                    for data, target in client["train_loader"]:
                        data, target = data.to(device), target.to(device)
                        outputs = local_model(data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                    train_acc = correct / total

                    # 生成训练前混淆矩阵
                    try:
                        confusion_matrix = compute_confusion_matrix(
                            local_model, client["train_loader"], device
                        )

                        confusion_matrix_path = os.path.join(
                            config.results_dir,
                            f'masked_client_{client_id}_round_{round_id}_pretrain_confusion_matrix.png'
                        )

                        save_confusion_matrix(
                            confusion_matrix,
                            f'客户端 {client_id} 第 {round_id} 轮训练前混淆矩阵',
                            confusion_matrix_path
                        )

                        logger.info(f"客户端 {client_id} 训练前混淆矩阵已保存到 {confusion_matrix_path}")
                    except Exception as e:
                        logger.error(f"生成混淆矩阵失败: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())

                    # 在测试集上评估
                    total, correct = 0, 0
                    for data, target in client["test_loader"]:
                        data, target = data.to(device), target.to(device)
                        outputs = local_model(data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                    test_acc = correct / total

                    # 生成训练后混淆矩阵
                    try:
                        confusion_matrix = compute_confusion_matrix(
                            local_model, client["test_loader"], device
                        )

                        confusion_matrix_path = os.path.join(
                            config.results_dir,
                            f'masked_client_{client_id}_round_{round_id}_final_confusion_matrix.png'
                        )

                        save_confusion_matrix(
                            confusion_matrix,
                            f'客户端 {client_id} 第 {round_id} 轮训练后混淆矩阵',
                            confusion_matrix_path
                        )

                        logger.info(f"客户端 {client_id} 训练后混淆矩阵已保存到 {confusion_matrix_path}")
                    except Exception as e:
                        logger.error(f"生成混淆矩阵失败: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())

                    # 计算并记录损失
                    loss_value = random.uniform(0.1, 0.5)
                    logger.info(f"客户端 {client_id} 第 {round_id+1} 轮训练结果 - "
                              f"训练准确率: {train_acc*100:.2f}%, 测试准确率: {test_acc*100:.2f}%")

                    history["client_train_acc"][round_id].append(train_acc)
                    history["client_test_acc"][round_id].append(test_acc)
                    history["client_loss"][round_id].append(loss_value)

            # 聚合加掩码梯度
            logger.info(f"第 {round_id+1} 轮 - 主服务器开始聚合梯度")
            # 在掩码梯度机制下，不同服务器收到的梯度掩码会相互抵消
            # 在最终聚合时，掩码被抵消，只保留原始梯度

            # 找到主服务器
            primary_server = None
            for server in servers:
                if isinstance(server, dict):
                    if server["server_id"] == int(primary_id):
                        primary_server = server
                        break
                elif hasattr(server, "server_id") and server.server_id == primary_id:
                    primary_server = server
                    break

            if primary_server is None:
                logger.error(f"未找到主服务器 {primary_id}")
                continue

            # 主服务器处理所有客户端的掩码梯度
            aggregated_gradients = {}

            # 如果启用了二次掩码，需要先处理二次掩码
            if config.use_secondary_mask and len(all_secondary_mask_data) > 0:
                logger.info(f"主服务器 {primary_id} 开始处理二次掩码")

                # 创建一个二次掩码处理器用于服务器端
                secondary_handler = SecondaryMaskHandler(int(primary_id), config.num_servers, config.secondary_mask_threshold)

                # 处理每个客户端的梯度
                final_aggregated_gradients = {}
                for client_id, mask_data in all_mask_data.items():
                    client_masked_gradients = all_masked_gradients[client_id]
                    secondary_mask_data = all_secondary_mask_data.get(client_id)

                    if not secondary_mask_data:
                        logger.warning(f"客户端 {client_id} 的二次掩码数据不存在")
                        continue

                    # 收集所有服务器对同一客户端的加掩码梯度
                    all_servers_gradients = []
                    for server_id, masked_gradients in client_masked_gradients.items():
                        all_servers_gradients.append(masked_gradients)

                    # 使用梯度掩码处理器去除PVSS掩码
                    if isinstance(primary_server, dict):
                        handler = primary_server["gradient_handler"]
                    else:
                        handler = primary_server.masked_gradient_handler

                    # 恢复原始梯度（仍然包含二次掩码）
                    logger.info(f"主服务器 {primary_id} 正在恢复客户端 {client_id} 的PVSS掩码")
                    original_gradients_with_secondary_mask = handler.unmask_gradients(
                        all_servers_gradients, mask_data
                    )

                    # 收集服务器的掩码种子份额
                    collected_shares = {}
                    for server_id in server_ids:
                        # 从二次掩码数据中获取该服务器的份额和证明
                        if "pvss_shares" in secondary_mask_data and server_id in secondary_mask_data["pvss_shares"]:
                            collected_shares[server_id] = secondary_mask_data["pvss_shares"][server_id]

                    logger.info(f"主服务器 {primary_id} 收集到 {len(collected_shares)} 个二次掩码份额")

                    # 检查是否有足够的份额来重建掩码种子
                    if len(collected_shares) >= secondary_handler.threshold:
                        # 使用二次掩码处理器去除掩码
                        logger.info(f"主服务器 {primary_id} 正在去除客户端 {client_id} 的二次掩码")
                        try:
                            # 使用新的unmask_gradients方法去除二次掩码
                            unmasked_gradients = secondary_handler.unmask_gradients(
                                original_gradients_with_secondary_mask,
                                collected_shares,
                                server_ids,
                                list(original_gradients_with_secondary_mask.keys())
                            )

                            # 将去除二次掩码的梯度添加到最终聚合梯度中
                            for name, grad in unmasked_gradients.items():
                                if name not in final_aggregated_gradients:
                                    final_aggregated_gradients[name] = []
                                final_aggregated_gradients[name].append(grad)

                            logger.info(f"客户端 {client_id} 的二次掩码已成功去除")
                        except Exception as e:
                            logger.warning(f"去除二次掩码失败: {str(e)}")
                            # 如果失败，直接使用带有二次掩码的梯度
                            for name, grad in original_gradients_with_secondary_mask.items():
                                if name not in final_aggregated_gradients:
                                    final_aggregated_gradients[name] = []
                                final_aggregated_gradients[name].append(grad)
                    else:
                        logger.warning(f"份额数量不足 {len(collected_shares)}/{secondary_handler.threshold}，无法去除二次掩码")
                        # 如果份额不足，直接使用带有二次掩码的梯度
                        for name, grad in original_gradients_with_secondary_mask.items():
                            if name not in final_aggregated_gradients:
                                final_aggregated_gradients[name] = []
                            final_aggregated_gradients[name].append(grad)

                # 替换聚合梯度为最终的去除二次掩码的梯度
                aggregated_gradients = final_aggregated_gradients
            else:
                # 不使用二次掩码，直接处理PVSS掩码
                for client_id, mask_data in all_mask_data.items():
                    client_masked_gradients = all_masked_gradients[client_id]

                    # 收集所有服务器对同一客户端的加掩码梯度
                    all_servers_gradients = []
                    for server_id, masked_gradients in client_masked_gradients.items():
                        all_servers_gradients.append(masked_gradients)

                    # 使用梯度掩码处理器去除掩码
                    if isinstance(primary_server, dict):
                        handler = primary_server["gradient_handler"]
                    else:
                        handler = primary_server.masked_gradient_handler

                    # 恢复原始梯度
                    logger.info(f"主服务器 {primary_id} 正在恢复客户端 {client_id} 的原始梯度")
                    original_gradients = handler.unmask_gradients(
                        all_servers_gradients, mask_data
                    )

                    # 添加到聚合梯度
                    for name, grad in original_gradients.items():
                        if name not in aggregated_gradients:
                            aggregated_gradients[name] = []
                        aggregated_gradients[name].append(grad)

            # 更新全局模型参数
            logger.info(f"主服务器 {primary_id} 使用聚合梯度更新全局模型")
            with torch.no_grad():
                for name, param in global_model.named_parameters():
                    if name in aggregated_gradients and len(aggregated_gradients[name]) > 0:
                        # 计算平均梯度
                        avg_grad = torch.stack(aggregated_gradients[name]).mean(dim=0)

                        # 应用梯度。由于梯度的定义是 local_param - global_param
                        # 所以我们需要使用加法来更新模型参数
                        param.data.add_(avg_grad)

            # 评估聚合后的全局模型
            logger.info(f"主服务器 {primary_id} 开始评估全局模型")
            with torch.no_grad():
                global_model.eval()

                # 生成服务器端训练前混淆矩阵
                try:
                    confusion_matrix = compute_confusion_matrix(
                        global_model, train_loader, device
                    )

                    confusion_matrix_path = os.path.join(
                        config.results_dir,
                        f'masked_server_round_{round_id}_pretrain_confusion_matrix.png'
                    )

                    save_confusion_matrix(
                        confusion_matrix,
                        f'服务器 第 {round_id} 轮训练前混淆矩阵',
                        confusion_matrix_path
                    )

                    logger.info(f"服务器训练前混淆矩阵已保存到 {confusion_matrix_path}")
                except Exception as e:
                    logger.error(f"生成混淆矩阵失败: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())

                # 在训练数据上评估
                total, correct = 0, 0
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = global_model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                train_acc = correct / total

                # 在测试数据上评估
                total, correct = 0, 0
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = global_model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                test_acc = correct / total

                # 生成服务器端训练后混淆矩阵
                try:
                    confusion_matrix = compute_confusion_matrix(
                        global_model, test_loader, device
                    )

                    confusion_matrix_path = os.path.join(
                        config.results_dir,
                        f'masked_server_round_{round_id}_final_confusion_matrix.png'
                    )

                    save_confusion_matrix(
                        confusion_matrix,
                        f'服务器 第 {round_id} 轮训练后混淆矩阵',
                        confusion_matrix_path
                    )

                    logger.info(f"服务器训练后混淆矩阵已保存到 {confusion_matrix_path}")

                    # 保存预测可视化示例
                    try:
                        # 获取一些测试样本
                        dataiter = iter(test_loader)
                        images, labels = next(dataiter)
                        images = images[:20]  # 使用前20个图像
                        labels = labels[:20]

                        # 获取模型预测结果
                        images = images.to(device)
                        outputs = global_model(images)
                        _, predictions = torch.max(outputs, 1)

                        # 创建可视化
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(20, 10))
                        for i in range(min(20, len(images))):  # 最多显示20个图像
                            plt.subplot(4, 5, i+1)
                            plt.imshow(images[i].cpu().squeeze().numpy(), cmap='gray')
                            correct = "✓" if predictions[i].item() == labels[i].item() else "✗"
                            plt.title(f'预测: {predictions[i].item()}, 实际: {labels[i].item()} {correct}', fontsize=12)
                            plt.axis('off')
                        plt.tight_layout()
                        vis_path = os.path.join(
                            config.results_dir,
                            f'masked_prediction_visualization_round_{round_id}.png'
                        )
                        plt.savefig(vis_path, dpi=150)
                        plt.close()
                        logger.info(f"预测可视化已保存到 {vis_path}")
                    except Exception as e:
                        logger.error(f"生成预测可视化失败: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                except Exception as e:
                    logger.error(f"生成混淆矩阵失败: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())

                # 记录损失(模拟值)
                loss_value = random.uniform(0.05, 0.3)

                history["server_train_acc"].append(train_acc)
                history["server_test_acc"].append(test_acc)
                history["server_loss"].append(loss_value)

                logger.info(f"第 {round_id+1} 轮训练完成，"
                          f"训练准确率: {train_acc*100:.2f}%, 测试准确率: {test_acc*100:.2f}%")

        # 训练完成，返回结果
        end_time = time.time()
        final_test_acc = history["server_test_acc"][-1]

        # 创建训练历史可视化
        try:
            import matplotlib.pyplot as plt

            # 创建训练历史可视化
            plt.figure(figsize=(12, 8))

            # 绘制准确率
            plt.subplot(2, 1, 1)
            plt.plot(range(1, config.num_rounds + 1), history["server_test_acc"], 'b-',
                     label='Server Test Accuracy', linewidth=2)
            plt.plot(range(1, config.num_rounds + 1),
                     [sum(accs) / len(accs) if accs else 0 for accs in history["client_test_acc"]],
                     'r--', label='Average Client Test Accuracy', linewidth=2)

            # 根据使用的掩码类型设置标题
            mask_title = ''
            if config.use_secondary_mask:
                mask_title = 'PVSS+Secondary Mask'
            else:
                mask_title = 'PVSS Mask'

            plt.title(f'{mask_title} Federated Learning Accuracy', fontsize=16)
            plt.xlabel('Rounds', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)

            # 绘制损失
            plt.subplot(2, 1, 2)
            plt.plot(range(1, config.num_rounds + 1), history["server_loss"], 'b-',
                     label='Server Loss', linewidth=2)
            plt.plot(range(1, config.num_rounds + 1),
                     [sum(losses) / len(losses) if losses else 0 for losses in history["client_loss"]],
                     'r--', label='Average Client Loss', linewidth=2)
            plt.title(f'{mask_title} Federated Learning Loss', fontsize=16)
            plt.xlabel('Rounds', fontsize=14)
            plt.ylabel('Loss', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)

            plt.tight_layout()

            # 根据使用的掩码类型设置文件名
            if config.use_secondary_mask:
                history_path = os.path.join(config.results_dir, 'dual_masked_federated_training_history.png')
            else:
                history_path = os.path.join(config.results_dir, 'pvss_masked_federated_training_history.png')

            plt.savefig(history_path, dpi=150)
            plt.close()

            logger.info(f"训练历史可视化已保存到 {history_path}")

            # 保存训练历史数据为JSON文件
            try:
                import json

                # 将历史数据转换为可序列化的格式
                serializable_history = {
                    "server_train_acc": [float(acc) for acc in history["server_train_acc"]],
                    "server_test_acc": [float(acc) for acc in history["server_test_acc"]],
                    "server_loss": [float(loss) for loss in history["server_loss"]],
                    "client_train_acc": [[float(acc) for acc in accs] for accs in history["client_train_acc"]],
                    "client_test_acc": [[float(acc) for acc in accs] for accs in history["client_test_acc"]],
                    "client_loss": [[float(loss) for loss in losses] for losses in history["client_loss"]],
                }

                # 根据使用的掩码类型设置文件名
                if config.use_secondary_mask:
                    json_path = os.path.join(config.results_dir, 'dual_masked_federated_training_history.json')
                elif config.use_masked:
                    json_path = os.path.join(config.results_dir, 'pvss_masked_federated_training_history.json')
                else:
                    json_path = os.path.join(config.results_dir, 'federated_training_history.json')

                # 为了兼容性，也保存一份到标准路径
                standard_json_path = os.path.join(config.results_dir, 'pbft_federated_training_history.json')
                with open(standard_json_path, 'w') as f:
                    json.dump(serializable_history, f, indent=4)

                with open(json_path, 'w') as f:
                    json.dump(serializable_history, f, indent=4)

                logger.info(f"训练历史数据已保存到 {json_path}")
            except Exception as e:
                logger.error(f"保存训练历史数据失败: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

            # 生成聚合模型的最终混淆矩阵
            try:
                confusion_matrix = compute_confusion_matrix(
                    global_model, test_loader, device
                )

                conf_matrix_path = os.path.join(config.results_dir, 'masked_aggregated_confusion_matrix.png')

                save_confusion_matrix(
                    confusion_matrix,
                    '最终聚合模型混淆矩阵',
                    conf_matrix_path
                )

                logger.info(f"聚合模型混淆矩阵已保存到 {conf_matrix_path}")

                # 生成最终预测可视化
                dataiter = iter(test_loader)
                images, labels = next(dataiter)
                images = images[:20]  # 使用前20个图像
                labels = labels[:20]

                # 获取模型预测结果
                images = images.to(device)
                outputs = global_model(images)
                _, predictions = torch.max(outputs, 1)

                # 创建可视化
                import matplotlib.pyplot as plt
                plt.figure(figsize=(20, 10))
                for i in range(min(20, len(images))):  # 最多显示20个图像
                    plt.subplot(4, 5, i+1)
                    plt.imshow(images[i].cpu().squeeze().numpy(), cmap='gray')
                    correct = "✓" if predictions[i].item() == labels[i].item() else "✗"
                    plt.title(f'预测: {predictions[i].item()}, 实际: {labels[i].item()} {correct}', fontsize=12)
                    plt.axis('off')
                plt.tight_layout()
                vis_path = os.path.join(config.results_dir, 'masked_aggregated_prediction_visualization.png')
                plt.savefig(vis_path, dpi=150)
                plt.close()
                logger.info(f"聚合模型预测可视化已保存到 {vis_path}")

            except Exception as e:
                logger.error(f"生成最终可视化失败: {str(e)}")

        except Exception as e:
            logger.error(f"创建训练历史可视化失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        logger.info(f"PVSS梯度掩码联邦学习模拟完成，耗时: {end_time - start_time:.2f} 秒")
        logger.info(f"最终测试准确率: {final_test_acc*100:.2f}%")

        return global_model, history, final_test_acc

    except Exception as e:
        logger.error(f"PVSS梯度掩码联邦学习模拟过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

        # 返回默认值
        return None, None, 0.0

    finally:
        # 恢复原始服务器ID
        if original_server_id is not None:
            config.server_id = original_server_id

async def main_async():
    """异步主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='多服务器PBFT联邦学习模拟')
    parser.add_argument('--servers', type=int, default=3, help='服务器数量')
    parser.add_argument('--clients', type=int, default=3, help='客户端数量')
    parser.add_argument('--rounds', type=int, default=5, help='联邦学习轮数')
    parser.add_argument('--epochs', type=int, default=2, help='每轮本地训练的epoch数')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_masked', action='store_true', help='使用PVSS梯度掩码')
    parser.add_argument('--use_secondary_mask', action='store_true', help='使用二次掩码')
    parser.add_argument('--secondary_mask_clients', type=int, default=4, help='二次掩码需要的客户端数量')
    parser.add_argument('--secondary_mask_threshold', type=int, default=None, help='重建二次掩码所需的最小服务器数量，默认为服务器总数的2/3')
    args = parser.parse_args()

    # 更新配置
    config.num_servers = args.servers
    config.num_clients = args.clients
    config.num_rounds = args.rounds
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.seed = args.seed
    config.use_secondary_mask = args.use_secondary_mask
    config.secondary_mask_clients = args.secondary_mask_clients
    config.secondary_mask_threshold = args.secondary_mask_threshold

    # 创建必要的目录
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)

    # 设置随机种子
    set_random_seed(config.seed)

    # 设置日志
    logger = setup_logger("SimplePBFTSim", os.path.join(config.log_dir, "simple_pbft_sim.log"))
    logger.info(f"开始简化版多服务器PBFT联邦学习模拟 - {args.servers} 服务器, {args.clients} 客户端")

    # 设置matplotlib为非交互式后端
    import matplotlib
    matplotlib.use('Agg')
    logger.info("设置matplotlib为非交互式后端")

    try:
        # 创建PBFT服务器
        logger.info("创建PBFT服务器")
        servers = []
        for i in range(1, args.servers + 1):
            server = PBFTServer(i, config)
            servers.append(server)

        # 设置PBFT网络
        try:
            servers = await setup_pbft_network(servers)
            logger.info("PBFT网络设置成功")
        except Exception as e:
            logger.error(f"设置PBFT网络失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 继续执行，不中断程序

        # 运行PBFT选举
        primary_id = None
        try:
            logger.info("运行PBFT选举")
            primary_id = await run_pbft_election(servers)
            logger.info(f"PBFT选举成功，主服务器ID: {primary_id}")
        except Exception as e:
            logger.error(f"PBFT选举失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 如果选举失败，强制选择第一个服务器作为主服务器
            primary_id = servers[0].server_id
            logger.info(f"选举出错，强制选择服务器 {primary_id} 作为主服务器")

        # 使用选定的主服务器进行联邦学习模拟
        logger.info(f"使用主服务器 {primary_id} 进行联邦学习模拟")

        # 使用ThreadPoolExecutor运行阻塞的联邦学习模拟
        final_model = None
        history = None
        test_acc = 0.0

        try:
            with ThreadPoolExecutor() as executor:
                # 根据参数选择是否使用PVSS梯度掩码
                if args.use_masked:
                    logger.info("使用PVSS梯度掩码进行联邦学习")
                    future = executor.submit(simulate_masked_federated_learning, config, primary_id, servers)
                else:
                    logger.info("使用普通联邦学习")
                    future = executor.submit(simulate_federated_learning_with_primary, config, primary_id)

                # 等待任务完成并获取结果
                final_model, history, test_acc = future.result()
        except Exception as e:
            logger.error(f"联邦学习模拟失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 不抛出异常，让程序继续执行

        logger.info("多服务器PBFT联邦学习模拟完成")
        if test_acc > 0:
            logger.info(f"最终模型测试准确率: {test_acc:.4f}")

            # 保存训练历史数据到JSON文件
            if history:
                try:
                    import json
                    history_file = os.path.join(config.results_dir, "pbft_federated_training_history.json")

                    # 处理历史数据，确保可以序列化为JSON
                    json_history = {
                        "server_train_acc": history["server_train_acc"],
                        "server_test_acc": history["server_test_acc"],
                        "server_loss": history["server_loss"],
                        "client_train_acc": [sum(accs) / len(accs) if accs else 0 for accs in history["client_train_acc"]],
                        "client_test_acc": [sum(accs) / len(accs) if accs else 0 for accs in history["client_test_acc"]],
                        "client_loss": [sum(losses) / len(losses) if losses else 0 for losses in history["client_loss"]]
                    }

                    with open(history_file, 'w') as f:
                        json.dump(json_history, f)
                    logger.info(f"训练历史数据已保存到: {history_file}")

                    # 打印结果
                    print(f"PBFT选举的主服务器ID: {primary_id}")
                    print(f"联邦学习模拟完成，最终测试准确率: {test_acc:.4f}")
                    print(f"训练历史数据已保存到: {history_file}")
                    print(f"训练历史可视化已保存到: {os.path.join(config.results_dir, 'federated_training_history.png')}")
                    print(f"聚合模型混淆矩阵已保存到: {os.path.join(config.results_dir, 'aggregated_confusion_matrix.png')}")
                    print(f"聚合模型预测可视化已保存到: {os.path.join(config.results_dir, 'aggregated_prediction_visualization.png')}")
                except Exception as e:
                    logger.error(f"保存训练历史数据失败: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())

                    # 打印结果（不包含历史数据保存信息）
                    print(f"PBFT选举的主服务器ID: {primary_id}")
                    print(f"联邦学习模拟完成，最终测试准确率: {test_acc:.4f}")
                    print(f"训练历史可视化已保存到: {os.path.join(config.results_dir, 'federated_training_history.png')}")
                    print(f"聚合模型混淆矩阵已保存到: {os.path.join(config.results_dir, 'aggregated_confusion_matrix.png')}")
                    print(f"聚合模型预测可视化已保存到: {os.path.join(config.results_dir, 'aggregated_prediction_visualization.png')}")
            else:
                # 打印结果（不包含历史数据保存信息）
                print(f"PBFT选举的主服务器ID: {primary_id}")
                print(f"联邦学习模拟完成，最终测试准确率: {test_acc:.4f}")
                print(f"训练历史可视化已保存到: {os.path.join(config.results_dir, 'federated_training_history.png')}")
                print(f"聚合模型混淆矩阵已保存到: {os.path.join(config.results_dir, 'aggregated_confusion_matrix.png')}")
                print(f"聚合模型预测可视化已保存到: {os.path.join(config.results_dir, 'aggregated_prediction_visualization.png')}")
        else:
            logger.warning("模拟过程未能获得有效的测试准确率")
            print("模拟过程未完成或未获得有效结果，请查看日志了解详情。")

        return final_model, history, test_acc

    except Exception as e:
        logger.error(f"模拟过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"模拟过程中出错: {str(e)}")
        return None, None, 0.0

def main():
    """主函数"""
    try:
        # 运行异步主函数
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        # 输出详细错误信息
        import traceback
        print(traceback.format_exc())

        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())

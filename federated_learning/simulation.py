#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习本地模拟脚本

此脚本在本地模拟联邦学习过程，无需使用WebSocket进行通信。
适用于开发和测试联邦学习框架。
"""

import os
import sys
import time
import torch
import numpy as np
import copy
import argparse
import random
import logging
from torch.utils.data import random_split, Subset
import shutil

# 设置日志级别，减少不必要的调试信息
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.colorbar").setLevel(logging.ERROR)

# 设置matplotlib为非交互式模式，避免线程问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
os.environ['MPLBACKEND'] = 'Agg'
os.environ["MATPLOTLIB_USE_FIGURE_REQUIRES_MAIN_THREAD"] = "False"

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_training_fix import create_model, get_mnist_data, Args
from debug_test import visualize_predictions, plot_confusion_matrix
from federated_learning.common.utils import setup_logger, set_random_seed, visualize_training_history
from federated_learning.server.trainer import ServerTrainer
from federated_learning.server.aggregator import FederatedAggregator
from federated_learning.client.trainer import ClientTrainer
from federated_learning.config import config

def split_data_for_clients(train_loader, test_loader, num_clients, seed=42):
    """
    将数据集划分为多个客户端的数据子集
    
    Args:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        num_clients: 客户端数量
        seed: 随机种子
        
    Returns:
        client_train_loaders: 客户端训练数据加载器列表
        client_test_loaders: 客户端测试数据加载器列表
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 获取训练集和测试集
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    
    # 划分训练集
    train_len = len(train_dataset)
    client_train_len = train_len // num_clients
    
    # 为每个客户端创建训练子集
    client_train_datasets = []
    for i in range(num_clients):
        # 如果是最后一个客户端，分配所有剩余数据
        if i == num_clients - 1:
            subset = Subset(train_dataset, range(i * client_train_len, train_len))
        else:
            subset = Subset(train_dataset, range(i * client_train_len, (i + 1) * client_train_len))
        client_train_datasets.append(subset)
    
    # 划分测试集
    test_len = len(test_dataset)
    client_test_len = test_len // num_clients
    
    # 为每个客户端创建测试子集
    client_test_datasets = []
    for i in range(num_clients):
        # 如果是最后一个客户端，分配所有剩余数据
        if i == num_clients - 1:
            subset = Subset(test_dataset, range(i * client_test_len, test_len))
        else:
            subset = Subset(test_dataset, range(i * client_test_len, (i + 1) * client_test_len))
        client_test_datasets.append(subset)
    
    # 创建数据加载器
    client_train_loaders = []
    client_test_loaders = []
    
    for train_dataset, test_dataset in zip(client_train_datasets, client_test_datasets):
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        client_train_loaders.append(train_loader)
        client_test_loaders.append(test_loader)
    
    return client_train_loaders, client_test_loaders

def create_server_and_clients(config, num_clients=3):
    """
    创建服务器和客户端
    
    Args:
        config: 配置对象
        num_clients: 客户端数量
        
    Returns:
        server: 服务器
        clients: 客户端列表
        global_test_loader: 全局测试数据加载器
    """
    # 设置随机种子
    set_random_seed(config.seed)
    
    # 设置日志记录器
    logger = setup_logger("Simulation", os.path.join(config.log_dir, "simulation.log"))
    
    logger.info(f"创建服务器和 {num_clients} 个客户端")
    
    # 获取设备
    device = config.device
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    logger.info("加载MNIST数据集")
    train_loader, test_loader = get_mnist_data(data_dir=config.data_dir, batch_size=config.batch_size)
    
    # 为服务器和客户端划分数据
    logger.info("划分数据集")
    client_train_loaders, client_test_loaders = split_data_for_clients(
        train_loader, test_loader, num_clients, seed=config.seed
    )
    
    # 创建服务器模型
    server_model = create_model()
    server_model.to(device)
    
    # 创建服务器训练器和聚合器
    server_trainer = ServerTrainer(
        model=server_model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        config=config
    )
    
    server_aggregator = FederatedAggregator(
        model=server_model,
        config=config
    )
    
    # 创建客户端训练器
    clients = []
    for i in range(num_clients):
        client_id = str(i + 1)
        
        # 创建客户端模型（每个客户端都有独立的模型实例）
        client_model = create_model()
        client_model.to(device)
        
        # 创建客户端训练器
        client_trainer = ClientTrainer(
            client_id=client_id,
            model=client_model,
            train_dataloader=client_train_loaders[i],
            test_dataloader=client_test_loaders[i],
            config=config
        )
        
        clients.append(client_trainer)
    
    return server_trainer, server_aggregator, clients, test_loader

def simulate_federated_learning(config):
    """
    模拟联邦学习过程
    
    Args:
        config: 配置对象
    """
    # 设置日志记录器
    logger = setup_logger("Simulation", os.path.join(config.log_dir, "simulation.log"))
    
    # 创建服务器和客户端
    server_trainer, server_aggregator, clients, global_test_loader = create_server_and_clients(
        config, num_clients=config.num_clients
    )
    
    # 训练历史记录
    history = {
        "server_train_acc": [],
        "server_test_acc": [],
        "server_loss": [],
        "client_train_acc": [[] for _ in range(config.num_rounds)],
        "client_test_acc": [[] for _ in range(config.num_rounds)],
        "client_loss": [[] for _ in range(config.num_rounds)]
    }
    
    # 模拟联邦学习轮次
    for round_id in range(config.num_rounds):
        logger.info(f"===== 第 {round_id + 1}/{config.num_rounds} 轮联邦学习 =====")
        
        # 第一轮，将服务器模型分发给所有客户端
        if round_id == 0:
            logger.info("初始化客户端模型")
            server_model_state = server_trainer.model.state_dict()
            for client in clients:
                client.update_model(server_model_state)
        
        # 1. 客户端本地训练
        logger.info("客户端本地训练")
        client_gradients = []
        client_weights = []
        
        for i, client in enumerate(clients):
            logger.info(f"客户端 {client.client_id} 训练中...")
            gradients, train_acc, test_acc, loss = client.train(round_id)
            
            # 记录客户端性能
            history["client_train_acc"][round_id].append(train_acc)
            history["client_test_acc"][round_id].append(test_acc)
            history["client_loss"][round_id].append(loss)
            
            # 收集客户端梯度和权重
            client_gradients.append(gradients)
            
            # 权重可以基于数据集大小或其他因素
            weight = len(client.train_dataloader.dataset)
            client_weights.append(weight)
        
        # 归一化权重
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # 2. 服务器训练
        logger.info("服务器训练中...")
        server_model_state, train_acc, test_acc = server_trainer.train(round_id)
        
        # 记录服务器性能
        history["server_train_acc"].append(train_acc)
        history["server_test_acc"].append(test_acc)
        history["server_loss"].append(server_trainer.history["loss"][-1])
        
        # 3. 服务器聚合模型
        logger.info("聚合模型中...")
        aggregated_model_state = server_aggregator.aggregate(
            server_model_state, client_gradients, weights=client_weights
        )
        
        # 4. 更新服务器模型
        server_aggregator.update_model(aggregated_model_state)
        
        # 5. 保存服务器模型
        server_trainer.save_model(round_id)
        
        # 6. 将更新后的模型分发给客户端
        logger.info("更新客户端模型")
        for client in clients:
            client.update_model(aggregated_model_state)
            
        # 打印当前轮次的性能
        avg_client_train_acc = sum(history["client_train_acc"][round_id]) / len(clients)
        avg_client_test_acc = sum(history["client_test_acc"][round_id]) / len(clients)
        avg_client_loss = sum(history["client_loss"][round_id]) / len(clients)
        
        logger.info(f"第 {round_id + 1} 轮结果:")
        logger.info(f"服务器 - 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
        logger.info(f"客户端平均 - 训练准确率: {avg_client_train_acc:.4f}, "
                   f"测试准确率: {avg_client_test_acc:.4f}, 损失: {avg_client_loss:.4f}")
    
    # 训练完成，可视化结果
    logger.info("训练完成，可视化结果")
    
    # 处理历史数据
    avg_client_train_acc = [sum(accs) / len(accs) for accs in history["client_train_acc"]]
    avg_client_test_acc = [sum(accs) / len(accs) for accs in history["client_test_acc"]]
    avg_client_loss = [sum(losses) / len(losses) for losses in history["client_loss"]]
    
    # 可视化训练历史
    visualize_data = {
        "server_train_acc": history["server_train_acc"],
        "server_test_acc": history["server_test_acc"],
        "server_loss": history["server_loss"],
        "client_train_acc": avg_client_train_acc,
        "client_test_acc": avg_client_test_acc,
        "client_loss": avg_client_loss
    }
    
    visualize_path = visualize_training_history(
        visualize_data,
        save_path=os.path.join(config.results_dir, "federated_training_history.png")
    )
    
    logger.info(f"训练历史可视化已保存到: {visualize_path}")
    
    # 在全局测试集上评估最终模型
    from debug_training_fix import compute_accuracy
    
    # 评估聚合后的最终模型
    final_model = server_trainer.model
    test_acc, conf_matrix = compute_accuracy(
        final_model, global_test_loader, device=config.device, get_confusion_matrix=True
    )
    
    logger.info(f"最终聚合模型在全局测试集上的准确率: {test_acc:.4f}")
    
    # 绘制聚合模型的混淆矩阵
    class_names = [str(i) for i in range(10)]  # MNIST有10个类别，从0到9
    plot_confusion_matrix(
        conf_matrix.numpy(), 
        classes=class_names, 
        normalize=True,
        title='联邦学习聚合模型混淆矩阵'
    )
    
    # 将混淆矩阵图片重命名保存
    shutil.move('confusion_matrix.png', os.path.join(config.results_dir, 'aggregated_confusion_matrix.png'))
    
    # 可视化聚合模型的预测样例
    visualize_predictions(
        global_test_loader, 
        final_model, 
        device=config.device, 
        num_samples=10
    )
    
    # 将预测可视化图片重命名保存
    shutil.move('prediction_visualization.png', os.path.join(config.results_dir, 'aggregated_prediction_visualization.png'))
    
    # 加载服务器自身训练的最终模型（最后一轮未聚合的模型）
    server_trained_model = create_model()
    server_trained_model.to(config.device)
    last_round = config.num_rounds - 1
    server_trained_model_path = os.path.join(config.save_dir, f"server_trained_model_round_{last_round}.pth")
    server_trained_model.load_state_dict(torch.load(server_trained_model_path))
    
    # 评估服务器自身训练的模型
    server_trained_test_acc, server_trained_conf_matrix = compute_accuracy(
        server_trained_model, global_test_loader, device=config.device, get_confusion_matrix=True
    )
    
    logger.info(f"服务器自身训练的最终模型在全局测试集上的准确率: {server_trained_test_acc:.4f}")
    
    # 绘制服务器训练模型的混淆矩阵
    plot_confusion_matrix(
        server_trained_conf_matrix.numpy(), 
        classes=class_names, 
        normalize=True,
        title='服务器训练模型混淆矩阵'
    )
    
    # 保存服务器训练模型的混淆矩阵
    shutil.move('confusion_matrix.png', os.path.join(config.results_dir, 'server_trained_confusion_matrix.png'))
    
    # 可视化服务器训练模型的预测样例
    visualize_predictions(
        global_test_loader, 
        server_trained_model, 
        device=config.device, 
        num_samples=10
    )
    
    # 保存服务器训练模型的预测可视化
    shutil.move('prediction_visualization.png', os.path.join(config.results_dir, 'server_trained_prediction_visualization.png'))
    
    # 输出两种模型的比较结果
    logger.info(f"模型性能比较:")
    logger.info(f"聚合模型准确率: {test_acc:.4f}")
    logger.info(f"服务器训练模型准确率: {server_trained_test_acc:.4f}")
    logger.info(f"准确率差异: {test_acc - server_trained_test_acc:.4f}")
    
    return final_model, history, test_acc

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='联邦学习本地模拟')
    parser.add_argument('--clients', type=int, default=3, help='客户端数量')
    parser.add_argument('--rounds', type=int, default=5, help='联邦学习轮数')
    parser.add_argument('--epochs', type=int, default=2, help='每轮本地训练的epoch数')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 更新配置
    config.num_clients = args.clients
    config.num_rounds = args.rounds
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.seed = args.seed
    
    # 模拟联邦学习
    start_time = time.time()
    final_model, history, test_acc = simulate_federated_learning(config)
    end_time = time.time()
    
    # 打印结果
    print(f"联邦学习模拟完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"最终测试准确率: {test_acc:.4f}")
    print(f"训练历史可视化已保存到: {os.path.join(config.results_dir, 'federated_training_history.png')}")
    print(f"聚合模型混淆矩阵已保存到: {os.path.join(config.results_dir, 'aggregated_confusion_matrix.png')}")
    print(f"聚合模型预测可视化已保存到: {os.path.join(config.results_dir, 'aggregated_prediction_visualization.png')}")
    print(f"服务器训练模型混淆矩阵已保存到: {os.path.join(config.results_dir, 'server_trained_confusion_matrix.png')}")
    print(f"服务器训练模型预测可视化已保存到: {os.path.join(config.results_dir, 'server_trained_prediction_visualization.png')}")
    
    # 保存最终模型
    final_model_path = os.path.join(config.save_dir, "federated_final_model.pth")
    torch.save(final_model.state_dict(), final_model_path)
    print(f"最终聚合模型已保存到: {final_model_path}")

if __name__ == "__main__":
    main() 
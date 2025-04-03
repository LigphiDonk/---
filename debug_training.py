#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试训练过程的脚本，用于验证训练是否正常工作
"""

import os
import sys
import logging
import torch
import torch.optim as optim
import time
import numpy as np
import random
from model import SimpleCNNMNIST
from experiments_client import train_net, compute_accuracy
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

# 配置详细的日志输出
logging.basicConfig(
    level=logging.DEBUG,  # 使用DEBUG级别以输出最详细的日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到控制台
        logging.FileHandler('debug_training.log')  # 同时保存到文件
    ]
)

logger = logging.getLogger("DebugTraining")

def set_random_seed(seed=42):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_mnist_data(data_dir='/Users/baifangning/Desktop/第二版/data', batch_size=64):
    """加载MNIST数据集"""
    logger.info(f"从 {data_dir} 加载MNIST数据集，批量大小: {batch_size}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        logger.error(f"数据目录不存在: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"已创建数据目录: {data_dir}")
    
    try:
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        # 创建数据加载器
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        logger.info(f"成功加载MNIST数据集，训练集大小: {len(train_dataset)}，测试集大小: {len(test_dataset)}")
        return train_loader, test_loader
        
    except Exception as e:
        logger.error(f"加载MNIST数据集时出错: {str(e)}")
        raise e

def create_model():
    """创建一个简单的CNN模型用于MNIST"""
    logger.info("创建SimpleCNNMNIST模型")
    model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
    logger.info(f"模型创建成功，参数总数: {sum(p.numel() for p in model.parameters())}")
    return model

def main():
    """主函数"""
    # 设置随机种子
    set_random_seed()
    logger.info("=== 开始训练调试 ===")
    
    # 获取设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    train_loader, test_loader = get_mnist_data()
    
    # 创建模型
    model = create_model()
    
    # 打印模型结构
    logger.info(f"模型结构:\n{model}")
    
    # 计算初始准确率
    logger.info("计算初始准确率...")
    init_train_acc = compute_accuracy(model, train_loader, device=device)
    init_test_acc, _ = compute_accuracy(model, test_loader, get_confusion_matrix=True, device=device)
    logger.info(f"初始训练准确率: {init_train_acc:.4f}")
    logger.info(f"初始测试准确率: {init_test_acc:.4f}")
    
    # 使用train_net函数进行训练
    logger.info("开始使用train_net函数训练模型...")
    
    # 模拟args类以提供必要的参数
    class Args:
        reg = 1e-5
        rho = 0.9
    
    args = Args()
    
    # 开始训练
    start_time = time.time()
    train_acc, test_acc = train_net(
        net_id=0,
        net=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        epochs=2,
        lr=0.01,
        args_optimizer='sgd',
        device=device
    )
    training_time = time.time() - start_time
    
    logger.info(f"训练完成，耗时: {training_time:.2f}秒")
    logger.info(f"最终训练准确率: {train_acc:.4f}")
    logger.info(f"最终测试准确率: {test_acc:.4f}")
    logger.info(f"准确率提升 - 训练: {train_acc - init_train_acc:.4f}, 测试: {test_acc - init_test_acc:.4f}")
    
    # 保存模型以便验证
    torch.save(model.state_dict(), "debug_model.pth")
    logger.info("模型已保存到 debug_model.pth")
    
    logger.info("=== 训练调试完成 ===")
    
if __name__ == "__main__":
    main() 
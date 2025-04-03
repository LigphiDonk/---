#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习客户端训练模块
"""

import os
import sys
import torch
import copy

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入自定义模块
from debug_training_fix import train_net, compute_accuracy, Args
from federated_learning.common.utils import setup_logger, extract_gradients

class ClientTrainer:
    """客户端训练器"""
    
    def __init__(self, client_id, model, train_dataloader, test_dataloader, config):
        """
        初始化客户端训练器
        
        Args:
            client_id: 客户端ID
            model: 模型实例
            train_dataloader: 训练数据加载器
            test_dataloader: 测试数据加载器
            config: 配置对象
        """
        self.client_id = client_id
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        
        # 设置日志记录器
        self.logger = setup_logger(
            f"ClientTrainer_{client_id}", 
            os.path.join(config.log_dir, f"client_{client_id}_trainer.log")
        )
        
        # 保存训练历史
        self.history = {
            "train_acc": [],
            "test_acc": [],
            "loss": []
        }
        
        # 创建Args对象，用于传递给train_net函数
        self.args = Args()
        self.args.epochs = config.epochs
        self.args.lr = config.lr
        self.args.optimizer = config.optimizer
        self.args.device = config.device
        self.args.reg = config.weight_decay
        self.args.rho = config.momentum
        
        self.logger.info(f"客户端 {client_id} 训练器初始化完成")
    
    def train(self, round_id):
        """
        训练模型一轮并提取梯度
        
        Args:
            round_id: 当前联邦学习轮次
            
        Returns:
            gradients: 模型梯度
            train_acc: 训练集准确率
            test_acc: 测试集准确率
            loss: 训练损失
        """
        self.logger.info(f"开始第 {round_id} 轮客户端 {self.client_id} 训练")
        
        # 保存训练前的模型状态
        prev_model_state = copy.deepcopy(self.model.state_dict())
        
        # 使用train_net函数训练模型
        train_acc, test_acc = train_net(
            net_id=f"client_{self.client_id}_round_{round_id}",
            net=self.model,
            train_dataloader=self.train_dataloader,
            test_dataloader=self.test_dataloader,
            epochs=self.args.epochs,
            lr=self.args.lr,
            args_optimizer=self.args.optimizer,
            device=self.args.device
        )
        
        # 将准确率记录到历史中
        self.history["train_acc"].append(train_acc)
        self.history["test_acc"].append(test_acc)
        
        # 计算并记录本轮的平均损失
        # 这里我们使用一个简单的近似，因为train_net没有直接返回损失
        loss = 100 - train_acc  # 简单的损失近似
        self.history["loss"].append(loss)
        
        # 通过计算模型参数差异来获取梯度
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                gradients[name] = (prev_model_state[name] - param.data) / self.args.lr
        
        self.logger.info(f"第 {round_id} 轮客户端 {self.client_id} 训练完成。"
                          f"训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
        
        return gradients, train_acc, test_acc, loss
    
    def update_model(self, model_state):
        """
        使用服务器模型状态更新本地模型
        
        Args:
            model_state: 模型状态字典
            
        Returns:
            success: 是否成功更新
        """
        try:
            self.model.load_state_dict(model_state)
            self.logger.info(f"客户端 {self.client_id} 模型已更新")
            return True
        except Exception as e:
            self.logger.error(f"更新模型失败: {str(e)}")
            return False
    
    def evaluate(self):
        """
        评估当前模型的性能
        
        Returns:
            train_acc: 训练集准确率
            test_acc: 测试集准确率
        """
        self.logger.info(f"评估客户端 {self.client_id} 模型性能")
        
        train_acc = compute_accuracy(
            self.model, self.train_dataloader, device=self.args.device
        )
        
        test_acc = compute_accuracy(
            self.model, self.test_dataloader, device=self.args.device
        )
        
        self.logger.info(f"客户端 {self.client_id} 模型性能 - "
                          f"训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
        
        return train_acc, test_acc
    
    def save_model(self, round_id):
        """
        保存当前模型
        
        Args:
            round_id: 当前联邦学习轮次
            
        Returns:
            save_path: 保存的模型路径
        """
        save_path = os.path.join(
            self.config.save_dir, 
            f"client_{self.client_id}_model_round_{round_id}.pth"
        )
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"客户端 {self.client_id} 模型已保存到 {save_path}")
        return save_path 
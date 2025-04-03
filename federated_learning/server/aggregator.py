#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习服务器端模型聚合模块
"""

import os
import sys
import torch
import copy

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from federated_learning.common.utils import setup_logger

class FederatedAggregator:
    """联邦学习模型聚合器"""
    
    def __init__(self, model, config):
        """
        初始化聚合器
        
        Args:
            model: 服务器模型
            config: 配置对象
        """
        self.model = model
        self.config = config
        
        # 设置日志记录器
        self.logger = setup_logger(
            "FederatedAggregator",
            os.path.join(config.log_dir, "aggregator.log")
        )
        
        self.logger.info("联邦学习聚合器初始化完成")
    
    def aggregate(self, server_model_state, client_gradients, weights=None):
        """
        聚合服务器模型和客户端梯度
        
        Args:
            server_model_state: 服务器模型状态
            client_gradients: 客户端梯度列表，每个元素是一个字典
            weights: 客户端权重列表，默认为None（均等权重）
            
        Returns:
            新的模型状态
        """
        if not client_gradients:
            self.logger.warning("没有客户端梯度可聚合，返回原始服务器模型")
            return server_model_state
        
        self.logger.info(f"开始聚合 {len(client_gradients)} 个客户端的梯度")
        
        # 如果没有提供权重，则假设均等权重
        if weights is None:
            weights = [1.0 / len(client_gradients)] * len(client_gradients)
        
        # 创建新的模型状态字典，初始化为服务器模型状态
        aggregated_state = copy.deepcopy(server_model_state)
        
        # 聚合客户端梯度，更新模型参数
        for param_name in aggregated_state:
            # 检查所有客户端梯度中是否都有这个参数
            if all(param_name in grad for grad in client_gradients):
                # 计算权重平均的梯度
                weighted_gradients = torch.stack([
                    grad[param_name] * weight 
                    for grad, weight in zip(client_gradients, weights)
                ], dim=0)
                
                avg_gradient = torch.sum(weighted_gradients, dim=0)
                
                # 应用梯度更新
                aggregated_state[param_name] = server_model_state[param_name] - self.config.lr * avg_gradient
        
        self.logger.info("梯度聚合完成")
        
        return aggregated_state
    
    def update_model(self, new_model_state):
        """
        使用聚合后的模型状态更新模型
        
        Args:
            new_model_state: 聚合后的模型状态
            
        Returns:
            success: 是否成功更新
        """
        try:
            self.model.load_state_dict(new_model_state)
            self.logger.info("模型已更新为聚合后的状态")
            return True
        except Exception as e:
            self.logger.error(f"更新模型失败: {str(e)}")
            return False 
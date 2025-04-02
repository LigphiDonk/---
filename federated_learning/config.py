#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习配置参数
"""

import os

class FederatedConfig:
    """联邦学习配置类"""
    
    def __init__(self):
        # 基本路径配置
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(self.base_dir), 'data')
        
        # 模型配置
        self.model_type = 'simple-cnn'
        self.input_dim = 16 * 4 * 4
        self.hidden_dims = [120, 84]
        self.output_dim = 10
        
        # 训练参数
        self.batch_size = 64
        self.epochs = 2
        self.lr = 0.01
        self.optimizer = 'sgd'
        self.momentum = 0.9
        self.weight_decay = 1e-5
        
        # 联邦学习特定参数
        self.num_rounds = 5              # 联邦学习轮数
        self.num_clients = 3             # 客户端数量
        self.fraction_clients = 1.0      # 每轮参与训练的客户端比例
        self.server_train_ratio = 0.5    # 服务器训练数据比例
        self.client_train_ratio = 0.5    # 每个客户端的训练数据比例
        
        # 通信参数
        self.server_host = 'localhost'
        self.server_port = 8765
        self.timeout = 60                # 通信超时时间（秒）
        
        # 其他参数
        self.seed = 42                   # 随机种子
        self.device = 'cpu'              # 计算设备，'cpu'或'cuda'
        
        # 日志和保存配置
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.save_dir = os.path.join(self.base_dir, 'saved_models')
        self.results_dir = os.path.join(self.base_dir, 'results')
        
        # 确保必要的目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # === 多服务器配置 ===
        self.num_servers = 3             # 服务器数量
        self.server_base_port = 8765     # 服务器基础端口号，每个服务器使用 base_port + server_id
        
        # === PBFT共识配置 ===
        self.view_change_timeout = 10    # 视图更改超时时间（秒）
        self.election_timeout = 5        # 选举超时时间（秒）
        self.heartbeat_interval = 3      # 心跳间隔（秒）
        self.consensus_threshold = 2/3   # 共识阈值，至少需要多少比例的节点达成一致
        self.pbft_log_limit = 1000       # PBFT日志条目限制
        self.checkpoint_interval = 10    # 检查点间隔（轮次）

# 创建默认配置对象
config = FederatedConfig() 
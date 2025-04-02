#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习服务器端主模块
"""

import os
import sys
import time
import torch
import asyncio
import websockets
import json

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from debug_training_fix import create_model, get_mnist_data
from federated_learning.common.utils import setup_logger, set_random_seed, visualize_training_history
from federated_learning.common.protocol import MessageType, Message, serialize_model, deserialize_gradient
from federated_learning.server.trainer import ServerTrainer
from federated_learning.server.aggregator import FederatedAggregator
from federated_learning.config import config

class FederatedServer:
    """联邦学习服务器"""
    
    def __init__(self, config):
        """
        初始化联邦学习服务器
        
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 设置日志记录器
        self.logger = setup_logger(
            "FederatedServer",
            os.path.join(config.log_dir, "server.log")
        )
        
        # 设置随机种子以确保结果可重现
        set_random_seed(config.seed)
        
        self.logger.info(f"使用设备: {config.device}")
        
        # 创建模型
        self.model = create_model()
        self.model.to(config.device)
        self.logger.info(f"创建模型: {type(self.model).__name__}")
        
        # 加载数据
        self.logger.info("加载MNIST数据集")
        self.train_loader, self.test_loader = get_mnist_data(
            data_dir=config.data_dir,
            batch_size=config.batch_size
        )
        
        # 创建训练器
        self.trainer = ServerTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            test_dataloader=self.test_loader,
            config=config
        )
        
        # 创建聚合器
        self.aggregator = FederatedAggregator(
            model=self.model,
            config=config
        )
        
        # 存储客户端连接
        self.clients = {}
        
        # 存储当前轮次
        self.current_round = 0
        
        # 存储客户端梯度
        self.client_gradients = {}
        
        # 训练历史记录
        self.history = {
            "server_train_acc": [],
            "server_test_acc": [],
            "server_loss": [],
            "client_train_acc": [],
            "client_test_acc": [],
            "client_loss": []
        }
        
        self.logger.info("联邦学习服务器初始化完成")
    
    async def handle_client(self, websocket, path):
        """
        处理客户端连接
        
        Args:
            websocket: WebSocket连接
            path: 请求路径
        """
        # 客户端连接
        client_id = str(len(self.clients) + 1)
        self.clients[client_id] = websocket
        self.logger.info(f"客户端 {client_id} 已连接")
        
        try:
            # 发送初始模型给客户端
            await self.send_model(websocket, client_id)
            
            # 持续处理客户端消息
            async for message in websocket:
                msg = Message.from_json(message)
                
                if msg.type == MessageType.GRADIENT:
                    # 接收客户端梯度
                    await self.receive_gradient(client_id, msg.data)
                elif msg.type == MessageType.STOP:
                    # 客户端请求停止
                    self.logger.info(f"客户端 {client_id} 请求停止训练")
                    break
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"客户端 {client_id} 断开连接")
        finally:
            # 客户端断开连接
            if client_id in self.clients:
                del self.clients[client_id]
    
    async def send_model(self, websocket, client_id):
        """
        向客户端发送模型
        
        Args:
            websocket: WebSocket连接
            client_id: 客户端ID
        """
        # 序列化模型
        serialized_model = serialize_model(self.model.state_dict())
        
        # 创建消息
        msg = Message(
            MessageType.MODEL_UPDATE,
            {
                "model": serialized_model,
                "round": self.current_round
            }
        )
        
        # 发送消息
        await websocket.send(msg.to_json())
        self.logger.info(f"已向客户端 {client_id} 发送模型 (轮次 {self.current_round})")
    
    async def receive_gradient(self, client_id, data):
        """
        接收客户端梯度
        
        Args:
            client_id: 客户端ID
            data: 消息数据
        """
        # 获取轮次和梯度
        round_id = data.get("round")
        serialized_gradients = data.get("gradients")
        train_acc = data.get("train_acc", 0)
        test_acc = data.get("test_acc", 0)
        loss = data.get("loss", 0)
        
        # 检查轮次是否匹配
        if round_id != self.current_round:
            self.logger.warning(f"客户端 {client_id} 轮次不匹配: 期望 {self.current_round}, 收到 {round_id}")
            return
        
        # 反序列化梯度
        gradients = deserialize_gradient(serialized_gradients)
        
        # 存储客户端梯度和性能指标
        self.client_gradients[client_id] = gradients
        
        # 记录客户端性能
        if round_id == len(self.history["client_train_acc"]):
            self.history["client_train_acc"].append([])
            self.history["client_test_acc"].append([])
            self.history["client_loss"].append([])
            
        self.history["client_train_acc"][round_id].append(train_acc)
        self.history["client_test_acc"][round_id].append(test_acc)
        self.history["client_loss"][round_id].append(loss)
        
        self.logger.info(f"收到客户端 {client_id} 梯度 (轮次 {round_id})")
        self.logger.info(f"客户端 {client_id} 性能 - 训练准确率: {train_acc:.4f}, "
                          f"测试准确率: {test_acc:.4f}, 损失: {loss:.4f}")
        
        # 检查是否已收到所有客户端的梯度
        if len(self.client_gradients) == len(self.clients):
            self.logger.info(f"已收到所有客户端的梯度，开始聚合 (轮次 {round_id})")
            # 聚合梯度并更新模型
            await self.aggregate_and_update()
    
    async def aggregate_and_update(self):
        """
        聚合梯度并更新模型
        """
        # 训练服务器模型
        server_model_state, train_acc, test_acc = self.trainer.train(self.current_round)
        
        # 记录服务器性能
        self.history["server_train_acc"].append(train_acc)
        self.history["server_test_acc"].append(test_acc)
        self.history["server_loss"].append(self.trainer.history["loss"][-1])
        
        # 聚合服务器模型和客户端梯度
        client_gradients_list = list(self.client_gradients.values())
        aggregated_state = self.aggregator.aggregate(server_model_state, client_gradients_list)
        
        # 更新模型
        self.aggregator.update_model(aggregated_state)
        
        # 保存模型
        save_path = self.trainer.save_model(self.current_round)
        
        # 清空客户端梯度
        self.client_gradients = {}
        
        # 进入下一轮
        self.current_round += 1
        
        if self.current_round < self.config.num_rounds:
            # 更新客户端模型
            for client_id, websocket in self.clients.items():
                await self.send_model(websocket, client_id)
        else:
            # 训练完成
            self.logger.info(f"联邦学习训练完成 ({self.config.num_rounds} 轮)")
            
            # 可视化训练历史
            # 处理客户端数据，计算平均值
            avg_client_train_acc = [sum(accs)/len(accs) if accs else 0 for accs in self.history["client_train_acc"]]
            avg_client_test_acc = [sum(accs)/len(accs) if accs else 0 for accs in self.history["client_test_acc"]]
            avg_client_loss = [sum(losses)/len(losses) if losses else 0 for losses in self.history["client_loss"]]
            
            history_data = {
                "server_train_acc": self.history["server_train_acc"],
                "server_test_acc": self.history["server_test_acc"],
                "server_loss": self.history["server_loss"],
                "client_train_acc": avg_client_train_acc,
                "client_test_acc": avg_client_test_acc,
                "client_loss": avg_client_loss
            }
            
            visualize_path = visualize_training_history(
                history_data,
                save_path=os.path.join(self.config.results_dir, "federated_training_history.png")
            )
            
            self.logger.info(f"训练历史已可视化并保存到 {visualize_path}")
            
            # 向所有客户端发送停止消息
            stop_msg = Message(MessageType.STOP, {"message": "训练完成"})
            for client_id, websocket in self.clients.items():
                await websocket.send(stop_msg.to_json())
    
    async def start(self):
        """
        启动服务器
        """
        self.logger.info(f"启动服务器，监听 {self.config.server_host}:{self.config.server_port}")
        
        # 启动WebSocket服务器
        async with websockets.serve(
            self.handle_client,
            self.config.server_host,
            self.config.server_port
        ):
            # 保持服务器运行
            await asyncio.Future()
            
def main():
    """主函数"""
    # 创建服务器
    server = FederatedServer(config)
    
    # 启动服务器
    asyncio.run(server.start())

if __name__ == "__main__":
    main() 
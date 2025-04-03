#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习客户端主模块
"""

import os
import sys
import time
import torch
import asyncio
import websockets
import json
import random

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from debug_training_fix import create_model, get_mnist_data
from federated_learning.common.utils import setup_logger, set_random_seed
from federated_learning.common.protocol import MessageType, Message, serialize_gradient, deserialize_model
from federated_learning.client.trainer import ClientTrainer
from federated_learning.config import config

class FederatedClient:
    """联邦学习客户端"""
    
    def __init__(self, client_id, config):
        """
        初始化联邦学习客户端
        
        Args:
            client_id: 客户端ID
            config: 配置对象
        """
        self.client_id = client_id
        self.config = config
        
        # 设置日志记录器
        self.logger = setup_logger(
            f"FederatedClient_{client_id}",
            os.path.join(config.log_dir, f"client_{client_id}.log")
        )
        
        # 设置随机种子
        seed = config.seed + int(client_id)  # 为每个客户端设置不同的随机种子
        set_random_seed(seed)
        
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
        self.trainer = ClientTrainer(
            client_id=client_id,
            model=self.model,
            train_dataloader=self.train_loader,
            test_dataloader=self.test_loader,
            config=config
        )
        
        # 当前轮次
        self.current_round = 0
        
        # 是否正在训练
        self.is_training = False
        
        self.logger.info(f"联邦学习客户端 {client_id} 初始化完成")
    
    async def connect_to_server(self):
        """
        连接到服务器
        
        Returns:
            websocket: WebSocket连接
        """
        # 检查环境变量中是否有主服务器端口
        primary_port = os.environ.get('FL_PRIMARY_SERVER_PORT')
        if primary_port:
            self.logger.info(f"从环境变量中获取主服务器端口: {primary_port}")
            uri = f"ws://{self.config.server_host}:{primary_port}"
            self.logger.info(f"直接连接到主服务器: {uri}")
            
            try:
                websocket = await websockets.connect(uri)
                
                # 发送初始化消息
                init_msg = Message(
                    MessageType.INIT,
                    {"client_id": self.client_id}
                )
                await websocket.send(init_msg.to_json())
                
                # 接收响应
                resp = await websocket.recv()
                resp_msg = Message.from_json(resp)
                
                self.logger.info("成功连接到主服务器")
                return websocket
            except Exception as e:
                self.logger.error(f"连接主服务器失败: {str(e)}")
                # 继续尝试普通连接方式
        
        # 先尝试连接默认服务器
        uri = f"ws://{self.config.server_host}:{self.config.server_base_port + 1}"  # 连接到服务器1的端口（8766）
        self.logger.info(f"正在连接到服务器1: {uri}")
        
        try:
            # 更详细的异常捕获和日志记录
            try:
                websocket = await websockets.connect(uri)
                self.logger.info(f"WebSocket连接创建成功: {uri}")
            except Exception as e:
                self.logger.error(f"WebSocket连接失败: {uri}, 错误: {str(e)}")
                raise e
            
            # 发送初始化消息
            init_msg = Message(
                MessageType.INIT,
                {"client_id": self.client_id}
            )
            
            try:
                await websocket.send(init_msg.to_json())
                self.logger.info(f"发送初始化消息成功: {init_msg.to_json()}")
            except Exception as e:
                self.logger.error(f"发送初始化消息失败: {str(e)}")
                await websocket.close()
                raise e
            
            # 接收响应
            try:
                resp = await websocket.recv()
                self.logger.info(f"收到服务器响应: {resp}")
                resp_msg = Message.from_json(resp)
            except Exception as e:
                self.logger.error(f"接收或解析响应失败: {str(e)}")
                await websocket.close()
                raise e
            
            # 检查是否被拒绝（不是主服务器）
            if resp_msg.type == MessageType.INIT and resp_msg.data.get("status") == "rejected":
                reason = resp_msg.data.get("reason", "unknown")
                self.logger.info(f"连接被拒绝，原因: {reason}")
                
                if reason == "not_primary":
                    primary_id = resp_msg.data.get("primary_id")
                    if primary_id:
                        # 计算主服务器地址
                        primary_port = self.config.server_base_port + int(primary_id)
                        primary_uri = f"ws://{self.config.server_host}:{primary_port}"
                        self.logger.info(f"重定向到主服务器: {primary_uri}")
                        
                        # 关闭当前连接
                        await websocket.close()
                        
                        # 连接到主服务器
                        try:
                            websocket = await websockets.connect(primary_uri)
                            self.logger.info(f"主服务器WebSocket连接创建成功: {primary_uri}")
                        except Exception as e:
                            self.logger.error(f"连接主服务器失败: {primary_uri}, 错误: {str(e)}")
                            raise e
                        
                        try:
                            await websocket.send(init_msg.to_json())
                            self.logger.info(f"向主服务器发送初始化消息成功")
                        except Exception as e:
                            self.logger.error(f"向主服务器发送初始化消息失败: {str(e)}")
                            await websocket.close()
                            raise e
                        
                        # 接收响应但不处理，假设主服务器会接受连接
                        try:
                            resp = await websocket.recv()
                            self.logger.info(f"收到主服务器响应: {resp}")
                        except Exception as e:
                            self.logger.error(f"接收主服务器响应失败: {str(e)}")
                            await websocket.close()
                            raise e
            
            self.logger.info("成功连接到服务器")
            return websocket
        except Exception as e:
            self.logger.error(f"连接服务器失败: {str(e)}")
            
            # 如果配置了多服务器，尝试连接其他服务器
            if hasattr(self.config, 'num_servers') and self.config.num_servers > 1:
                self.logger.info(f"尝试连接其他服务器，共 {self.config.num_servers} 个服务器")
                
                for server_id in range(1, self.config.num_servers + 1):
                    # 跳过已尝试过的默认服务器
                    default_port = self.config.server_port
                    current_port = self.config.server_base_port + server_id
                    
                    if current_port == default_port:
                        self.logger.info(f"跳过已尝试过的默认服务器 {server_id}")
                        continue
                    
                    server_uri = f"ws://{self.config.server_host}:{current_port}"
                    self.logger.info(f"尝试连接到服务器 {server_id}: {server_uri}")
                    
                    try:
                        # 更详细的异常捕获
                        try:
                            websocket = await websockets.connect(server_uri)
                            self.logger.info(f"WebSocket连接创建成功: {server_uri}")
                        except Exception as e:
                            self.logger.error(f"连接服务器 {server_id} 失败: {server_uri}, 错误: {str(e)}")
                            continue
                        
                        # 发送初始化消息
                        init_msg = Message(
                            MessageType.INIT,
                            {"client_id": self.client_id}
                        )
                        
                        try:
                            await websocket.send(init_msg.to_json())
                            self.logger.info(f"发送初始化消息成功: {init_msg.to_json()}")
                        except Exception as e:
                            self.logger.error(f"发送初始化消息失败: {str(e)}")
                            await websocket.close()
                            continue
                        
                        # 接收响应
                        try:
                            resp = await websocket.recv()
                            self.logger.info(f"收到服务器 {server_id} 响应: {resp}")
                            resp_msg = Message.from_json(resp)
                        except Exception as e:
                            self.logger.error(f"接收或解析服务器 {server_id} 响应失败: {str(e)}")
                            await websocket.close()
                            continue
                        
                        # 检查是否被拒绝（不是主服务器）
                        if resp_msg.type == MessageType.INIT and resp_msg.data.get("status") == "rejected":
                            reason = resp_msg.data.get("reason", "unknown")
                            self.logger.info(f"连接被服务器 {server_id} 拒绝，原因: {reason}")
                            
                            if reason == "not_primary":
                                primary_id = resp_msg.data.get("primary_id")
                                if primary_id:
                                    # 计算主服务器地址
                                    primary_port = self.config.server_base_port + int(primary_id)
                                    primary_uri = f"ws://{self.config.server_host}:{primary_port}"
                                    self.logger.info(f"被重定向到主服务器: {primary_uri}")
                                    
                                    # 关闭当前连接
                                    await websocket.close()
                                    
                                    # 连接到主服务器
                                    try:
                                        websocket = await websockets.connect(primary_uri)
                                        self.logger.info(f"主服务器WebSocket连接创建成功: {primary_uri}")
                                    except Exception as e:
                                        self.logger.error(f"连接主服务器失败: {primary_uri}, 错误: {str(e)}")
                                        continue
                                    
                                    try:
                                        await websocket.send(init_msg.to_json())
                                        self.logger.info(f"向主服务器发送初始化消息成功")
                                    except Exception as e:
                                        self.logger.error(f"向主服务器发送初始化消息失败: {str(e)}")
                                        await websocket.close()
                                        continue
                                    
                                    # 接收响应但不处理，假设主服务器会接受连接
                                    try:
                                        resp = await websocket.recv()
                                        self.logger.info(f"收到主服务器响应: {resp}")
                                    except Exception as e:
                                        self.logger.error(f"接收主服务器响应失败: {str(e)}")
                                        await websocket.close()
                                        continue
                        
                        self.logger.info(f"成功连接到服务器 {server_id}")
                        return websocket
                    except Exception as e:
                        self.logger.error(f"连接服务器 {server_id} 过程中发生错误: {str(e)}")
            
            # 如果所有尝试都失败，抛出异常
            self.logger.error("所有服务器连接尝试均失败")
            raise Exception("无法连接到任何服务器")
    
    async def receive_model(self, websocket):
        """
        从服务器接收模型
        
        Args:
            websocket: WebSocket连接
            
        Returns:
            round_id: 轮次ID
        """
        try:
            # 接收消息
            message = await websocket.recv()
            msg = Message.from_json(message)
            
            if msg.type == MessageType.MODEL_UPDATE:
                # 接收模型更新
                round_id = msg.data.get("round")
                serialized_model = msg.data.get("model")
                
                # 反序列化模型
                model_state = deserialize_model(serialized_model)
                
                # 更新本地模型
                self.trainer.update_model(model_state)
                
                self.logger.info(f"已接收服务器模型 (轮次 {round_id})")
                
                return round_id
            elif msg.type == MessageType.STOP:
                # 训练停止
                self.logger.info("服务器通知停止训练")
                return None
            elif msg.type == MessageType.FINISH:
                # 服务器通知联邦学习已完成
                message = msg.data.get("message", "联邦学习已完成")
                self.logger.info(f"收到服务器完成消息: {message}")
                self.logger.info("===================================")
                self.logger.info("=== 联邦学习已完成，客户端将退出 ===")
                self.logger.info("===================================")
                return None
            else:
                self.logger.warning(f"收到未知类型的消息: {msg.type}")
                return None
        except Exception as e:
            self.logger.error(f"接收模型失败: {str(e)}")
            raise e
    
    async def send_gradient(self, websocket, round_id, gradients, train_acc, test_acc, loss):
        """
        向服务器发送梯度
        
        Args:
            websocket: WebSocket连接
            round_id: 轮次ID
            gradients: 梯度字典
            train_acc: 训练准确率
            test_acc: 测试准确率
            loss: 训练损失
        """
        try:
            # 序列化梯度
            serialized_gradients = serialize_gradient(gradients)
            
            # 创建消息
            msg = Message(
                MessageType.GRADIENT,
                {
                    "round": round_id,
                    "gradients": serialized_gradients,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "loss": loss
                }
            )
            
            # 发送消息
            await websocket.send(msg.to_json())
            self.logger.info(f"已向服务器发送梯度 (轮次 {round_id})")
        except Exception as e:
            self.logger.error(f"发送梯度失败: {str(e)}")
            raise e
    
    async def start(self):
        """
        启动客户端
        """
        try:
            # 连接到服务器
            websocket = await self.connect_to_server()
            
            # 持续运行
            while True:
                # 接收模型
                round_id = await self.receive_model(websocket)
                
                if round_id is None:
                    self.logger.info("训练结束")
                    break
                
                self.current_round = round_id
                
                # 训练模型
                self.is_training = True
                gradients, train_acc, test_acc, loss = self.trainer.train(round_id)
                self.is_training = False
                
                # 保存模型
                self.trainer.save_model(round_id)
                
                # 发送梯度
                await self.send_gradient(websocket, round_id, gradients, train_acc, test_acc, loss)
                
            # 关闭连接
            await websocket.close()
            self.logger.info("与服务器的连接已关闭")
            
        except Exception as e:
            self.logger.error(f"客户端运行时出错: {str(e)}")
            raise e

async def run_client(client_id, config):
    """
    运行联邦学习客户端
    
    Args:
        client_id: 客户端ID
        config: 配置对象
    """
    # 设置日志记录器
    logger = setup_logger(f"Client-{client_id}", os.path.join(config.log_dir, f"client_run_{client_id}.log"))
    
    try:
        # 创建客户端
        client = FederatedClient(client_id, config)
        
        # 启动客户端
        await client.start()
    except asyncio.CancelledError:
        # 捕获CancelledError并正常退出
        logger.info(f"客户端 {client_id} 正在优雅地关闭...")
    except Exception as e:
        logger.error(f"客户端 {client_id} 发生错误: {str(e)}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='联邦学习客户端')
    parser.add_argument('--id', type=str, required=True, help='客户端ID')
    args = parser.parse_args()
    
    client_id = args.id
    
    # 启动客户端
    asyncio.run(run_client(client_id, config))

if __name__ == "__main__":
    main() 
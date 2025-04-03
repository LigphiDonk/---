#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习服务器集群管理模块

此模块用于协调多服务器联邦学习，包括：
1. 服务器之间的通信
2. 基于PBFT的共识机制
3. 随机选择主服务器进行联邦学习
"""

import os
import sys
import time
import json
import asyncio
import random
import websockets
import torch
from copy import deepcopy

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from federated_learning.common.utils import setup_logger, set_random_seed
from federated_learning.common.protocol import MessageType, Message, serialize_model, deserialize_model
from federated_learning.common.pbft import PBFTNode, PBFTMessage, PBFTMessageType
from federated_learning.server.server import FederatedServer
from federated_learning.config import config

class ServerMessage:
    """服务器间通信消息"""
    
    def __init__(self, msg_type, sender_id, data=None):
        self.type = msg_type
        self.sender_id = sender_id
        self.data = data if data is not None else {}
    
    def to_json(self):
        """将消息转换为JSON格式"""
        return json.dumps({
            "type": self.type,
            "sender_id": self.sender_id,
            "data": self.data
        })
    
    @classmethod
    def from_json(cls, json_str):
        """从JSON字符串创建消息"""
        data = json.loads(json_str)
        return cls(data["type"], data["sender_id"], data["data"])

class ServerMessageType:
    """服务器间通信消息类型"""
    CONNECT = "connect"              # 服务器连接
    DISCONNECT = "disconnect"        # 服务器断开连接
    HEARTBEAT = "heartbeat"          # 心跳检测
    SERVER_LIST = "server_list"      # 服务器列表更新
    SYNC_MODEL = "sync_model"        # 模型同步
    PRIMARY_ELECTED = "primary"      # 主服务器选举结果
    START_FL_ROUND = "start_fl"      # 开始联邦学习轮次
    END_FL_ROUND = "end_fl"          # 结束联邦学习轮次
    FL_RESULT = "fl_result"          # 联邦学习结果

class ServerCluster:
    """联邦学习服务器集群"""
    
    def __init__(self, server_id, config):
        """
        初始化服务器集群
        
        Args:
            server_id: 服务器ID
            config: 配置对象
        """
        self.server_id = server_id
        self.config = config
        
        # 设置日志记录器
        self.logger = setup_logger(
            f"ServerCluster-{server_id}",
            os.path.join(config.log_dir, f"server_cluster_{server_id}.log")
        )
        
        # 设置随机种子
        set_random_seed(config.seed + int(server_id))
        
        # 创建联邦学习服务器
        self.fl_server = FederatedServer(config)
        
        # 方便访问的引用
        self.trainer = self.fl_server.trainer
        self.aggregator = self.fl_server.aggregator
        
        # 创建PBFT节点
        self.pbft_node = PBFTNode(server_id, config, logger=self.logger)
        
        # 设置PBFT节点的回调函数
        self.pbft_node.on_election_completed = self.on_primary_elected
        
        # 服务器连接（WebSocket连接）
        self.server_connections = {}
        
        # 已连接的客户端
        self.connected_clients = {}
        
        # 同时充当集群服务器和集群客户端
        self.server = None
        self.clients = {}
        
        # 当前联邦学习轮次
        self.current_round = 0
        
        # 是否是主服务器
        self.is_primary = False
        
        # 联邦学习状态
        self.fl_in_progress = False
        
        # 保存其他服务器的地址
        self.server_addresses = {}
        
        # 记录已处理的选举消息，避免消息循环
        self.processed_election_messages = set()
        
        self.logger.info(f"服务器集群节点 {server_id} 初始化完成")
    
    async def start_server(self):
        """启动服务器，监听其他服务器和客户端的连接"""
        # 计算服务器监听地址
        host = "localhost"
        port = self.config.server_base_port + int(self.server_id)
        
        # 创建WebSocket服务器
        self.server = await websockets.serve(
            self.handle_connection,
            host,
            port
        )
        
        self.logger.info(f"服务器集群节点 {self.server_id} 启动，监听地址: {host}:{port}")
        
        # 连接到其他服务器
        await self.connect_to_other_servers()
        
        # 等待所有服务器连接完成
        # 如果是服务器1，则负责检测所有服务器是否已连接并开始选举
        if self.server_id == "1":
            self.logger.info("服务器1启动，等待所有服务器连接后开始选举")
            # 创建定时任务检查服务器连接状态
            asyncio.create_task(self.check_all_servers_connected())
    
    async def connect_to_other_servers(self):
        """连接到其他服务器节点"""
        self.logger.info("尝试连接到其他服务器节点")
        
        # 获取服务器数量
        num_servers = self.config.num_servers
        
        # 连接到ID小于自己的服务器
        for i in range(1, int(self.server_id)):
            server_id = str(i)
            host = "localhost"
            port = self.config.server_base_port + i
            uri = f"ws://{host}:{port}"
            
            self.server_addresses[server_id] = uri
            
            try:
                self.logger.info(f"连接到服务器 {server_id}: {uri}")
                websocket = await websockets.connect(uri)
                self.server_connections[server_id] = websocket
                
                # 发送连接消息
                connect_msg = ServerMessage(
                    ServerMessageType.CONNECT,
                    self.server_id,
                    {"server_id": self.server_id}
                )
                await websocket.send(connect_msg.to_json())
                
                # 设置消息处理任务
                asyncio.create_task(self.handle_server_messages(server_id, websocket))
                
                # 更新PBFT节点的服务器连接
                self.pbft_node.server_nodes[server_id] = websocket
                
            except Exception as e:
                self.logger.error(f"连接到服务器 {server_id} 失败: {str(e)}")
    
    async def handle_connection(self, websocket):
        """
        处理新连接
        
        Args:
            websocket: WebSocket连接
        """
        try:
            # 记录连接信息
            self.logger.info(f"收到新连接")
            
            # 接收第一个消息以确定连接类型
            message = await websocket.recv()
            
            # 尝试解析为服务器消息
            try:
                server_msg = ServerMessage.from_json(message)
                
                # 处理服务器连接
                if server_msg.type == ServerMessageType.CONNECT:
                    server_id = server_msg.data.get("server_id")
                    self.logger.info(f"识别为服务器连接，ID: {server_id}")
                    await self.handle_server_connection(server_id, websocket)
                    return
            except Exception as e:
                self.logger.debug(f"解析服务器消息失败: {str(e)}")
                pass
            
            # 尝试解析为客户端消息
            try:
                client_msg = Message.from_json(message)
                
                # 如果是客户端连接
                if client_msg.type == MessageType.INIT:
                    client_id = client_msg.data.get("client_id")
                    self.logger.info(f"识别为客户端连接，ID: {client_id}")
                    await self.handle_client_connection(client_id, websocket, first_message=client_msg)
                    return
            except Exception as e:
                self.logger.debug(f"解析客户端消息失败: {str(e)}")
                pass
            
            # 无法识别的连接类型
            self.logger.warning(f"收到无法识别的连接: {message[:100]}...")  # 只记录消息的前100个字符
            await websocket.close(1003, "Unrecognized connection type")
            
        except Exception as e:
            self.logger.error(f"处理连接时出错: {str(e)}")
            await websocket.close(1011, "Internal error")
    
    async def handle_server_connection(self, server_id, websocket):
        """
        处理服务器连接
        
        Args:
            server_id: 服务器ID
            websocket: WebSocket连接
        """
        self.logger.info(f"服务器 {server_id} 已连接")
        
        self.server_connections[server_id] = websocket
        
        # 更新PBFT节点的服务器连接
        self.pbft_node.server_nodes[server_id] = websocket
        
        # 发送当前服务器列表
        servers_list = list(self.server_connections.keys()) + [self.server_id]
        server_list_msg = ServerMessage(
            ServerMessageType.SERVER_LIST,
            self.server_id,
            {"servers": servers_list}
        )
        await websocket.send(server_list_msg.to_json())
        
        # 如果不是服务器1，向服务器1广播连接状态更新
        if self.server_id != "1" and "1" in self.server_connections:
            self.logger.info(f"向服务器1发送连接状态更新")
            status_msg = ServerMessage(
                ServerMessageType.SERVER_LIST,
                self.server_id,
                {"servers": servers_list, "connected": True}
            )
            await self.server_connections["1"].send(status_msg.to_json())
        
        try:
            # 处理服务器消息
            await self.handle_server_messages(server_id, websocket)
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"服务器 {server_id} 断开连接")
        finally:
            # 移除服务器连接
            if server_id in self.server_connections:
                del self.server_connections[server_id]
            
            # 更新PBFT节点的服务器连接
            if server_id in self.pbft_node.server_nodes:
                del self.pbft_node.server_nodes[server_id]
            
            # 如果是主服务器断开连接，启动新的选举
            if self.pbft_node.elected_primary == server_id:
                self.logger.info(f"主服务器 {server_id} 断开连接，启动新的选举")
                await self.pbft_node.start_election()
    
    async def handle_client_connection(self, client_id, websocket, first_message=None):
        """
        处理客户端连接
        
        Args:
            client_id: 客户端ID
            websocket: WebSocket连接
            first_message: 第一个消息
        """
        try:
            # 检查是否已经完成选举
            if not self.pbft_node.elected_primary and self.server_id == "1":
                # 如果是服务器1且没有选出主服务器，则强制选择
                self.logger.warning(f"收到客户端连接但尚未选出主服务器，强制选择一个主服务器")
                self.force_select_primary()
            
            # 转发给联邦学习服务器
            if self.is_primary:
                # 如果当前服务器是主服务器，直接处理客户端
                self.connected_clients[client_id] = websocket
                
                # 发送接受连接的消息
                accept_msg = Message(
                    MessageType.INIT,
                    {"status": "accepted", "primary_id": self.server_id}
                )
                await websocket.send(accept_msg.to_json())
                
                # 转发到FL服务器
                self.logger.info(f"将客户端 {client_id} 连接转发到联邦学习服务器")
                try:
                    await self.fl_server.handle_client(websocket, "")
                except Exception as e:
                    self.logger.error(f"处理客户端 {client_id} 时出错: {str(e)}")
            else:
                # 如果不是主服务器，告知客户端主服务器地址
                if not self.pbft_node.elected_primary:
                    # 如果没有选出主服务器，告知客户端稍后重试
                    self.logger.warning(f"收到客户端 {client_id} 连接，但尚未选出主服务器")
                    reject_msg = Message(
                        MessageType.INIT,
                        {
                            "status": "rejected", 
                            "reason": "no_primary", 
                            "message": "No primary server elected yet, please try again later"
                        }
                    )
                else:
                    self.logger.info(f"收到客户端 {client_id} 连接，但当前不是主服务器，重定向到主服务器 {self.pbft_node.elected_primary}")
                    # 发送拒绝消息，指明主服务器
                    reject_msg = Message(
                        MessageType.INIT,
                        {
                            "status": "rejected", 
                            "reason": "not_primary", 
                            "primary_id": self.pbft_node.elected_primary
                        }
                    )
                await websocket.send(reject_msg.to_json())
                await websocket.close()
        except Exception as e:
            self.logger.error(f"处理客户端连接时出错: {str(e)}")
            try:
                await websocket.close(1011, "Internal server error")
            except:
                pass
    
    async def handle_server_messages(self, server_id, websocket):
        """
        处理来自服务器的消息
        
        Args:
            server_id: 服务器ID
            websocket: WebSocket连接
        """
        try:
            async for message in websocket:
                # 尝试解析为服务器消息
                try:
                    server_msg = ServerMessage.from_json(message)
                    await self.process_server_message(server_id, server_msg)
                    continue
                except:
                    pass
                
                # 尝试解析为PBFT消息
                try:
                    pbft_msg = PBFTMessage.from_json(message)
                    await self.pbft_node.process_message(pbft_msg)
                    continue
                except:
                    pass
                
                self.logger.warning(f"收到来自服务器 {server_id} 的无法识别的消息")
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"服务器 {server_id} 断开连接")
            # 处理断开连接
            if server_id in self.server_connections:
                del self.server_connections[server_id]
            
            # 更新PBFT节点的服务器连接
            if server_id in self.pbft_node.server_nodes:
                del self.pbft_node.server_nodes[server_id]
    
    async def process_server_message(self, server_id, message):
        """
        处理服务器消息
        
        Args:
            server_id: 服务器ID
            message: 服务器消息
        """
        msg_type = message.type
        
        if msg_type == ServerMessageType.CONNECT:
            # 处理服务器连接消息（应该已经在handle_connection中处理）
            pass
        
        elif msg_type == ServerMessageType.DISCONNECT:
            # 处理服务器断开连接消息
            self.logger.info(f"服务器 {server_id} 主动断开连接")
            
            # 移除服务器连接
            if server_id in self.server_connections:
                await self.server_connections[server_id].close()
                del self.server_connections[server_id]
            
            # 更新PBFT节点的服务器连接
            if server_id in self.pbft_node.server_nodes:
                del self.pbft_node.server_nodes[server_id]
        
        elif msg_type == ServerMessageType.SERVER_LIST:
            # 处理服务器列表更新消息
            server_list = message.data.get("servers", [])
            connected_status = message.data.get("connected", False)
            self.logger.info(f"收到服务器列表更新: {server_list}, 连接状态变更: {connected_status}")
            
            # 连接到列表中新增的服务器
            for new_server_id in server_list:
                if (new_server_id != self.server_id and 
                    new_server_id not in self.server_connections and
                    int(new_server_id) > int(self.server_id)):
                    
                    # 计算服务器地址
                    host = "localhost"
                    port = self.config.server_base_port + int(new_server_id)
                    uri = f"ws://{host}:{port}"
                    
                    self.server_addresses[new_server_id] = uri
                    
                    self.logger.info(f"尝试连接到服务器列表中的新服务器 {new_server_id}: {uri}")
                    asyncio.create_task(self.connect_to_server(new_server_id, uri))
            
            # 如果是服务器1，并且收到连接状态更新，检查是否所有服务器都已连接
            if self.server_id == "1" and connected_status and not self.pbft_node.elected_primary:
                asyncio.create_task(self.check_all_servers_connected())
        
        elif msg_type == ServerMessageType.PRIMARY_ELECTED:
            # 处理主服务器选举结果
            primary_id = message.data.get("primary_id")
            force_selected = message.data.get("force_selected", False)
            
            # 创建消息唯一标识，用于去重
            message_id = f"{primary_id}_{server_id}_{force_selected}"
            
            # 检查是否已处理过该消息，避免消息循环
            if message_id in self.processed_election_messages:
                self.logger.debug(f"忽略重复的选举消息: {message_id}")
                return
            
            # 记录此消息已被处理
            self.processed_election_messages.add(message_id)
            
            # 根据消息类型记录不同的日志
            if force_selected:
                self.logger.warning(f"收到强制选择的主服务器结果: {primary_id}, 来自: {server_id}")
            else:
                self.logger.info(f"收到PBFT共识选举的主服务器结果: {primary_id}, 来自: {server_id}")
            
            # 更新PBFT节点状态，强制接受选举结果
            old_primary = self.pbft_node.elected_primary
            self.pbft_node.elected_primary = primary_id
            self.pbft_node.election_in_progress = False
            
            # 如果已经有相同的主服务器，不需要进一步处理
            if old_primary == primary_id and old_primary is not None:
                self.logger.debug(f"主服务器没有变化，仍为: {primary_id}")
                return
            
            # 更新主服务器状态
            old_is_primary = self.is_primary
            self.is_primary = (primary_id == self.server_id)
            
            # 如果主服务器状态发生变化，记录并处理
            if self.is_primary != old_is_primary:
                if self.is_primary:
                    self.logger.info("====================================================")
                    self.logger.info("====== 本服务器被选为主服务器! ======")
                    self.logger.info("====================================================")
                    
                    # 启动联邦学习进程
                    if not self.fl_in_progress:
                        self.logger.info("====================================================")
                        self.logger.info("====== 主服务器即将开始联邦学习过程 ======")
                        self.logger.info("====================================================")
                        asyncio.create_task(self.run_federated_learning())
                else:
                    self.logger.info("====================================================")
                    self.logger.info(f"====== 服务器 {primary_id} 被选为主服务器 ======")
                    self.logger.info("====================================================")
            
            # 只向尚未发送过的服务器转发一次
            for other_id, other_ws in self.server_connections.items():
                if other_id != server_id:  # 不发送给消息来源
                    # 为每个转发目标创建一个唯一标识
                    forward_id = f"{primary_id}_{other_id}_{force_selected}"
                    if forward_id in self.processed_election_messages:
                        self.logger.debug(f"跳过向{other_id}的重复转发")
                        continue
                    
                    # 记录此转发已完成
                    self.processed_election_messages.add(forward_id)
                    
                    try:
                        await other_ws.send(json.dumps({
                            "type": ServerMessageType.PRIMARY_ELECTED,
                            "sender_id": self.server_id,
                            "data": {"primary_id": primary_id, "force_selected": force_selected}
                        }))
                        self.logger.debug(f"转发选举结果到服务器 {other_id}")
                    except Exception as e:
                        self.logger.error(f"转发选举结果到服务器 {other_id} 失败: {str(e)}")
        
        elif msg_type == ServerMessageType.START_FL_ROUND:
            # 处理开始联邦学习轮次消息
            round_id = message.data.get("round_id", 0)
            self.logger.info(f"收到开始联邦学习轮次消息: 轮次 {round_id}")
            
            # 如果是主服务器，则开始联邦学习
            if self.is_primary:
                self.fl_in_progress = True
                self.current_round = round_id
                
                # 这里可以添加开始联邦学习的逻辑
                self.logger.info(f"主服务器开始联邦学习轮次 {round_id}")
            
        elif msg_type == ServerMessageType.END_FL_ROUND:
            # 处理结束联邦学习轮次消息
            round_id = message.data.get("round_id", 0)
            self.logger.info(f"收到结束联邦学习轮次消息: 轮次 {round_id}")
            
            # 结束联邦学习
            self.fl_in_progress = False
            
            # 如果是主服务器，则处理联邦学习结果
            if self.is_primary:
                self.logger.info(f"主服务器结束联邦学习轮次 {round_id}")
        
        elif msg_type == ServerMessageType.SYNC_MODEL:
            # 处理模型同步消息
            round_id = message.data.get("round_id", 0)
            serialized_model = message.data.get("model")
            
            self.logger.info(f"收到模型同步消息: 轮次 {round_id}")
            
            if serialized_model:
                # 反序列化模型
                model_state = deserialize_model(serialized_model)
                
                # 更新本地模型
                self.fl_server.model.load_state_dict(model_state)
                self.logger.info(f"已同步轮次 {round_id} 的模型")
            else:
                self.logger.warning(f"模型同步消息中没有模型数据")
    
    async def connect_to_server(self, server_id, uri):
        """
        连接到指定服务器
        
        Args:
            server_id: 服务器ID
            uri: 服务器URI
        """
        try:
            self.logger.info(f"连接到服务器 {server_id}: {uri}")
            websocket = await websockets.connect(uri)
            
            # 发送连接消息
            connect_msg = ServerMessage(
                ServerMessageType.CONNECT,
                self.server_id,
                {"server_id": self.server_id}
            )
            await websocket.send(connect_msg.to_json())
            
            # 保存连接
            self.server_connections[server_id] = websocket
            
            # 更新PBFT节点的服务器连接
            self.pbft_node.server_nodes[server_id] = websocket
            
            # 设置消息处理任务
            asyncio.create_task(self.handle_server_messages(server_id, websocket))
            
            self.logger.info(f"成功连接到服务器 {server_id}")
            
        except Exception as e:
            self.logger.error(f"连接到服务器 {server_id} 失败: {str(e)}")
    
    async def on_primary_elected(self, primary_id):
        """
        当PBFT选举完成时的回调函数
        
        Args:
            primary_id: 主服务器ID
        """
        self.logger.info("====================================================")
        self.logger.info(f"PBFT共识选举完成！结果: 服务器 {primary_id} 被选为主服务器")
        self.logger.info("====================================================")
        
        # 更新主服务器状态
        self.is_primary = (primary_id == self.server_id)
        
        if self.is_primary:
            self.logger.info("====================================================")
            self.logger.info("====== 本服务器被选为主服务器! ======")
            self.logger.info("====================================================")
            
            # 启动联邦学习进程
            if not self.fl_in_progress:
                self.logger.info("====================================================")
                self.logger.info("====== 主服务器即将开始联邦学习过程 ======")
                self.logger.info("====================================================")
                asyncio.create_task(self.run_federated_learning())
        else:
            self.logger.info("====================================================")
            self.logger.info(f"====== 服务器 {primary_id} 被选为主服务器 ======")
            self.logger.info("====================================================")
        
        # 广播选举结果给所有服务器
        message = ServerMessage(
            ServerMessageType.PRIMARY_ELECTED,
            self.server_id,
            {"primary_id": primary_id}
        )
        
        self.logger.info(f"广播主服务器选举结果: {primary_id}")
        await self.broadcast_server_message(ServerMessageType.PRIMARY_ELECTED, {"primary_id": primary_id})
    
    async def broadcast_server_message(self, msg_type, data=None):
        """
        广播服务器消息给所有连接的服务器
        
        Args:
            msg_type: 消息类型
            data: 消息数据
        """
        message = ServerMessage(msg_type, self.server_id, data)
        json_msg = message.to_json()
        
        # 处理PRIMARY_ELECTED消息的特殊逻辑，确保去重
        if msg_type == ServerMessageType.PRIMARY_ELECTED:
            primary_id = data.get("primary_id")
            force_selected = data.get("force_selected", False)
            
            tasks = []
            for server_id, websocket in self.server_connections.items():
                # 创建消息唯一标识
                message_id = f"{primary_id}_{server_id}_{force_selected}"
                
                # 检查是否已经向该服务器发送过此消息
                if message_id in self.processed_election_messages:
                    self.logger.debug(f"跳过向{server_id}广播已发送的选举消息")
                    continue
                
                # 记录此消息已被处理
                self.processed_election_messages.add(message_id)
                
                self.logger.debug(f"广播选举消息到服务器 {server_id}")
                task = asyncio.create_task(websocket.send(json_msg))
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks)
            return
        
        # 其他类型消息的常规广播
        tasks = []
        for server_id, websocket in self.server_connections.items():
            self.logger.debug(f"广播 {msg_type} 消息到服务器 {server_id}")
            task = asyncio.create_task(websocket.send(json_msg))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def run_federated_learning(self):
        """
        运行联邦学习进程
        
        这是一个包装方法，实际调用start_federated_learning
        """
        self.logger.info("====================================================")
        self.logger.info("====== 联邦学习过程正式开始 ======")
        self.logger.info("====================================================")
        await self.start_federated_learning()

    async def start_federated_learning(self):
        """主服务器开始联邦学习过程"""
        if not self.is_primary:
            self.logger.warning("只有主服务器可以启动联邦学习")
            return
        
        self.logger.info("====================================================")
        self.logger.info("====== 主服务器正式开始联邦学习过程 ======")
        self.logger.info("====================================================")
        
        # 重置联邦学习状态
        self.fl_in_progress = True
        self.current_round = 0
        
        # 训练历史记录
        history = {
            "server_train_acc": [],
            "server_test_acc": [],
            "server_loss": [],
            "client_train_acc": [[]],  # 只有一轮
            "client_test_acc": [[]],   # 只有一轮
            "client_loss": [[]]        # 只有一轮
        }
        
        # 等待客户端连接
        self.logger.info("等待客户端连接...")
        await asyncio.sleep(3)  # 给客户端连接的时间
        
        # 检查连接的客户端数量
        if not self.connected_clients:
            self.logger.warning("没有客户端连接，无法开始联邦学习")
            self.fl_in_progress = False
            return
        
        self.logger.info(f"已连接 {len(self.connected_clients)} 个客户端")
        
        # 开始单轮联邦学习
        self.logger.info("====================================================")
        self.logger.info("====== 开始单轮联邦学习过程 ======")
        self.logger.info("====================================================")
        
        # 广播开始轮次消息
        await self.broadcast_server_message(
            ServerMessageType.START_FL_ROUND,
            {"round_id": self.current_round}
        )
        
        self.logger.info(f"===== 单轮联邦学习 =====")
        
        # 初始化客户端模型
        self.logger.info("初始化客户端模型")
        
        # 1. 客户端训练 - 通过WebSocket直接处理
        self.logger.info("等待客户端训练完成...")
        
        # 等待客户端完成训练并返回梯度 - 这是由WebSocket服务器中的handle_client处理的
        # 等待足够的时间，让客户端完成训练
        await asyncio.sleep(self.config.epochs * 2)  # 模拟等待时间
        
        # 2. 服务器训练
        self.logger.info("服务器训练中...")
        server_model_state, train_acc, test_acc = self.trainer.train(self.current_round)
        
        # 记录服务器性能
        history["server_train_acc"].append(train_acc)
        history["server_test_acc"].append(test_acc)
        history["server_loss"].append(self.trainer.history["loss"][-1])
        
        # 3. 收集客户端梯度 - 已由WebSocket处理，从fl_server中获取
        client_gradients = list(self.fl_server.client_gradients.values())
        
        # 如果没有收到客户端梯度，则跳过聚合
        if not client_gradients:
            self.logger.warning("没有收到客户端梯度，跳过聚合")
            self.fl_in_progress = False
            return
        
        # 计算客户端权重 - 使用平均权重
        client_weights = [1.0 / len(client_gradients)] * len(client_gradients)
        
        # 从fl_server中获取客户端性能数据
        if self.current_round < len(self.fl_server.history["client_train_acc"]):
            client_train_accs = self.fl_server.history["client_train_acc"][self.current_round]
            client_test_accs = self.fl_server.history["client_test_acc"][self.current_round]
            client_losses = self.fl_server.history["client_loss"][self.current_round]
            
            # 记录客户端性能
            history["client_train_acc"][self.current_round] = client_train_accs
            history["client_test_acc"][self.current_round] = client_test_accs
            history["client_loss"][self.current_round] = client_losses
        
        # 4. 服务器聚合模型
        self.logger.info("聚合模型中...")
        aggregated_model_state = self.aggregator.aggregate(
            server_model_state, client_gradients, weights=client_weights
        )
        
        # 5. 更新服务器模型
        self.aggregator.update_model(aggregated_model_state)
        
        # 6. 保存服务器模型
        self.trainer.save_model(self.current_round)
        
        # 7. 清空客户端梯度并关闭所有客户端连接
        self.fl_server.client_gradients = {}
        
        # 8. 向所有客户端发送训练完成消息并关闭连接
        self.logger.info("向所有客户端发送训练完成消息并关闭连接")
        close_tasks = []
        for client_id, websocket in list(self.connected_clients.items()):
            try:
                # 发送完成消息
                finish_msg = Message(
                    MessageType.FINISH,
                    {"message": "联邦学习训练已完成"}
                )
                await websocket.send(finish_msg.to_json())
                
                # 关闭连接
                await websocket.close()
                
                # 从已连接客户端列表中移除
                if client_id in self.connected_clients:
                    del self.connected_clients[client_id]
                
                self.logger.info(f"客户端 {client_id} 连接已关闭")
            except Exception as e:
                self.logger.error(f"关闭客户端 {client_id} 连接时出错: {str(e)}")
        
        # 9. 同步更新后的模型给其他服务器
        serialized_model = serialize_model(aggregated_model_state)
        await self.broadcast_server_message(
            ServerMessageType.SYNC_MODEL,
            {
                "round_id": self.current_round,
                "model": serialized_model
            }
        )
        
        # 10. 打印单轮训练结果
        if history["client_train_acc"][0]:
            avg_client_train_acc = sum(history["client_train_acc"][0]) / len(history["client_train_acc"][0])
            avg_client_test_acc = sum(history["client_test_acc"][0]) / len(history["client_test_acc"][0])
            avg_client_loss = sum(history["client_loss"][0]) / len(history["client_loss"][0])
            
            self.logger.info("====================================================")
            self.logger.info("====== 联邦学习单轮训练结果 ======")
            self.logger.info(f"服务器 - 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
            self.logger.info(f"客户端平均 - 训练准确率: {avg_client_train_acc:.4f}, "
                           f"测试准确率: {avg_client_test_acc:.4f}, 损失: {avg_client_loss:.4f}")
            self.logger.info("====================================================")
        else:
            self.logger.info("====================================================")
            self.logger.info("====== 联邦学习单轮训练结果 ======")
            self.logger.info(f"服务器 - 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
            self.logger.info("客户端性能数据不可用")
            self.logger.info("====================================================")
        
        # 11. 结束联邦学习过程
        self.fl_in_progress = False
        self.logger.info("====================================================")
        self.logger.info("====== 联邦学习过程结束 ======")
        self.logger.info("====================================================")
        
        # 12. 生成可视化结果
        try:
            self.logger.info("生成训练结果可视化")
            
            # 可视化训练结果
            from federated_learning.common.utils import visualize_training_history
            
            visualize_data = {
                "server_train_acc": history["server_train_acc"],
                "server_test_acc": history["server_test_acc"],
                "server_loss": history["server_loss"],
                "client_train_acc": history["client_train_acc"][0] if history["client_train_acc"][0] else [],
                "client_test_acc": history["client_test_acc"][0] if history["client_test_acc"][0] else [],
                "client_loss": history["client_loss"][0] if history["client_loss"][0] else []
            }
            
            visualize_path = visualize_training_history(
                visualize_data,
                save_path=os.path.join(self.config.results_dir, f"single_round_training_history_{self.server_id}.png")
            )
            
            self.logger.info(f"训练历史可视化已保存到: {visualize_path}")
            
            # 评估最终模型
            from debug_training_fix import compute_accuracy
            from debug_test import visualize_predictions, plot_confusion_matrix
            
            # 评估聚合后的最终模型
            test_loader = self.fl_server.test_loader
            final_model = self.fl_server.model
            
            test_acc, conf_matrix = compute_accuracy(
                final_model, test_loader, device=self.config.device, get_confusion_matrix=True
            )
            
            self.logger.info(f"最终聚合模型在全局测试集上的准确率: {test_acc:.4f}")
            
            # 绘制聚合模型的混淆矩阵
            class_names = [str(i) for i in range(10)]  # MNIST有10个类别，从0到9
            plot_confusion_matrix(
                conf_matrix.numpy(), 
                classes=class_names, 
                normalize=True,
                title=f'单轮联邦学习聚合模型混淆矩阵 (服务器 {self.server_id})'
            )
            
            # 将混淆矩阵图片重命名保存
            import shutil
            shutil.move('confusion_matrix.png', os.path.join(self.config.results_dir, f'single_round_confusion_matrix_{self.server_id}.png'))
            
            # 可视化聚合模型的预测样例
            visualize_predictions(
                test_loader, 
                final_model, 
                device=self.config.device, 
                num_samples=10
            )
            
            # 将预测可视化图片重命名保存
            shutil.move('prediction_visualization.png', os.path.join(self.config.results_dir, f'single_round_prediction_visualization_{self.server_id}.png'))
            
            self.logger.info(f"可视化结果已保存到结果目录")
            
        except Exception as e:
            self.logger.error(f"生成可视化结果时出错: {str(e)}")

    async def sync_server_data(self):
        """同步服务器数据"""
        for server_id, uri in self.server_addresses.items():
            try:
                self.logger.info(f"连接到服务器 {server_id}: {uri}")
                websocket = await websockets.connect(uri)
                
                # 发送SYNC消息
                message = PBFTMessage(
                    msg_type="SYNC", 
                    view=self.view, 
                    sequence_no=0,
                    sender_id=self.server_id
                )
                await websocket.send(message.to_json())
                
                # 接收响应
                response = await websocket.recv()
                response_data = json.loads(response)
                
                # 更新服务器数据
                if "server_data" in response_data:
                    server_data = response_data["server_data"]
                    self.update_server_data(server_data)
                
                await websocket.close()
                self.logger.info(f"与服务器 {server_id} 同步完成")
            except Exception as e:
                self.logger.error(f"同步服务器 {server_id} 数据失败: {e}")
                continue

    async def check_all_servers_connected(self):
        """检查是否所有服务器都已连接，如果是则开始选举"""
        # 首先等待所有服务器连接，不设超时
        self.logger.info("开始等待所有服务器连接...")
        
        # 第一阶段：等待所有服务器连接
        all_connected = False
        while not all_connected:
            # 检查当前连接的服务器
            connected_servers = set(self.server_connections.keys()) | {self.server_id}
            expected_servers = {str(i) for i in range(1, self.config.num_servers + 1)}
            
            self.logger.info(f"当前已连接服务器: {connected_servers}, 预期服务器: {expected_servers}")
            
            # 比较集合的长度和内容
            all_connected = len(connected_servers) == len(expected_servers) and all(server_id in expected_servers for server_id in connected_servers)
            
            if all_connected:
                self.logger.info("所有服务器已完全连接！")
                break
                
            # 短暂等待后再次检查
            await asyncio.sleep(2)
        
        # 第二阶段：确认所有服务器连接后，开始PBFT选举
        if not self.pbft_node.election_in_progress and not self.pbft_node.elected_primary:
            self.logger.info("所有服务器已连接，开始PBFT共识选举过程")
            self.pbft_node.election_in_progress = False
            self.pbft_node.elected_primary = None
            await self.pbft_node.start_election()
            
            # 开始选举计时
            election_timeout = 20  # 给选举20秒时间
            self.logger.info(f"开始选举计时，超时时间：{election_timeout}秒")
            
            start_time = time.time()
            while time.time() - start_time < election_timeout:
                # 检查选举是否已完成
                if self.pbft_node.elected_primary:
                    self.logger.info(f"选举已完成，主服务器为: {self.pbft_node.elected_primary}")
                    return
                
                # 等待一段时间再检查
                await asyncio.sleep(2)
                self.logger.info(f"选举进行中...已等待 {int(time.time() - start_time)} 秒")
            
            # 如果超时仍未完成选举，强制选择一个主服务器
            if not self.pbft_node.elected_primary:
                self.logger.warning(f"选举超时（{election_timeout}秒），强制选择一个主服务器")
                self.force_select_primary()
        elif self.pbft_node.elected_primary:
            self.logger.info(f"选举已完成，主服务器为: {self.pbft_node.elected_primary}")
        else:
            self.logger.warning("选举已在进行中，不再启动新的选举")

    def force_select_primary(self):
        """强制选择一个主服务器"""
        # 将已连接的服务器列表排序
        server_list = sorted([self.server_id] + list(self.server_connections.keys()))
        
        # 简单地选择ID最小的服务器作为主服务器
        primary_id = server_list[0]
        
        self.logger.warning("====================================================")
        self.logger.warning(f"!!! 强制选择服务器 {primary_id} 作为主服务器 !!!")
        self.logger.warning("====================================================")
        
        # 更新PBFT节点状态
        self.pbft_node.elected_primary = primary_id
        self.pbft_node.election_in_progress = False
        
        # 更新自身状态
        self.is_primary = (primary_id == self.server_id)
        
        if self.is_primary:
            self.logger.info("====================================================")
            self.logger.info("====== 本服务器被强制选为主服务器! ======")
            self.logger.info("====================================================")
            
            # 启动联邦学习进程
            if not self.fl_in_progress:
                self.logger.info("====================================================")
                self.logger.info("====== 主服务器即将开始联邦学习过程 ======")
                self.logger.info("====================================================")
                asyncio.create_task(self.run_federated_learning())
        else:
            self.logger.info("====================================================")
            self.logger.info(f"====== 服务器 {primary_id} 被强制选为主服务器 ======")
            self.logger.info("====================================================")
            
        # 为该选举结果创建一个唯一标识并标记为已处理
        election_id = f"{primary_id}_{self.server_id}_True"
        self.processed_election_messages.add(election_id)
        
        # 向所有服务器广播一次选举结果
        asyncio.create_task(self.broadcast_server_message(
            ServerMessageType.PRIMARY_ELECTED, 
            {"primary_id": primary_id, "force_selected": True}
        ))
            
        return primary_id

async def run_server(server_id, config):
    """
    运行服务器集群节点
    
    Args:
        server_id: 服务器ID
        config: 配置对象
    """
    # 设置日志记录器
    logger = setup_logger(f"Server-{server_id}", os.path.join(config.log_dir, f"server_{server_id}.log"))
    
    try:
        # 创建服务器集群
        server_cluster = ServerCluster(server_id, config)
        
        # 启动服务器
        await server_cluster.start_server()
        
        # 等待服务器运行
        await asyncio.Future()  # 永久运行
    except asyncio.CancelledError:
        # 捕获CancelledError并正常退出
        logger.info(f"服务器 {server_id} 正在优雅地关闭...")
    except Exception as e:
        logger.error(f"服务器 {server_id} 发生错误: {str(e)}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='联邦学习服务器集群节点')
    parser.add_argument('--id', type=str, required=True, help='服务器ID')
    args = parser.parse_args()
    
    server_id = args.id
    
    # 启动服务器
    asyncio.run(run_server(server_id, config))

if __name__ == "__main__":
    main() 
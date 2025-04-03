#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PBFT共识模块

此模块实现了PBFT（实用拜占庭容错）共识算法
用于多服务器联邦学习中的共识达成
"""

import os
import sys
import time
import json
import pickle
import base64
import random
import hashlib
import asyncio
from enum import Enum
import websockets
import logging

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from federated_learning.common.utils import setup_logger
from federated_learning.common.protocol import Message

class PBFTMessageType(Enum):
    """PBFT消息类型"""
    REQUEST = "request"           # 客户端请求
    PRE_PREPARE = "pre-prepare"   # 主节点预准备
    PREPARE = "prepare"           # 节点准备
    COMMIT = "commit"             # 节点提交
    REPLY = "reply"               # 回复客户端
    VIEW_CHANGE = "view-change"   # 视图更改
    NEW_VIEW = "new-view"         # 新视图
    CHECKPOINT = "checkpoint"     # 检查点
    ELECTION = "election"         # 选举消息
    ELECTION_RESULT = "election-result"  # 选举结果

class PBFTMessage:
    """PBFT消息"""
    
    def __init__(self, msg_type, view_or_node_id, seq_num_or_data=None, data=None, digest=None, node_id=None):
        """
        初始化PBFT消息
        
        支持两种调用模式:
        1. 新模式: (msg_type, view, seq_num, data, digest, node_id)
        2. 旧模式: (msg_type, node_id, data)
        
        Args:
            msg_type: 消息类型
            view_or_node_id: 当前视图编号或节点ID
            seq_num_or_data: 序列号或消息数据
            data: 消息数据
            digest: 消息摘要(如果为None，则自动计算)
            node_id: 发送节点ID
        """
        # 检测调用模式，兼容两种调用方式
        if data is None and isinstance(seq_num_or_data, dict):
            # 旧模式: (msg_type, node_id, data)
            self.type = msg_type
            self.node_id = view_or_node_id
            self.sender_id = view_or_node_id  # 添加sender_id属性
            self.data = seq_num_or_data
            self.view = 0  # 使用默认值
            self.seq_num = 0  # 使用默认值
            # 计算摘要
            self.digest = self._calculate_digest()
        else:
            # 新模式: (msg_type, view, seq_num, data, digest, node_id)
            self.type = msg_type
            self.view = view_or_node_id
            self.seq_num = seq_num_or_data
            self.data = data
            self.node_id = node_id
            self.sender_id = node_id  # 添加sender_id属性，与node_id保持一致
            
            # 如果没有提供摘要，则计算
            if digest is None:
                self.digest = self._calculate_digest()
            else:
                self.digest = digest
    
    def _calculate_digest(self):
        """计算消息摘要"""
        # 序列化数据
        serialized_data = json.dumps(self.data, sort_keys=True).encode()
        # 计算SHA256摘要
        return hashlib.sha256(serialized_data).hexdigest()
    
    def to_dict(self):
        """将消息转换为字典"""
        return {
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "view": self.view,
            "seq_num": self.seq_num,
            "data": self.data,
            "digest": self.digest,
            "node_id": self.node_id,
            "sender_id": self.sender_id  # 添加sender_id到字典
        }
    
    @classmethod
    def from_dict(cls, message_dict):
        """从字典创建消息"""
        # 将字符串类型转换为枚举
        msg_type = message_dict["type"]
        if isinstance(msg_type, str):
            for t in PBFTMessageType:
                if t.value == msg_type:
                    msg_type = t
                    break
        
        # 获取node_id，如果不存在尝试使用sender_id
        node_id = message_dict.get("node_id")
        if node_id is None:
            node_id = message_dict.get("sender_id")
        
        # 创建新消息
        msg = cls(
            msg_type=msg_type,
            view=message_dict["view"],
            seq_num=message_dict["seq_num"],
            data=message_dict["data"],
            digest=message_dict["digest"],
            node_id=node_id
        )
        
        # 确保sender_id和node_id一致
        if "sender_id" in message_dict:
            msg.sender_id = message_dict["sender_id"]
        
        return msg
    
    def to_json(self):
        """将消息转换为JSON字符串"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str):
        """从JSON字符串创建消息"""
        message_dict = json.loads(json_str)
        return cls.from_dict(message_dict)

class PBFTNode:
    """PBFT节点"""
    
    def __init__(self, node_id, config, logger=None):
        """
        初始化PBFT节点
        
        Args:
            node_id: 节点ID
            config: 配置对象
            logger: 日志记录器(如果为None，则创建新的)
        """
        self.node_id = node_id
        self.config = config
        
        # 设置日志记录器
        if logger is None:
            self.logger = setup_logger(
                f"PBFT-{node_id}",
                os.path.join(config.log_dir, f"pbft_node_{node_id}.log")
            )
        else:
            self.logger = logger
        
        # 当前视图编号
        self.view = 0
        
        # 是否是主节点
        self.is_primary = (int(node_id) % config.num_servers == self.view % config.num_servers)
        
        # 序列号计数器
        self.seq_counter = 0
        
        # 存储消息日志
        self.message_log = {
            PBFTMessageType.REQUEST: {},
            PBFTMessageType.PRE_PREPARE: {},
            PBFTMessageType.PREPARE: {},
            PBFTMessageType.COMMIT: {},
            PBFTMessageType.CHECKPOINT: {},
            PBFTMessageType.VIEW_CHANGE: {},
            PBFTMessageType.NEW_VIEW: {},
            PBFTMessageType.ELECTION: {},
            PBFTMessageType.ELECTION_RESULT: {}
        }
        
        # 存储已完成的请求
        self.completed_requests = set()
        
        # 服务器列表
        self.server_nodes = {}
        
        # 选举状态
        self.election_in_progress = False
        self.votes = {}
        self.elected_primary = None
        
        # 视图更改超时
        self.view_change_timeout = config.view_change_timeout
        
        # 当前处理的请求
        self.current_request = None
        
        # 当前选举信息
        self.current_election = None
        
        # 存储已处理的选举ID，防止重复处理
        self.processed_elections = set()
        
        # 回调函数
        self.on_election_completed = None
        
        self.logger.info(f"PBFT节点 {node_id} 初始化完成，当前视图: {self.view}，是否为主节点: {self.is_primary}")
    
    def next_seq_num(self):
        """获取下一个序列号"""
        self.seq_counter += 1
        return self.seq_counter
    
    def get_primary_id(self, view=None):
        """
        获取当前视图的主节点ID
        
        Args:
            view: 视图编号，如果为None则使用当前视图
            
        Returns:
            主节点ID
        """
        if view is None:
            view = self.view
        
        # 主节点的选择规则：view mod n (n为节点数量)
        primary_idx = view % self.config.num_servers
        
        # 获取所有服务器节点ID列表并排序
        server_ids = sorted([int(server_id) for server_id in self.server_nodes.keys()])
        
        if primary_idx < len(server_ids):
            return str(server_ids[primary_idx])
        
        # 如果索引超出范围，则回到第一个节点
        return str(server_ids[0]) if server_ids else self.node_id
    
    async def broadcast(self, message):
        """
        广播消息给所有服务器节点
        
        Args:
            message: 要广播的消息
        """
        message.node_id = self.node_id
        json_msg = message.to_json()
        
        tasks = []
        for server_id, websocket in self.server_nodes.items():
            if server_id != self.node_id:  # 不向自己发送
                self.logger.debug(f"广播消息到服务器 {server_id}: {message.type}")
                task = asyncio.create_task(websocket.send(json_msg))
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def start_election(self):
        """开始选举过程"""
        self.logger.info("开始PBFT共识选举过程")
        
        # 强制重置选举状态
        self.election_in_progress = True
        self.processed_elections.clear()  # 清空已处理的选举记录
        
        # 随机生成一个选举编号
        election_id = str(int(time.time() * 1000))  # 使用时间戳作为选举ID
        self.logger.info(f"生成选举ID: {election_id}")
        
        # 记录当前选举
        self.current_election = {
            "id": election_id,
            "votes": {},           # 投票结果
            "prepared": set(),     # prepare阶段完成的节点
            "committed": set(),    # commit阶段完成的节点
        }
        
        # 确保所有服务器节点连接
        self.logger.info(f"当前连接的服务器节点: {list(self.server_nodes.keys())}")
        
        # 随机选择一个候选服务器（可以是自己）
        candidates = list(self.server_nodes.keys()) + [self.node_id]
        candidates = list(set(candidates))  # 去重
        candidates.sort()  # 确保选择顺序是确定的
        
        # 确保至少有一个节点
        if not candidates:
            self.logger.warning("没有可用的候选服务器，选举失败")
            self.election_in_progress = False
            return
        
        self.logger.info(f"候选服务器列表: {candidates}")
        
        # 使用确定性算法选择候选人（基于选举ID的哈希）
        import hashlib
        hash_input = election_id + ''.join(candidates)
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        index = int(hash_value, 16) % len(candidates)
        candidate = candidates[index]
        
        self.logger.info(f"选择候选服务器: {candidate}")
        
        # 创建提案消息
        proposal = {
            "election_id": election_id,
            "candidate": candidate
        }
        
        # 发送pre-prepare消息
        pre_prepare_msg = PBFTMessage(
            PBFTMessageType.PRE_PREPARE,
            self.view,  # 使用view字段来存储视图编号
            self.seq_counter,  # 使用自增的序列号
            proposal,  # 数据
            None,  # 自动计算摘要
            self.node_id  # 发送者ID
        )
        
        # 更新序列号
        self.seq_counter += 1
        
        # 处理本地消息
        await self.handle_pre_prepare(pre_prepare_msg)
        
        # 向其他节点广播消息
        broadcast_tasks = []
        for node_id, websocket in self.server_nodes.items():
            if node_id != self.node_id:  # 不向自己发送
                self.logger.info(f"向节点 {node_id} 发送pre-prepare消息")
                try:
                    task = asyncio.create_task(websocket.send(pre_prepare_msg.to_json()))
                    broadcast_tasks.append(task)
                except Exception as e:
                    self.logger.error(f"发送pre-prepare消息到节点 {node_id} 失败: {str(e)}")
        
        # 等待所有广播任务完成
        if broadcast_tasks:
            await asyncio.gather(*broadcast_tasks, return_exceptions=True)
            
        self.logger.info("所有pre-prepare消息已发送，等待prepare阶段响应")
    
    async def handle_pre_prepare(self, message):
        """处理pre-prepare消息"""
        # 检查是否已经处理过该选举
        election_id = message.data.get("election_id")
        if election_id in self.processed_elections:
            self.logger.info(f"已处理过选举 {election_id}，忽略pre-prepare消息")
            return
        
        # 更新当前选举信息
        self.current_election = {
            "id": election_id,
            "votes": {},
            "prepared": set(),
            "committed": set(),
        }
        
        # 记录提案
        candidate = message.data.get("candidate")
        self.logger.info(f"收到pre-prepare消息: 选举 {election_id}, 候选服务器 {candidate}")
        
        # 安全获取sender_id
        sender_id = getattr(message, 'sender_id', None) or getattr(message, 'node_id', None)
        if not sender_id:
            self.logger.warning("无法获取消息发送者ID")
            sender_id = "unknown"
        
        self.logger.info(f"消息发送者ID: {sender_id}")
        
        # 记录投票
        self.current_election["votes"][sender_id] = candidate
        
        # 发送prepare消息
        prepare_msg = PBFTMessage(
            PBFTMessageType.PREPARE,
            self.view,  # 使用view字段
            self.seq_counter,  # 使用序列号
            {
                "election_id": election_id,
                "candidate": candidate,
                "voter": self.node_id
            },
            None,  # 自动计算摘要
            self.node_id  # 发送者ID
        )
        
        # 更新序列号
        self.seq_counter += 1
        
        # 处理本地prepare消息
        await self.handle_prepare(prepare_msg)
        
        # 广播prepare消息到其他节点
        broadcast_tasks = []
        for node_id, websocket in self.server_nodes.items():
            # 不发送给自己或原始消息的发送者
            if node_id != self.node_id and node_id != sender_id:
                self.logger.info(f"向节点 {node_id} 发送prepare消息")
                try:
                    task = asyncio.create_task(websocket.send(prepare_msg.to_json()))
                    broadcast_tasks.append(task)
                except Exception as e:
                    self.logger.error(f"发送prepare消息到节点 {node_id} 失败: {str(e)}")
        
        # 等待所有广播任务完成
        if broadcast_tasks:
            await asyncio.gather(*broadcast_tasks, return_exceptions=True)
            
        self.logger.info("所有prepare消息已发送，等待其他节点的prepare响应")
    
    async def handle_prepare(self, message):
        """处理prepare消息"""
        election_id = message.data.get("election_id")
        candidate = message.data.get("candidate")
        voter = message.data.get("voter")
        
        # 检查是否是当前选举
        if not self.current_election or self.current_election.get("id") != election_id:
            self.logger.warning(f"收到未知选举 {election_id} 的prepare消息，忽略")
            return
        
        self.logger.info(f"收到prepare消息: 选举 {election_id}, 候选服务器 {candidate}, 投票者 {voter}")
        
        # 记录投票
        self.current_election["votes"][voter] = candidate
        
        # 标记该节点已完成prepare阶段
        self.current_election["prepared"].add(voter)
        
        # 检查是否收到足够的prepare消息（>= 2f + 1）
        # 在这里，f是最大容错节点数量，通常为 (n-1)/3，其中n是总节点数
        # 简化起见，我们使用超过半数的节点作为阈值
        n = len(self.server_nodes) + 1  # 加上自己
        threshold = n // 2 + 1  # 超过半数
        
        prepared_count = len(self.current_election["prepared"])
        self.logger.info(f"当前prepare阶段完成节点数: {prepared_count}/{n}, 阈值: {threshold}")
        
        if prepared_count >= threshold:
            self.logger.info(f"收到足够的prepare消息，进入commit阶段")
            
            # 发送commit消息
            commit_msg = PBFTMessage(
                PBFTMessageType.COMMIT,
                self.view,  # 使用view字段
                self.seq_counter,  # 使用序列号
                {
                    "election_id": election_id,
                    "candidate": candidate,
                    "committer": self.node_id
                },
                None,  # 自动计算摘要
                self.node_id  # 发送者ID
            )
            
            # 更新序列号
            self.seq_counter += 1
            
            # 处理本地commit消息
            await self.handle_commit(commit_msg)
            
            # 广播commit消息
            broadcast_tasks = []
            for node_id, websocket in self.server_nodes.items():
                if node_id != self.node_id:  # 不向自己发送
                    self.logger.info(f"向节点 {node_id} 发送commit消息")
                    try:
                        task = asyncio.create_task(websocket.send(commit_msg.to_json()))
                        broadcast_tasks.append(task)
                    except Exception as e:
                        self.logger.error(f"发送commit消息到节点 {node_id} 失败: {str(e)}")
            
            # 等待所有广播任务完成
            if broadcast_tasks:
                await asyncio.gather(*broadcast_tasks, return_exceptions=True)
                
            self.logger.info("所有commit消息已发送，等待其他节点的commit响应")
    
    async def handle_commit(self, message):
        """处理commit消息"""
        election_id = message.data.get("election_id")
        candidate = message.data.get("candidate")
        committer = message.data.get("committer")
        
        # 检查是否是当前选举
        if not self.current_election or self.current_election.get("id") != election_id:
            self.logger.warning(f"收到未知选举 {election_id} 的commit消息，忽略")
            return
        
        self.logger.info(f"收到commit消息: 选举 {election_id}, 候选服务器 {candidate}, 提交者 {committer}")
        
        # 标记该节点已完成commit阶段
        self.current_election["committed"].add(committer)
        
        # 检查是否收到足够的commit消息
        n = len(self.server_nodes) + 1  # 加上自己
        threshold = n // 2 + 1  # 超过半数
        
        committed_count = len(self.current_election["committed"])
        self.logger.info(f"当前commit阶段完成节点数: {committed_count}/{n}, 阈值: {threshold}")
        
        if committed_count >= threshold:
            self.logger.info(f"收到足够的commit消息，选举完成！最终选出的主服务器是: {candidate}")
            
            # 完成选举
            self.elected_primary = candidate
            self.processed_elections.add(election_id)
            self.election_in_progress = False
            
            # 调用选举完成回调函数（如果有）
            if self.on_election_completed:
                self.logger.info(f"调用选举完成回调函数，传递选举结果: {candidate}")
                if asyncio.iscoroutinefunction(self.on_election_completed):
                    await self.on_election_completed(candidate)
                else:
                    self.on_election_completed(candidate)
            
            # 清理当前选举状态
            self.current_election = None
    
    async def process_message(self, message):
        """处理接收到的消息"""
        try:
            # 获取消息类型
            msg_type = message.type
            
            # 记录消息接收
            sender_id = getattr(message, 'sender_id', None) or getattr(message, 'node_id', "unknown")
            self.logger.info(f"处理来自节点 {sender_id} 的 {msg_type} 消息")
            
            # 根据消息类型选择处理方法
            if msg_type == PBFTMessageType.PRE_PREPARE or msg_type == "pre-prepare":
                await self.handle_pre_prepare(message)
            elif msg_type == PBFTMessageType.PREPARE or msg_type == "prepare":
                await self.handle_prepare(message)
            elif msg_type == PBFTMessageType.COMMIT or msg_type == "commit":
                await self.handle_commit(message)
            else:
                self.logger.warning(f"未知消息类型: {msg_type}")
        except Exception as e:
            self.logger.error(f"处理消息时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc()) 
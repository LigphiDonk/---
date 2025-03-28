import torch
import logging
import time
import threading
import copy
import json
from typing import Dict, List, Set, Any, Tuple, Optional

from network import PBFTNode, Message
from consensus import PBFTConsensus
from election import ElectionManager

logger = logging.getLogger("PBFTClient")

class PBFTFederatedClient(PBFTNode):
    """基于PBFT的联邦学习客户端"""
    
    def __init__(self, node_id: int, host: str = '127.0.0.1', port: int = None, bootstrap_nodes: List[Tuple[int, str, int]] = None):
        """初始化PBFT联邦学习客户端
        
        Args:
            node_id: 节点ID
            host: 节点主机地址
            port: 节点监听端口
            bootstrap_nodes: 引导节点列表 [(node_id, host, port), ...]
        """
        super().__init__(node_id, host, port)
        
        # 联邦学习相关状态
        self.local_model = None       # 本地训练模型
        self.global_model = None      # 全局模型
        self.training_args = None     # 训练参数
        self.train_dataloader = None  # 训练数据
        self.test_dataloader = None   # 测试数据
        self.net_dataidx_map = None   # 数据分配映射
        
        # 已知节点映射 {node_id: (host, port)}
        self.known_nodes = {}
        
        # PBFT共识
        self.consensus = PBFTConsensus(node_id)
        
        # 选举管理器
        self.election = ElectionManager(node_id)
        
        # 当前训练轮次
        self.current_round = 0
        
        # 聚合的模型提案
        self.model_proposals: Dict[int, Dict[int, Any]] = {}  # {round: {node_id: model}}
        
        # 注册消息处理器
        self.register_handler("TRAINING_CONFIG", self._handle_training_config)
        self.register_handler("GLOBAL_MODEL", self._handle_global_model)
        self.register_handler("MODEL_PROPOSAL", self._handle_model_proposal)
        self.register_handler("AGGREGATION_REQUEST", self._handle_aggregation_request)
        
        # 选举相关消息处理器
        self.register_handler("ELECTION_START", self._handle_election_start)
        self.register_handler("ELECTION_VOTE", self._handle_election_vote)
        self.register_handler("ELECTION_RESULT", self._handle_election_result)
        
        # PBFT共识相关消息处理器
        self.register_handler("PBFT_MESSAGE", self._handle_pbft_message)
        
        # 设置PBFT共识回调
        self.consensus.set_callbacks(
            on_broadcast=self._broadcast_pbft_message,
            on_send=self._send_pbft_message,
            on_consensus_complete=self._on_consensus_complete
        )
        
        # 引导节点
        self.bootstrap_nodes = bootstrap_nodes
        
        # 选举和超时检查线程
        self.check_thread = None
    
    def start(self):
        """启动客户端"""
        super().start()
        
        # 连接到引导节点
        if self.bootstrap_nodes:
            self.discover_peers(self.bootstrap_nodes)
        
        # 启动选举和超时检查线程
        self.check_thread = threading.Thread(target=self._periodic_check, daemon=True)
        self.check_thread.start()
        
        logger.info(f"PBFT联邦学习客户端 {self.node_id} 启动完成")
    
    def stop(self):
        """停止客户端"""
        super().stop()
        logger.info(f"PBFT联邦学习客户端 {self.node_id} 已停止")
    
    def set_training_config(self, config: Dict):
        """设置训练配置
        
        Args:
            config: 训练配置
        """
        self.training_args = config.get("args")
        self.net_dataidx_map = config.get("net_dataidx_map")
        
        logger.info(f"客户端 {self.node_id} 设置训练配置")
    
    def set_model(self, model):
        """设置模型
        
        Args:
            model: 模型
        """
        self.global_model = copy.deepcopy(model)
        self.local_model = copy.deepcopy(model)
        
        logger.info(f"客户端 {self.node_id} 设置模型")
    
    def set_data(self, train_dataloader, test_dataloader):
        """设置数据
        
        Args:
            train_dataloader: 训练数据加载器
            test_dataloader: 测试数据加载器
        """
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
        logger.info(f"客户端 {self.node_id} 设置数据")
    
    def train(self):
        """训练本地模型"""
        if self.local_model is None or self.training_args is None:
            logger.error("模型或训练参数未设置")
            return False
        
        # 导入必要的模块（确保在需要时导入）
        from experiments_client import train_net
        
        # 获取设备
        device = self.training_args.device if hasattr(self.training_args, "device") else "cpu"
        
        # 准备训练参数
        net = copy.deepcopy(self.local_model)
        nets = {self.node_id: net}
        
        try:
            logger.info(f"客户端 {self.node_id} 开始本地训练")
            start_time = time.time()
            
            # 训练模型
            train_acc, test_acc = train_net(
                self.node_id, 
                net, 
                self.train_dataloader, 
                self.test_dataloader, 
                self.training_args.epochs, 
                self.training_args.lr, 
                self.training_args.optimizer,
                device=device
            )
            
            training_time = time.time() - start_time
            
            # 更新本地模型
            self.local_model = copy.deepcopy(net)
            
            # 记录训练事件
            self.election.record_event(self.node_id, "training", {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "time": training_time,
                "round": self.current_round
            })
            
            # 记录模型质量事件
            self.election.record_event(self.node_id, "model", {
                "quality": test_acc / 100.0,  # 归一化到0-1
                "round": self.current_round
            })
            
            logger.info(f"客户端 {self.node_id} 完成本地训练, 训练准确率: {train_acc}, 测试准确率: {test_acc}")
            return True
        
        except Exception as e:
            logger.error(f"客户端 {self.node_id} 训练出错: {str(e)}")
            return False
    
    def propose_model(self):
        """提议本地模型"""
        if self.local_model is None:
            logger.error("本地模型未训练")
            return False
        
        # 创建模型提案消息
        proposal_msg = {
            "type": "MODEL_PROPOSAL",
            "round": self.current_round,
            "model": self.local_model.state_dict(),
            "node_id": self.node_id,
            "timestamp": time.time()
        }
        
        # 广播模型提案
        self.broadcast("MODEL_PROPOSAL", proposal_msg)
        
        # 将自己的提案添加到提案集
        if self.current_round not in self.model_proposals:
            self.model_proposals[self.current_round] = {}
        self.model_proposals[self.current_round][self.node_id] = self.local_model.state_dict()
        
        logger.info(f"客户端 {self.node_id} 提议本地模型, 轮次: {self.current_round}")
        return True
    
    def aggregate_models(self, round_num: int = None):
        """聚合模型
        
        Args:
            round_num: 要聚合的轮次，默认为当前轮次
        
        Returns:
            模型状态字典
        """
        round_num = round_num if round_num is not None else self.current_round
        
        if round_num not in self.model_proposals:
            logger.error(f"轮次 {round_num} 没有模型提案")
            return None
        
        proposals = self.model_proposals[round_num]
        if not proposals:
            logger.error(f"轮次 {round_num} 没有模型提案")
            return None
        
        # 聚合模型（简单平均）
        logger.info(f"客户端 {self.node_id} 聚合轮次 {round_num} 的 {len(proposals)} 个模型")
        
        # 初始化聚合模型
        aggregated_model = copy.deepcopy(list(proposals.values())[0])
        
        # 对每个参数求平均
        for key in aggregated_model.keys():
            aggregated_model[key] = torch.zeros_like(aggregated_model[key])
            for model_dict in proposals.values():
                aggregated_model[key] += model_dict[key]
            aggregated_model[key] /= len(proposals)
        
        return aggregated_model
    
    def request_model_aggregation(self):
        """请求模型聚合（主节点调用）"""
        if not self.consensus.is_primary():
            logger.warning(f"客户端 {self.node_id} 不是主节点，不能请求模型聚合")
            return False
        
        # 创建聚合请求
        request = {
            "type": "AGGREGATION_REQUEST",
            "round": self.current_round,
            "node_ids": list(self.model_proposals.get(self.current_round, {}).keys()),
            "timestamp": time.time()
        }
        
        # 启动共识流程
        success = self.consensus.start_consensus(request)
        if success:
            logger.info(f"主节点 {self.node_id} 请求轮次 {self.current_round} 的模型聚合")
        else:
            logger.error(f"主节点 {self.node_id} 请求模型聚合失败")
        
        return success
    
    def _broadcast_pbft_message(self, message: Dict):
        """广播PBFT消息"""
        self.broadcast("PBFT_MESSAGE", message)
    
    def _send_pbft_message(self, node_id: int, message: Dict):
        """发送PBFT消息给指定节点"""
        self.send_to_peer(node_id, Message("PBFT_MESSAGE", self.node_id, message))
    
    def _on_consensus_complete(self, request: Dict) -> Any:
        """共识完成回调"""
        if request["type"] == "AGGREGATION_REQUEST":
            # 聚合模型
            round_num = request["round"]
            aggregated_model = self.aggregate_models(round_num)
            
            if aggregated_model:
                # 更新全局模型
                self.global_model.load_state_dict(aggregated_model)
                
                # 广播新的全局模型
                self._broadcast_global_model(round_num)
                
                # 记录共识成功事件
                self.election.record_event(self.node_id, "consensus", {
                    "success": True,
                    "round": round_num
                })
                
                logger.info(f"客户端 {self.node_id} 完成轮次 {round_num} 的模型聚合共识")
                return aggregated_model
            else:
                logger.error(f"客户端 {self.node_id} 聚合轮次 {round_num} 的模型失败")
                return None
        
        return None
    
    def _broadcast_global_model(self, round_num: int):
        """广播全局模型"""
        if self.global_model is None:
            return
        
        # 创建全局模型消息
        model_msg = {
            "type": "GLOBAL_MODEL",
            "round": round_num,
            "model": self.global_model.state_dict(),
            "source_node": self.node_id,
            "timestamp": time.time()
        }
        
        # 广播全局模型
        self.broadcast("GLOBAL_MODEL", model_msg)
        
        logger.info(f"客户端 {self.node_id} 广播轮次 {round_num} 的全局模型")
    
    def _periodic_check(self):
        """定期检查选举和超时"""
        while self.running:
            # 记录在线事件
            self.election.record_event(self.node_id, "online", {
                "duration": 60  # 60秒在线时间
            })
            
            # 检查是否应该启动选举
            if self.election.should_start_election():
                election_msg = self.election.start_election()
                self.broadcast("ELECTION_START", election_msg)
            
            # 检查共识超时
            self.consensus.check_timeout()
            
            # 每分钟检查一次
            time.sleep(60)
    
    def _handle_training_config(self, sender_id: int, content: Dict):
        """处理训练配置消息"""
        logger.info(f"客户端 {self.node_id} 从节点 {sender_id} 接收训练配置")
        self.set_training_config(content)
        
        # 记录配置响应事件
        self.election.record_event(self.node_id, "response", {
            "type": "config",
            "time": 0.1,  # 假设响应时间为0.1秒
            "sender": sender_id
        })
    
    def _handle_global_model(self, sender_id: int, content: Dict):
        """处理全局模型消息"""
        round_num = content["round"]
        model_dict = content["model"]
        source_node = content.get("source_node", sender_id)
        
        logger.info(f"客户端 {self.node_id} 从节点 {sender_id} 接收轮次 {round_num} 的全局模型")
        
        # 检查轮次
        if round_num < self.current_round:
            logger.warning(f"收到旧轮次的全局模型: {round_num} < {self.current_round}")
            return
        
        # 更新全局模型
        if self.global_model is not None:
            self.global_model.load_state_dict(model_dict)
            
            # 更新本地模型为全局模型
            self.local_model = copy.deepcopy(self.global_model)
            
            # 更新轮次
            if round_num >= self.current_round:
                self.current_round = round_num + 1
                
                # 清理旧的提案
                if round_num in self.model_proposals:
                    del self.model_proposals[round_num]
            
            logger.info(f"客户端 {self.node_id} 更新全局模型，当前轮次: {self.current_round}")
        else:
            logger.error("全局模型未初始化")
        
        # 记录模型响应事件
        self.election.record_event(self.node_id, "response", {
            "type": "model",
            "time": 0.2,  # 假设响应时间为0.2秒
            "sender": sender_id,
            "source": source_node
        })
    
    def _handle_model_proposal(self, sender_id: int, content: Dict):
        """处理模型提案消息"""
        round_num = content["round"]
        model_dict = content["model"]
        node_id = content["node_id"]
        
        logger.info(f"客户端 {self.node_id} 从节点 {sender_id} 接收节点 {node_id} 轮次 {round_num} 的模型提案")
        
        # 存储模型提案
        if round_num not in self.model_proposals:
            self.model_proposals[round_num] = {}
        
        self.model_proposals[round_num][node_id] = model_dict
        
        # 如果是主节点，并且收到足够多的提案，启动聚合共识
        if self.consensus.is_primary() and round_num == self.current_round:
            # 判断是否收到足够多的提案
            min_proposals = max(2, len(self.consensus.state.validators) // 2 + 1)
            if len(self.model_proposals[round_num]) >= min_proposals:
                self.request_model_aggregation()
        
        # 记录提案响应事件
        self.election.record_event(self.node_id, "response", {
            "type": "proposal",
            "time": 0.15,  # 假设响应时间为0.15秒
            "sender": sender_id,
            "proposer": node_id
        })
    
    def _handle_aggregation_request(self, sender_id: int, content: Dict):
        """处理聚合请求消息"""
        if not self.consensus.is_validator():
            return
        
        round_num = content["round"]
        node_ids = content["node_ids"]
        
        logger.info(f"客户端 {self.node_id} 从节点 {sender_id} 接收轮次 {round_num} 的聚合请求")
        
        # 将聚合请求转发给共识模块
        self.consensus.handle_request(content, sender_id)
    
    def _handle_pbft_message(self, sender_id: int, content: Dict):
        """处理PBFT共识消息"""
        # 将PBFT消息传递给共识模块
        self.consensus.handle_message(content, sender_id)
    
    def _handle_election_start(self, sender_id: int, content: Dict):
        """处理选举开始消息"""
        term = content["term"]
        candidates = content["candidates"]
        
        logger.info(f"客户端 {self.node_id} 从节点 {sender_id} 接收第 {term} 轮选举开始消息")
        
        # 参与投票
        vote_msg = self.election.vote(term, candidates)
        self.broadcast("ELECTION_VOTE", vote_msg)
    
    def _handle_election_vote(self, sender_id: int, content: Dict):
        """处理选举投票消息"""
        self.election.receive_vote(content)
        
        term = content["term"]
        
        # 检查是否收到足够的投票
        # 修改：使用peers或连接计数而不是known_nodes
        connected_nodes_count = len(self.peers) + 1  # +1表示自己
        if len(self.election.votes.get(term, {}).get(self.node_id, {})) >= connected_nodes_count // 2 + 1:
            result_msg = self.election.finalize_election(term)
            self.broadcast("ELECTION_RESULT", result_msg)
    
    def _handle_election_result(self, sender_id: int, content: Dict):
        """处理选举结果消息"""
        term = content["term"]
        validators = content["validators"]
        
        logger.info(f"客户端 {self.node_id} 从节点 {sender_id} 接收第 {term} 轮选举结果，验证节点: {validators}")
        
        # 更新本地验证节点集合
        self.election.validators = set(validators)
        
        # 更新共识模块的验证节点集合
        self.consensus.set_validators(set(validators))
        
        # 更新本节点角色
        is_validator = self.node_id in validators
        self.set_as_validator(is_validator)
        
        # 更新是否是主节点
        is_primary = self.consensus.is_primary()
        self.set_as_primary(is_primary)
        
        # 记录选举结果事件
        self.election.record_event(self.node_id, "election", {
            "term": term,
            "result": "validator" if is_validator else "normal",
            "is_primary": is_primary
        })

    def set_as_validator(self, is_validator: bool):
        """设置节点是否为验证节点
        
        Args:
            is_validator: 是否为验证节点
        """
        if is_validator:
            logger.info(f"节点 {self.node_id} 成为验证节点")
        else:
            logger.info(f"节点 {self.node_id} 成为普通节点")
        
        # 更新节点状态
        self.consensus.state.is_validator = is_validator

    def set_as_primary(self, is_primary: bool):
        """设置节点是否为主节点
        
        Args:
            is_primary: 是否为主节点
        """
        if is_primary:
            logger.info(f"节点 {self.node_id} 成为主节点")
        else:
            logger.info(f"节点 {self.node_id} 成为从节点")
        
        # 更新节点状态
        self.consensus.state.is_primary = is_primary 
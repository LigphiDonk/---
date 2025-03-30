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
# 导入PVSS工具
from pvss_utils import PVSSHandler, apply_mask, batch_generate_masks

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
        self.model = None      # 全局模型
        self.previous_global_model = None  # 上一轮的全局模型，用于余弦相似度计算
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
        
        # PVSS消息处理器
        self.register_handler("PVSS_KEY", self._handle_pvss_key)
        
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
        
        # PVSS配置
        self.use_mask = True  # 是否使用掩码
        self.pvss_handler = PVSSHandler(node_id, f=1)  # PVSS处理器 (f默认为1)
        self.pvss_public_keys = {}  # 所有节点的PVSS公钥 {node_id -> public_key}
        
        # 设置共识模块的掩码开关
        self.consensus.set_use_mask(self.use_mask)
        
        # 余弦相似度防御相关
        self.use_cosine_defense = False
        self.cosine_threshold = -0.1  # 默认阈值
    
    def start(self):
        """启动客户端"""
        super().start()
        
        # 连接到引导节点
        if self.bootstrap_nodes:
            self.discover_peers(self.bootstrap_nodes)
        
        # 启动选举和超时检查线程
        self.check_thread = threading.Thread(target=self._periodic_check, daemon=True)
        self.check_thread.start()
        
        # 广播PVSS公钥
        self._broadcast_pvss_key()
        
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
        self.model = copy.deepcopy(model)
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
        
        # 保存全局模型副本用于计算更新
        global_model_copy = copy.deepcopy(self.model)
        
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
            
            # 计算模型更新（delta weights）
            delta_w = {}
            global_state_dict = global_model_copy.state_dict()
            local_state_dict = self.local_model.state_dict()
            
            for key in local_state_dict:
                if key in global_state_dict:
                    # 计算差值：本地模型 - 全局模型
                    delta_w[key] = local_state_dict[key] - global_state_dict[key]
            
            # 计算本地更新与上一轮全局模型之间的余弦相似度
            similarity_score = None
            if self.previous_global_model is not None:
                try:
                    # 展平delta_w和previous_global_model为1D张量
                    flattened_delta = torch.cat([delta_w[key].flatten() for key in delta_w])
                    
                    # 从上一轮全局模型中提取相应参数
                    previous_global_state = self.previous_global_model.state_dict()
                    flattened_prev_global = torch.cat([previous_global_state[key].flatten() for key in delta_w if key in previous_global_state])
                    
                    # 计算余弦相似度
                    if flattened_prev_global.shape == flattened_delta.shape:
                        similarity_score = torch.nn.functional.cosine_similarity(
                            flattened_delta.unsqueeze(0), 
                            flattened_prev_global.unsqueeze(0)
                        ).item()
                        logger.info(f"节点 {self.node_id} 第 {self.current_round} 轮计算的余弦相似度: {similarity_score:.4f}")
                    else:
                        logger.warning(f"无法计算余弦相似度: 张量形状不匹配: {flattened_delta.shape} vs {flattened_prev_global.shape}")
                except Exception as e:
                    logger.error(f"计算余弦相似度时出错: {str(e)}")
            else:
                logger.info(f"节点 {self.node_id} 第 {self.current_round} 轮没有上一轮全局模型，跳过余弦相似度计算")
            
            # 检查余弦相似度是否满足阈值要求
            cosine_threshold = getattr(self.training_args, 'cosine_threshold', -1.0)  # 默认阈值为-1.0（允许所有更新）
            use_cosine_defense = getattr(self.training_args, 'use_cosine_defense', False)
            
            # 根据余弦相似度决定是否应用掩码和提交更新
            should_submit_update = True
            if use_cosine_defense and similarity_score is not None:
                if similarity_score < cosine_threshold:
                    logger.warning(f"节点 {self.node_id} 第 {self.current_round} 轮更新的余弦相似度({similarity_score:.4f})低于阈值({cosine_threshold})，被标记为潜在恶意更新")
                    should_submit_update = False
                else:
                    logger.info(f"节点 {self.node_id} 第 {self.current_round} 轮更新的余弦相似度({similarity_score:.4f})满足阈值要求({cosine_threshold})")
            
            # 应用掩码到模型更新
            if should_submit_update:
                if self.use_mask:
                    masked_delta_w = self.apply_mask_to_updates(delta_w)
                    
                    # 将掩码后的更新应用到本地模型（用于提案）
                    masked_model = copy.deepcopy(global_model_copy)
                    masked_state_dict = masked_model.state_dict()
                    
                    for key in masked_delta_w:
                        # 全局模型 + 掩码后的更新
                        masked_state_dict[key] = global_state_dict[key] + masked_delta_w[key]
                    
                    # 用于提案的掩码模型
                    self.masked_model_for_proposal = masked_model
                    logger.info(f"节点 {self.node_id} 已应用掩码到模型更新")
                else:
                    self.masked_model_for_proposal = self.local_model
                
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
            else:
                # 如果不提交更新，记录但不参与模型聚合
                logger.info(f"客户端 {self.node_id} 完成本地训练，但因余弦相似度检查未通过，不提交更新")
                # 避免将self.masked_model_for_proposal设置为任何值
                self.masked_model_for_proposal = None
                return True
        
        except Exception as e:
            logger.error(f"客户端 {self.node_id} 训练出错: {str(e)}")
            return False
    
    def propose_model(self):
        """提交本地训练的模型更新
        
        1. 计算本地模型与全局模型的差
        2. 如果启用了余弦相似度防御，检查更新是否可信
        3. 如果启用了掩码，应用掩码到模型更新
        4. 将掩码后的更新广播给所有节点
        """
        if self.model is None or self.local_model is None:
            logger.error(f"客户端 {self.node_id} 尚未接收全局模型或完成本地训练")
            return
        
        # 计算差值 delta_w = local_model - global_model
        delta_w = {}
        for key in self.local_model.state_dict():
            if key in self.model.state_dict():
                delta_w[key] = self.local_model.state_dict()[key] - self.model.state_dict()[key]
        
        # 如果启用了余弦相似度防御且不是第一轮，检查相似度
        if self.use_cosine_defense and self.previous_global_model and self.current_round > 0:
            similarity = self.calculate_update_similarity(delta_w)
            if similarity is not None:
                logger.info(f"客户端 {self.node_id} 本地更新与上一轮全局模型的余弦相似度: {similarity:.4f}")
                
                if similarity < self.cosine_threshold:
                    logger.warning(f"客户端 {self.node_id} 的更新相似度 {similarity:.4f} 低于阈值 {self.cosine_threshold}，" 
                                 f"被视为潜在恶意更新并丢弃")
                    return
                else:
                    logger.info(f"客户端 {self.node_id} 的更新通过余弦相似度检查")
        
        # 应用掩码到模型更新
        if self.use_mask:
            masked_delta_w = self.apply_mask_to_updates(delta_w)
        else:
            masked_delta_w = delta_w
        
        # 构造模型提案消息
        proposal_id = f"{self.node_id}_{self.current_round}_{time.time()}"
        model_proposal = {
            "node_id": self.node_id,
            "round": self.current_round,
            "delta_w": masked_delta_w,
            "timestamp": time.time(),
            "proposal_id": proposal_id
        }
        
        # 添加到本轮提案
        if self.current_round not in self.model_proposals:
            self.model_proposals[self.current_round] = {}
        self.model_proposals[self.current_round][self.node_id] = model_proposal
        
        # 广播模型提案
        self.broadcast("MODEL_PROPOSAL", model_proposal)
        logger.info(f"客户端 {self.node_id} 提交第 {self.current_round} 轮的模型更新")
    
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
        
        # 聚合模型
        logger.info(f"客户端 {self.node_id} 聚合轮次 {round_num} 的 {len(proposals)} 个模型")
        
        # 如果使用掩码，则采用安全聚合
        if self.use_mask:
            from masked_aggregation import secure_federated_aggregation
            
            # 获取当前轮次的掩码种子和符号映射
            mask_seed = self.consensus.get_mask_seed(round_num)
            sign_map = self.consensus.get_sign_map(round_num)
            
            if mask_seed is None or sign_map is None:
                logger.warning(f"无法获取轮次 {round_num} 的掩码种子或符号映射，将使用普通聚合")
            else:
                logger.info(f"使用掩码种子 {mask_seed} 和符号映射进行安全聚合")
                
                # 过滤出有效的节点和提案
                valid_proposals = {}
                for node_id, model_dict in proposals.items():
                    if node_id in sign_map:
                        valid_proposals[node_id] = model_dict
                
                # 调用安全聚合函数
                aggregated_model = secure_federated_aggregation(
                    valid_proposals, 
                    mask_seed, 
                    sign_map
                )
                
                if aggregated_model:
                    logger.info(f"安全聚合成功，聚合了 {len(valid_proposals)} 个模型")
                    return aggregated_model
                
                logger.warning("安全聚合失败，将回退到普通聚合")
        
        # 普通聚合（简单平均）
        logger.info(f"使用普通聚合方式（简单平均）聚合 {len(proposals)} 个模型")
        
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
        """请求模型聚合"""
        if not self.consensus.is_primary():
            logger.warning(f"非主节点 {self.node_id} 请求模型聚合，忽略")
            return False
        
        if self.current_round not in self.model_proposals:
            logger.warning(f"当前轮次 {self.current_round} 没有可用的模型提案")
            return False
        
        # 准备聚合请求
        request = {
            "type": "AGGREGATION",  # 修改为与_on_consensus_complete中匹配的类型
            "round": self.current_round,
            "node_ids": list(self.model_proposals[self.current_round].keys()),
            "timestamp": time.time()
        }
        
        logger.info(f"主节点 {self.node_id} 请求轮次 {self.current_round} 的模型聚合")
        
        # 提交请求给共识模块
        success = self.consensus.start_consensus(request)
        
        return success
    
    def _broadcast_pbft_message(self, message: Dict):
        """广播PBFT消息"""
        self.broadcast("PBFT_MESSAGE", message)
    
    def _send_pbft_message(self, node_id: int, message: Dict):
        """发送PBFT消息给指定节点"""
        self.send_to_peer(node_id, Message("PBFT_MESSAGE", self.node_id, message))
    
    def _on_consensus_complete(self, request: Dict) -> Any:
        """当共识完成时的回调函数"""
        if "type" not in request:
            return None
        
        # 处理模型聚合请求
        if request["type"] == "AGGREGATION":
            round_num = request["round"]
            node_ids = request["node_ids"]
            
            logger.info(f"节点 {self.node_id} 开始聚合轮次 {round_num} 的模型")
            
            # 聚合模型
            global_model = self.aggregate_models(round_num)
            
            if global_model is not None:
                # 更新全局模型
                self.model = global_model
                # 更新本地模型为全局模型
                self.local_model = copy.deepcopy(global_model)
                
                # 更新轮次
                if round_num >= self.current_round:
                    self.current_round = round_num + 1
                
                # 广播全局模型给所有客户端
                self._broadcast_global_model(round_num)
                
                logger.info(f"节点 {self.node_id} 完成轮次 {round_num} 的模型聚合，当前轮次: {self.current_round}")
                
                # 返回聚合结果
                return {
                    "status": "success",
                    "round": round_num,
                    "message": f"模型聚合成功，当前轮次: {self.current_round}"
                }
            else:
                logger.error(f"客户端 {self.node_id} 聚合轮次 {round_num} 的模型失败")
                return None
        
        return None
    
    def _broadcast_global_model(self, round_num: int):
        """广播全局模型"""
        if self.model is None:
            logger.error(f"客户端 {self.node_id} 无法广播全局模型：模型未初始化")
            return
        
        # 创建全局模型消息
        model_msg = {
            "type": "GLOBAL_MODEL",
            "round": round_num,
            "model": self.model.state_dict(),
            "source_node": self.node_id,
            "timestamp": time.time(),
            "is_new_round": True  # 标记为新一轮开始的全局模型
        }
        
        # 广播全局模型
        self.broadcast("GLOBAL_MODEL", model_msg)
        
        logger.info(f"客户端 {self.node_id} 广播轮次 {round_num} 的全局模型作为第 {round_num + 1} 轮的起始模型")
    
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
        """处理全局模型消息
        
        Args:
            sender_id: 发送者ID
            content: 消息内容
        """
        round_num = content.get("round", 0)
        model_state = content.get("model_state", None)
        
        logger.info(f"客户端 {self.node_id} 从节点 {sender_id} 接收第 {round_num} 轮的全局模型")
        
        if model_state is None:
            logger.error(f"接收到的全局模型状态为空")
            return
        
        # 如果是新一轮，保存当前全局模型为上一轮全局模型
        if round_num > self.current_round and self.model is not None:
            self.previous_global_model = copy.deepcopy(self.model)
            logger.info(f"客户端 {self.node_id} 保存第 {self.current_round} 轮的全局模型作为上一轮模型")
        
        # 更新当前回合
        self.current_round = round_num
        
        # 更新全局模型
        if self.model is None:
            # 第一次接收全局模型，需要初始化
            try:
                # 尝试根据模型状态重建模型
                # 这里假设模型结构在所有节点上都是一致的
                self.model = copy.deepcopy(self.local_model)
                self.model.load_state_dict(model_state)
            except Exception as e:
                logger.error(f"加载全局模型失败: {str(e)}")
                return
        else:
            # 更新现有模型
            try:
                self.model.load_state_dict(model_state)
            except Exception as e:
                logger.error(f"更新全局模型失败: {str(e)}")
                return
        
        # 重置本地模型为全局模型
        if self.local_model is not None:
            self.local_model = copy.deepcopy(self.model)
        
        logger.info(f"客户端 {self.node_id} 成功更新第 {round_num} 轮的全局模型")
    
    def _handle_model_proposal(self, sender_id: int, content: Dict):
        """处理模型提案消息"""
        round_num = content["round"]
        model_dict = content["model"]
        node_id = content["node_id"]
        is_masked = content.get("masked", False)
        
        logger.info(f"客户端 {self.node_id} 从节点 {sender_id} 接收轮次 {round_num} 的模型提案 (掩码: {is_masked})")
        
        # 初始化该轮的提案集
        if round_num not in self.model_proposals:
            self.model_proposals[round_num] = {}
        
        # 保存提案
        self.model_proposals[round_num][node_id] = model_dict
        
        # 如果是主节点，并且已收集足够的提案，则启动共识
        if self.consensus.is_primary():
            min_proposals_needed = max(2, len(self.consensus.state.validators) // 2 + 1)
            if len(self.model_proposals[round_num]) >= min_proposals_needed:
                # 检查是否有正在进行的共识
                if self.consensus.state.current_request is None:
                    self.request_model_aggregation()
                    logger.info(f"主节点 {self.node_id} 自动请求轮次 {round_num} 的模型聚合，已收到 {len(self.model_proposals[round_num])} 个提案")
    
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
        # 简单实现：任一节点收到所有节点的投票后，完成选举
        if len(self.election.votes.get(term, {}).get(self.node_id, {})) >= len(self.known_nodes) // 2 + 1:
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
    
    def _broadcast_pvss_key(self):
        """广播本节点的PVSS公钥"""
        pubkey = self.pvss_handler.get_public_key()
        
        key_msg = {
            "type": "PVSS_KEY",
            "node_id": self.node_id,
            "public_key": pubkey,
            "timestamp": time.time()
        }
        
        self.broadcast("PVSS_KEY", key_msg)
        logger.info(f"节点 {self.node_id} 广播PVSS公钥")
    
    def _handle_pvss_key(self, sender_id: int, content: Dict):
        """处理PVSS公钥消息"""
        node_id = content["node_id"]
        pubkey = content["public_key"]
        
        # 保存公钥
        self.pvss_public_keys[node_id] = pubkey
        
        # 设置到PVSS处理器
        self.pvss_handler.set_public_key(node_id, pubkey)
        
        logger.info(f"节点 {self.node_id} 接收到节点 {node_id} 的PVSS公钥")
    
    def set_use_mask(self, use_mask: bool):
        """设置是否使用掩码
        
        Args:
            use_mask: 是否使用掩码
        """
        self.use_mask = use_mask
        self.consensus.set_use_mask(use_mask)
        logger.info(f"节点 {self.node_id} 设置使用掩码: {use_mask}")
    
    def apply_mask_to_updates(self, delta_w):
        """应用掩码到模型更新（增量权重）
        
        步骤3：应用掩码到模型更新
        - 从共识达成的sign_map中查找本节点的符号值
        - 使用mask_seed生成掩码
        - 对模型更新应用掩码
        
        Args:
            delta_w: 模型参数更新（字典形式）
            
        Returns:
            Dict: 应用掩码后的模型更新
        """
        if not self.use_mask:
            logger.info(f"节点 {self.node_id} 未启用掩码，返回原始更新")
            return delta_w
        
        # 从masked_aggregation模块导入函数
        from masked_aggregation import apply_mask_to_model_update
        
        # 获取当前轮次的掩码种子和符号映射
        mask_seed = self.consensus.get_mask_seed(self.current_round)
        sign_map = self.consensus.get_sign_map(self.current_round)
        
        if mask_seed is None or sign_map is None:
            logger.warning(f"节点 {self.node_id} 无法获取掩码种子或符号映射，返回原始更新")
            return delta_w
            
        # 获取当前节点的符号
        if self.node_id not in sign_map:
            logger.warning(f"节点 {self.node_id} 不在符号映射中，返回原始更新")
            return delta_w
            
        sign = sign_map[self.node_id]
        logger.info(f"节点 {self.node_id} 使用掩码种子 {mask_seed} 和符号 {sign} 应用掩码到模型更新")
        
        # 应用掩码到模型更新
        masked_delta_w = apply_mask_to_model_update(delta_w, mask_seed, sign)
        
        return masked_delta_w
    
    def set_cosine_defense(self, use_defense: bool, threshold: float = -0.1):
        """设置是否使用余弦相似度防御及阈值
        
        Args:
            use_defense: 是否使用余弦相似度防御
            threshold: 余弦相似度阈值，低于此值的更新将被丢弃
        """
        self.use_cosine_defense = use_defense
        self.cosine_threshold = threshold
        logger.info(f"节点 {self.node_id} 设置余弦相似度防御: 启用={use_defense}, 阈值={threshold}")
    
    def calculate_update_similarity(self, delta_w):
        """计算本地更新与上一轮全局模型的余弦相似度
        
        Args:
            delta_w: 模型参数更新（字典形式）
            
        Returns:
            float: 计算的余弦相似度，如果无法计算则返回None
        """
        if not self.previous_global_model:
            logger.info(f"节点 {self.node_id} 没有上一轮全局模型，无法计算相似度")
            return None
        
        try:
            # 展平delta_w为1D张量
            delta_w_keys = list(delta_w.keys())
            flattened_delta = torch.cat([delta_w[key].flatten() for key in delta_w_keys])
            
            # 从上一轮全局模型中提取相应参数
            previous_global_state = self.previous_global_model.state_dict()
            global_params = []
            for key in delta_w_keys:
                if key in previous_global_state:
                    global_params.append(previous_global_state[key].flatten())
            
            if not global_params:
                logger.warning(f"节点 {self.node_id} 无法在上一轮全局模型中找到匹配的参数")
                return None
            
            flattened_prev_global = torch.cat(global_params)
            
            # 计算余弦相似度
            similarity_score = torch.nn.functional.cosine_similarity(
                flattened_delta.unsqueeze(0), 
                flattened_prev_global.unsqueeze(0)
            ).item()
            
            return similarity_score
        except Exception as e:
            logger.error(f"计算余弦相似度时出错: {str(e)}")
            return None 
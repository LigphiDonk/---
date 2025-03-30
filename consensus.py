import time
import hashlib
import logging
import copy
from typing import Dict, List, Set, Any, Optional, Tuple
from enum import Enum
import random
import torch

# 导入PVSS工具
from pvss_utils import generate_sign_map, generate_mask, PVSSHandler

logger = logging.getLogger("PBFT")

class MessageType(Enum):
    """PBFT协议中的消息类型"""
    REQUEST = "REQUEST"           # 请求更新
    PRE_PREPARE = "PRE_PREPARE"   # 预准备
    PREPARE = "PREPARE"           # 准备
    COMMIT = "COMMIT"             # 提交
    REPLY = "REPLY"               # 回复
    VIEW_CHANGE = "VIEW_CHANGE"   # 视图变更
    NEW_VIEW = "NEW_VIEW"         # 新视图


class PBFTState:
    """PBFT共识状态"""
    
    def __init__(self, node_id: int, f: int = 1):
        """初始化PBFT状态
        
        Args:
            node_id: 节点ID
            f: 容错数量，系统最多容忍f个恶意节点
        """
        self.node_id = node_id
        self.f = f  # 容错数量
        
        # 共识相关状态
        self.view = 0  # 当前视图编号
        self.sequence_number = 0  # 当前序列号
        
        # 主节点映射 {view -> primary_id}
        self.primary_nodes: Dict[int, int] = {}
        
        # 验证节点集合
        self.validators: Set[int] = set()
        
        # 消息日志
        self.pre_prepare_log: Dict[Tuple[int, int], Dict] = {}  # (view, seq) -> pre-prepare消息
        self.prepare_log: Dict[Tuple[int, int], Dict[int, Dict]] = {}  # (view, seq) -> {node_id -> prepare消息}
        self.commit_log: Dict[Tuple[int, int], Dict[int, Dict]] = {}  # (view, seq) -> {node_id -> commit消息}
        
        # 已完成共识的请求
        self.completed_requests: Set[str] = set()
        
        # 视图变更相关
        self.view_change_log: Dict[int, Dict[int, Dict]] = {}  # view -> {node_id -> view-change消息}
        self.view_change_timeout = 10.0  # 视图变更超时时间(秒)
        self.last_request_time = time.time()
        
        # 当前待处理的请求
        self.current_request = None
        
        # 处理完成的回调函数
        self.on_consensus_complete = None
        
        # PVSS掩码相关
        self.mask_seeds: Dict[Tuple[int, int], int] = {}  # (view, seq) -> 掩码种子
        self.sign_maps: Dict[Tuple[int, int], Dict[int, int]] = {}  # (view, seq) -> 符号映射 {node_id -> sign}
        self.use_mask = True  # 是否使用掩码，可通过配置开关
        
        # PVSS处理器
        self.pvss_handler = PVSSHandler(node_id, f)
        
        # PVSS份额和证明存储
        self.pvss_shares: Dict[Tuple[int, int], Dict[int, Dict]] = {}  # (view, seq) -> {node_id -> share}
        self.pvss_proofs: Dict[Tuple[int, int], Dict[int, bytes]] = {}  # (view, seq) -> {node_id -> proof}
        self.pvss_verified: Dict[Tuple[int, int], bool] = {}  # (view, seq) -> 是否已验证PVSS
    
    def set_validators(self, validators: Set[int]):
        """设置验证节点集合"""
        self.validators = validators
        # 重新计算主节点映射
        for v in range(100):  # 预计算100个视图的主节点
            if validators:
                self.primary_nodes[v] = list(validators)[v % len(validators)]
            
        logger.info(f"节点 {self.node_id} 设置验证节点集合: {validators}")
    
    def get_primary(self, view: int = None) -> Optional[int]:
        """获取指定视图的主节点ID"""
        view = self.view if view is None else view
        return self.primary_nodes.get(view)
    
    def is_primary(self) -> bool:
        """当前节点是否是主节点"""
        return self.node_id == self.get_primary()
    
    def create_request_id(self, request: Dict) -> str:
        """创建请求ID，用于唯一标识一个请求"""
        # 使用请求的哈希作为ID
        request_str = str(request).encode('utf-8')
        return hashlib.sha256(request_str).hexdigest()
    
    def create_message_digest(self, message: Dict) -> str:
        """创建消息摘要"""
        message_str = str(message).encode('utf-8')
        return hashlib.sha256(message_str).hexdigest()
    
    def handle_request(self, request: Dict) -> Tuple[bool, Optional[Dict]]:
        """处理新的请求
        
        如果当前节点是主节点，则发送PRE-PREPARE消息
        
        Args:
            request: 请求内容
            
        Returns:
            Tuple[bool, Optional[Dict]]: 是否成功处理, PRE-PREPARE消息(如有)
        """
        request_id = self.create_request_id(request)
        
        # 检查请求是否已经完成
        if request_id in self.completed_requests:
            logger.info(f"请求 {request_id} 已处理完成，忽略")
            return False, None
        
        # 只有主节点处理请求
        if not self.is_primary():
            logger.warning(f"非主节点收到请求，忽略")
            return False, None
        
        # 生成PVSS掩码相关数据
        mask_seed = None
        sign_map = None
        pvss_shares = None
        pvss_proofs = None
        
        if self.use_mask and "node_ids" in request:
            # 从请求中获取参与节点列表
            node_ids = request.get("node_ids", [])
            if node_ids:
                # 生成随机掩码种子
                mask_seed = random.randint(10000, 9999999)
                # 生成符号映射
                sign_map = generate_sign_map(node_ids)
                logger.info(f"主节点 {self.node_id} 生成掩码种子 {mask_seed} 和符号映射 {sign_map}")
                
                # 使用PVSS分发掩码种子
                pvss_shares, pvss_proofs = self.pvss_handler.deal_secret(mask_seed, node_ids)
                logger.info(f"主节点 {self.node_id} 使用PVSS分发掩码种子，生成了 {len(pvss_shares)} 个份额和证明")
        
        # 创建PRE-PREPARE消息
        pre_prepare = {
            "type": MessageType.PRE_PREPARE.value,
            "view": self.view,
            "sequence": self.sequence_number,
            "request_id": request_id,
            "digest": self.create_message_digest(request),
            "timestamp": time.time(),
            "request": request
        }
        
        # 如果启用了掩码，添加掩码相关字段
        if mask_seed is not None and sign_map is not None:
            pre_prepare["mask_seed"] = mask_seed
            pre_prepare["sign_map"] = sign_map
            
            # 如果有PVSS份额和证明，添加到消息中
            if pvss_shares and pvss_proofs:
                pre_prepare["pvss_shares"] = pvss_shares
                pre_prepare["pvss_proofs"] = pvss_proofs
            
            # 保存掩码种子和符号映射
            self.mask_seeds[(self.view, self.sequence_number)] = mask_seed
            self.sign_maps[(self.view, self.sequence_number)] = sign_map
            
            # 保存PVSS份额和证明
            self.pvss_shares[(self.view, self.sequence_number)] = pvss_shares
            self.pvss_proofs[(self.view, self.sequence_number)] = pvss_proofs
            
            # 标记为已验证（对于主节点来说）
            self.pvss_verified[(self.view, self.sequence_number)] = True
        
        # 保存PRE-PREPARE消息
        self.pre_prepare_log[(self.view, self.sequence_number)] = pre_prepare
        self.current_request = request
        
        # 更新序列号
        self.sequence_number += 1
        self.last_request_time = time.time()
        
        logger.info(f"主节点 {self.node_id} 生成PRE-PREPARE消息: view={self.view}, seq={self.sequence_number-1}")
        
        # 返回PRE-PREPARE消息，由调用者负责广播
        return True, pre_prepare
    
    def handle_pre_prepare(self, pre_prepare: Dict, sender_id: int) -> Tuple[bool, Optional[Dict]]:
        """处理PRE-PREPARE消息
        
        如果消息有效，则生成PREPARE消息
        
        PVSS验证流程:
        1. 从PRE-PREPARE消息中提取PVSS份额和证明
        2. 验证自己的份额是否有效（使用当前节点的私钥和主节点的公钥）
        3. 如果验证通过，则在PREPARE消息中包含PVSS验证状态
        4. 其他节点收到PREPARE消息后，会检查PVSS验证状态
        5. 只有PVSS验证通过的PREPARE消息才会被接受
        6. 当收集到足够的PREPARE消息后，节点会生成COMMIT消息
        7. 共识达成后，每个节点可以使用收集到的份额重构掩码种子
        
        Args:
            pre_prepare: PRE-PREPARE消息
            sender_id: 发送者ID
            
        Returns:
            Tuple[bool, Optional[Dict]]: 是否有效, PREPARE消息(如有)
        """
        # 检查发送者是否是主节点
        view = pre_prepare["view"]
        if sender_id != self.get_primary(view):
            logger.warning(f"PRE-PREPARE消息发送者 {sender_id} 不是主节点")
            return False, None
        
        # 提取消息字段
        seq = pre_prepare["sequence"]
        request_id = pre_prepare["request_id"]
        digest = pre_prepare["digest"]
        
        # 检查视图是否正确
        if view != self.view:
            logger.warning(f"PRE-PREPARE消息视图不一致: {view} != {self.view}")
            return False, None
        
        # 检查是否已接收过该请求的PRE-PREPARE
        if (view, seq) in self.pre_prepare_log:
            logger.warning(f"已接收过视图{view}序列{seq}的PRE-PREPARE消息")
            return False, None
        
        # 检查摘要是否正确
        request = pre_prepare["request"]
        if digest != self.create_message_digest(request):
            logger.warning(f"PRE-PREPARE消息摘要不匹配")
            return False, None
        
        # 验证掩码种子和符号映射（如果存在）
        pvss_verified = False
        if self.use_mask and "mask_seed" in pre_prepare and "sign_map" in pre_prepare:
            mask_seed = pre_prepare["mask_seed"]
            sign_map = pre_prepare["sign_map"]
            
            # 验证符号映射
            if "node_ids" in request:
                node_ids = request.get("node_ids", [])
                # 检查所有参与节点是否都在符号映射中
                for node_id in node_ids:
                    if node_id not in sign_map:
                        logger.warning(f"符号映射中缺少节点 {node_id}")
                        return False, None
                
                # 检查符号映射是否正确（如果节点数是4的倍数，总和应为0）
                if len(node_ids) % 4 == 0 and sum(sign_map.values()) != 0:
                    logger.warning(f"符号映射总和不为0: {sum(sign_map.values())}")
                    return False, None
                
                # 处理PVSS份额和证明验证
                if "pvss_shares" in pre_prepare and "pvss_proofs" in pre_prepare:
                    pvss_shares = pre_prepare["pvss_shares"]
                    pvss_proofs = pre_prepare["pvss_proofs"]
                    
                    # 获取当前节点的份额和证明，注意处理字符串键
                    node_id_str = str(self.node_id)
                    if node_id_str in pvss_shares and node_id_str in pvss_proofs:
                        my_share = pvss_shares[node_id_str]
                        my_proof = pvss_proofs[node_id_str]
                        
                        logger.info(f"节点 {self.node_id} 从PRE-PREPARE获取份额: {my_share}")
                        
                        # 使用PVSS验证份额
                        if self.pvss_handler.verify_share(my_share, my_proof, sender_id):
                            logger.info(f"节点 {self.node_id} 成功验证PVSS份额")
                            pvss_verified = True
                        else:
                            logger.warning(f"节点 {self.node_id} 验证PVSS份额失败")
                            return False, None
                    else:
                        logger.warning(f"PRE-PREPARE消息中缺少节点 {self.node_id} 的PVSS份额或证明")
                        logger.warning(f"可用份额节点: {list(pvss_shares.keys())}")
                        return False, None
                    
                    # 保存PVSS份额和证明
                    self.pvss_shares[(view, seq)] = pvss_shares
                    self.pvss_proofs[(view, seq)] = pvss_proofs
                    # 标记为已验证
                    self.pvss_verified[(view, seq)] = pvss_verified
                
                # 保存掩码种子和符号映射
                self.mask_seeds[(view, seq)] = mask_seed
                self.sign_maps[(view, seq)] = sign_map
                
                logger.info(f"节点 {self.node_id} 接收掩码种子 {mask_seed} 和符号映射 {sign_map}")
            else:
                logger.warning("请求中缺少node_ids字段，无法验证符号映射")
                return False, None
        
        # 保存PRE-PREPARE消息
        self.pre_prepare_log[(view, seq)] = pre_prepare
        self.current_request = request
        
        # 创建PREPARE消息
        prepare = {
            "type": MessageType.PREPARE.value,
            "view": view,
            "sequence": seq,
            "request_id": request_id,
            "digest": digest,
            "node_id": self.node_id,
            "timestamp": time.time(),
            "pvss_verified": pvss_verified  # 添加PVSS验证状态
        }
        
        # 初始化prepare_log
        if (view, seq) not in self.prepare_log:
            self.prepare_log[(view, seq)] = {}
        
        # 保存自己的PREPARE消息
        self.prepare_log[(view, seq)][self.node_id] = prepare
        
        logger.info(f"节点 {self.node_id} 生成PREPARE消息: view={view}, seq={seq}")
        
        # 返回PREPARE消息，由调用者负责广播
        return True, prepare
    
    def handle_prepare(self, prepare: Dict, sender_id: int) -> Tuple[bool, Optional[Dict]]:
        """处理PREPARE消息
        
        当接收到2f个不同节点的PREPARE消息后，生成COMMIT消息
        
        Args:
            prepare: PREPARE消息
            sender_id: 发送者ID
            
        Returns:
            Tuple[bool, Optional[Dict]]: 是否有效, COMMIT消息(如有)
        """
        # 提取消息字段
        view = prepare["view"]
        seq = prepare["sequence"]
        digest = prepare["digest"]
        
        # 检查视图是否正确
        if view != self.view:
            logger.warning(f"PREPARE消息视图不一致: {view} != {self.view}")
            return False, None
        
        # 检查是否已接收过PRE-PREPARE消息
        if (view, seq) not in self.pre_prepare_log:
            logger.warning(f"未接收过视图{view}序列{seq}的PRE-PREPARE消息")
            return False, None
        
        # 检查摘要是否与PRE-PREPARE一致
        pre_prepare = self.pre_prepare_log[(view, seq)]
        if digest != pre_prepare["digest"]:
            logger.warning(f"PREPARE消息摘要与PRE-PREPARE不一致")
            return False, None
        
        # 检查PVSS验证状态（如果消息中包含）
        if "pvss_verified" in prepare and not prepare["pvss_verified"]:
            logger.warning(f"节点 {sender_id} 的PREPARE消息中PVSS验证失败")
            return False, None
        
        # 初始化prepare_log
        if (view, seq) not in self.prepare_log:
            self.prepare_log[(view, seq)] = {}
        
        # 保存PREPARE消息
        self.prepare_log[(view, seq)][sender_id] = prepare
        
        # 检查是否已收到足够的PREPARE消息
        if not self.prepared(view, seq):
            return True, None
        
        # 已收到足够的PREPARE消息，进入prepared状态
        logger.info(f"节点 {self.node_id} 已收到足够PREPARE消息: view={view}, seq={seq}")
        
        # 创建COMMIT消息
        commit = {
            "type": MessageType.COMMIT.value,
            "view": view,
            "sequence": seq,
            "request_id": pre_prepare["request_id"],
            "digest": digest,
            "node_id": self.node_id,
            "timestamp": time.time()
        }
        
        # 初始化commit_log
        if (view, seq) not in self.commit_log:
            self.commit_log[(view, seq)] = {}
        
        # 保存自己的COMMIT消息
        self.commit_log[(view, seq)][self.node_id] = commit
        
        logger.info(f"节点 {self.node_id} 生成COMMIT消息: view={view}, seq={seq}")
        
        # 返回COMMIT消息，由调用者负责广播
        return True, commit
    
    def handle_commit(self, commit: Dict, sender_id: int) -> Tuple[bool, Optional[Dict]]:
        """处理COMMIT消息
        
        当接收到2f+1个不同节点的COMMIT消息后，请求达成共识，执行请求
        
        Args:
            commit: COMMIT消息
            sender_id: 发送者ID
            
        Returns:
            Tuple[bool, Optional[Dict]]: 是否有效, 执行结果(如有)
        """
        # 提取消息字段
        view = commit["view"]
        seq = commit["sequence"]
        digest = commit["digest"]
        
        # 检查视图是否正确
        if view != self.view:
            logger.warning(f"COMMIT消息视图不一致: {view} != {self.view}")
            return False, None
        
        # 检查是否已接收过PRE-PREPARE消息
        if (view, seq) not in self.pre_prepare_log:
            logger.warning(f"未接收过视图{view}序列{seq}的PRE-PREPARE消息")
            return False, None
        
        # 检查摘要是否与PRE-PREPARE一致
        pre_prepare = self.pre_prepare_log[(view, seq)]
        if digest != pre_prepare["digest"]:
            logger.warning(f"COMMIT消息摘要与PRE-PREPARE不一致")
            return False, None
        
        # 初始化commit_log
        if (view, seq) not in self.commit_log:
            self.commit_log[(view, seq)] = {}
        
        # 保存COMMIT消息
        self.commit_log[(view, seq)][sender_id] = commit
        
        # 检查是否已收到足够的COMMIT消息
        if not self.committed(view, seq):
            return True, None
        
        # 已收到足够的COMMIT消息，请求达成共识
        logger.info(f"节点 {self.node_id} 请求达成共识: view={view}, seq={seq}")
        
        # 获取请求
        request = pre_prepare["request"]
        request_id = pre_prepare["request_id"]
        
        # 尝试重构PVSS秘密（掩码种子）
        if self.use_mask and (view, seq) in self.pvss_shares and (view, seq) in self.pvss_verified:
            if self.pvss_verified.get((view, seq), False):
                # 重构掩码种子
                self._reconstruct_mask_seed(view, seq)
        
        # 标记请求已完成
        self.completed_requests.add(request_id)
        
        # 如果设置了回调函数，则调用
        result = None
        if self.on_consensus_complete:
            result = self.on_consensus_complete(request)
        
        # 创建REPLY消息
        reply = {
            "type": MessageType.REPLY.value,
            "view": view,
            "sequence": seq,
            "request_id": request_id,
            "result": result,
            "node_id": self.node_id,
            "timestamp": time.time()
        }
        
        logger.info(f"节点 {self.node_id} 生成REPLY消息: view={view}, seq={seq}")
        
        # 返回REPLY消息，由调用者决定是否发送
        return True, reply
    
    def _reconstruct_mask_seed(self, view: int, seq: int) -> bool:
        """重构掩码种子
        
        使用PVSS收集的份额重构掩码种子
        
        Args:
            view: 视图编号
            seq: 序列号
            
        Returns:
            bool: 是否成功重构
        """
        # 检查是否有PVSS份额
        if (view, seq) not in self.pvss_shares:
            logger.warning(f"找不到视图 {view} 序列 {seq} 的PVSS份额")
            return False
        
        # 已经有种子则直接返回
        if (view, seq) in self.mask_seeds and self.mask_seeds[(view, seq)] is not None:
            logger.info(f"已有掩码种子: {self.mask_seeds[(view, seq)]}")
            return True
        
        # 获取所有验证过的份额（这里简化处理，假设所有PREPARE通过的节点都有有效份额）
        shares = {}
        for node_id in self.prepare_log.get((view, seq), {}):
            if str(node_id) in self.pvss_shares.get((view, seq), {}):
                share = self.pvss_shares[(view, seq)][str(node_id)]
                shares[node_id] = share
        
        # 尝试重构秘密
        seed = self.pvss_handler.reconstruct_secret(shares)
        if seed is not None:
            logger.info(f"节点 {self.node_id} 成功重构掩码种子: {seed}")
            self.mask_seeds[(view, seq)] = seed
            return True
        else:
            logger.warning(f"节点 {self.node_id} 重构掩码种子失败")
            return False
    
    def prepared(self, view: int, seq: int) -> bool:
        """检查是否已收到足够的PREPARE消息
        
        需要满足条件:
        1. 已收到PRE-PREPARE消息
        2. 已收到至少2f个不同节点的PREPARE消息(包括自己)
        
        Args:
            view: 视图编号
            seq: 序列号
            
        Returns:
            bool: 是否已准备好
        """
        # 检查是否已接收过PRE-PREPARE消息
        if (view, seq) not in self.pre_prepare_log:
            return False
        
        # 检查PREPARE消息数量
        prepare_count = len(self.prepare_log.get((view, seq), {}))
        return prepare_count >= 2 * self.f
    
    def committed(self, view: int, seq: int) -> bool:
        """检查是否已收到足够的COMMIT消息
        
        需要满足条件:
        1. 已进入prepared状态
        2. 已收到至少2f+1个不同节点的COMMIT消息(包括自己)
        
        Args:
            view: 视图编号
            seq: 序列号
            
        Returns:
            bool: 是否已提交
        """
        # 检查是否已进入prepared状态
        if not self.prepared(view, seq):
            return False
        
        # 检查COMMIT消息数量
        commit_count = len(self.commit_log.get((view, seq), {}))
        return commit_count >= 2 * self.f + 1
    
    def check_view_change(self) -> bool:
        """检查是否需要发起视图变更
        
        当主节点长时间未处理请求时，发起视图变更
        
        Returns:
            bool: 是否需要视图变更
        """
        # 如果是主节点，则不发起视图变更
        if self.is_primary():
            return False
        
        # 检查距离上次请求的时间
        time_since_last_request = time.time() - self.last_request_time
        return time_since_last_request > self.view_change_timeout
    
    def start_view_change(self) -> Dict:
        """发起视图变更
        
        Returns:
            Dict: VIEW-CHANGE消息
        """
        # 计算新视图编号
        new_view = self.view + 1
        
        # 收集已准备好但未提交的请求
        prepared_requests = []
        for (view, seq), pre_prepare in self.pre_prepare_log.items():
            if view < self.view and self.prepared(view, seq) and not self.committed(view, seq):
                prepared_requests.append({
                    "view": view,
                    "sequence": seq,
                    "digest": pre_prepare["digest"],
                    "prepares": list(self.prepare_log.get((view, seq), {}).values())
                })
        
        # 创建VIEW-CHANGE消息
        view_change = {
            "type": MessageType.VIEW_CHANGE.value,
            "new_view": new_view,
            "last_sequence": self.sequence_number - 1,
            "prepared_requests": prepared_requests,
            "node_id": self.node_id,
            "timestamp": time.time()
        }
        
        # 初始化view_change_log
        if new_view not in self.view_change_log:
            self.view_change_log[new_view] = {}
        
        # 保存自己的VIEW-CHANGE消息
        self.view_change_log[new_view][self.node_id] = view_change
        
        logger.info(f"节点 {self.node_id} 发起视图变更: {self.view} -> {new_view}")
        
        # 更新视图编号
        self.view = new_view
        
        # 返回VIEW-CHANGE消息，由调用者负责广播
        return view_change
    
    def handle_view_change(self, view_change: Dict, sender_id: int) -> Tuple[bool, Optional[Dict]]:
        """处理VIEW-CHANGE消息
        
        当收到2f+1个不同节点的VIEW-CHANGE消息后，如果当前节点是新视图的主节点，则生成NEW-VIEW消息
        
        Args:
            view_change: VIEW-CHANGE消息
            sender_id: 发送者ID
            
        Returns:
            Tuple[bool, Optional[Dict]]: 是否有效, NEW-VIEW消息(如有)
        """
        # 提取消息字段
        new_view = view_change["new_view"]
        
        # 如果新视图小于当前视图，则忽略
        if new_view < self.view:
            logger.warning(f"视图变更消息的新视图 {new_view} 小于当前视图 {self.view}")
            return False, None
        
        # 初始化view_change_log
        if new_view not in self.view_change_log:
            self.view_change_log[new_view] = {}
        
        # 保存VIEW-CHANGE消息
        self.view_change_log[new_view][sender_id] = view_change
        
        # 如果新视图大于当前视图，则更新当前视图
        if new_view > self.view:
            logger.info(f"节点 {self.node_id} 视图变更: {self.view} -> {new_view}")
            self.view = new_view
        
        # 检查是否已收到足够的VIEW-CHANGE消息
        view_change_count = len(self.view_change_log.get(new_view, {}))
        if view_change_count < 2 * self.f + 1:
            return True, None
        
        # 检查当前节点是否是新视图的主节点
        if self.node_id != self.get_primary(new_view):
            return True, None
        
        # 已收到足够的VIEW-CHANGE消息，且当前节点是新视图的主节点，生成NEW-VIEW消息
        logger.info(f"节点 {self.node_id} 是新视图 {new_view} 的主节点，生成NEW-VIEW消息")
        
        # 收集VIEW-CHANGE消息
        view_changes = list(self.view_change_log[new_view].values())
        
        # 收集需要在新视图中重新执行的请求
        reexecute_requests = []
        for view_change in view_changes:
            for prepared in view_change["prepared_requests"]:
                # 检查是否已包含相同的请求
                exists = False
                for req in reexecute_requests:
                    if req["sequence"] == prepared["sequence"] and req["digest"] == prepared["digest"]:
                        exists = True
                        break
                
                if not exists:
                    reexecute_requests.append(prepared)
        
        # 对请求按序列号排序
        reexecute_requests.sort(key=lambda x: x["sequence"])
        
        # 创建NEW-VIEW消息
        new_view_msg = {
            "type": MessageType.NEW_VIEW.value,
            "new_view": new_view,
            "view_changes": view_changes,
            "reexecute_requests": reexecute_requests,
            "node_id": self.node_id,
            "timestamp": time.time()
        }
        
        logger.info(f"节点 {self.node_id} 生成NEW-VIEW消息: view={new_view}")
        
        # 返回NEW-VIEW消息，由调用者负责广播
        return True, new_view_msg
    
    def handle_new_view(self, new_view_msg: Dict, sender_id: int) -> bool:
        """处理NEW-VIEW消息
        
        处理新视图消息，重新执行未完成的请求
        
        Args:
            new_view_msg: NEW-VIEW消息
            sender_id: 发送者ID
            
        Returns:
            bool: 是否有效
        """
        # 提取消息字段
        new_view = new_view_msg["new_view"]
        
        # 检查发送者是否是新视图的主节点
        if sender_id != self.get_primary(new_view):
            logger.warning(f"NEW-VIEW消息发送者 {sender_id} 不是新视图 {new_view} 的主节点")
            return False
        
        # 检查新视图是否大于等于当前视图
        if new_view < self.view:
            logger.warning(f"NEW-VIEW消息的新视图 {new_view} 小于当前视图 {self.view}")
            return False
        
        # 检查VIEW-CHANGE消息数量
        view_changes = new_view_msg["view_changes"]
        if len(view_changes) < 2 * self.f + 1:
            logger.warning(f"NEW-VIEW消息中的VIEW-CHANGE消息数量不足: {len(view_changes)} < {2*self.f+1}")
            return False
        
        # 更新视图
        logger.info(f"节点 {self.node_id} 接受新视图: {self.view} -> {new_view}")
        self.view = new_view
        
        # 重新执行未完成的请求
        reexecute_requests = new_view_msg["reexecute_requests"]
        for request in reexecute_requests:
            # TODO: 重新执行请求
            pass
        
        # 清理旧视图的日志
        self._clean_logs_before_view(new_view)
        
        return True
    
    def _clean_logs_before_view(self, view: int):
        """清理指定视图之前的日志"""
        # 清理PRE-PREPARE日志
        for (v, seq) in list(self.pre_prepare_log.keys()):
            if v < view:
                del self.pre_prepare_log[(v, seq)]
        
        # 清理PREPARE日志
        for (v, seq) in list(self.prepare_log.keys()):
            if v < view:
                del self.prepare_log[(v, seq)]
        
        # 清理COMMIT日志
        for (v, seq) in list(self.commit_log.keys()):
            if v < view:
                del self.commit_log[(v, seq)]
        
        # 清理视图变更日志
        for v in list(self.view_change_log.keys()):
            if v < view:
                del self.view_change_log[v]

    def get_mask_seed(self, view: int, seq: int) -> Optional[int]:
        """获取指定视图和序列号的掩码种子"""
        return self.mask_seeds.get((view, seq))
    
    def get_sign_map(self, view: int, seq: int) -> Optional[Dict[int, int]]:
        """获取指定视图和序列号的符号映射"""
        return self.sign_maps.get((view, seq))
    
    def get_sign(self, view: int, seq: int, node_id: int) -> Optional[int]:
        """获取指定节点的符号值"""
        sign_map = self.get_sign_map(view, seq)
        if sign_map:
            return sign_map.get(node_id)
        return None
    
    def set_use_mask(self, use_mask: bool):
        """设置是否使用掩码
        
        Args:
            use_mask: 是否使用掩码
        """
        self.use_mask = use_mask
        logger.info(f"节点 {self.node_id} 设置使用掩码: {use_mask}")


class PBFTConsensus:
    """PBFT共识处理器"""
    
    def __init__(self, node_id: int, validators: Set[int] = None, f: int = 1):
        """初始化PBFT共识处理器
        
        Args:
            node_id: 节点ID
            validators: 验证节点集合
            f: 容错数量，系统最多容忍f个恶意节点
        """
        self.node_id = node_id
        self.state = PBFTState(node_id, f)
        
        # 设置验证节点
        if validators:
            self.set_validators(validators)
        
        # 消息处理回调
        self.on_broadcast = None  # 广播消息回调
        self.on_send = None       # 发送消息回调
        self.on_consensus_complete = None  # 共识完成回调
        
        logger.info(f"初始化节点 {node_id} 的PBFT共识处理器")
    
    def set_validators(self, validators: Set[int]):
        """设置验证节点集合"""
        self.state.set_validators(validators)
    
    def is_validator(self) -> bool:
        """当前节点是否是验证节点"""
        return self.node_id in self.state.validators
    
    def is_primary(self) -> bool:
        """当前节点是否是主节点"""
        return self.state.is_primary()
    
    def set_callbacks(self, on_broadcast=None, on_send=None, on_consensus_complete=None):
        """设置回调函数"""
        self.on_broadcast = on_broadcast
        self.on_send = on_send
        self.state.on_consensus_complete = on_consensus_complete
    
    def _on_consensus_complete(self, request: Dict) -> Any:
        """共识完成回调"""
        if self.on_consensus_complete:
            return self.on_consensus_complete(request)
        return None
    
    def _broadcast(self, message: Dict):
        """广播消息"""
        if self.on_broadcast:
            self.on_broadcast(message)
    
    def _send(self, node_id: int, message: Dict):
        """发送消息给指定节点"""
        if self.on_send:
            self.on_send(node_id, message)
    
    def set_pvss_public_keys(self, public_keys: Dict[int, bytes]):
        """设置所有节点的PVSS公钥
        
        Args:
            public_keys: 节点ID到公钥的映射 {node_id -> public_key_bytes}
        """
        self.state.pvss_handler.set_public_keys(public_keys)
        logger.info(f"节点 {self.node_id} 设置 {len(public_keys)} 个节点的PVSS公钥")
    
    def get_pvss_public_key(self) -> bytes:
        """获取当前节点的PVSS公钥
        
        Returns:
            bytes: 当前节点的PVSS公钥字节
        """
        return self.state.pvss_handler.get_public_key()
    
    def start_consensus(self, request: Dict) -> bool:
        """启动新的共识流程
        
        主节点调用此方法启动共识
        
        Args:
            request: 请求内容
            
        Returns:
            bool: 是否成功启动共识
        """
        # 只有主节点可以启动共识
        if not self.is_primary():
            logger.warning(f"非主节点 {self.node_id} 尝试启动共识，已拒绝")
            return False
        
        # 处理请求
        success, pre_prepare = self.state.handle_request(request)
        if not success or not pre_prepare:
            logger.warning(f"主节点 {self.node_id} 处理请求失败")
            return False
        
        # 广播PRE-PREPARE消息
        self._broadcast(pre_prepare)
        
        # 主节点也处理自己的PRE-PREPARE消息
        success, prepare = self.state.handle_pre_prepare(pre_prepare, self.node_id)
        if success and prepare:
            # 广播PREPARE消息
            self._broadcast(prepare)
        
        return True
    
    def handle_message(self, message: Dict, sender_id: int) -> None:
        """处理接收到的消息
        
        Args:
            message: 消息内容
            sender_id: 发送者ID
        """
        msg_type = message["type"]
        
        # 根据消息类型处理
        if msg_type == MessageType.REQUEST.value:
            self.handle_request(message, sender_id)
        elif msg_type == MessageType.PRE_PREPARE.value:
            self.handle_pre_prepare(message, sender_id)
        elif msg_type == MessageType.PREPARE.value:
            self.handle_prepare(message, sender_id)
        elif msg_type == MessageType.COMMIT.value:
            self.handle_commit(message, sender_id)
        elif msg_type == MessageType.VIEW_CHANGE.value:
            self.handle_view_change(message, sender_id)
        elif msg_type == MessageType.NEW_VIEW.value:
            self.handle_new_view(message, sender_id)
        else:
            logger.warning(f"未知消息类型: {msg_type}")
    
    def handle_request(self, request: Dict, sender_id: int) -> None:
        """处理请求消息"""
        # 只有验证节点处理请求
        if not self.is_validator():
            return
        
        # 主节点处理请求，非主节点转发请求
        if self.is_primary():
            success, pre_prepare = self.state.handle_request(request)
            if success:
                self._broadcast(pre_prepare)
        else:
            primary = self.state.get_primary()
            if primary is not None:
                self._send(primary, request)
    
    def handle_pre_prepare(self, pre_prepare: Dict, sender_id: int) -> None:
        """处理PRE-PREPARE消息"""
        # 只有验证节点处理PRE-PREPARE消息
        if not self.is_validator():
            return
        
        # 处理PRE-PREPARE消息，生成PREPARE消息
        success, prepare = self.state.handle_pre_prepare(pre_prepare, sender_id)
        if success and prepare:
            # 广播PREPARE消息
            self._broadcast(prepare)
    
    def handle_prepare(self, prepare: Dict, sender_id: int) -> None:
        """处理PREPARE消息"""
        # 只有验证节点处理PREPARE消息
        if not self.is_validator():
            return
        
        # 处理PREPARE消息，生成COMMIT消息
        success, commit = self.state.handle_prepare(prepare, sender_id)
        if success and commit:
            # 广播COMMIT消息
            self._broadcast(commit)
    
    def handle_commit(self, commit: Dict, sender_id: int) -> None:
        """处理COMMIT消息"""
        # 只有验证节点处理COMMIT消息
        if not self.is_validator():
            return
        
        # 处理COMMIT消息，执行请求
        success, reply = self.state.handle_commit(commit, sender_id)
        if success and reply:
            # 发送REPLY消息给客户端
            # 在联邦学习场景中，这里可能是通知所有节点共识结果
            self._broadcast(reply)
    
    def handle_view_change(self, view_change: Dict, sender_id: int) -> None:
        """处理VIEW-CHANGE消息"""
        # 只有验证节点处理VIEW-CHANGE消息
        if not self.is_validator():
            return
        
        # 处理VIEW-CHANGE消息，生成NEW-VIEW消息
        success, new_view = self.state.handle_view_change(view_change, sender_id)
        if success and new_view:
            # 广播NEW-VIEW消息
            self._broadcast(new_view)
    
    def handle_new_view(self, new_view: Dict, sender_id: int) -> None:
        """处理NEW-VIEW消息"""
        # 只有验证节点处理NEW-VIEW消息
        if not self.is_validator():
            return
        
        # 处理NEW-VIEW消息
        self.state.handle_new_view(new_view, sender_id)
    
    def check_timeout(self) -> None:
        """检查超时，发起视图变更"""
        # 只有验证节点检查超时
        if not self.is_validator():
            return
        
        # 检查是否需要视图变更
        if self.state.check_view_change():
            # 发起视图变更
            view_change = self.state.start_view_change()
            # 广播VIEW-CHANGE消息
            self._broadcast(view_change)
    
    def set_use_mask(self, use_mask: bool):
        """设置是否使用掩码
        
        Args:
            use_mask: 是否使用掩码
        """
        self.state.use_mask = use_mask
        logger.info(f"节点 {self.node_id} 设置使用掩码: {use_mask}")
    
    def get_mask_seed(self, round_num: int) -> Any:
        """获取指定轮次的掩码种子
        
        Args:
            round_num: 轮次号
            
        Returns:
            Any: 掩码种子，如果不存在则返回None
        """
        if not self.state.use_mask:
            return None
            
        # 查找与该轮次对应的视图和序列号
        for (view, seq), request in self.state.completed_requests.items():
            if request and "type" in request and request["type"] == "AGGREGATION_REQUEST" and request.get("round") == round_num:
                # 找到对应的请求，返回该(view, seq)的掩码种子
                if (view, seq) in self.state.mask_seeds:
                    return self.state.mask_seeds[(view, seq)]
        
        logger.warning(f"无法找到轮次 {round_num} 的掩码种子")
        return None
    
    def get_sign_map(self, round_num: int) -> Dict[int, int]:
        """获取指定轮次的符号映射
        
        Args:
            round_num: 轮次号
            
        Returns:
            Dict[int, int]: 符号映射 {node_id: sign}，如果不存在则返回None
        """
        if not self.state.use_mask:
            return None
            
        # 查找与该轮次对应的视图和序列号
        for (view, seq), request in self.state.completed_requests.items():
            if request and "type" in request and request["type"] == "AGGREGATION_REQUEST" and request.get("round") == round_num:
                # 找到对应的请求，返回该(view, seq)的符号映射
                if (view, seq) in self.state.sign_maps:
                    return self.state.sign_maps[(view, seq)]
        
        logger.warning(f"无法找到轮次 {round_num} 的符号映射")
        return None
    
    def generate_mask(self, seed: int, param_tensor: torch.Tensor, sign: int = 1) -> torch.Tensor:
        """根据种子生成掩码
        
        Args:
            seed: 掩码种子
            param_tensor: 模型参数张量
            sign: 符号 (+1 或 -1)，默认为 1
            
        Returns:
            torch.Tensor: 掩码张量
        """
        return generate_mask(seed, param_tensor, sign=sign) 
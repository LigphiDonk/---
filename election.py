import time
import random
import logging
import math
from typing import Dict, List, Set, Any, Tuple

logger = logging.getLogger("Election")

class ReputationScore:
    """节点声誉评分系统"""
    
    def __init__(self):
        # 节点声誉分数 {node_id: score}
        self.scores: Dict[int, float] = {}
        
        # 节点历史表现记录 {node_id: [records]}
        self.history: Dict[int, List[Dict]] = {}
        
        # 权重参数
        self.weights = {
            "online_time": 0.2,          # 在线时长权重
            "response_time": 0.15,       # 响应时间权重
            "consensus_contribution": 0.3,  # 共识贡献权重
            "model_quality": 0.35,       # 模型质量权重
        }
        
        # 衰减参数
        self.decay_factor = 0.95  # 历史记录衰减因子
        self.max_history = 20     # 保留的最大历史记录数
    
    def add_node(self, node_id: int, initial_score: float = 0.5):
        """添加节点到声誉系统"""
        if node_id not in self.scores:
            self.scores[node_id] = initial_score
            self.history[node_id] = []
    
    def record_event(self, node_id: int, event_type: str, data: Dict):
        """记录节点事件
        
        Args:
            node_id: 节点ID
            event_type: 事件类型
            data: 事件数据
        """
        if node_id not in self.history:
            self.add_node(node_id)
        
        # 记录事件
        record = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data
        }
        
        # 添加到历史记录
        self.history[node_id].append(record)
        
        # 如果历史记录过长，截取最近的max_history条
        if len(self.history[node_id]) > self.max_history:
            self.history[node_id] = self.history[node_id][-self.max_history:]
    
    def update_score(self, node_id: int):
        """更新节点的声誉分数"""
        if node_id not in self.history:
            return
        
        history = self.history[node_id]
        if not history:
            return
        
        # 初始化各项指标
        metrics = {
            "online_time": 0.0,
            "response_time": 0.0,
            "consensus_contribution": 0.0,
            "model_quality": 0.0
        }
        
        # 计算在线时长指标
        online_events = [record for record in history if record["type"] == "online"]
        if online_events:
            total_online_time = sum(record["data"].get("duration", 0) for record in online_events)
            metrics["online_time"] = min(1.0, total_online_time / (24 * 3600))  # 标准化为0-1，以一天为上限
        
        # 计算响应时间指标
        response_events = [record for record in history if record["type"] == "response"]
        if response_events:
            avg_response_time = sum(record["data"].get("time", 1.0) for record in response_events) / len(response_events)
            metrics["response_time"] = 1.0 / (1.0 + avg_response_time)  # 响应时间越短，分数越高
        
        # 计算共识贡献指标
        consensus_events = [record for record in history if record["type"] == "consensus"]
        if consensus_events:
            successful = sum(1 for record in consensus_events if record["data"].get("success", False))
            metrics["consensus_contribution"] = successful / len(consensus_events)
        
        # 计算模型质量指标
        model_events = [record for record in history if record["type"] == "model"]
        if model_events:
            avg_quality = sum(record["data"].get("quality", 0.0) for record in model_events) / len(model_events)
            metrics["model_quality"] = avg_quality
        
        # 计算加权分数
        score = sum(metrics[key] * self.weights[key] for key in metrics)
        
        # 应用衰减因子，与历史分数融合
        old_score = self.scores.get(node_id, 0.5)
        self.scores[node_id] = old_score * self.decay_factor + score * (1 - self.decay_factor)
        
        logger.debug(f"更新节点 {node_id} 的声誉分数: {old_score:.3f} -> {self.scores[node_id]:.3f}")
    
    def get_score(self, node_id: int) -> float:
        """获取节点的声誉分数"""
        return self.scores.get(node_id, 0.0)
    
    def get_top_nodes(self, n: int) -> List[Tuple[int, float]]:
        """获取声誉分数最高的n个节点
        
        Args:
            n: 获取的节点数量
            
        Returns:
            List[Tuple[int, float]]: (node_id, score) 元组列表
        """
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:n]


class ElectionManager:
    """选举管理器"""
    
    def __init__(self, node_id: int):
        self.node_id = node_id
        
        # 声誉评分系统
        self.reputation = ReputationScore()
        
        # 验证节点集合
        self.validators: Set[int] = set()
        
        # 当前选举周期
        self.election_term = 0
        
        # 选举相关参数
        self.max_validators = 7  # 最大验证节点数量
        self.election_interval = 30  # 选举间隔（秒）
        self.last_election_time = 0  # 上次选举时间
        
        # 选举投票 {term: {node_id: {voter_id: vote}}}
        self.votes: Dict[int, Dict[int, Dict[int, bool]]] = {}
        
        # 已知节点列表
        self.known_nodes: Set[int] = set()
    
    def add_node(self, node_id: int, initial_score: float = 0.5):
        """添加节点"""
        self.known_nodes.add(node_id)
        self.reputation.add_node(node_id, initial_score)
    
    def record_event(self, node_id: int, event_type: str, data: Dict):
        """记录节点事件"""
        if node_id not in self.known_nodes:
            self.add_node(node_id)
        
        self.reputation.record_event(node_id, event_type, data)
        self.reputation.update_score(node_id)
    
    def should_start_election(self) -> bool:
        """检查是否应该开始新一轮选举"""
        current_time = time.time()
        return (current_time - self.last_election_time) >= self.election_interval
    
    def start_election(self) -> Dict:
        """开始新一轮选举
        
        Returns:
            Dict: 选举消息
        """
        self.election_term += 1
        self.last_election_time = time.time()
        
        # 初始化投票记录
        if self.election_term not in self.votes:
            self.votes[self.election_term] = {}
        
        # 提名候选人（根据声誉分数）
        candidates = self.reputation.get_top_nodes(self.max_validators * 2)
        candidate_ids = [node_id for node_id, _ in candidates]
        
        # 创建选举消息
        election_msg = {
            "type": "ELECTION_START",
            "term": self.election_term,
            "candidates": candidate_ids,
            "timestamp": time.time(),
            "initiator": self.node_id
        }
        
        logger.info(f"节点 {self.node_id} 开始第 {self.election_term} 轮选举, 候选人: {candidate_ids}")
        
        return election_msg
    
    def vote(self, election_term: int, candidates: List[int]) -> Dict:
        """对候选人进行投票
        
        Args:
            election_term: 选举周期
            candidates: 候选人列表
            
        Returns:
            Dict: 投票消息
        """
        # 确保选举周期存在
        if election_term not in self.votes:
            self.votes[election_term] = {}
        
        # 获取候选人得分
        scores = [(node_id, self.reputation.get_score(node_id)) for node_id in candidates]
        
        # 按分数排序
        ranked_candidates = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # 选出得分最高的max_validators个节点
        selected = [node_id for node_id, _ in ranked_candidates[:self.max_validators]]
        
        # 记录投票
        for node_id in candidates:
            if node_id not in self.votes[election_term]:
                self.votes[election_term][node_id] = {}
            
            # 对选中的候选人投赞成票，其他投反对票
            self.votes[election_term][node_id][self.node_id] = (node_id in selected)
        
        # 创建投票消息
        vote_msg = {
            "type": "ELECTION_VOTE",
            "term": election_term,
            "votes": {node_id: (node_id in selected) for node_id in candidates},
            "voter": self.node_id,
            "timestamp": time.time()
        }
        
        logger.info(f"节点 {self.node_id} 在第 {election_term} 轮选举中投票, 支持: {selected}")
        
        return vote_msg
    
    def receive_vote(self, vote_msg: Dict):
        """接收投票消息
        
        Args:
            vote_msg: 投票消息
        """
        election_term = vote_msg["term"]
        voter_id = vote_msg["voter"]
        votes = vote_msg["votes"]
        
        # 确保选举周期存在
        if election_term not in self.votes:
            self.votes[election_term] = {}
        
        # 记录投票
        for node_id, vote in votes.items():
            if node_id not in self.votes[election_term]:
                self.votes[election_term][node_id] = {}
            
            self.votes[election_term][node_id][voter_id] = vote
    
    def count_votes(self, election_term: int) -> List[int]:
        """计票
        
        Args:
            election_term: 选举周期
            
        Returns:
            List[int]: 当选的验证节点列表
        """
        if election_term not in self.votes:
            return []
        
        results = {}
        for node_id, votes in self.votes[election_term].items():
            # 计算赞成票数
            approve_count = sum(1 for vote in votes.values() if vote)
            total_count = len(votes)
            
            # 计算赞成率
            approval_rate = approve_count / total_count if total_count > 0 else 0
            
            results[node_id] = (approve_count, approval_rate)
        
        # 按赞成票数和赞成率排序
        ranked_results = sorted(results.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)
        
        # 选出赞成票数最多的max_validators个节点
        elected = [node_id for node_id, _ in ranked_results[:self.max_validators]]
        
        logger.info(f"第 {election_term} 轮选举结果: {elected}")
        
        return elected
    
    def finalize_election(self, election_term: int) -> Dict:
        """完成选举，更新验证节点集合
        
        Args:
            election_term: 选举周期
            
        Returns:
            Dict: 选举结果消息
        """
        # 计票
        elected = self.count_votes(election_term)
        
        # 更新验证节点集合
        old_validators = self.validators.copy()
        self.validators = set(elected)
        
        # 记录节点成为验证节点的事件
        for node_id in self.validators:
            if node_id not in old_validators:
                self.record_event(node_id, "role_change", {
                    "role": "validator",
                    "term": election_term
                })
        
        # 记录节点失去验证节点身份的事件
        for node_id in old_validators:
            if node_id not in self.validators:
                self.record_event(node_id, "role_change", {
                    "role": "normal",
                    "term": election_term
                })
        
        # 创建选举结果消息
        result_msg = {
            "type": "ELECTION_RESULT",
            "term": election_term,
            "validators": list(self.validators),
            "timestamp": time.time()
        }
        
        logger.info(f"选举完成, 第 {election_term} 轮, 验证节点: {self.validators}")
        
        return result_msg
    
    def is_validator(self, node_id: int) -> bool:
        """检查节点是否是验证节点"""
        return node_id in self.validators 
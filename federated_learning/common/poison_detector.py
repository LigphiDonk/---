import torch
from typing import Dict, List
import numpy as np

class GradientPoisonDetector:
    """梯度投毒检测器，使用余弦相似度检测异常梯度"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        初始化检测器
        
        Args:
            similarity_threshold: 余弦相似度阈值，低于此值的梯度会被视为可能的投毒攻击
        """
        self.similarity_threshold = similarity_threshold
        
    def flatten_gradients(self, gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将梯度字典展平为一维向量"""
        return torch.cat([grad.flatten() for grad in gradients.values()])
    
    def compute_cosine_similarity(self, grad1: torch.Tensor, grad2: torch.Tensor) -> float:
        """计算两个梯度向量间的余弦相似度"""
        return torch.nn.functional.cosine_similarity(grad1, grad2, dim=0).item()
    
    def detect_poisoned_gradients(
        self,
        all_client_gradients: Dict[int, Dict[str, torch.Tensor]]
    ) -> List[int]:
        """
        检测可能被投毒的客户端梯度
        
        Args:
            all_client_gradients: 所有客户端的梯度 {client_id: gradients}
            
        Returns:
            可能被投毒的客户端ID列表
        """
        # 展平所有客户端的梯度
        flattened_gradients = {
            client_id: self.flatten_gradients(grads)
            for client_id, grads in all_client_gradients.items()
        }
        
        # 计算平均梯度
        avg_gradient = torch.mean(
            torch.stack(list(flattened_gradients.values())), dim=0
        )
        
        # 检测异常梯度
        poisoned_clients = []
        for client_id, grad in flattened_gradients.items():
            similarity = self.compute_cosine_similarity(grad, avg_gradient)
            if similarity < self.similarity_threshold:
                poisoned_clients.append(client_id)
                
        return poisoned_clients
    
    def filter_gradients(
        self,
        all_client_gradients: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        过滤掉可能被投毒的梯度
        
        Args:
            all_client_gradients: 所有客户端的梯度
            
        Returns:
            过滤后的梯度字典
        """
        poisoned_clients = self.detect_poisoned_gradients(all_client_gradients)
        return {
            client_id: grads
            for client_id, grads in all_client_gradients.items()
            if client_id not in poisoned_clients
        }
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
梯度掩码处理模块

此模块实现基于PVSS的梯度掩码机制，用于保护联邦学习中客户端向服务器传输的梯度
"""

import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Any
from .pvss import PVSSHandler

class MaskedGradientHandler:
    """梯度掩码处理器，实现梯度的安全传输"""
    
    def __init__(self, node_id: int, total_nodes: int, f: int):
        """
        初始化梯度掩码处理器
        
        Args:
            node_id: 当前节点ID
            total_nodes: 总节点数
            f: 容错上限，表示系统可以容忍的最大故障节点数
        """
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.f = f
        self.pvss_handler = PVSSHandler(node_id, f)
        self.mask_cache = {}  # 缓存掩码信息 {(round_id, client_id): mask_info}
        
    def generate_mask(self, round_id: int, client_id: int, server_ids: List[int]) -> Dict:
        """
        生成梯度掩码数据
        
        Args:
            round_id: 训练轮次
            client_id: 客户端ID
            server_ids: 服务器节点ID列表
            
        Returns:
            包含掩码信息的字典
        """
        # 生成随机掩码种子
        mask_seed = random.randint(100000, 999999999)
        
        # 生成掩码符号映射 (一半服务器用+1，一半用-1)
        sign_map = self._generate_sign_map(server_ids)
        
        # 使用PVSS分发掩码种子
        pvss_shares, pvss_proofs = self.pvss_handler.deal_secret(mask_seed, server_ids)
        
        # 构造掩码数据
        mask_data = {
            "round_id": round_id,
            "client_id": client_id,
            "mask_seed": mask_seed,
            "sign_map": sign_map,
            "pvss_shares": pvss_shares,
            "pvss_proofs": pvss_proofs
        }
        
        # 缓存掩码数据
        cache_key = (round_id, client_id)
        self.mask_cache[cache_key] = mask_data
        
        return mask_data
    
    def apply_mask(self, gradients: Dict[str, torch.Tensor], 
                  mask_seed: int, sign: int) -> Dict[str, torch.Tensor]:
        """
        应用掩码到梯度
        
        Args:
            gradients: 梯度字典
            mask_seed: 掩码种子
            sign: 掩码符号(+1/-1)
            
        Returns:
            加掩码后的梯度
        """
        masked_gradients = {}
        
        # 对每个梯度张量应用掩码
        for name, grad in gradients.items():
            # 生成与梯度形状相同的随机掩码
            mask = self._generate_mask_tensor(mask_seed + hash(name) % 10000, 
                                            grad.shape, grad.device)
            
            # 应用掩码: gradient + sign * mask
            masked_gradients[name] = grad + sign * mask
            
        return masked_gradients
    
    def unmask_gradients(self, masked_gradients_list: List[Dict[str, torch.Tensor]], 
                        mask_data: Dict) -> Dict[str, torch.Tensor]:
        """
        去除掩码，恢复原始梯度
        
        Args:
            masked_gradients_list: 多个服务器的加掩码梯度列表
            mask_data: 掩码数据
            
        Returns:
            恢复后的原始梯度
        """
        if len(masked_gradients_list) == 0:
            raise ValueError("Empty masked gradients list")
            
        # 获取掩码种子和符号映射
        mask_seed = mask_data["mask_seed"]
        sign_map = mask_data["sign_map"]
        
        # 初始化结果梯度字典
        result_gradients = {}
        
        # 获取第一个梯度字典的键，作为所有梯度的键
        first_gradients = masked_gradients_list[0]
        
        # 对每个梯度参数进行处理
        for name in first_gradients.keys():
            # 收集所有服务器对该参数的梯度
            param_gradients = []
            for server_gradients in masked_gradients_list:
                if name in server_gradients:
                    param_gradients.append(server_gradients[name])
            
            # 计算平均梯度（掩码会相互抵消）
            if len(param_gradients) > 0:
                # 对所有服务器的梯度取平均
                avg_gradient = torch.stack(param_gradients).mean(dim=0)
                result_gradients[name] = avg_gradient
        
        return result_gradients
    
    def verify_mask_data(self, mask_data: Dict, server_ids: List[int]) -> bool:
        """
        验证掩码数据
        
        Args:
            mask_data: 掩码数据字典
            server_ids: 服务器节点ID列表
            
        Returns:
            验证是否通过
        """
        # 验证基本字段是否存在
        required_fields = ["round_id", "client_id", "mask_seed", 
                          "sign_map", "pvss_shares", "pvss_proofs"]
        for field in required_fields:
            if field not in mask_data:
                return False
        
        # 验证符号映射
        sign_map = mask_data["sign_map"]
        for server_id in server_ids:
            if str(server_id) not in sign_map and server_id not in sign_map:
                return False
        
        # 验证PVSS份额和证明
        return self.pvss_handler.verify_shares(
            mask_data["pvss_shares"],
            mask_data["pvss_proofs"],
            server_ids
        )
    
    def _generate_sign_map(self, node_ids: List[int]) -> Dict[int, int]:
        """
        生成符号映射，一半节点用+1，一半节点用-1
        
        Args:
            node_ids: 节点ID列表
            
        Returns:
            符号映射字典
        """
        sign_map = {}
        half = len(node_ids) // 2
        
        # 随机打乱节点ID
        shuffled_ids = node_ids.copy()
        random.shuffle(shuffled_ids)
        
        # 前半部分节点使用+1，后半部分使用-1
        for i, node_id in enumerate(shuffled_ids):
            sign_map[node_id] = 1 if i < half else -1
            
        return sign_map
    
    def _generate_mask_tensor(self, seed: int, shape: torch.Size, 
                             device: torch.device) -> torch.Tensor:
        """
        生成随机掩码张量
        
        Args:
            seed: 随机种子
            shape: 张量形状
            device: 设备(CPU/GPU)
            
        Returns:
            随机掩码张量
        """
        # 设置随机种子以确保可重复性
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 生成标准正态分布的随机张量作为掩码
        # 缩小掩码幅度，避免对原始梯度影响过大
        scale = 0.01  # 掩码缩放因子
        mask = torch.randn(shape, device=device) * scale
        
        return mask 
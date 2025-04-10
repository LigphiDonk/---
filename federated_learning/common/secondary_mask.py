#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二次掩码处理模块

此模块实现了一个二次掩码机制，在PVSS掩码之后添加，客户端自己生成掩码种子并通过Shamir秘密分享分享出去
"""

import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Any
from federated_learning.common.pvss import PVSSHandler

class SecondaryMaskHandler:
    """二次掩码处理器，实现客户端自己生成掩码种子并通过Shamir秘密分享分享出去的掩码机制"""

    def __init__(self, client_id: int, total_servers: int = 4, threshold: int = None):
        """
        初始化二次掩码处理器

        Args:
            client_id: 客户端ID
            total_servers: 服务器总数，默认为4
            threshold: 重建掩码所需的最小服务器数量，默认为total_servers的2/3
        """
        self.client_id = client_id
        self.total_servers = total_servers
        self.threshold = threshold if threshold is not None else max(2, int(total_servers * 2 / 3))
        self.mask_cache = {}  # 缓存掩码信息 {round_id: mask_info}

        # 初始化PVSS处理器，用于Shamir秘密分享
        # f参数表示容错上限，设置为total_servers - threshold
        self.pvss_handler = PVSSHandler(client_id, total_servers - self.threshold)

    def generate_mask(self, round_id: int, server_ids: List[int]) -> Dict:
        """
        生成二次掩码数据，并通过Shamir秘密分享分享掩码种子

        Args:
            round_id: 训练轮次
            server_ids: 服务器节点ID列表

        Returns:
            包含掩码信息的字典
        """
        # 生成随机掩码种子
        mask_seed = random.randint(100000, 999999999)

        # 使用PVSS分发掩码种子到各个服务器
        pvss_shares, pvss_proofs = self.pvss_handler.deal_secret(mask_seed, server_ids)

        # 构造掩码数据
        mask_data = {
            "round_id": round_id,
            "client_id": self.client_id,
            "mask_seed": mask_seed,  # 客户端自己保存原始种子（实际部署时可以考虑移除）
            "pvss_shares": pvss_shares,  # 分享给各服务器的份额
            "pvss_proofs": pvss_proofs,  # 份额的证明
            "server_ids": server_ids,  # 参与的服务器ID列表
            "threshold": self.threshold  # 重建所需的最小服务器数量
        }

        # 缓存掩码数据
        self.mask_cache[round_id] = mask_data

        return mask_data

    def apply_mask(self, gradients: Dict[str, torch.Tensor],
                  mask_data: Dict) -> Dict[str, torch.Tensor]:
        """
        应用二次掩码到梯度

        Args:
            gradients: 梯度字典
            mask_data: 掩码数据

        Returns:
            加掩码后的梯度
        """
        masked_gradients = {}

        # 获取掩码种子
        mask_seed = mask_data["mask_seed"]

        # 对每个梯度张量应用掩码
        for name, grad in gradients.items():
            # 为每个参数生成特定的掩码种子
            param_seed = mask_seed + hash(name) % 10000

            # 生成随机掩码
            mask = self._generate_mask_tensor(param_seed, grad.shape, grad.device)

            # 应用掩码: gradient + mask
            masked_gradients[name] = grad + mask

        return masked_gradients

    def unmask_gradients(self, masked_gradients: Dict[str, torch.Tensor],
                      shares: Dict[int, int], server_ids: List[int],
                      param_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        去除二次掩码，恢复原始梯度

        Args:
            masked_gradients: 加掩码的梯度字典
            shares: 服务器收集到的掩码种子份额 {server_id: share}
            server_ids: 参与的服务器ID列表
            param_names: 参数名称列表

        Returns:
            恢复后的原始梯度
        """
        # 检查份额数量是否足够重建
        if len(shares) < self.threshold:
            # 如果份额数量不足，无法去除掩码，直接返回输入的梯度
            return masked_gradients

        # 使用收集到的份额重建掩码种子
        try:
            mask_seed = self.pvss_handler.reconstruct_secret(shares)
        except Exception as e:
            # 重建失败，返回原始梯度
            print(f"重建掩码种子失败: {e}")
            return masked_gradients

        # 初始化结果梯度字典
        unmasked_gradients = {}

        # 对每个梯度去除掩码
        for name in param_names:
            if name not in masked_gradients:
                continue

            grad = masked_gradients[name]
            # 为每个参数生成特定的掩码种子
            param_seed = mask_seed + hash(name) % 10000

            # 生成相同的掩码
            mask = self._generate_mask_tensor(param_seed, grad.shape, grad.device)

            # 去除掩码: gradient - mask
            unmasked_gradients[name] = grad - mask

        return unmasked_gradients

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
        scale = 0.02  # 掩码缩放因子
        mask = torch.randn(shape, device=device) * scale

        return mask

    def collect_share(self, round_id: int, client_id: int, server_id: int) -> Dict:
        """
        获取特定服务器的掩码种子份额

        Args:
            round_id: 训练轮次
            client_id: 客户端ID
            server_id: 服务器ID

        Returns:
            包含份额和证明的字典，如果没有找到则返回None
        """
        # 从缓存中获取掩码数据
        cache_key = round_id
        if cache_key not in self.mask_cache:
            return None

        mask_data = self.mask_cache[cache_key]
        if mask_data["client_id"] != client_id:
            return None

        # 获取特定服务器的份额和证明
        pvss_shares = mask_data.get("pvss_shares", {})
        pvss_proofs = mask_data.get("pvss_proofs", {})

        if server_id not in pvss_shares or server_id not in pvss_proofs:
            return None

        return {
            "share": pvss_shares[server_id],
            "proof": pvss_proofs[server_id],
            "round_id": round_id,
            "client_id": client_id,
            "server_id": server_id
        }
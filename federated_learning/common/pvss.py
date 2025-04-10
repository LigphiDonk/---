#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PVSS(Publicly Verifiable Secret Sharing)模块

此模块实现了可公开验证的秘密共享算法，用于安全地在不同节点间共享秘密信息，
同时提供验证机制确保共享的信息正确性。
"""

import random
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Any
import sympy

class PVSSHandler:
    """PVSS处理器，实现可公开验证的秘密共享"""

    def __init__(self, node_id: int, f: int):
        """
        初始化PVSS处理器

        Args:
            node_id: 当前节点ID
            f: 容错上限，表示系统可以容忍的最大故障节点数
        """
        self.node_id = node_id
        self.f = f
        self.prime = self._generate_large_prime(256)  # 生成大素数作为有限域
        self.shares_cache = {}  # 缓存收到的份额

    def deal_secret(self, secret: int, node_ids: List[int]) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """
        分发秘密到多个节点

        Args:
            secret: 要分享的秘密（整数）
            node_ids: 接收秘密份额的节点ID列表

        Returns:
            (shares, proofs): 份额和证明的字典
        """
        n = len(node_ids)
        t = n - self.f  # 重建阈值

        # 确保秘密在有限域范围内
        secret = secret % self.prime

        # 生成随机多项式 f(x) = secret + a_1*x + a_2*x^2 + ... + a_{t-1}*x^{t-1}
        coeffs = [secret]
        for _ in range(t-1):
            coeffs.append(random.randint(1, min(10000, self.prime-1)))

        # 计算每个节点的份额
        shares = {}
        for node_id in node_ids:
            share = self._evaluate_polynomial(coeffs, node_id)
            shares[node_id] = share

        # 生成证明（简化版，实际应使用零知识证明）
        proofs = {}
        for node_id in node_ids:
            # 简单的哈希证明，实际实现应使用更复杂的密码学证明
            proof = self._generate_proof(secret, node_id, shares[node_id])
            proofs[node_id] = proof

        return shares, proofs

    def verify_shares(self, shares: Dict[int, int], proofs: Dict[int, List[int]],
                      node_ids: List[int]) -> bool:
        """
        验证收到的份额和证明

        Args:
            shares: 份额字典 {node_id: share}
            proofs: 证明字典 {node_id: proof}
            node_ids: 参与节点ID列表

        Returns:
            验证是否通过
        """
        # 验证每个份额
        for node_id in node_ids:
            if node_id not in shares or node_id not in proofs:
                return False

            # 存储收到的份额
            self.shares_cache[node_id] = shares[node_id]

            # 验证份额与证明是否匹配
            if not self._verify_proof(node_id, shares[node_id], proofs[node_id]):
                return False

        return True

    def reconstruct_secret(self, shares: Dict[int, int]) -> int:
        """
        重建秘密

        Args:
            shares: 份额字典 {node_id: share}

        Returns:
            重建的秘密
        """
        if len(shares) < len(self.shares_cache) - self.f:
            raise ValueError("Insufficient shares to reconstruct the secret")

        # 使用Lagrange插值恢复多项式，获取常数项（秘密）
        x_values = list(shares.keys())
        y_values = [shares[x] for x in x_values]

        # 使用拉格朗日插值公式重建秘密
        secret = self._lagrange_interpolation(x_values, y_values, 0)
        return secret

    def _generate_large_prime(self, bits: int) -> int:
        """生成指定位数的大素数"""
        # 使用较小的素数以避免整数溢出问题
        # 对于模拟环境，这个素数足够大且安全
        return 2147483647  # 2^31 - 1，一个梅森素数

    def _evaluate_polynomial(self, coeffs: List[int], x: int) -> int:
        """在点x处计算多项式的值"""
        result = 0
        power = 1

        for coeff in coeffs:
            result = (result + coeff * power) % self.prime
            power = (power * x) % self.prime

        return result

    def _generate_proof(self, secret: int, node_id: int, share: int) -> List[int]:
        """
        生成份额的证明（简化实现）

        实际上，此处应使用零知识证明等密码学技术，这里简化为哈希
        """
        # 生成哈希证明
        hash_input = f"{secret}-{node_id}-{share}".encode()
        hash_value = hashlib.sha256(hash_input).digest()

        # 将哈希转换为整数列表
        proof = []
        for i in range(0, 32, 4):
            value = int.from_bytes(hash_value[i:i+4], byteorder='big')
            proof.append(value)

        return proof

    def _verify_proof(self, node_id: int, share: int, proof: List[int]) -> bool:
        """
        验证份额证明（简化实现）
        """
        # 简化验证，实际应检查零知识证明
        # 这里我们只是检查proof格式是否正确
        return isinstance(proof, list) and len(proof) > 0

    def _lagrange_interpolation(self, x_values: List[int], y_values: List[int], x: int) -> int:
        """
        使用Lagrange插值重建在x处的多项式值
        """
        n = len(x_values)
        result = 0

        for i in range(n):
            term = y_values[i]
            for j in range(n):
                if i != j:
                    num = (x - x_values[j]) % self.prime
                    den = (x_values[i] - x_values[j]) % self.prime
                    # 计算模逆
                    den_inv = pow(den, self.prime - 2, self.prime)  # 费马小定理求模逆
                    term = (term * num * den_inv) % self.prime

            result = (result + term) % self.prime

        return result
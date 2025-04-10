#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试PVSS修复

此脚本用于测试PVSS模块的修复是否有效
"""

import sys
import os

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_learning.common.pvss import PVSSHandler

def test_pvss():
    """测试PVSS模块"""
    print("测试PVSS模块...")
    
    # 创建PVSS处理器
    node_id = 1
    f = 1  # 容错上限
    pvss_handler = PVSSHandler(node_id, f)
    
    # 测试大整数
    secret = 987654321
    node_ids = [101, 102, 103]
    
    print(f"原始秘密: {secret}")
    print(f"使用的素数: {pvss_handler.prime}")
    
    # 分发秘密
    try:
        shares, proofs = pvss_handler.deal_secret(secret, node_ids)
        print(f"生成的份额: {shares}")
        print(f"生成的证明: {proofs}")
        
        # 验证份额
        verified = pvss_handler.verify_shares(shares, proofs, node_ids)
        print(f"验证结果: {verified}")
        
        # 重建秘密
        reconstructed_secret = pvss_handler.reconstruct_secret(shares)
        print(f"重建的秘密: {reconstructed_secret}")
        
        # 检查重建的秘密是否正确
        if reconstructed_secret == secret % pvss_handler.prime:
            print("测试成功: 重建的秘密与原始秘密匹配")
        else:
            print(f"测试失败: 重建的秘密 ({reconstructed_secret}) 与原始秘密 ({secret % pvss_handler.prime}) 不匹配")
    
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_pvss()

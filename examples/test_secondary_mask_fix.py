#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试二次掩码修复

此脚本用于测试二次掩码功能的修复是否有效
"""

import sys
import os
import torch
import random
import numpy as np

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_learning.common.secondary_mask import SecondaryMaskHandler

def test_secondary_mask():
    """测试二次掩码功能"""
    print("测试二次掩码功能...")
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 模拟参数
    total_servers = 4
    threshold = 3  # 需要至少3个服务器的份额才能重建掩码种子
    client_id = 1
    round_id = 1
    server_ids = [101, 102, 103, 104]  # 服务器ID列表
    
    # 创建二次掩码处理器
    mask_handler = SecondaryMaskHandler(client_id, total_servers, threshold)
    
    # 生成掩码数据（包括Shamir秘密分享的份额）
    try:
        mask_data = mask_handler.generate_mask(round_id, server_ids)
        print(f"生成的掩码数据: {mask_data['round_id']}, 客户端ID: {mask_data['client_id']}")
        print(f"原始掩码种子: {mask_data['mask_seed']}")
        print(f"生成了 {len(mask_data['pvss_shares'])} 个份额")
        
        # 创建模拟梯度
        gradients = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(5, 10),
            "layer2.bias": torch.randn(5)
        }
        
        # 应用掩码
        masked_gradients = mask_handler.apply_mask(gradients, mask_data)
        
        # 模拟服务器收集份额
        collected_shares = {}
        for server_id in server_ids:
            share_data = mask_handler.collect_share(round_id, client_id, server_id)
            if share_data:
                collected_shares[server_id] = share_data["share"]
        
        print(f"收集到 {len(collected_shares)} 个份额")
        
        # 使用所有服务器的份额
        print("使用所有服务器的份额去除掩码")
        unmasked_gradients = mask_handler.unmask_gradients(
            masked_gradients, 
            collected_shares,
            server_ids,
            list(gradients.keys())
        )
        
        # 检查去掩码后的梯度与原始梯度的差异
        success = True
        for name in gradients:
            diff = torch.abs(unmasked_gradients[name] - gradients[name]).mean().item()
            print(f"{name}: 与原始梯度的平均差异: {diff:.10f}")
            if diff > 1e-5:
                success = False
        
        if success:
            print("测试成功: 去掩码后的梯度与原始梯度匹配")
        else:
            print("测试失败: 去掩码后的梯度与原始梯度不匹配")
        
        # 使用阈值数量的份额
        print("\n使用阈值数量的份额去除掩码")
        threshold_shares = {k: collected_shares[k] for k in list(collected_shares.keys())[:threshold]}
        print(f"使用 {len(threshold_shares)} 个份额")
        
        unmasked_gradients_threshold = mask_handler.unmask_gradients(
            masked_gradients, 
            threshold_shares,
            server_ids[:threshold],
            list(gradients.keys())
        )
        
        # 检查去掩码后的梯度与原始梯度的差异
        success = True
        for name in gradients:
            diff = torch.abs(unmasked_gradients_threshold[name] - gradients[name]).mean().item()
            print(f"{name}: 与原始梯度的平均差异: {diff:.10f}")
            if diff > 1e-5:
                success = False
        
        if success:
            print("测试成功: 使用阈值份额去掩码后的梯度与原始梯度匹配")
        else:
            print("测试失败: 使用阈值份额去掩码后的梯度与原始梯度不匹配")
    
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_secondary_mask()

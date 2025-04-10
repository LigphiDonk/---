#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二次掩码示例

此示例展示了如何使用改进后的二次掩码机制，客户端自己生成掩码种子并通过Shamir秘密分享分享出去
"""

import torch
import random
import numpy as np
from federated_learning.common.secondary_mask import SecondaryMaskHandler

def main():
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
    
    # 打印原始梯度
    print("\n原始梯度:")
    for name, grad in gradients.items():
        print(f"{name}: 均值={grad.mean().item():.6f}, 标准差={grad.std().item():.6f}")
    
    # 应用掩码
    masked_gradients = mask_handler.apply_mask(gradients, mask_data)
    
    # 打印加掩码后的梯度
    print("\n加掩码后的梯度:")
    for name, grad in masked_gradients.items():
        print(f"{name}: 均值={grad.mean().item():.6f}, 标准差={grad.std().item():.6f}")
    
    # 模拟服务器收集份额
    # 在实际应用中，这些份额会分发给不同的服务器
    collected_shares = {}
    for server_id in server_ids:
        share_data = mask_handler.collect_share(round_id, client_id, server_id)
        if share_data:
            collected_shares[server_id] = share_data["share"]
    
    print(f"\n收集到 {len(collected_shares)} 个份额")
    
    # 模拟服务器使用份额重建掩码种子并去除掩码
    # 场景1: 使用所有服务器的份额
    print("\n场景1: 使用所有服务器的份额")
    unmasked_gradients = mask_handler.unmask_gradients(
        masked_gradients, 
        collected_shares,
        server_ids,
        list(gradients.keys())
    )
    
    # 打印去掩码后的梯度
    print("\n去掩码后的梯度:")
    for name, grad in unmasked_gradients.items():
        print(f"{name}: 均值={grad.mean().item():.6f}, 标准差={grad.std().item():.6f}")
        # 计算与原始梯度的差异
        diff = torch.abs(grad - gradients[name]).mean().item()
        print(f"  与原始梯度的平均差异: {diff:.10f}")
    
    # 场景2: 只使用阈值数量的份额
    print("\n场景2: 只使用阈值数量的份额")
    threshold_shares = {k: collected_shares[k] for k in list(collected_shares.keys())[:threshold]}
    print(f"使用 {len(threshold_shares)} 个份额")
    
    unmasked_gradients_threshold = mask_handler.unmask_gradients(
        masked_gradients, 
        threshold_shares,
        server_ids[:threshold],
        list(gradients.keys())
    )
    
    # 打印去掩码后的梯度
    print("\n使用阈值份额去掩码后的梯度:")
    for name, grad in unmasked_gradients_threshold.items():
        print(f"{name}: 均值={grad.mean().item():.6f}, 标准差={grad.std().item():.6f}")
        # 计算与原始梯度的差异
        diff = torch.abs(grad - gradients[name]).mean().item()
        print(f"  与原始梯度的平均差异: {diff:.10f}")
    
    # 场景3: 使用少于阈值的份额（应该无法正确重建）
    print("\n场景3: 使用少于阈值的份额（应该无法正确重建）")
    insufficient_shares = {k: collected_shares[k] for k in list(collected_shares.keys())[:threshold-1]}
    print(f"使用 {len(insufficient_shares)} 个份额")
    
    unmasked_gradients_insufficient = mask_handler.unmask_gradients(
        masked_gradients, 
        insufficient_shares,
        server_ids[:threshold-1],
        list(gradients.keys())
    )
    
    # 在这种情况下，应该返回原始的加掩码梯度
    print("\n使用不足份额尝试去掩码的结果:")
    if unmasked_gradients_insufficient is masked_gradients:
        print("正确处理: 返回了原始的加掩码梯度")
    else:
        print("错误: 应该返回原始的加掩码梯度")

if __name__ == "__main__":
    main()

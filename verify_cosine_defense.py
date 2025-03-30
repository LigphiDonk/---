#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证余弦相似度防御机制

通过模拟正常和恶意更新，测试余弦相似度防御是否正确工作。
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import copy
import random
import argparse
import sys
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CosineDefenseVerify")

# 简单的测试模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.fc3 = nn.Linear(5, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def calculate_cosine_similarity(model1_state, model2_state):
    """计算两个模型状态之间的余弦相似度"""
    # 展平参数
    flattened_model1 = torch.cat([model1_state[key].flatten() for key in model1_state])
    flattened_model2 = torch.cat([model2_state[key].flatten() for key in model2_state])
    
    # 计算余弦相似度
    similarity = torch.nn.functional.cosine_similarity(
        flattened_model1.unsqueeze(0), 
        flattened_model2.unsqueeze(0)
    ).item()
    
    return similarity

def calculate_model_update_similarity(global_model, local_model, previous_global_model):
    """计算本地模型更新与上一轮全局模型的余弦相似度"""
    # 计算模型更新（delta weights）
    delta_w = {}
    global_state_dict = global_model.state_dict()
    local_state_dict = local_model.state_dict()
    
    for key in local_state_dict:
        if key in global_state_dict:
            # 计算差值：本地模型 - 全局模型
            delta_w[key] = local_state_dict[key] - global_state_dict[key]
    
    # 展平delta_w和previous_global_model为1D张量
    flattened_delta = torch.cat([delta_w[key].flatten() for key in delta_w])
    
    # 从上一轮全局模型中提取相应参数
    previous_global_state = previous_global_model.state_dict()
    flattened_prev_global = torch.cat([previous_global_state[key].flatten() for key in delta_w if key in previous_global_state])
    
    # 计算余弦相似度
    similarity_score = torch.nn.functional.cosine_similarity(
        flattened_delta.unsqueeze(0), 
        flattened_prev_global.unsqueeze(0)
    ).item()
    
    return similarity_score

def test_normal_update(threshold):
    """测试正常更新（应该通过余弦相似度检查）"""
    logger.info("=== 测试正常更新 ===")
    
    # 创建模型
    previous_global_model = SimpleModel()  # 第n-1轮的全局模型
    global_model = SimpleModel()          # 第n轮的全局模型
    local_model = SimpleModel()           # 客户端基于global_model训练的本地模型
    
    # 对第n-1轮全局模型进行一些修改，使其与第n轮不同
    with torch.no_grad():
        for param in previous_global_model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    
    # 对第n轮全局模型进行一些修改，使其与第n-1轮不同
    with torch.no_grad():
        for param in global_model.parameters():
            param.add_(torch.randn_like(param) * 0.05)
    
    # 对本地模型进行类似于正常训练的修改
    # 在global_model基础上添加与previous_global_model类似方向的小变化
    with torch.no_grad():
        for (name1, param1), (name2, param2) in zip(
            local_model.named_parameters(), previous_global_model.named_parameters()):
            # 添加一个方向相似的小变化
            param1.add_(param2 * 0.02 + torch.randn_like(param1) * 0.01)
    
    # 计算模型更新与上一轮全局模型的余弦相似度
    similarity = calculate_model_update_similarity(global_model, local_model, previous_global_model)
    
    logger.info(f"正常更新的余弦相似度: {similarity:.4f}")
    logger.info(f"阈值: {threshold:.4f}")
    logger.info(f"检查结果: {'通过' if similarity >= threshold else '拒绝'}")
    
    # 验证正常更新是否通过检查
    assert similarity >= threshold, "正常更新应该通过余弦相似度检查"
    
    return similarity

def test_malicious_update(threshold):
    """测试恶意更新（应该不通过余弦相似度检查）"""
    logger.info("=== 测试恶意更新 ===")
    
    # 创建模型
    previous_global_model = SimpleModel()  # 第n-1轮的全局模型
    global_model = SimpleModel()          # 第n轮的全局模型
    local_model = SimpleModel()           # 客户端基于global_model训练的本地模型
    
    # 对第n-1轮全局模型进行一些修改
    with torch.no_grad():
        for param in previous_global_model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    
    # 对第n轮全局模型进行一些修改
    with torch.no_grad():
        for param in global_model.parameters():
            param.add_(torch.randn_like(param) * 0.05)
    
    # 对本地模型进行恶意修改（与previous_global_model相反方向的大变化）
    with torch.no_grad():
        for (name1, param1), (name2, param2) in zip(
            local_model.named_parameters(), previous_global_model.named_parameters()):
            # 添加一个方向相反的大变化
            param1.add_(-param2 * 3.0 + torch.randn_like(param1) * 0.5)
    
    # 计算模型更新与上一轮全局模型的余弦相似度
    similarity = calculate_model_update_similarity(global_model, local_model, previous_global_model)
    
    logger.info(f"恶意更新的余弦相似度: {similarity:.4f}")
    logger.info(f"阈值: {threshold:.4f}")
    logger.info(f"检查结果: {'通过' if similarity >= threshold else '拒绝'}")
    
    # 验证恶意更新是否被拒绝
    assert similarity < threshold, "恶意更新应该不通过余弦相似度检查"
    
    return similarity

def verify_implementation():
    """验证余弦相似度防御机制是否已成功实现"""
    logger.info("=== 验证余弦相似度防御实现 ===")
    
    # 1. 验证参数已添加到启动脚本
    try:
        import argparse
        from pbft_startup import get_args
        
        parser = argparse.ArgumentParser()
        args = parser.parse_args([])
        cosine_defense_arg = [a for a in dir(args) if a == 'use_cosine_defense']
        cosine_threshold_arg = [a for a in dir(args) if a == 'cosine_threshold']
        
        if 'use_cosine_defense' in cosine_defense_arg and 'cosine_threshold' in cosine_threshold_arg:
            logger.info("✓ 启动脚本中包含余弦相似度防御参数")
        else:
            logger.error("✗ 启动脚本中缺少余弦相似度防御参数")
    except ImportError:
        logger.error("无法导入pbft_startup模块")
    
    # 2. 验证客户端添加了余弦相似度方法
    try:
        from pbft_client import PBFTFederatedClient
        client_methods = dir(PBFTFederatedClient)
        
        if 'set_cosine_defense' in client_methods and 'calculate_update_similarity' in client_methods:
            logger.info("✓ PBFT客户端实现了余弦相似度防御方法")
        else:
            logger.error("✗ PBFT客户端缺少余弦相似度防御方法")
    except ImportError:
        logger.error("无法导入PBFTFederatedClient类")
    
    # 3. 验证余弦相似度计算函数
    model = SimpleModel()
    model2 = SimpleModel()
    
    # 修改模型2以创建差异
    with torch.no_grad():
        for param in model2.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    
    # 计算余弦相似度
    try:
        similarity = calculate_cosine_similarity(model.state_dict(), model2.state_dict())
        logger.info(f"✓ 相似度计算正常，测试相似度: {similarity:.4f}")
    except Exception as e:
        logger.error(f"✗ 相似度计算失败: {str(e)}")
    
    # 总结
    logger.info("\n=== 实现验证总结 ===")
    logger.info("余弦相似度防御机制验证完成。若所有检查都通过，则表明实现已成功添加到系统中。")
    logger.info("可通过以下参数启用余弦相似度防御:")
    logger.info("python pbft_startup.py --use_cosine_defense --cosine_threshold -0.1")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='验证余弦相似度防御机制')
    parser.add_argument('--threshold', type=float, default=-0.1, help='余弦相似度阈值，默认为-0.1')
    parser.add_argument('--verify_implementation', action='store_true', help='验证余弦相似度防御的实现')
    args = parser.parse_args()
    
    # 设置随机种子以确保结果可重现
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 根据参数选择验证方式
    if args.verify_implementation:
        verify_implementation()
        return
    
    try:
        # 测试正常更新
        normal_similarity = test_normal_update(args.threshold)
        
        # 测试恶意更新
        malicious_similarity = test_malicious_update(args.threshold)
        
        # 总结
        logger.info("\n=== 测试结果汇总 ===")
        logger.info(f"正常更新余弦相似度: {normal_similarity:.4f} (应 >= {args.threshold})")
        logger.info(f"恶意更新余弦相似度: {malicious_similarity:.4f} (应 < {args.threshold})")
        logger.info(f"余弦相似度防御机制验证{'成功' if normal_similarity >= args.threshold and malicious_similarity < args.threshold else '失败'}")
        
        # 如果阈值设置不合理，给出建议
        if normal_similarity < args.threshold:
            logger.warning(f"阈值 {args.threshold} 过高，导致正常更新也被拒绝。建议降低阈值。")
        elif malicious_similarity >= args.threshold:
            logger.warning(f"阈值 {args.threshold} 过低，导致恶意更新也被接受。建议提高阈值。")
        
        # 计算良好区分正常和恶意更新的阈值建议
        suggested_threshold = (normal_similarity + malicious_similarity) / 2
        logger.info(f"建议的余弦相似度阈值: {suggested_threshold:.4f}")
    
    except AssertionError as e:
        logger.error(f"测试失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
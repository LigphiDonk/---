import torch
import logging
import numpy as np
from typing import Dict, List, Any, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CosineDeviationGrouping")

def calculate_cosine_deviation(local_model, global_model):
    """
    计算本地模型与全局模型之间的偏离度（基于余弦相似度）
    
    Args:
        local_model: 本地模型
        global_model: 全局模型
        
    Returns:
        float: 偏离度 (1 - 余弦相似度)
    """
    # 展平模型参数
    local_params = torch.cat([p.flatten() for p in local_model.parameters()])
    global_params = torch.cat([p.flatten() for p in global_model.parameters()])
    
    # 计算余弦相似度
    cosine_sim = torch.nn.functional.cosine_similarity(
        local_params.unsqueeze(0),
        global_params.unsqueeze(0)
    ).item()
    
    # 计算偏离度（1 - 余弦相似度）
    deviation = 1 - cosine_sim
    
    logger.info(f"计算得到的余弦相似度: {cosine_sim:.4f}, 偏离度: {deviation:.4f}")
    
    return deviation

def calculate_update_deviation(delta_w, previous_global_model):
    """
    计算模型更新与上一轮全局模型之间的偏离度
    
    Args:
        delta_w: 模型更新
        previous_global_model: 上一轮全局模型
        
    Returns:
        float: 偏离度 (1 - 余弦相似度)
    """
    # 展平delta_w
    flattened_delta = torch.cat([delta_w[key].flatten() for key in delta_w])
    
    # 从上一轮全局模型中提取相应参数
    previous_global_state = previous_global_model.state_dict()
    flattened_prev_global = torch.cat([previous_global_state[key].flatten() for key in delta_w if key in previous_global_state])
    
    # 计算余弦相似度
    similarity_score = torch.nn.functional.cosine_similarity(
        flattened_delta.unsqueeze(0), 
        flattened_prev_global.unsqueeze(0)
    ).item()
    
    # 计算偏离度
    deviation = 1 - similarity_score
    
    logger.info(f"模型更新偏离度: {deviation:.4f} (余弦相似度: {similarity_score:.4f})")
    
    return deviation

def group_clients_by_deviation(clients_deviations: Dict[int, float]) -> List[List[int]]:
    """
    根据偏离度将客户端分组，每组4个客户端
    
    Args:
        clients_deviations: 客户端ID到偏离度的映射 {node_id: deviation}
        
    Returns:
        List[List[int]]: 客户端分组，每个子列表包含4个客户端ID
    """
    # 按偏离度排序
    sorted_clients = sorted(clients_deviations.items(), key=lambda x: x[1])
    
    # 4个一组分组
    groups = []
    for i in range(0, len(sorted_clients), 4):
        group = sorted_clients[i:i+4]
        if len(group) == 4:  # 确保每组都有4个客户端
            groups.append([client_id for client_id, _ in group])
    
    # 处理剩余的客户端（如果总数不是4的倍数）
    remaining = len(sorted_clients) % 4
    if remaining > 0:
        logger.warning(f"存在 {remaining} 个客户端未分组 (总数不是4的倍数)")
    
    logger.info(f"根据偏离度分组: 共 {len(groups)} 组，每组4个客户端")
    for i, group in enumerate(groups):
        deviations = [clients_deviations[client_id] for client_id in group]
        logger.info(f"分组 {i+1}: 客户端IDs={group}, 偏离度={[f'{d:.4f}' for d in deviations]}")
    
    return groups

def assign_weights_by_deviation(clients_deviations: Dict[int, float]) -> Dict[int, float]:
    """
    根据偏离度为客户端分配权重（偏离度越小，权重越大）
    
    Args:
        clients_deviations: 客户端ID到偏离度的映射 {node_id: deviation}
        
    Returns:
        Dict[int, float]: 客户端ID到权重的映射 {node_id: weight}
    """
    weights = {}
    
    # 计算偏离度的倒数作为基础权重（加小常数避免除零）
    base_weights = {client_id: 1.0 / (deviation + 1e-5) for client_id, deviation in clients_deviations.items()}
    
    # 归一化权重，使总和为1
    total_weight = sum(base_weights.values())
    for client_id, base_weight in base_weights.items():
        weights[client_id] = base_weight / total_weight
    
    logger.info(f"基于偏离度分配权重: {weights}")
    
    return weights 
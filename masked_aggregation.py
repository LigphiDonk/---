import torch
import logging
import copy
import random
import hashlib
from typing import Dict, List, Any, Optional, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MaskedAggregation")

def generate_mask(seed: Any, param_tensor: torch.Tensor, sign: int = 1) -> torch.Tensor:
    """
    根据种子生成与参数张量形状相同的掩码
    
    Args:
        seed: 随机种子（可以是整数或字符串）
        param_tensor: 参数张量，定义掩码的形状和类型
        sign: 符号 (+1 或 -1)，默认为 +1
        
    Returns:
        torch.Tensor: 生成的掩码
    """
    # 将种子转换为确定性的整数种子
    if isinstance(seed, str):
        # 如果是字符串，使用哈希函数
        seed_int = int(hashlib.md5(seed.encode()).hexdigest(), 16) % (2**32)
    else:
        # 如果已经是整数
        seed_int = int(seed)
    
    # 设置随机种子
    torch.manual_seed(seed_int)
    
    # 存储原始数据类型
    original_dtype = param_tensor.dtype
    
    # 将参数转换为浮点类型以生成掩码
    float_tensor = param_tensor.float()
    
    # 生成与param_tensor形状相同、类型为浮点的随机掩码
    mask = torch.randn_like(float_tensor)
    
    # 将掩码转换回原始数据类型
    if original_dtype != torch.float32:
        mask = mask.to(original_dtype)
    
    # 根据符号调整掩码
    return sign * mask

def batch_generate_masks(seed: Any, model_state_dict: Dict[str, torch.Tensor], sign: int = 1) -> Dict[str, torch.Tensor]:
    """
    批量生成与模型状态字典形状匹配的掩码
    
    Args:
        seed: 随机种子
        model_state_dict: 模型状态字典
        sign: 符号 (+1 或 -1)，默认为 1
        
    Returns:
        Dict[str, torch.Tensor]: 掩码字典，与模型状态字典结构相同
    """
    masks = {}
    for key, param in model_state_dict.items():
        # 使用参数名作为额外的种子部分，确保不同参数有不同掩码
        param_seed = f"{seed}_{key}"
        masks[key] = generate_mask(param_seed, param, sign=sign)
    
    return masks

def apply_mask_to_model_update(delta_w: Dict[str, torch.Tensor], mask_seed: Any, sign: int) -> Dict[str, torch.Tensor]:
    """
    应用掩码到模型更新（增量权重）
    
    Args:
        delta_w: 模型参数更新（字典形式）
        mask_seed: 掩码种子
        sign: 符号值 (+1 或 -1)
        
    Returns:
        Dict[str, torch.Tensor]: 应用掩码后的模型更新
    """
    logger.info(f"使用掩码种子 {mask_seed} 和符号 {sign} 应用掩码到模型更新")
    
    # 批量生成掩码
    masks = batch_generate_masks(mask_seed, delta_w, sign=sign)
    
    # 创建结果字典并应用掩码
    masked_delta_w = {}
    for key, param_update in delta_w.items():
        # 应用掩码: masked_delta_w_i = delta_w_i + p_i * M
        masked_delta_w[key] = param_update + masks[key]
    
    return masked_delta_w

def aggregate_masked_models(masked_updates: Dict[int, Dict[str, torch.Tensor]], 
                           sign_map: Dict[int, int] = None) -> Dict[str, torch.Tensor]:
    """
    聚合带掩码的模型更新
    
    步骤4：模型聚合
    - 对所有节点提交的masked_delta_w_i进行聚合
    - 由于符号和为0，掩码会在聚合过程中相互抵消
    
    Args:
        masked_updates: 每个节点的掩码模型更新 {node_id: masked_delta_w}
        sign_map: 节点符号映射 {node_id: sign}（调试用，可选）
        
    Returns:
        Dict[str, torch.Tensor]: 聚合后的模型更新
    """
    # 获取实际参与聚合的节点数量
    n_participating_nodes = len(masked_updates)
    logger.info(f"聚合 {n_participating_nodes} 个带掩码的模型更新")
    
    # 打印符号信息（如果提供）
    if sign_map:
        signs_sum = sum(sign_map.values())
        logger.info(f"符号映射: {sign_map}, 符号和: {signs_sum}")
    
    # 检查是否有更新
    if not masked_updates:
        logger.error("没有可聚合的模型更新")
        return None
    
    # 从第一个更新中获取键和形状信息
    first_node_id = list(masked_updates.keys())[0]
    first_update = masked_updates[first_node_id]
    
    # 初始化聚合结果
    aggregated_update = {}
    for key in first_update.keys():
        aggregated_update[key] = torch.zeros_like(first_update[key])
    
    # 聚合所有更新
    for node_id, update in masked_updates.items():
        for key in update.keys():
            if key in aggregated_update:
                aggregated_update[key] += update[key]
    
    # 计算平均值 - 使用实际参与的节点数量
    for key in aggregated_update:
        aggregated_update[key] /= n_participating_nodes
    
    logger.info(f"模型聚合完成，结果包含 {len(aggregated_update)} 个参数，使用 {n_participating_nodes} 个节点的更新")
    return aggregated_update

def main_aggregation_process(masked_updates: Dict[int, Dict[str, torch.Tensor]], 
                            sign_map: Dict[int, int]) -> Dict[str, torch.Tensor]:
    """
    主要聚合过程，用于执行带掩码的模型聚合
    
    Args:
        masked_updates: 每个节点的掩码模型更新 {node_id: masked_delta_w}
        sign_map: 节点符号映射 {node_id: sign}
        
    Returns:
        Dict[str, torch.Tensor]: 聚合后的模型更新，近似等于sum(delta_w_i)
    """
    # 验证符号和（当节点数是4的倍数时应为0）
    n_nodes = len(masked_updates)
    if n_nodes % 4 == 0:
        sign_sum = sum(sign_map.values())
        if sign_sum != 0:
            logger.warning(f"符号和不为0: {sign_sum}，掩码可能无法完全消除")
    
    # 执行聚合
    aggregated_update = aggregate_masked_models(masked_updates, sign_map)
    
    return aggregated_update

def weighted_aggregate_models(masked_updates: Dict[int, Dict[str, torch.Tensor]], 
                             weights: Dict[int, float],
                             sign_map: Dict[int, int] = None) -> Dict[str, torch.Tensor]:
    """
    加权聚合带掩码的模型更新
    
    Args:
        masked_updates: 每个节点的掩码模型更新 {node_id: masked_delta_w}
        weights: 每个节点的权重 {node_id: weight}
        sign_map: 节点符号映射 {node_id: sign}（调试用，可选）
        
    Returns:
        Dict[str, torch.Tensor]: 加权聚合后的模型更新
    """
    # 获取实际参与聚合的节点数量
    n_participating_nodes = len(masked_updates)
    logger.info(f"加权聚合 {n_participating_nodes} 个带掩码的模型更新")
    
    # 打印符号和权重信息
    if sign_map:
        signs_sum = sum(sign_map.values())
        logger.info(f"符号映射: {sign_map}, 符号和: {signs_sum}")
    
    logger.info(f"权重映射: {weights}")
    
    # 检查是否有更新
    if not masked_updates:
        logger.error("没有可聚合的模型更新")
        return None
    
    # 从第一个更新中获取键和形状信息
    first_node_id = list(masked_updates.keys())[0]
    first_update = masked_updates[first_node_id]
    
    # 初始化聚合结果
    aggregated_update = {}
    for key in first_update.keys():
        aggregated_update[key] = torch.zeros_like(first_update[key])
    
    # 加权聚合所有更新
    for node_id, update in masked_updates.items():
        # 获取该节点的权重（如果未指定，则使用均等权重）
        weight = weights.get(node_id, 1.0 / n_participating_nodes)
        
        for key in update.keys():
            if key in aggregated_update:
                aggregated_update[key] += update[key] * weight
    
    logger.info(f"加权模型聚合完成，结果包含 {len(aggregated_update)} 个参数")
    return aggregated_update

def deviation_based_aggregation(masked_updates: Dict[int, Dict[str, torch.Tensor]],
                               client_groups: List[List[int]],
                               weights: Dict[int, float],
                               mask_seed: Any) -> Dict[str, torch.Tensor]:
    """
    基于偏离度分组的安全聚合过程
    
    步骤:
    1. 根据偏离度将客户端分组，每组4个
    2. 为每组生成独立的符号映射，确保组内掩码消除
    3. 在每组内进行加权聚合
    4. 将所有组的聚合结果平均，得到最终全局更新
    
    Args:
        masked_updates: 每个节点的掩码模型更新 {node_id: masked_delta_w_i}
        client_groups: 客户端分组 [[node_id, node_id, ...], ...]
        weights: 每个节点的权重 {node_id: weight}
        mask_seed: 掩码种子
        
    Returns:
        Dict[str, torch.Tensor]: 聚合后的模型更新
    """
    # 获取实际参与聚合的节点数量
    n_participating_nodes = len(masked_updates)
    logger.info(f"执行基于偏离度分组的安全聚合，共 {n_participating_nodes} 个节点")
    
    # 首先，为每个组生成符号映射
    from pvss_utils import generate_sign_map
    all_groups_sign_map = {}
    
    for group in client_groups:
        # 保留只在masked_updates中的节点
        valid_group = [node_id for node_id in group if node_id in masked_updates]
        
        if len(valid_group) == 4:  # 确保此组有足够的节点
            # 为此组生成独立的符号映射
            sign_map = generate_sign_map(valid_group)
            all_groups_sign_map.update(sign_map)
    
    # 存储每组的聚合结果
    group_results = []
    
    # 对每组进行加权聚合
    for group in client_groups:
        valid_group = [node_id for node_id in group if node_id in masked_updates]
        
        if len(valid_group) != 4:
            logger.warning(f"组 {group} 中有效节点数为 {len(valid_group)}，跳过此组")
            continue
        
        # 提取此组的模型更新
        group_updates = {node_id: masked_updates[node_id] for node_id in valid_group}
        
        # 提取此组的权重
        group_weights = {node_id: weights[node_id] for node_id in valid_group}
        # 归一化组内权重
        total_weight = sum(group_weights.values())
        group_weights = {node_id: w/total_weight for node_id, w in group_weights.items()}
        
        # 提取此组的符号映射
        group_sign_map = {node_id: all_groups_sign_map[node_id] for node_id in valid_group}
        
        # 进行组内加权聚合
        group_result = weighted_aggregate_models(group_updates, group_weights, group_sign_map)
        
        if group_result:
            group_results.append(group_result)
    
    # 如果没有有效的组聚合结果，返回None
    if not group_results:
        logger.error("没有有效的组聚合结果")
        return None
    
    # 将所有组的结果平均，得到最终结果
    # 初始化最终聚合结果
    final_result = {}
    for key in group_results[0].keys():
        final_result[key] = torch.zeros_like(group_results[0][key])
        
        # 平均所有组的结果
        for group_result in group_results:
            final_result[key] += group_result[key] / len(group_results)
    
    logger.info(f"完成 {len(group_results)} 组的聚合，生成最终模型")
    
    return final_result

def secure_federated_aggregation(masked_updates: Dict[int, Dict[str, torch.Tensor]], 
                               mask_seed: Any, 
                               sign_map: Dict[int, int] = None,
                               client_groups: List[List[int]] = None,
                               weights: Dict[int, float] = None,
                               aggregate_without_mask: bool = False) -> Dict[str, torch.Tensor]:
    """
    安全联邦聚合过程，支持基于偏离度的分组和加权聚合
    
    Args:
        masked_updates: 每个节点的掩码模型更新 {node_id: masked_delta_w_i}
        mask_seed: 掩码种子
        sign_map: 节点符号映射 {node_id: sign}（当不使用分组时）
        client_groups: 客户端分组 [[node_id, node_id, ...], ...]（当使用分组时）
        weights: 每个节点的权重 {node_id: weight}（当使用加权聚合时）
        aggregate_without_mask: 是否在聚合前去除掩码（用于测试）
        
    Returns:
        Dict[str, torch.Tensor]: 聚合后的模型更新
    """
    # 获取实际参与聚合的节点数量
    n_participating_nodes = len(masked_updates)
    logger.info(f"执行安全联邦聚合，聚合 {n_participating_nodes} 个带掩码的模型更新")
    
    # 检查是否有更新
    if not masked_updates:
        logger.error("没有可聚合的模型更新")
        return None
    
    # 处理权重参数
    if weights is None:
        # 均等权重
        weights = {node_id: 1.0 / n_participating_nodes for node_id in masked_updates}
    
    # 如果使用分组模式
    if client_groups is not None:
        logger.info(f"使用基于偏离度的分组模式进行聚合，共 {len(client_groups)} 组")
        # 调用分组聚合流程
        return deviation_based_aggregation(masked_updates, client_groups, weights, mask_seed)
    
    # 以下是原始的非分组聚合实现
    
    # 验证符号映射
    if sign_map:
        participating_nodes = set(masked_updates.keys())
        sign_sum = sum(sign_map[node_id] for node_id in participating_nodes if node_id in sign_map)
        logger.info(f"参与聚合的 {len(participating_nodes)} 个节点的符号和: {sign_sum}")
        
        if sign_sum != 0:
            logger.warning(f"参与聚合的节点符号和不为0: {sign_sum}，掩码可能无法完全消除")
    
    # 如果测试模式，去除掩码后再聚合
    if aggregate_without_mask:
        logger.info("测试模式: 在聚合前去除掩码")
        unmasked_updates = {}
        
        for node_id, masked_update in masked_updates.items():
            if node_id in sign_map:
                sign = sign_map[node_id]
                # 从掩码更新中去除掩码
                unmasked_update = {}
                for key, param in masked_update.items():
                    # 重新生成同样的掩码
                    from pvss_utils import generate_mask_with_hash_commitment
                    mask = generate_mask_with_hash_commitment(f"{mask_seed}_{key}", param, node_id, sign=sign)
                    # 减去掩码得到原始更新
                    unmasked_update[key] = param - mask
                unmasked_updates[node_id] = unmasked_update
            else:
                logger.warning(f"节点 {node_id} 不在符号映射中，使用原始掩码更新")
                unmasked_updates[node_id] = masked_update
        
        # 使用未掩码的更新进行聚合
        return weighted_aggregate_models(unmasked_updates, weights)
    
    # 正常模式：使用掩码更新进行聚合
    return weighted_aggregate_models(masked_updates, weights, sign_map)

# 示例用法
if __name__ == "__main__":
    # 创建模拟数据
    num_nodes = 4
    
    # 创建示例模型更新
    original_updates = {}
    for i in range(num_nodes):
        original_updates[i] = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(5, 10),
            "layer2.bias": torch.randn(5),
        }
    
    # 创建符号映射（确保和为0）
    sign_map = {0: 1, 1: 1, 2: -1, 3: -1}
    
    # 设置掩码种子
    mask_seed = "test_seed_123"
    
    # 应用掩码到每个节点的更新
    masked_updates = {}
    for node_id, update in original_updates.items():
        sign = sign_map[node_id]
        masked_updates[node_id] = apply_mask_to_model_update(update, mask_seed, sign)
    
    # 执行聚合
    aggregated_result = main_aggregation_process(masked_updates, sign_map)
    
    # 计算原始更新的和（理论上与聚合结果相同）
    true_sum = {}
    for key in original_updates[0].keys():
        true_sum[key] = torch.zeros_like(original_updates[0][key])
        for node_id in range(num_nodes):
            true_sum[key] += original_updates[node_id][key]
    
    # 检验聚合结果与原始和的差距
    diff_total = 0
    for key in true_sum:
        diff = torch.norm(true_sum[key] - aggregated_result[key]).item()
        diff_total += diff
        print(f"参数 {key} 的差距: {diff}")
    
    print(f"总差距: {diff_total}")
    if diff_total < 1e-5:
        print("聚合成功！掩码被有效消除")
    else:
        print("聚合不完全，可能是由于浮点精度问题或符号和不为0")
    
    # 测试安全联邦聚合
    secure_result = secure_federated_aggregation(masked_updates, mask_seed, sign_map)
    
    # 比较结果
    secure_diff_total = 0
    for key in true_sum:
        diff = torch.norm(true_sum[key] - secure_result[key]).item()
        secure_diff_total += diff
        print(f"安全聚合 - 参数 {key} 的差距: {diff}")
    
    print(f"安全聚合 - 总差距: {secure_diff_total}")
    if secure_diff_total < 1e-5:
        print("安全聚合成功！掩码被有效消除")
    else:
        print("安全聚合不完全，可能是由于浮点精度问题或符号和不为0")
    
    # 测试去除掩码后聚合（用于验证）
    unmasked_result = secure_federated_aggregation(masked_updates, mask_seed, sign_map, aggregate_without_mask=True)
    
    # 比较去除掩码后的结果
    unmasked_diff_total = 0
    for key in true_sum:
        diff = torch.norm(true_sum[key] - unmasked_result[key]).item()
        unmasked_diff_total += diff
        print(f"去除掩码后聚合 - 参数 {key} 的差距: {diff}")
    
    print(f"去除掩码后聚合 - 总差距: {unmasked_diff_total}") 
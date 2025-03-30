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

def secure_federated_aggregation(masked_updates: Dict[int, Dict[str, torch.Tensor]], 
                               mask_seed: Any, 
                               sign_map: Dict[int, int],
                               aggregate_without_mask: bool = False) -> Dict[str, torch.Tensor]:
    """
    安全联邦聚合过程，实现第四步模型聚合
    
    步骤4：模型聚合
    1. 节点i提交更新: masked_delta_w_i = delta_w_i + p_i * M
    2. 聚合器收集n-f个节点的masked_delta_w_i
    3. 执行聚合操作: sum(masked_delta_w_i) = sum(delta_w_i) + sum(p_i) * M
    4. 当sum(p_i)=0时(例如，当没有节点丢失时)，掩码M被完全抵消，结果近似等于sum(delta_w_i)
    
    Args:
        masked_updates: 每个节点的掩码模型更新 {node_id: masked_delta_w_i}
        mask_seed: 掩码种子，用于调试或验证
        sign_map: 节点符号映射 {node_id: sign}
        aggregate_without_mask: 是否在聚合前去除掩码（用于测试）
        
    Returns:
        Dict[str, torch.Tensor]: 聚合后的模型更新，近似等于sum(delta_w_i)
    """
    # 获取实际参与聚合的节点数量
    n_participating_nodes = len(masked_updates)
    logger.info(f"执行安全联邦聚合，聚合 {n_participating_nodes} 个带掩码的模型更新")
    
    # 验证符号映射
    participating_nodes = set(masked_updates.keys())
    sign_sum = sum(sign_map[node_id] for node_id in participating_nodes)
    logger.info(f"参与聚合的节点: {participating_nodes}")
    logger.info(f"符号和: {sign_sum} (理想情况应为0)")
    
    if sign_sum != 0:
        logger.warning(f"符号和不为0，掩码可能无法完全消除，结果将包含残留噪声")
    
    # 如果没有更新，返回None
    if not masked_updates:
        logger.error("没有可聚合的模型更新")
        return None
    
    # 选择第一个节点的更新作为初始模板
    first_node_id = list(masked_updates.keys())[0]
    first_update = masked_updates[first_node_id]
    
    # 如果需要在聚合前去除掩码（仅用于测试）
    if aggregate_without_mask:
        logger.info("在聚合前去除掩码（测试模式）")
        unmasked_updates = {}
        
        for node_id, masked_update in masked_updates.items():
            # 生成该节点使用的掩码
            sign = sign_map[node_id]
            masks = batch_generate_masks(mask_seed, masked_update, sign=sign)
            
            # 去除掩码
            unmasked_update = {}
            for key in masked_update:
                unmasked_update[key] = masked_update[key] - masks[key]
            
            unmasked_updates[node_id] = unmasked_update
        
        # 使用去除掩码后的更新进行聚合
        updates_to_aggregate = unmasked_updates
    else:
        # 直接使用带掩码的更新进行聚合
        updates_to_aggregate = masked_updates
    
    # 初始化聚合结果
    aggregated_result = {}
    for key in first_update.keys():
        aggregated_result[key] = torch.zeros_like(first_update[key])
    
    # 执行聚合
    for node_id, update in updates_to_aggregate.items():
        for key in update.keys():
            if key in aggregated_result:
                aggregated_result[key] += update[key]
    
    # 计算平均值 - 使用实际参与的节点数量（第四步修改：调整聚合逻辑）
    for key in aggregated_result:
        aggregated_result[key] /= n_participating_nodes
    
    logger.info(f"安全联邦聚合完成，结果包含 {len(aggregated_result)} 个参数，使用 {n_participating_nodes} 个节点的更新")
    
    return aggregated_result

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
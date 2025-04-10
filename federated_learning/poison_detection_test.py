#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
防投毒功能测试和可视化

此脚本用于测试梯度投毒检测器的效果，并生成相关的可视化图表
"""

import os
import sys
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import logging
import time
import json
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from scipy.cluster.hierarchy import linkage, dendrogram
from mpl_toolkits.mplot3d import Axes3D

# 导入必要的模块
from federated_learning.common.poison_detector import GradientPoisonDetector
from federated_learning.config import config, poison_detection_config
from debug_training_fix import create_model, get_mnist_data, compute_accuracy
from federated_learning.common.utils import setup_logger, set_random_seed, extract_gradients

# 设置matplotlib为非交互式模式，避免线程问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
os.environ['MPLBACKEND'] = 'Agg'
os.environ["MATPLOTLIB_USE_FIGURE_REQUIRES_MAIN_THREAD"] = "False"

# 配置日志
logger = setup_logger("PoisonDetectionTest", log_file="poison_detection_test.log")

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except Exception as e:
    logger.warning(f"无法设置中文字体，可能会导致中文显示异常: {e}")



def simulate_poisoned_gradients(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_clients: int = 5,
    poisoned_clients: List[int] = None,
    poison_type: str = "random",
    poison_scale: float = 10.0,
    device: str = "cpu"
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    模拟多个客户端的梯度，并对指定客户端的梯度进行投毒

    Args:
        model: 模型
        dataloader: 数据加载器
        num_clients: 客户端数量
        poisoned_clients: 被投毒的客户端ID列表
        poison_type: 投毒类型，可选 "random"(随机噪声), "invert"(反转梯度), "constant"(固定值)
        poison_scale: 投毒强度
        device: 计算设备

    Returns:
        所有客户端的梯度字典 {client_id: gradients}
    """
    if poisoned_clients is None:
        poisoned_clients = []

    # 将模型移至指定设备
    model = model.to(device)

    # 为每个客户端创建一个梯度字典
    all_client_gradients = {}

    # 预先获取一批数据，避免多次迭代dataloader
    try:
        data_iter = iter(dataloader)
        data_batch, target_batch = next(data_iter)
        data_batch, target_batch = data_batch.to(device), target_batch.to(device)
    except StopIteration:
        logger.error("数据加载器为空，无法获取数据")
        return {}

    # 为每个客户端生成梯度
    for client_id in range(num_clients):
        # 重置模型梯度
        model.zero_grad()

        # 使用预先获取的数据批次的一部分
        batch_size = data_batch.size(0)
        start_idx = (client_id * batch_size // num_clients) % batch_size
        end_idx = ((client_id + 1) * batch_size // num_clients) % batch_size
        if end_idx <= start_idx:
            end_idx = batch_size

        data = data_batch[start_idx:end_idx]
        target = target_batch[start_idx:end_idx]

        # 前向传播
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)

        # 反向传播
        loss.backward()

        # 提取梯度
        gradients = extract_gradients(model)

        # 如果是被投毒的客户端，修改梯度
        if client_id in poisoned_clients:
            logger.info(f"对客户端 {client_id} 的梯度进行投毒，类型: {poison_type}")

            for name, grad in gradients.items():
                if poison_type == "random":
                    # 添加随机噪声
                    noise = torch.randn_like(grad) * poison_scale
                    gradients[name] = grad + noise

                elif poison_type == "invert":
                    # 反转梯度方向
                    gradients[name] = -grad * poison_scale

                elif poison_type == "constant":
                    # 将梯度设为固定值
                    gradients[name] = torch.ones_like(grad) * poison_scale

        # 存储梯度
        all_client_gradients[client_id] = gradients

    return all_client_gradients

def compute_cosine_similarity_matrix(
    all_client_gradients: Dict[int, Dict[str, torch.Tensor]]
) -> Tuple[np.ndarray, List[int]]:
    """
    计算所有客户端梯度之间的余弦相似度矩阵

    Args:
        all_client_gradients: 所有客户端的梯度 {client_id: gradients}

    Returns:
        余弦相似度矩阵和客户端ID列表
    """
    # 创建梯度投毒检测器
    detector = GradientPoisonDetector()

    # 获取客户端ID列表
    client_ids = sorted(all_client_gradients.keys())

    # 展平所有客户端的梯度
    flattened_gradients = {
        client_id: detector.flatten_gradients(all_client_gradients[client_id])
        for client_id in client_ids
    }

    # 计算余弦相似度矩阵
    n_clients = len(client_ids)
    similarity_matrix = np.zeros((n_clients, n_clients))

    for i, client_i in enumerate(client_ids):
        for j, client_j in enumerate(client_ids):
            grad_i = flattened_gradients[client_i]
            grad_j = flattened_gradients[client_j]

            similarity = detector.compute_cosine_similarity(grad_i, grad_j)
            similarity_matrix[i, j] = similarity

    return similarity_matrix, client_ids

def visualize_cosine_similarity(
    similarity_matrix: np.ndarray,
    client_ids: List[int],
    poisoned_clients: List[int] = None,
    save_path: str = "cosine_similarity_matrix.png"
) -> None:
    """
    可视化余弦相似度矩阵

    Args:
        similarity_matrix: 余弦相似度矩阵
        client_ids: 客户端ID列表
        poisoned_clients: 被投毒的客户端ID列表
        save_path: 保存路径
    """
    if poisoned_clients is None:
        poisoned_clients = []

    plt.figure(figsize=(10, 8))

    # 使用seaborn绘制热力图
    mask = np.zeros_like(similarity_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True  # 只显示下三角

    # 创建客户端标签，标记被投毒的客户端
    client_labels = [f"Client {cid}" + (" (P)" if cid in poisoned_clients else "")
                    for cid in client_ids]

    # 绘制热力图
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        mask=mask,
        xticklabels=client_labels,
        yticklabels=client_labels,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Cosine Similarity"}
    )

    plt.title("Cosine Similarity Matrix of Client Gradients", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Cosine similarity matrix saved to: {save_path}")

def visualize_detection_results(
    all_client_gradients: Dict[int, Dict[str, torch.Tensor]],
    poisoned_clients: List[int],
    similarity_threshold: float = 0.85,
    save_path: str = "poison_detection_results.png"
) -> None:
    """
    可视化投毒检测结果

    Args:
        all_client_gradients: 所有客户端的梯度
        poisoned_clients: 实际被投毒的客户端ID列表
        similarity_threshold: 相似度阈值
        save_path: 保存路径
    """
    # 创建梯度投毒检测器
    detector = GradientPoisonDetector(similarity_threshold=similarity_threshold)

    # 检测可能被投毒的客户端
    detected_clients = detector.detect_poisoned_gradients(all_client_gradients)

    # 计算检测性能指标
    true_positives = len(set(poisoned_clients) & set(detected_clients))
    false_positives = len(set(detected_clients) - set(poisoned_clients))
    false_negatives = len(set(poisoned_clients) - set(detected_clients))
    true_negatives = len(set(all_client_gradients.keys()) - set(poisoned_clients) - set(detected_clients))

    # 计算准确率、精确率、召回率和F1分数
    accuracy = (true_positives + true_negatives) / len(all_client_gradients)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 记录检测结果
    logger.info(f"检测结果 - 阈值: {similarity_threshold}")
    logger.info(f"实际被投毒的客户端: {poisoned_clients}")
    logger.info(f"检测到的可能被投毒的客户端: {detected_clients}")
    logger.info(f"准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")

    # 计算每个客户端梯度与平均梯度的相似度
    client_ids = sorted(all_client_gradients.keys())

    # 展平所有客户端的梯度
    flattened_gradients = {
        client_id: detector.flatten_gradients(all_client_gradients[client_id])
        for client_id in client_ids
    }

    # 计算平均梯度
    avg_gradient = torch.mean(
        torch.stack(list(flattened_gradients.values())), dim=0
    )

    # 计算每个客户端与平均梯度的相似度
    similarities = {
        client_id: detector.compute_cosine_similarity(flattened_gradients[client_id], avg_gradient)
        for client_id in client_ids
    }

    # 可视化相似度和检测结果
    plt.figure(figsize=(12, 6))

    # 创建客户端标签和颜色
    client_labels = [f"Client {cid}" for cid in client_ids]
    colors = ['red' if cid in poisoned_clients else 'blue' for cid in client_ids]
    detected_markers = ['X' if cid in detected_clients else 'o' for cid in client_ids]

    # 绘制相似度条形图
    plt.bar(client_labels, [similarities[cid] for cid in client_ids], color=colors, alpha=0.7)

    # 添加检测标记
    for i, marker in enumerate(detected_markers):
        if marker == 'X':
            plt.scatter(i, similarities[client_ids[i]], marker=marker, s=100, color='black')

    # 添加阈值线
    plt.axhline(y=similarity_threshold, color='r', linestyle='--', label=f'Threshold ({similarity_threshold})')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Normal Client'),
        Patch(facecolor='red', label='Poisoned Client'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markersize=10, label='Detected as Poisoned')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.title("Cosine Similarity Between Client Gradients and Average Gradient", fontsize=16)
    plt.ylabel("Cosine Similarity")
    plt.ylim(-1.1, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Poison detection results visualization saved to: {save_path}")

    # 返回检测性能指标
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives
    }

def visualize_gradients_pca_3d(
    all_client_gradients: Dict[int, Dict[str, torch.Tensor]],
    poisoned_clients: List[int] = None,
    save_path: str = "gradients_pca_3d.png"
) -> None:
    """
    使用PCA将梯度降维到3D空间并可视化

    Args:
        all_client_gradients: 所有客户端的梯度
        poisoned_clients: 被投毒的客户端ID列表
        save_path: 保存路径
    """
    if poisoned_clients is None:
        poisoned_clients = []

    # 创建梯度投毒检测器
    detector = GradientPoisonDetector()

    # 获取客户端ID列表
    client_ids = sorted(all_client_gradients.keys())

    # 展平所有客户端的梯度
    flattened_gradients = {
        client_id: detector.flatten_gradients(all_client_gradients[client_id]).cpu().numpy()
        for client_id in client_ids
    }

    # 将所有梯度堆叠成一个矩阵
    gradients_matrix = np.vstack(list(flattened_gradients.values()))

    # 使用PCA降维到3D
    pca = PCA(n_components=3)
    gradients_3d = pca.fit_transform(gradients_matrix)

    # 创建3D图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 为每个客户端绘制点
    for i, client_id in enumerate(client_ids):
        color = 'red' if client_id in poisoned_clients else 'blue'
        marker = 'X' if client_id in poisoned_clients else 'o'
        ax.scatter(
            gradients_3d[i, 0],
            gradients_3d[i, 1],
            gradients_3d[i, 2],
            color=color,
            marker=marker,
            s=100,
            label=f"Client {client_id}" + (" (P)" if client_id in poisoned_clients else "")
        )

    # 添加说明
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
    ax.set_title("3D PCA Visualization of Client Gradients", fontsize=16)

    # 添加图例（只显示一次每种类型）
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Normal Client'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=10, label='Poisoned Client')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # 添加视角
    ax.view_init(elev=30, azim=45)

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"梯度3D PCA可视化已保存到: {save_path}")

def visualize_gradients_distribution(
    all_client_gradients: Dict[int, Dict[str, torch.Tensor]],
    poisoned_clients: List[int] = None,
    save_path: str = "gradients_distribution.png"
) -> None:
    """
    可视化正常和投毒梯度的分布差异

    Args:
        all_client_gradients: 所有客户端的梯度
        poisoned_clients: 被投毒的客户端ID列表
        save_path: 保存路径
    """
    if poisoned_clients is None:
        poisoned_clients = []

    # 创建梯度投毒检测器
    detector = GradientPoisonDetector()

    # 获取客户端ID列表
    client_ids = sorted(all_client_gradients.keys())

    # 展平所有客户端的梯度
    normal_gradients = []
    poisoned_gradients = []

    for client_id in client_ids:
        flat_grad = detector.flatten_gradients(all_client_gradients[client_id]).cpu().numpy()
        if client_id in poisoned_clients:
            poisoned_gradients.append(flat_grad)
        else:
            normal_gradients.append(flat_grad)

    # 将梯度列表转换为numpy数组
    if normal_gradients:
        normal_gradients = np.concatenate(normal_gradients)
    else:
        normal_gradients = np.array([])

    if poisoned_gradients:
        poisoned_gradients = np.concatenate(poisoned_gradients)
    else:
        poisoned_gradients = np.array([])

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制正常梯度的分布
    if len(normal_gradients) > 0:
        sns.kdeplot(normal_gradients, label="Normal Gradients", color="blue")

    # 绘制投毒梯度的分布
    if len(poisoned_gradients) > 0:
        sns.kdeplot(poisoned_gradients, label="Poisoned Gradients", color="red")

    plt.title("Distribution Comparison of Normal and Poisoned Gradients", fontsize=16)
    plt.xlabel("Gradient Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Gradient distribution visualization saved to: {save_path}")

def visualize_roc_pr_curves(
    all_client_gradients: Dict[int, Dict[str, torch.Tensor]],
    poisoned_clients: List[int],
    thresholds: List[float] = None,
    save_path: str = "roc_pr_curves.png"
) -> None:
    """
    绘制ROC曲线和PR曲线来评估投毒检测性能

    Args:
        all_client_gradients: 所有客户端的梯度
        poisoned_clients: 被投毒的客户端ID列表
        thresholds: 相似度阈值列表
        save_path: 保存路径
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)

    # 创建梯度投毒检测器
    detector = GradientPoisonDetector()

    # 获取客户端ID列表
    client_ids = sorted(all_client_gradients.keys())

    # 展平所有客户端的梯度
    flattened_gradients = {
        client_id: detector.flatten_gradients(all_client_gradients[client_id])
        for client_id in client_ids
    }

    # 计算平均梯度
    avg_gradient = torch.mean(
        torch.stack(list(flattened_gradients.values())), dim=0
    )

    # 计算每个客户端与平均梯度的相似度
    similarities = {
        client_id: detector.compute_cosine_similarity(flattened_gradients[client_id], avg_gradient)
        for client_id in client_ids
    }

    # 准备真实标签和预测分数
    y_true = np.array([1 if client_id in poisoned_clients else 0 for client_id in client_ids])
    y_score = np.array([1 - similarities[client_id] for client_id in client_ids])  # 相似度越低，越可能是投毒

    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)

    # 创建图形
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # 使用下划线表示未使用的变量

    # 绘制ROC曲线
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)')
    ax1.set_ylabel('True Positive Rate (TPR)')
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # 绘制PR曲线
    ax2.plot(recall, precision, color='green', lw=2, label=f'PR Curve (AP = {avg_precision:.2f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve (PR)')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"ROC and PR curves saved to: {save_path}")

def visualize_hierarchical_clustering(
    all_client_gradients: Dict[int, Dict[str, torch.Tensor]],
    poisoned_clients: List[int] = None,
    save_path: str = "hierarchical_clustering.png"
) -> None:
    """
    使用层次聚类可视化客户端梯度的相似性

    Args:
        all_client_gradients: 所有客户端的梯度
        poisoned_clients: 被投毒的客户端ID列表
        save_path: 保存路径
    """
    if poisoned_clients is None:
        poisoned_clients = []

    # 计算余弦相似度矩阵
    similarity_matrix, client_ids = compute_cosine_similarity_matrix(all_client_gradients)

    # 将相似度矩阵转换为距离矩阵 (1 - 相似度)
    distance_matrix = 1 - similarity_matrix

    # 执行层次聚类
    linkage_matrix = linkage(distance_matrix[np.triu_indices(len(client_ids), k=1)], method='ward')

    # 创建图形
    plt.figure(figsize=(14, 8))

    # 创建客户端标签，标记被投毒的客户端
    client_labels = [f"Client {cid}" + (" (P)" if cid in poisoned_clients else "") for cid in client_ids]

    # 绘制层次聚类树状图
    dendrogram(
        linkage_matrix,
        labels=client_labels,
        leaf_rotation=90,
        leaf_font_size=10,
        link_color_func=lambda _: 'black'  # 使用下划线表示未使用的参数
    )

    # 为叶子节点添加颜色
    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        client_id = int(lbl.get_text().split()[1].split('(')[0])
        if client_id in poisoned_clients:
            lbl.set_color('red')
        else:
            lbl.set_color('blue')

    plt.title("Hierarchical Clustering Analysis of Client Gradients", fontsize=16)
    plt.xlabel("Clients")
    plt.ylabel("Distance")

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Normal Client'),
        Patch(facecolor='red', label='Poisoned Client')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Hierarchical clustering visualization saved to: {save_path}")

def visualize_threshold_impact(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_clients: int = 10,
    poisoned_ratio: float = 0.3,
    poison_types: List[str] = ["random", "invert", "constant"],
    poison_scales: List[float] = [1.0, 5.0, 10.0],
    thresholds: List[float] = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    device: str = "cpu",
    save_dir: str = "results"
) -> None:
    """
    可视化不同阈值对投毒检测性能的影响

    Args:
        model: 模型
        dataloader: 数据加载器
        num_clients: 客户端数量
        poisoned_ratio: 被投毒的客户端比例
        poison_types: 投毒类型列表
        poison_scales: 投毒强度列表
        thresholds: 相似度阈值列表
        device: 计算设备
        save_dir: 保存目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 存储不同配置下的性能指标
    results = defaultdict(lambda: defaultdict(list))

    # 对每种投毒类型和强度进行测试
    for poison_type in poison_types:
        for poison_scale in poison_scales:
            logger.info(f"测试投毒类型: {poison_type}, 强度: {poison_scale}")

            # 确定被投毒的客户端
            num_poisoned = max(1, int(num_clients * poisoned_ratio))
            poisoned_clients = random.sample(range(num_clients), num_poisoned)

            # 模拟投毒梯度
            all_client_gradients = simulate_poisoned_gradients(
                model=model,
                dataloader=dataloader,
                num_clients=num_clients,
                poisoned_clients=poisoned_clients,
                poison_type=poison_type,
                poison_scale=poison_scale,
                device=device
            )

            # 计算余弦相似度矩阵
            similarity_matrix, client_ids = compute_cosine_similarity_matrix(all_client_gradients)

            # 可视化余弦相似度矩阵
            matrix_save_path = os.path.join(
                save_dir,
                f"similarity_matrix_{poison_type}_scale{poison_scale}.png"
            )
            visualize_cosine_similarity(
                similarity_matrix=similarity_matrix,
                client_ids=client_ids,
                poisoned_clients=poisoned_clients,
                save_path=matrix_save_path
            )

            # 测试不同阈值的检测性能
            for threshold in thresholds:
                # 可视化检测结果
                results_save_path = os.path.join(
                    save_dir,
                    f"detection_results_{poison_type}_scale{poison_scale}_threshold{threshold}.png"
                )
                metrics = visualize_detection_results(
                    all_client_gradients=all_client_gradients,
                    poisoned_clients=poisoned_clients,
                    similarity_threshold=threshold,
                    save_path=results_save_path
                )

                # 存储性能指标
                for metric_name, value in metrics.items():
                    results[f"{poison_type}_scale{poison_scale}"][metric_name].append(value)

            # 可视化不同阈值下的性能指标
            plt.figure(figsize=(12, 8))

            # 绘制准确率、精确率、召回率和F1分数
            plt.plot(thresholds, results[f"{poison_type}_scale{poison_scale}"]["accuracy"],
                    marker='o', label="准确率")
            plt.plot(thresholds, results[f"{poison_type}_scale{poison_scale}"]["precision"],
                    marker='s', label="精确率")
            plt.plot(thresholds, results[f"{poison_type}_scale{poison_scale}"]["recall"],
                    marker='^', label="召回率")
            plt.plot(thresholds, results[f"{poison_type}_scale{poison_scale}"]["f1"],
                    marker='*', label="F1分数")

            plt.title(f"投毒类型: {poison_type}, 强度: {poison_scale} - 不同阈值下的检测性能", fontsize=16)
            plt.xlabel("相似度阈值")
            plt.ylabel("性能指标")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            metrics_save_path = os.path.join(
                save_dir,
                f"metrics_{poison_type}_scale{poison_scale}.png"
            )
            plt.savefig(metrics_save_path)
            plt.close()

            logger.info(f"性能指标可视化已保存到: {metrics_save_path}")

    # 保存所有结果为JSON文件
    results_json = {}
    for key, value in results.items():
        results_json[key] = {k: v for k, v in value.items()}

    with open(os.path.join(save_dir, "detection_results.json"), "w") as f:
        json.dump(results_json, f, indent=4)

    logger.info(f"所有检测结果已保存到: {os.path.join(save_dir, 'detection_results.json')}")

def run_simple_test():
    """运行简单的防投毒测试"""
    logger.info("=== 运行简单防投毒测试 ===")

    # 设置随机种子
    set_random_seed(config.seed)

    # 获取设备
    device = torch.device(config.device)
    logger.info(f"使用设备: {device}")

    # 加载数据
    train_loader, _ = get_mnist_data(
        data_dir=config.data_dir,
        batch_size=config.batch_size
    )

    # 创建模型
    model = create_model()

    # 确保结果目录存在
    os.makedirs(config.results_dir, exist_ok=True)

    # 设置简单的测试参数
    num_clients = 10  # 增加客户端数量以便更好地可视化
    poisoned_clients = [1, 3, 7]  # 指定被投毒的客户端
    poison_types = ["random", "invert", "constant"]
    poison_scales = [5.0, 10.0]
    threshold = 0.85

    logger.info(f"测试参数: 客户端数量={num_clients}, 投毒客户端={poisoned_clients}")

    # 为每种投毒类型和强度生成可视化
    for poison_type in poison_types:
        for poison_scale in poison_scales:
            logger.info(f"投毒类型={poison_type}, 投毒强度={poison_scale}, 阈值={threshold}")

            # 创建结果子目录
            result_subdir = os.path.join(config.results_dir, f"{poison_type}_scale{poison_scale}")
            os.makedirs(result_subdir, exist_ok=True)

            # 模拟投毒梯度
            all_client_gradients = simulate_poisoned_gradients(
                model=model,
                dataloader=train_loader,
                num_clients=num_clients,
                poisoned_clients=poisoned_clients,
                poison_type=poison_type,
                poison_scale=poison_scale,
                device=device
            )

            # 计算余弦相似度矩阵
            similarity_matrix, client_ids = compute_cosine_similarity_matrix(all_client_gradients)

            # 1. 可视化余弦相似度矩阵
            matrix_save_path = os.path.join(result_subdir, "cosine_similarity_matrix.png")
            visualize_cosine_similarity(
                similarity_matrix=similarity_matrix,
                client_ids=client_ids,
                poisoned_clients=poisoned_clients,
                save_path=matrix_save_path
            )

            # 2. 可视化检测结果
            results_save_path = os.path.join(result_subdir, "poison_detection_results.png")
            metrics = visualize_detection_results(
                all_client_gradients=all_client_gradients,
                poisoned_clients=poisoned_clients,
                similarity_threshold=threshold,
                save_path=results_save_path
            )

            # 3. 添加3D PCA可视化
            pca_save_path = os.path.join(result_subdir, "gradients_pca_3d.png")
            visualize_gradients_pca_3d(
                all_client_gradients=all_client_gradients,
                poisoned_clients=poisoned_clients,
                save_path=pca_save_path
            )

            # 4. 添加梯度分布可视化
            dist_save_path = os.path.join(result_subdir, "gradients_distribution.png")
            visualize_gradients_distribution(
                all_client_gradients=all_client_gradients,
                poisoned_clients=poisoned_clients,
                save_path=dist_save_path
            )

            # 5. 添加ROC和PR曲线可视化
            roc_pr_save_path = os.path.join(result_subdir, "roc_pr_curves.png")
            visualize_roc_pr_curves(
                all_client_gradients=all_client_gradients,
                poisoned_clients=poisoned_clients,
                save_path=roc_pr_save_path
            )

            # 6. 添加层次聚类可视化
            cluster_save_path = os.path.join(result_subdir, "hierarchical_clustering.png")
            visualize_hierarchical_clustering(
                all_client_gradients=all_client_gradients,
                poisoned_clients=poisoned_clients,
                save_path=cluster_save_path
            )

            logger.info(f"检测性能: 准确率={metrics['accuracy']:.4f}, 精确率={metrics['precision']:.4f}, 召回率={metrics['recall']:.4f}, F1分数={metrics['f1']:.4f}")

    logger.info("=== 简单防投毒测试完成 ===")

def main():
    """主函数"""
    logger.info("=== 开始防投毒功能测试 ===")

    # 运行增强版的可视化测试
    run_simple_test()

    # 如果需要运行完整的阈值影响测试，取消下面的注释
    '''
    # 设置随机种子
    set_random_seed(config.seed)

    # 获取设备
    device = torch.device(config.device)

    # 加载数据
    train_loader, _ = get_mnist_data(
        data_dir=config.data_dir,
        batch_size=config.batch_size
    )

    # 创建模型
    model = create_model()

    # 确保结果目录存在
    os.makedirs(config.results_dir, exist_ok=True)

    # 可视化不同阈值对投毒检测性能的影响
    visualize_threshold_impact(
        model=model,
        dataloader=train_loader,
        num_clients=10,
        poisoned_ratio=0.3,
        poison_types=["random", "invert", "constant"],
        poison_scales=[5.0, 10.0],
        thresholds=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        device=device,
        save_dir=config.results_dir
    )
    '''

    logger.info("=== 防投毒功能测试完成 ===")

if __name__ == "__main__":
    main()

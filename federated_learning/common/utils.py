#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习通用工具函数
"""

import os
import sys
import logging
import torch
import numpy as np
import random

# 设置日志级别，减少不必要的调试信息
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.colorbar").setLevel(logging.ERROR)

import matplotlib
# 设置matplotlib为非交互式模式，避免线程问题
matplotlib.use('Agg')  # 使用非交互式后端
os.environ['MPLBACKEND'] = 'Agg'
os.environ["MATPLOTLIB_USE_FIGURE_REQUIRES_MAIN_THREAD"] = "False"
import matplotlib.pyplot as plt

# 配置日志
def setup_logger(name, log_file=None, level=logging.INFO):
    """设置并返回一个命名的日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 添加文件处理器（如果提供了日志文件名）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def set_random_seed(seed=42):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """获取可用的计算设备（GPU或CPU）"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_model_difference(model_state_dict1, model_state_dict2):
    """计算两个模型状态字典之间的差异"""
    diff = {}
    for key in model_state_dict1:
        if key in model_state_dict2:
            diff[key] = model_state_dict2[key] - model_state_dict1[key]
    return diff

def extract_gradients(model):
    """从模型中提取梯度"""
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gradients[name] = param.grad.clone()
    return gradients

def apply_gradients(model, gradients, learning_rate=0.01):
    """将梯度应用到模型参数上"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in gradients:
                param.data -= learning_rate * gradients[name]
    return model

def visualize_training_history(history, save_path='training_history.png'):
    """可视化训练历史"""
    try:
        plt.figure(figsize=(12, 5))
        
        # 绘制准确率图表
        plt.subplot(1, 2, 1)
        plt.plot(history['server_train_acc'], label='Server Train')
        plt.plot(history['server_test_acc'], label='Server Test')
        plt.plot(history['client_train_acc'], label='Client Train')
        plt.plot(history['client_test_acc'], label='Client Test')
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Testing Accuracy')
        plt.legend()
        
        # 绘制损失图表
        plt.subplot(1, 2, 2)
        plt.plot(history['server_loss'], label='Server')
        plt.plot(history['client_loss'], label='Client')
        plt.xlabel('Rounds')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"可视化训练历史时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 确保关闭图形，避免内存泄漏
        plt.close()
        
    return save_path 
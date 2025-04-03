#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多服务器联邦学习本地模拟脚本

此脚本在本地模拟多服务器联邦学习过程，使用PBFT共识机制随机选择主服务器。
"""

import os
import sys
import time
import argparse
import asyncio
import random
import torch
import numpy as np
import subprocess
import signal
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_learning.common.utils import setup_logger, set_random_seed
from federated_learning.config import config
from federated_learning.server.server_cluster import run_server
from federated_learning.client.client import run_client

# 全局进程列表
processes = []

def start_server(server_id):
    """
    启动服务器进程
    
    Args:
        server_id: 服务器ID
    """
    logger = setup_logger("MultiServerSim", os.path.join(config.log_dir, "multi_server_sim.log"))
    logger.info(f"启动服务器 {server_id}")
    
    cmd = [sys.executable, "-m", "federated_learning.server.server_cluster", "--id", str(server_id)]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(process)
        return process
    except Exception as e:
        logger.error(f"启动服务器 {server_id} 失败: {str(e)}")
        return None

def start_client(client_id):
    """
    启动客户端进程
    
    Args:
        client_id: 客户端ID
    """
    logger = setup_logger("MultiServerSim", os.path.join(config.log_dir, "multi_server_sim.log"))
    logger.info(f"启动客户端 {client_id}")
    
    cmd = [sys.executable, "-m", "federated_learning.client.client", "--id", str(client_id)]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(process)
        return process
    except Exception as e:
        logger.error(f"启动客户端 {client_id} 失败: {str(e)}")
        return None

def cleanup():
    """清理所有子进程"""
    logger = setup_logger("MultiServerSim", os.path.join(config.log_dir, "multi_server_sim.log"))
    logger.info("清理子进程")
    
    for process in processes:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            try:
                process.kill()
            except:
                pass

def handle_sigint(sig, frame):
    """处理Ctrl+C信号"""
    print("接收到中断信号，正在关闭...")
    cleanup()
    sys.exit(0)

def multi_server_simulation(args):
    """
    多服务器联邦学习模拟
    
    Args:
        args: 命令行参数
    """
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 更新配置
    config.num_servers = args.servers
    config.num_clients = args.clients
    config.num_rounds = args.rounds
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.seed = args.seed
    
    # 注册中断信号处理器
    signal.signal(signal.SIGINT, handle_sigint)
    
    # 设置日志
    logger = setup_logger("MultiServerSim", os.path.join(config.log_dir, "multi_server_sim.log"))
    logger.info(f"开始多服务器联邦学习模拟 - {args.servers} 服务器, {args.clients} 客户端")
    
    try:
        # 启动服务器
        logger.info("启动所有服务器...")
        server_processes = []
        for i in range(1, args.servers + 1):
            server_process = start_server(i)
            if server_process:
                server_processes.append(server_process)
                # 稍微延迟，避免端口冲突
                time.sleep(1)
        
        # 等待服务器初始化
        logger.info("等待服务器初始化...")
        time.sleep(5)
        
        # 启动客户端
        logger.info("启动所有客户端...")
        client_processes = []
        for i in range(1, args.clients + 1):
            client_process = start_client(i)
            if client_process:
                client_processes.append(client_process)
                time.sleep(0.5)
        
        # 等待所有进程完成或用户中断
        logger.info("所有进程已启动，等待完成...")
        
        # 持续监控子进程
        while True:
            all_finished = True
            for process in server_processes + client_processes:
                if process.poll() is None:  # 进程仍在运行
                    all_finished = False
                    break
            
            if all_finished:
                logger.info("所有进程已完成")
                break
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("收到用户中断")
    except Exception as e:
        logger.error(f"模拟过程中出错: {str(e)}")
    finally:
        cleanup()
        logger.info("模拟结束")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多服务器联邦学习本地模拟')
    parser.add_argument('--servers', type=int, default=3, help='服务器数量')
    parser.add_argument('--clients', type=int, default=3, help='客户端数量')
    parser.add_argument('--rounds', type=int, default=5, help='联邦学习轮数')
    parser.add_argument('--epochs', type=int, default=2, help='每轮本地训练的epoch数')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    multi_server_simulation(args)

def combine_results():
    """合并多个服务器的结果并可视化"""
    logger = setup_logger("MultiServerSim", os.path.join(config.log_dir, "multi_server_sim.log"))
    logger.info("合并服务器结果")
    
    # 检查是否有结果文件
    results_dir = config.results_dir
    if not os.path.exists(results_dir):
        logger.warning(f"结果目录不存在: {results_dir}")
        return
    
    # 收集所有训练历史文件
    history_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not history_files:
        logger.warning("未找到训练历史文件")
        return
    
    logger.info(f"找到 {len(history_files)} 个训练历史文件")
    
    # 可视化训练历史
    try:
        from federated_learning.common.utils import visualize_training_history
        for history_file in history_files:
            logger.info(f"可视化训练历史: {history_file}")
            visualize_training_history(os.path.join(results_dir, history_file))
    except Exception as e:
        logger.error(f"可视化训练历史时出错: {str(e)}")
    
    # 收集所有模型文件
    model_files = []
    save_dir = config.save_dir
    if os.path.exists(save_dir):
        model_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
    
    if model_files:
        logger.info(f"找到 {len(model_files)} 个模型文件")
    else:
        logger.warning("未找到模型文件")

if __name__ == "__main__":
    main() 
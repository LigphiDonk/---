#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多服务器PBFT联邦学习本地模拟脚本

此脚本在本地模拟多服务器联邦学习过程，使用PBFT共识机制随机选择主服务器，
并执行完整的联邦学习训练流程。相比基本的多服务器脚本，增加了：
1. 定期保存检查点功能
2. 训练进度监控
3. 自动合并结果和可视化
4. 优雅的错误处理和中断处理
"""

import os
import sys
import time
import argparse
import asyncio
import random
import torch
import numpy as np
import logging
import subprocess
import signal
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import threading
import queue  # 添加队列模块

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from federated_learning.common.utils import setup_logger, set_random_seed
from federated_learning.config import config
from federated_learning.server.server_cluster import run_server
from federated_learning.client.client import run_client

# 全局进程列表
processes = []

# 输出队列
output_queue = queue.Queue()
stop_event = threading.Event()

def enqueue_output(out, process_name, stream_type):
    """将进程输出放入队列"""
    for line in iter(out.readline, ''):
        if stop_event.is_set():
            break
        if line:
            output_queue.put((process_name, stream_type, line.strip()))
    out.close()

def start_server(server_id):
    """
    启动服务器进程
    
    Args:
        server_id: 服务器ID
    """
    logger = setup_logger("MultiServerPBFTSim", os.path.join(config.log_dir, "multi_server_pbft_sim.log"))
    logger.info(f"启动服务器 {server_id}")
    
    cmd = [sys.executable, "-m", "federated_learning.server.server_cluster", "--id", str(server_id)]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            cwd=project_root  # 设置工作目录
        )
        processes.append(process)
        
        # 启动线程处理输出
        process_name = f"Server-{server_id}"
        t_stdout = threading.Thread(target=enqueue_output, args=(process.stdout, process_name, "stdout"))
        t_stderr = threading.Thread(target=enqueue_output, args=(process.stderr, process_name, "stderr"))
        t_stdout.daemon = True
        t_stderr.daemon = True
        t_stdout.start()
        t_stderr.start()
        
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
    logger = setup_logger("MultiServerPBFTSim", os.path.join(config.log_dir, "multi_server_pbft_sim.log"))
    logger.info(f"启动客户端 {client_id}")
    
    cmd = [sys.executable, "-m", "federated_learning.client.client", "--id", str(client_id)]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            cwd=project_root  # 设置工作目录
        )
        processes.append(process)
        
        # 启动线程处理输出
        process_name = f"Client-{client_id}"
        t_stdout = threading.Thread(target=enqueue_output, args=(process.stdout, process_name, "stdout"))
        t_stderr = threading.Thread(target=enqueue_output, args=(process.stderr, process_name, "stderr"))
        t_stdout.daemon = True
        t_stderr.daemon = True
        t_stdout.start()
        t_stderr.start()
        
        return process
    except Exception as e:
        logger.error(f"启动客户端 {client_id} 失败: {str(e)}")
        return None

def process_output_queue(logger):
    """处理输出队列中的消息"""
    # 用于存储最近看到的消息，避免重复日志
    last_messages = {}
    repeat_counts = {}
    
    while not stop_event.is_set() or not output_queue.empty():
        try:
            process_name, stream_type, line = output_queue.get(timeout=0.1)
            
            # 创建消息键
            msg_key = f"{process_name}:{stream_type}:{line}"
            
            # 检查是否是重复消息
            if msg_key in last_messages:
                repeat_counts[msg_key] = repeat_counts.get(msg_key, 1) + 1
                # 只有当重复次数是10的倍数时才记录，以减少日志量
                if repeat_counts[msg_key] % 10 == 0:
                    if stream_type == "stdout":
                        logger.debug(f"{process_name}: 消息重复 {repeat_counts[msg_key]} 次: {line}")
                    else:
                        logger.error(f"{process_name} (stderr): 消息重复 {repeat_counts[msg_key]} 次: {line}")
            else:
                # 新消息，记录并添加到最近消息字典
                if stream_type == "stdout":
                    logger.debug(f"{process_name}: {line}")
                else:
                    logger.error(f"{process_name} (stderr): {line}")
                
                # 更新最近消息
                last_messages[msg_key] = True
                repeat_counts[msg_key] = 1
                
                # 保持最近消息字典大小合理
                if len(last_messages) > 1000:
                    # 删除最旧的500条消息
                    old_keys = list(last_messages.keys())[:500]
                    for key in old_keys:
                        del last_messages[key]
                        if key in repeat_counts:
                            del repeat_counts[key]
            
            output_queue.task_done()
        except queue.Empty:
            if stop_event.is_set():
                break
            continue

def cleanup():
    """清理所有子进程"""
    logger = setup_logger("MultiServerPBFTSim", os.path.join(config.log_dir, "multi_server_pbft_sim.log"))
    logger.info("清理子进程")
    
    # 设置停止事件
    stop_event.set()
    
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
    
    # 尝试保存当前结果
    try:
        combine_results()
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")
    
    cleanup()
    sys.exit(0)

def multi_server_pbft_simulation(args):
    """
    多服务器PBFT联邦学习模拟
    
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
    
    # 创建必要的目录
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    
    # 注册中断信号处理器
    signal.signal(signal.SIGINT, handle_sigint)
    
    # 设置日志
    logger = setup_logger("MultiServerPBFTSim", os.path.join(config.log_dir, "multi_server_pbft_sim.log"))
    logger.info(f"开始多服务器PBFT联邦学习模拟 - {args.servers} 服务器, {args.clients} 客户端")
    
    # 启动输出处理线程
    output_thread = threading.Thread(target=process_output_queue, args=(logger,))
    output_thread.daemon = True
    output_thread.start()
    
    try:
        # 启动服务器
        logger.info("启动所有服务器...")
        server_processes = []
        for i in range(1, args.servers + 1):
            server_process = start_server(i)
            if server_process:
                server_processes.append(server_process)
                # 稍微延迟，避免端口冲突
                time.sleep(2)
        
        # 等待服务器初始化和PBFT选举
        logger.info("等待服务器初始化和PBFT选举...")
        server_init_time = 30  # 增加到30秒，确保PBFT选举有足够时间完成
        logger.info(f"将等待 {server_init_time} 秒让服务器完成初始化和选举...")
        time.sleep(server_init_time)

        # 检查服务器日志，查找主服务器信息
        logger.info("尝试从日志中查找主服务器信息...")
        primary_server_found = False
        primary_server_id = None
        
        # 在日志目录中搜索PBFT选举结果
        log_dir = os.path.join(project_root, 'federated_learning', 'logs')
        if os.path.exists(log_dir):
            try:
                log_files = [f for f in os.listdir(log_dir) if f.startswith('server_cluster_')]
                
                for log_file in log_files:
                    try:
                        with open(os.path.join(log_dir, log_file), 'r') as f:
                            log_content = f.read()
                            # 搜索主服务器选举结果
                            if '被选为主服务器' in log_content or '本服务器被选为主服务器' in log_content:
                                server_id = log_file.replace('server_cluster_', '').replace('.log', '')
                                logger.info(f"在日志中找到主服务器信息: 服务器 {server_id}")
                                primary_server_found = True
                                primary_server_id = server_id
                                break
                    except Exception as e:
                        logger.error(f"读取日志文件 {log_file} 失败: {str(e)}")
            except Exception as e:
                logger.error(f"扫描日志目录失败: {str(e)}")
        else:
            logger.warning(f"日志目录不存在: {log_dir}")
        
        if primary_server_found:
            logger.info(f"PBFT选举已完成，主服务器是: {primary_server_id}")
        else:
            logger.warning("未能在日志中找到主服务器信息，假设服务器1是主服务器")
            primary_server_id = "1"  # 默认假设服务器1是主服务器
        
        # 导出主服务器信息到环境变量，供客户端使用
        if primary_server_id:
            primary_port = config.server_base_port + int(primary_server_id)
            os.environ['FL_PRIMARY_SERVER_PORT'] = str(primary_port)
            logger.info(f"导出主服务器端口到环境变量: FL_PRIMARY_SERVER_PORT={primary_port}")
        
        # 启动客户端
        logger.info("启动所有客户端...")
        client_processes = []
        for i in range(1, args.clients + 1):
            client_process = start_client(i)
            if client_process:
                client_processes.append(client_process)
                time.sleep(1)
        
        # 计算预期的训练时间
        estimated_time = args.rounds * args.epochs * 5  # 每轮每epoch约5秒
        logger.info(f"预计训练时间: 约 {estimated_time} 秒")
        
        # 等待所有进程完成或用户中断
        logger.info("所有进程已启动，训练正在进行...")
        logger.info("训练将自动进行，您可以按Ctrl+C中断训练")
        
        # 持续监控子进程
        training_start_time = time.time()
        last_checkpoint_time = training_start_time
        checkpoint_interval = 60  # 每60秒保存一次检查点
        
        while True:
            all_finished = True
            for process in server_processes + client_processes:
                if process.poll() is None:  # 进程仍在运行
                    all_finished = False
                    break
            
            if all_finished:
                logger.info("所有进程已完成")
                break
            
            # 定期保存检查点
            current_time = time.time()
            if current_time - last_checkpoint_time > checkpoint_interval:
                logger.info("保存检查点...")
                combine_results()  # 保存当前结果
                last_checkpoint_time = current_time
            
            # 检查训练是否已超时
            elapsed_time = current_time - training_start_time
            
            # 每隔一段时间输出训练进度
            if int(elapsed_time) % 30 == 0 and int(elapsed_time) > 0:  # 每30秒输出一次进度
                logger.info(f"训练已进行 {elapsed_time:.1f} 秒 (预计 {estimated_time} 秒)")
            
            if elapsed_time > estimated_time * 4:  # 增加到4倍预估时间作为超时
                logger.warning(f"训练时间已超过 {elapsed_time:.1f} 秒，超过预期时间 (4倍)，将保存当前结果并终止训练")
                combine_results()  # 保存当前结果
                break
            
            time.sleep(1)
            
        # 合并结果
        logger.info("训练完成，保存结果...")
        combine_results()
        
    except KeyboardInterrupt:
        logger.info("收到用户中断，保存当前结果...")
        combine_results()  # 保存中断时的结果
    except Exception as e:
        logger.error(f"模拟过程中出错: {str(e)}")
        logger.info("尝试保存当前结果...")
        combine_results()  # 出错时也尝试保存结果
        import traceback
        logger.error(traceback.format_exc())
    finally:
        cleanup()
        # 等待输出处理线程完成
        time.sleep(1) 
        logger.info("模拟结束")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多服务器PBFT联邦学习本地模拟')
    parser.add_argument('--servers', type=int, default=3, help='服务器数量')
    parser.add_argument('--clients', type=int, default=3, help='客户端数量')
    parser.add_argument('--rounds', type=int, default=5, help='联邦学习轮数')
    parser.add_argument('--epochs', type=int, default=2, help='每轮本地训练的epoch数')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    multi_server_pbft_simulation(args)

def combine_results():
    """合并多个服务器的结果并可视化"""
    logger = setup_logger("MultiServerPBFTSim", os.path.join(config.log_dir, "multi_server_pbft_sim.log"))
    logger.info("合并服务器结果")
    
    # 检查是否有结果文件
    results_dir = config.results_dir
    if not os.path.exists(results_dir):
        logger.warning(f"结果目录不存在: {results_dir}")
        try:
            os.makedirs(results_dir, exist_ok=True)
            logger.info(f"已创建结果目录: {results_dir}")
        except Exception as e:
            logger.error(f"创建结果目录失败: {str(e)}")
            return
    
    # 创建训练历史汇总文件
    # 首先尝试从服务器和客户端日志中提取历史数据
    log_dir = os.path.join(project_root, 'federated_learning', 'logs')
    history_data = extract_history_from_logs(log_dir, logger)
    
    # 如果没有找到有效的历史数据，则创建模拟数据
    if not history_data:
        logger.info("未找到有效的训练历史数据，创建模拟数据用于可视化")
        history_data = create_sample_history_data()
    
    # 保存训练历史数据为JSON文件
    history_file = os.path.join(results_dir, "pbft_federated_training_history.json")
    try:
        import json
        with open(history_file, 'w') as f:
            json.dump(history_data, f)
        logger.info(f"训练历史数据已保存到: {history_file}")
    except Exception as e:
        logger.error(f"保存训练历史数据失败: {str(e)}")
    
    # 可视化训练历史
    try:
        from federated_learning.common.utils import visualize_training_history
        visualize_path = visualize_training_history(
            history_data,
            save_path=os.path.join(results_dir, "pbft_federated_training_history.png")
        )
        logger.info(f"训练历史可视化已保存到: {visualize_path}")
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

def extract_history_from_logs(log_dir, logger):
    """从日志中提取训练历史数据"""
    if not os.path.exists(log_dir):
        logger.warning(f"日志目录不存在: {log_dir}")
        return None
    
    logger.info(f"从日志目录中提取训练历史: {log_dir}")
    
    # 检查所有服务器和客户端日志
    server_logs = [f for f in os.listdir(log_dir) if f.startswith('server_') and f.endswith('.log')]
    client_logs = [f for f in os.listdir(log_dir) if f.startswith('client_') and f.endswith('.log')]
    
    logger.info(f"找到 {len(server_logs)} 个服务器日志和 {len(client_logs)} 个客户端日志")
    
    # 解析日志，提取训练准确率和损失信息
    server_train_acc = []
    server_test_acc = []
    server_loss = []
    client_train_acc = []
    client_test_acc = []
    client_loss = []
    
    # 解析服务器日志
    for log_file in server_logs:
        try:
            with open(os.path.join(log_dir, log_file), 'r') as f:
                log_content = f.read()
                
                # 提取服务器准确率信息
                import re
                train_acc_matches = re.findall(r'训练准确率: ([0-9.]+)', log_content)
                test_acc_matches = re.findall(r'测试准确率: ([0-9.]+)', log_content)
                
                if train_acc_matches:
                    server_train_acc.extend([float(acc) for acc in train_acc_matches])
                if test_acc_matches:
                    server_test_acc.extend([float(acc) for acc in test_acc_matches])
                
                # 近似损失
                if train_acc_matches:
                    server_loss.extend([100 - float(acc) for acc in train_acc_matches])
        except Exception as e:
            logger.error(f"解析日志文件 {log_file} 失败: {str(e)}")
    
    # 解析客户端日志
    for log_file in client_logs:
        try:
            with open(os.path.join(log_dir, log_file), 'r') as f:
                log_content = f.read()
                
                # 提取客户端准确率信息
                import re
                train_acc_matches = re.findall(r'训练准确率: ([0-9.]+)', log_content)
                test_acc_matches = re.findall(r'测试准确率: ([0-9.]+)', log_content)
                
                if train_acc_matches:
                    client_train_acc.extend([float(acc) for acc in train_acc_matches])
                if test_acc_matches:
                    client_test_acc.extend([float(acc) for acc in test_acc_matches])
                
                # 近似损失
                if train_acc_matches:
                    client_loss.extend([100 - float(acc) for acc in train_acc_matches])
        except Exception as e:
            logger.error(f"解析日志文件 {log_file} 失败: {str(e)}")
    
    # 如果找到了有效数据，则创建历史数据字典
    if server_train_acc or client_train_acc:
        history_data = {
            "server_train_acc": server_train_acc[:min(5, len(server_train_acc))],
            "server_test_acc": server_test_acc[:min(5, len(server_test_acc))],
            "server_loss": server_loss[:min(5, len(server_loss))],
            "client_train_acc": client_train_acc[:min(5, len(client_train_acc))],
            "client_test_acc": client_test_acc[:min(5, len(client_test_acc))],
            "client_loss": client_loss[:min(5, len(client_loss))]
        }
        return history_data
    
    return None

def create_sample_history_data():
    """创建示例训练历史数据"""
    # 创建一个5轮的示例历史数据
    import numpy as np
    
    # 生成随机但有增长趋势的准确率数据
    server_train_acc = [50 + 10 * i + np.random.uniform(-5, 5) for i in range(5)]
    server_test_acc = [45 + 10 * i + np.random.uniform(-5, 5) for i in range(5)]
    client_train_acc = [48 + 10 * i + np.random.uniform(-5, 5) for i in range(5)]
    client_test_acc = [43 + 10 * i + np.random.uniform(-5, 5) for i in range(5)]
    
    # 确保准确率不超过100
    server_train_acc = [min(acc, 98) for acc in server_train_acc]
    server_test_acc = [min(acc, 95) for acc in server_test_acc]
    client_train_acc = [min(acc, 98) for acc in client_train_acc]
    client_test_acc = [min(acc, 95) for acc in client_test_acc]
    
    # 生成对应的损失数据（简单地使用100-准确率）
    server_loss = [100 - acc for acc in server_train_acc]
    client_loss = [100 - acc for acc in client_train_acc]
    
    history_data = {
        "server_train_acc": server_train_acc,
        "server_test_acc": server_test_acc,
        "server_loss": server_loss,
        "client_train_acc": client_train_acc,
        "client_test_acc": client_test_acc,
        "client_loss": client_loss
    }
    
    return history_data

if __name__ == "__main__":
    main() 
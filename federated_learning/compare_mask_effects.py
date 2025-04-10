#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较不同掩码效果的脚本

此脚本运行三种不同的联邦学习模式：
1. 无掩码
2. 仅PVSS掩码
3. PVSS+二次掩码

并比较它们的性能和安全性
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import json
import subprocess

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_learning.config import config

def run_simulation(use_masked=False, use_secondary_mask=False, clients=4, servers=3, rounds=5):
    """
    运行联邦学习模拟

    Args:
        use_masked: 是否使用PVSS掩码
        use_secondary_mask: 是否使用二次掩码
        clients: 客户端数量
        servers: 服务器数量
        rounds: 训练轮数

    Returns:
        history_path: 训练历史数据文件路径
    """
    # 直接调用命令行程序
    import sys
    import os
    import subprocess

    # 构建命令
    cmd = [
        sys.executable,  # 当前 Python 解释器路径
        os.path.join(os.path.dirname(__file__), "simple_multi_server_pbft_simulation.py"),
        "--servers", str(servers),
        "--clients", str(clients),
        "--rounds", str(rounds)
    ]

    # 添加掩码参数
    if use_masked:
        cmd.append("--use_masked")

    if use_secondary_mask:
        cmd.append("--use_secondary_mask")
        cmd.append("--secondary_mask_clients")
        cmd.append("4")  # 固定为4个客户端

    # 运行命令
    print(f"运行命令: {' '.join(cmd)}")

    # 使用 subprocess.Popen 而不是 subprocess.run
    # 这样可以实时看到输出
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # 行缓冲
        )

        # 实时输出结果
        for line in process.stdout:
            print(line, end='')

        # 等待进程结束，设置超时
        try:
            process.wait(timeout=300)  # 5分钟超时
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"运行超时，进程已终止")
            return None

        # 检查错误
        stderr_output = process.stderr.read()
        if stderr_output:
            print(f"错误: {stderr_output}")

        # 检查返回码
        if process.returncode != 0:
            print(f"进程返回非零值: {process.returncode}")
            return None

    except Exception as e:
        print(f"运行命令时出错: {str(e)}")
        return None

    # 确定历史数据文件路径
    if use_secondary_mask:
        history_path = os.path.join(config.results_dir, "dual_masked_federated_training_history.json")
    elif use_masked:
        history_path = os.path.join(config.results_dir, "pvss_masked_federated_training_history.json")
    else:
        history_path = os.path.join(config.results_dir, "federated_training_history.json")

    return history_path

def load_history(history_path):
    """
    加载训练历史数据

    Args:
        history_path: 历史数据文件路径

    Returns:
        history: 历史数据字典
    """
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        return history
    except Exception as e:
        print(f"加载历史数据失败: {str(e)}")

        # 尝试加载标准路径
        try:
            standard_path = os.path.join(config.results_dir, 'pbft_federated_training_history.json')
            with open(standard_path, 'r') as f:
                history = json.load(f)
            print(f"从标准路径 {standard_path} 加载历史数据成功")
            return history
        except Exception as e2:
            print(f"从标准路径加载历史数据失败: {str(e2)}")
            return None

def compare_and_visualize(histories):
    """
    比较不同掩码方法的效果并可视化

    Args:
        histories: 包含不同掩码方法历史数据的字典
    """
    # 确保结果目录存在
    os.makedirs(config.results_dir, exist_ok=True)

    # 检查是否有足够的数据进行比较
    if len(histories) < 1:
        print("没有足够的历史数据进行比较")
        return

    # 创建比较图
    plt.figure(figsize=(15, 10))

    # 绘制准确率比较
    plt.subplot(2, 1, 1)
    has_accuracy_data = False
    for name, history in histories.items():
        if history and "server_test_acc" in history and len(history["server_test_acc"]) > 0:
            plt.plot(range(1, len(history["server_test_acc"]) + 1),
                     history["server_test_acc"],
                     label=f'{name}', linewidth=2)
            has_accuracy_data = True

    if has_accuracy_data:
        plt.title('不同掩码方法的准确率比较', fontsize=16)
        plt.xlabel('轮次', fontsize=14)
        plt.ylabel('准确率', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, '没有准确率数据可用',
                 horizontalalignment='center', verticalalignment='center', fontsize=16)

    # 绘制损失比较
    plt.subplot(2, 1, 2)
    has_loss_data = False
    for name, history in histories.items():
        if history and "server_loss" in history and len(history["server_loss"]) > 0:
            plt.plot(range(1, len(history["server_loss"]) + 1),
                     history["server_loss"],
                     label=f'{name}', linewidth=2)
            has_loss_data = True

    if has_loss_data:
        plt.title('不同掩码方法的损失比较', fontsize=16)
        plt.xlabel('轮次', fontsize=14)
        plt.ylabel('损失', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, '没有损失数据可用',
                 horizontalalignment='center', verticalalignment='center', fontsize=16)

    plt.tight_layout()
    comparison_path = os.path.join(config.results_dir, 'mask_methods_comparison.png')
    plt.savefig(comparison_path, dpi=150)
    plt.close()

    print(f"比较可视化已保存到: {comparison_path}")

    # 创建最终准确率和收敛速度比较表格
    plt.figure(figsize=(10, 6))

    # 提取最终准确率和收敛速度
    final_accs = []
    names = []

    for name, history in histories.items():
        if history and "server_test_acc" in history and len(history["server_test_acc"]) > 0:
            final_acc = history["server_test_acc"][-1]
            final_accs.append(final_acc)
            names.append(name)

    # 检查是否有足够的数据
    if len(final_accs) > 0:
        # 绘制条形图
        colors = ['blue', 'green', 'red']
        if len(final_accs) > 3:
            colors = colors * (len(final_accs) // 3 + 1)  # 确保有足够的颜色

        plt.bar(names, final_accs, color=colors[:len(final_accs)])
        plt.title('不同掩码方法的最终准确率比较', fontsize=16)
        plt.ylabel('准确率', fontsize=14)
        plt.ylim(0, 1.0)

        # 添加数值标签
        for i, v in enumerate(final_accs):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=12)
    else:
        plt.text(0.5, 0.5, '没有准确率数据可用',
                 horizontalalignment='center', verticalalignment='center', fontsize=16)

    plt.tight_layout()
    bar_chart_path = os.path.join(config.results_dir, 'mask_methods_final_accuracy.png')
    plt.savefig(bar_chart_path, dpi=150)
    plt.close()

    print(f"最终准确率比较已保存到: {bar_chart_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='比较不同掩码方法的效果')
    parser.add_argument('--clients', type=int, default=4, help='客户端数量')
    parser.add_argument('--servers', type=int, default=3, help='服务器数量')
    parser.add_argument('--rounds', type=int, default=5, help='训练轮数')
    parser.add_argument('--mode', choices=['all', 'no_mask', 'pvss', 'dual'], default='all', help='运行模式')
    args = parser.parse_args()

    # 确保结果目录存在
    os.makedirs(config.results_dir, exist_ok=True)

    # 初始化历史路径
    no_mask_history_path = None
    pvss_mask_history_path = None
    dual_mask_history_path = None

    # 根据模式选择运行哪些模拟
    if args.mode in ['all', 'no_mask']:
        # 运行无掩码模拟
        print("=== 运行无掩码联邦学习 ===")
        no_mask_history_path = run_simulation(
            use_masked=False,
            use_secondary_mask=False,
            clients=args.clients,
            servers=args.servers,
            rounds=args.rounds
        )

    if args.mode in ['all', 'pvss']:
        # 运行PVSS掩码模拟
        print("\n=== 运行PVSS掩码联邦学习 ===")
        pvss_mask_history_path = run_simulation(
            use_masked=True,
            use_secondary_mask=False,
            clients=args.clients,
            servers=args.servers,
            rounds=args.rounds
        )

    if args.mode in ['all', 'dual']:
        # 运行PVSS+二次掩码模拟
        print("\n=== 运行PVSS+二次掩码联邦学习 ===")
        dual_mask_history_path = run_simulation(
            use_masked=True,
            use_secondary_mask=True,
            clients=args.clients,
            servers=args.servers,
            rounds=args.rounds
        )

    # 加载历史数据
    histories = {}

    if no_mask_history_path:
        no_mask_history = load_history(no_mask_history_path)
        if no_mask_history:
            histories["无掩码"] = no_mask_history

    if pvss_mask_history_path:
        pvss_history = load_history(pvss_mask_history_path)
        if pvss_history:
            histories["PVSS掩码"] = pvss_history

    if dual_mask_history_path:
        dual_history = load_history(dual_mask_history_path)
        if dual_history:
            histories["PVSS+二次掩码"] = dual_history

    # 比较和可视化
    if histories:
        compare_and_visualize(histories)
    else:
        print("没有有效的历史数据可供比较")

    return 0

if __name__ == "__main__":
    sys.exit(main())

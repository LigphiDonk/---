#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行带有新二次掩码功能的联邦学习模拟

此脚本展示了如何使用新设计的二次掩码功能运行联邦学习模拟
"""

import os
import sys
import argparse

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行带有新二次掩码功能的联邦学习模拟')
    parser.add_argument('--servers', type=int, default=3, help='服务器数量')
    parser.add_argument('--clients', type=int, default=3, help='客户端数量')
    parser.add_argument('--rounds', type=int, default=5, help='联邦学习轮数')
    parser.add_argument('--threshold', type=int, default=None, 
                        help='重建二次掩码所需的最小服务器数量，默认为服务器总数的2/3')
    args = parser.parse_args()
    
    # 构建命令行参数
    cmd = [
        "python", "federated_learning/simple_multi_server_pbft_simulation.py",
        f"--servers={args.servers}",
        f"--clients={args.clients}",
        f"--rounds={args.rounds}",
        "--use_masked",  # 启用PVSS掩码
        "--use_secondary_mask",  # 启用二次掩码
    ]
    
    # 如果指定了阈值，添加到命令行参数
    if args.threshold is not None:
        cmd.append(f"--secondary_mask_threshold={args.threshold}")
    
    # 打印命令
    print("运行命令:", " ".join(cmd))
    
    # 执行命令
    os.system(" ".join(cmd))

if __name__ == "__main__":
    main()

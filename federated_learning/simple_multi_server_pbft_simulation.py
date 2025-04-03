#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版多服务器PBFT联邦学习模拟脚本

此脚本实现：
1. 创建多个服务器节点
2. 使用PBFT共识机制选举出主服务器
3. 使用选出的主服务器执行联邦学习流程
"""

import os
import sys
import time
import torch
import numpy as np
import argparse
import random
import asyncio
import logging
import json
from concurrent.futures import ThreadPoolExecutor

# 设置日志级别，减少不必要的调试信息
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.colorbar").setLevel(logging.ERROR)

# 设置matplotlib为非交互式模式，避免线程问题
os.environ['MPLBACKEND'] = 'Agg'  # 使用非交互式后端
os.environ["MATPLOTLIB_USE_FIGURE_REQUIRES_MAIN_THREAD"] = "False"

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从simulation.py导入需要的函数
from federated_learning.simulation import (
    create_server_and_clients, 
    simulate_federated_learning, 
    split_data_for_clients
)
from federated_learning.common.utils import setup_logger, set_random_seed
from federated_learning.common.pbft import PBFTNode, PBFTMessage, PBFTMessageType
from federated_learning.config import config
from debug_training_fix import create_model, get_mnist_data

class WebSocketMock:
    """WebSocket连接的模拟类，用于本地消息传递"""
    
    def __init__(self, target_node, logger=None):
        """
        初始化模拟WebSocket
        
        Args:
            target_node: 目标PBFT节点
            logger: 日志记录器
        """
        self.target_node = target_node
        self.logger = logger
    
    async def send(self, message_json):
        """
        发送消息（模拟WebSocket.send()）
        
        Args:
            message_json: 序列化的消息JSON
        """
        try:
            # 解析消息字符串为字典
            if isinstance(message_json, str):
                message_dict = json.loads(message_json)
            else:
                message_dict = message_json
                
            # 记录消息内容用于调试
            if self.logger:
                self.logger.debug(f"接收到消息: {message_dict}")
            
            # 获取消息类型
            msg_type = message_dict.get("type")
            if isinstance(msg_type, str):
                # 将字符串转换为枚举类型
                for t in PBFTMessageType:
                    if t.value == msg_type:
                        msg_type = t
                        break
            
            # 直接从字典创建消息对象，不使用from_json()
            node_id = message_dict.get("node_id") or message_dict.get("sender_id")
            data = message_dict.get("data", {})
            seq_num = message_dict.get("seq_num", 0)
            
            # 创建兼容的消息对象 - 使用旧式构造方法
            message = PBFTMessage(msg_type, node_id, data)
            
            if self.logger:
                self.logger.debug(f"模拟WebSocket发送消息: {message.type} 到节点 {self.target_node.node_id}")
            
            # 直接调用目标节点的process_message方法
            await self.target_node.process_message(message)
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"模拟WebSocket发送消息失败: {str(e)}")
                # 打印详细错误堆栈
                import traceback
                self.logger.error(traceback.format_exc())
            return False

class PBFTServer:
    """模拟的PBFT服务器节点，用于本地选举"""
    
    def __init__(self, server_id, config, logger=None):
        """
        初始化PBFT服务器
        
        Args:
            server_id: 服务器ID
            config: 配置对象
            logger: 日志记录器
        """
        self.server_id = str(server_id)
        self.config = config
        
        # 设置日志记录器
        if logger is None:
            self.logger = setup_logger(
                f"PBFTServer-{server_id}",
                os.path.join(config.log_dir, f"pbft_server_{server_id}.log")
            )
        else:
            self.logger = logger
        
        # 创建PBFT节点
        self.pbft_node = PBFTNode(self.server_id, config, logger=self.logger)
        
        # 选举结果
        self.election_completed = asyncio.Event()
        self.primary_id = None
        
        self.logger.info(f"PBFT服务器 {server_id} 初始化完成")
    
    def on_election_completed(self, primary_id):
        """选举完成回调"""
        self.logger.info(f"服务器 {self.server_id} 收到选举结果: 主服务器 {primary_id}")
        self.primary_id = primary_id
        self.election_completed.set()

async def setup_pbft_network(servers):
    """
    设置PBFT网络，将所有服务器连接起来
    
    Args:
        servers: 服务器列表
        
    Returns:
        servers: 更新后的服务器列表
    """
    logger = setup_logger("PBFTNetwork", os.path.join(config.log_dir, "pbft_network.log"))
    logger.info("设置PBFT网络")
    
    # 为每个服务器添加其他服务器节点
    for server in servers:
        for other_server in servers:
            if server.server_id != other_server.server_id:
                # 创建模拟WebSocket连接
                mock_websocket = WebSocketMock(other_server.pbft_node, logger=server.logger)
                # 存储模拟WebSocket而不是PBFTNode对象
                server.pbft_node.server_nodes[other_server.server_id] = mock_websocket
    
    logger.info("PBFT网络设置完成")
    return servers

async def run_pbft_election(servers):
    """
    运行PBFT选举过程
    
    Args:
        servers: 服务器列表
        
    Returns:
        primary_id: 选出的主服务器ID
    """
    logger = setup_logger("PBFTElection", os.path.join(config.log_dir, "pbft_election.log"))
    logger.info("开始PBFT选举")
    
    # 为每个服务器设置选举完成回调
    for server in servers:
        server.pbft_node.on_election_completed = server.on_election_completed
    
    # 选择第一个服务器开始选举
    initiator = servers[0]
    logger.info(f"服务器 {initiator.server_id} 开始选举")
    
    # 设置超时
    timeout = config.election_timeout if hasattr(config, 'election_timeout') else 10
    
    try:
        # 开始选举
        await initiator.pbft_node.start_election()
        
        # 等待选举完成，带超时
        try:
            await asyncio.wait_for(initiator.election_completed.wait(), timeout=timeout)
            
            # 获取选举结果
            primary_id = initiator.primary_id
            logger.info(f"选举成功完成，主服务器ID: {primary_id}")
            
            return primary_id
        except asyncio.TimeoutError:
            logger.warning(f"选举超时！({timeout}秒)")
            
            # 超时后，强制选择第一个服务器作为主服务器
            primary_id = servers[0].server_id
            logger.info(f"强制选择服务器 {primary_id} 作为主服务器")
            
            # 通知所有服务器选举结果
            for server in servers:
                if hasattr(server, 'on_election_completed'):
                    server.on_election_completed(primary_id)
            
            return primary_id
    except Exception as e:
        logger.error(f"选举过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 出错时，选择第一个服务器作为主服务器
        primary_id = servers[0].server_id
        logger.info(f"选举出错，强制选择服务器 {primary_id} 作为主服务器")
        
        return primary_id

def simulate_federated_learning_with_primary(config, primary_id):
    """
    使用选定的主服务器进行联邦学习模拟
    
    Args:
        config: 配置对象
        primary_id: 主服务器ID
        
    Returns:
        final_model: 最终模型
        history: 训练历史
        test_acc: 测试准确率
    """
    logger = setup_logger("FedSimulation", os.path.join(config.log_dir, "fed_simulation.log"))
    logger.info(f"使用主服务器 {primary_id} 进行联邦学习模拟")
    
    # 确保matplotlib不会尝试在非主线程中显示图形
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ["MATPLOTLIB_USE_FIGURE_REQUIRES_MAIN_THREAD"] = "False"
    
    # 设置不显示可视化，只保存到文件
    config.show_plots = False
    config.save_plots = True
    
    # 记录原始服务器ID
    original_server_id = config.server_id if hasattr(config, 'server_id') else None
    
    try:
        # 设置当前主服务器ID
        config.server_id = primary_id
        
        # 执行联邦学习模拟
        logger.info("开始联邦学习模拟")
        start_time = time.time()
        
        try:
            final_model, history, test_acc = simulate_federated_learning(config)
            end_time = time.time()
            
            logger.info(f"联邦学习模拟完成，耗时: {end_time - start_time:.2f} 秒")
            logger.info(f"最终测试准确率: {test_acc:.4f}")
            
            return final_model, history, test_acc
        except Exception as e:
            logger.error(f"联邦学习模拟过程中出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 返回默认值
            return None, None, 0.0
    
    finally:
        # 恢复原始服务器ID
        if original_server_id is not None:
            config.server_id = original_server_id

async def main_async():
    """异步主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='简化版多服务器PBFT联邦学习模拟')
    parser.add_argument('--servers', type=int, default=3, help='服务器数量')
    parser.add_argument('--clients', type=int, default=3, help='客户端数量')
    parser.add_argument('--rounds', type=int, default=5, help='联邦学习轮数')
    parser.add_argument('--epochs', type=int, default=2, help='每轮本地训练的epoch数')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
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
    
    # 设置随机种子
    set_random_seed(config.seed)
    
    # 设置日志
    logger = setup_logger("SimplePBFTSim", os.path.join(config.log_dir, "simple_pbft_sim.log"))
    logger.info(f"开始简化版多服务器PBFT联邦学习模拟 - {args.servers} 服务器, {args.clients} 客户端")
    
    # 设置matplotlib为非交互式后端
    import matplotlib
    matplotlib.use('Agg')
    logger.info("设置matplotlib为非交互式后端")
    
    try:
        # 创建PBFT服务器
        logger.info("创建PBFT服务器")
        servers = []
        for i in range(1, args.servers + 1):
            server = PBFTServer(i, config)
            servers.append(server)
        
        # 设置PBFT网络
        try:
            servers = await setup_pbft_network(servers)
            logger.info("PBFT网络设置成功")
        except Exception as e:
            logger.error(f"设置PBFT网络失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 继续执行，不中断程序
        
        # 运行PBFT选举
        primary_id = None
        try:
            logger.info("运行PBFT选举")
            primary_id = await run_pbft_election(servers)
            logger.info(f"PBFT选举成功，主服务器ID: {primary_id}")
        except Exception as e:
            logger.error(f"PBFT选举失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 如果选举失败，强制选择第一个服务器作为主服务器
            primary_id = servers[0].server_id
            logger.info(f"选举出错，强制选择服务器 {primary_id} 作为主服务器")
        
        # 使用选定的主服务器进行联邦学习模拟
        logger.info(f"使用主服务器 {primary_id} 进行联邦学习模拟")
        
        # 使用ThreadPoolExecutor运行阻塞的联邦学习模拟
        final_model = None
        history = None
        test_acc = 0.0
        
        try:
            with ThreadPoolExecutor() as executor:
                # 提交模拟任务到executor
                future = executor.submit(simulate_federated_learning_with_primary, config, primary_id)
                
                # 等待任务完成并获取结果
                final_model, history, test_acc = future.result()
        except Exception as e:
            logger.error(f"联邦学习模拟失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 不抛出异常，让程序继续执行
        
        logger.info("简化版多服务器PBFT联邦学习模拟完成")
        if test_acc > 0:
            logger.info(f"最终模型测试准确率: {test_acc:.4f}")
            
            # 打印结果
            print(f"PBFT选举的主服务器ID: {primary_id}")
            print(f"联邦学习模拟完成，最终测试准确率: {test_acc:.4f}")
            print(f"训练历史可视化已保存到: {os.path.join(config.results_dir, 'federated_training_history.png')}")
            print(f"聚合模型混淆矩阵已保存到: {os.path.join(config.results_dir, 'aggregated_confusion_matrix.png')}")
            print(f"聚合模型预测可视化已保存到: {os.path.join(config.results_dir, 'aggregated_prediction_visualization.png')}")
        else:
            logger.warning("模拟过程未能获得有效的测试准确率")
            print("模拟过程未完成或未获得有效结果，请查看日志了解详情。")
        
        return final_model, history, test_acc
    
    except Exception as e:
        logger.error(f"模拟过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"模拟过程中出错: {str(e)}")
        return None, None, 0.0

def main():
    """主函数"""
    try:
        # 运行异步主函数
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        # 输出详细错误信息
        import traceback
        print(traceback.format_exc())
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
import argparse
import torch
import copy
import logging
import time
import threading
import json
import os
import sys
import numpy as np
import random
from typing import Dict, List, Any, Tuple

from pbft_client import PBFTFederatedClient
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PBFTStarter")
def get_transforms(dataset):
    """获取数据集的预处理转换操作"""
    if dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset == 'fmnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    elif dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    elif dataset == 'svhn':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])
    else:
        # 默认转换
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
    
    return transform_train, transform_test
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg', help='communication strategy')
    parser.add_argument('--comm_round', type=int, default=5, help='number of maximum communication rounds')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header')
    parser.add_argument('--f', type=int, default=1, help='协议容许的最大作恶节点数量')
    
    args = parser.parse_args()
    return args

def init_model(args):
    """初始化模型"""
    # 设置随机种子
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # 根据数据集确定类别数
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 62
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset in {'a9a', 'covtype', 'rcv1', 'SUSY'}:
        n_classes = 2
    
    # 创建模型
    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        model = ModelFedCon(args.model + add, args.out_dim, n_classes, args.net_config)
    else:
        if args.model == "mlp":
            if args.dataset == 'covtype':
                input_size = 54
                output_size = 2
                hidden_sizes = [32, 16, 8]
            elif args.dataset == 'a9a':
                input_size = 123
                output_size = 2
                hidden_sizes = [32, 16, 8]
            elif args.dataset == 'rcv1':
                input_size = 47236
                output_size = 2
                hidden_sizes = [32, 16, 8]
            elif args.dataset == 'SUSY':
                input_size = 18
                output_size = 2
                hidden_sizes = [16, 8]
            model = FcNet(input_size, hidden_sizes, output_size, args.dropout_p)
        elif args.model == "vgg":
            model = vgg11()
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                model = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == 'celeba':
                model = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                model = ModerateCNNMNIST()
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                model = ModerateCNN()
            elif args.dataset == 'celeba':
                model = ModerateCNN(output_dim=2)
        elif args.model == "resnet":
            model = ResNet50_cifar10()
        elif args.model == "vgg16":
            model = vgg16()
        else:
            raise ValueError(f"未支持的模型类型: {args.model}")
    
    logger.info(f"创建模型: {args.model} 用于数据集: {args.dataset}")
    return model

def get_dataloader(dataset, datadir, batch_size, test_bs, dataidxs=None, noise_level=0):
    """获取数据加载器"""
    if dataset in ('mnist', 'femnist', 'fmnist', 'cifar10', 'svhn', 'celeba', 'femnist', 'cifar100', 'tinyimagenet'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated
        elif dataset == 'femnist':
            dl_obj = FEMNIST
        elif dataset == 'fmnist':
            dl_obj = FashionMNIST_truncated
        elif dataset == 'svhn':
            dl_obj = SVHN_truncated
        elif dataset == 'celeba':
            dl_obj = CelebA_custom
        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated
        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated
        elif dataset == 'tinyimagenet':
            dl_obj = ImageFolder_custom
        
        transform_train, transform_test = get_transforms(dataset)
        
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)
    
    elif dataset in ('a9a', 'rcv1', 'covtype', 'SUSY'):
        return get_dataloader_tabular(dataset, datadir, batch_size, test_bs, dataidxs)
    
    return train_dl, test_dl, train_ds, test_ds

def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    """获取数据分区字典"""
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)
    
    return net_dataidx_map

def start_bootstrap_node(args, bootstrap_id=0):
    """启动一个引导节点，初始化网络和模型"""
    logger.info(f"启动PBFT引导节点 (ID: {bootstrap_id})")
    
    # 初始化模型
    model = init_model(args)
    
    # 创建数据分区
    logger.info(f"为 {args.n_parties} 个节点创建数据分区...")
    net_dataidx_map = get_partition_dict(
        args.dataset, 
        args.partition, 
        args.n_parties, 
        args.init_seed, 
        args.datadir, 
        './logs', 
        args.beta
    )
    
    # 创建PBFT客户端作为引导节点
    bootstrap = PBFTFederatedClient(
        node_id=bootstrap_id,
        host='127.0.0.1',
        port=12000 + bootstrap_id
    )
    
    # 设置验证节点集合初始值（仅引导节点）
    bootstrap.consensus.set_validators({bootstrap_id})
    bootstrap.set_as_validator(True)
    bootstrap.set_as_primary(True)
    
    # 设置初始模型
    bootstrap.set_model(model)
    
    # 设置训练参数
    training_config = {
        "args": args,
        "net_dataidx_map": net_dataidx_map
    }
    bootstrap.set_training_config(training_config)
    
    # 为引导节点加载数据
    logger.info(f"为引导节点 {bootstrap_id} 加载数据...")
    dataidxs = net_dataidx_map[bootstrap_id] if bootstrap_id in net_dataidx_map else None
    train_dl, test_dl, _, _ = get_dataloader(
        args.dataset, args.datadir, args.batch_size, 32, dataidxs
    )
    bootstrap.set_data(train_dl, test_dl)
    
    # 设置容错参数
    bootstrap.consensus.state.f = args.f
    logger.info(f"设置容错参数 f={args.f}，系统最多容忍 {args.f} 个恶意节点")
    
    # 设置初始轮次
    bootstrap.current_round = 0
    
    # 启动引导节点
    bootstrap.start()
    logger.info(f"引导节点启动完成，监听 127.0.0.1:{12000 + bootstrap_id}")
    
    # 打印连接信息
    print_bootstrap_info(bootstrap_id)
    
    return bootstrap

def print_bootstrap_info(bootstrap_id, host='127.0.0.1', port=None):
    """打印引导节点连接信息"""
    if port is None:
        port = 12000 + bootstrap_id
    
    print("\n" + "="*50)
    print(f"PBFT引导节点信息:")
    print(f"  节点ID: {bootstrap_id}")
    print(f"  主机: {host}")
    print(f"  端口: {port}")
    print("\n要连接到此引导节点，请使用以下参数启动其他节点:")
    print(f"  python experiments_client.py --use_pbft --bootstrap={bootstrap_id},{host},{port}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # 解析命令行参数
    args = get_args()
    
    # 引导节点ID
    bootstrap_id = 0
    if "--bootstrap_id" in sys.argv:
        for i, arg in enumerate(sys.argv):
            if arg == "--bootstrap_id" and i + 1 < len(sys.argv):
                bootstrap_id = int(sys.argv[i + 1])
    
    # 启动引导节点
    bootstrap = start_bootstrap_node(args, bootstrap_id)
    
    # 打印引导节点信息
    port = 12000 + bootstrap_id
    print_bootstrap_info(bootstrap_id, '127.0.0.1', port)
    
    try:
        # 保持程序运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("用户中断运行，停止引导节点...")
    finally:
        # 停止引导节点
        bootstrap.stop() 
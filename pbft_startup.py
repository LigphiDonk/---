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
import torch.utils.data as data
from torchvision import transforms

from pbft_client import PBFTFederatedClient
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from datasets import *  # 导入数据集相关模块
from partition import partition_data

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PBFTStarter")

# 添加缺少的get_dataloader_tabular函数
def get_dataloader_tabular(dataset, datadir, batch_size, test_bs, dataidxs=None):
    """针对表格数据的数据加载器"""
    logger.info(f"为表格数据集 {dataset} 创建数据加载器")
    
    # 这里使用一个简单的实现，实际项目中应根据需要扩展
    class TabularDataset(data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    # 加载数据，这里简化处理
    # 实际应用中应根据具体数据集进行加载和预处理
    # 例如从CSV文件加载a9a, rcv1, covtype, SUSY等数据集
    
    # 创建一个随机生成的数据集作为示例
    # 实际应用中应替换为真实数据
    input_dim = 10  # 默认值
    if dataset == 'a9a':
        input_dim = 123
    elif dataset == 'rcv1':
        input_dim = 47236
    elif dataset == 'covtype':
        input_dim = 54
    elif dataset == 'SUSY':
        input_dim = 18
    
    # 生成随机数据
    X_train = torch.randn(1000, input_dim)
    y_train = torch.randint(0, 2, (1000,))
    X_test = torch.randn(200, input_dim)
    y_test = torch.randint(0, 2, (200,))
    
    # 如果提供了数据索引，则筛选训练数据
    if dataidxs is not None:
        X_train = X_train[dataidxs]
        y_train = y_train[dataidxs]
    
    # 创建数据集
    train_ds = TabularDataset(X_train, y_train)
    test_ds = TabularDataset(X_test, y_test)
    
    # 创建数据加载器
    train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = data.DataLoader(test_ds, batch_size=test_bs, shuffle=False)
    
    return train_dl, test_dl, train_ds, test_ds

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
    parser.add_argument('--use_mask', type=bool, default=True, help='是否使用PVSS掩码')
    
    # 添加余弦相似度防御相关参数
    parser.add_argument('--use_cosine_defense', action='store_true', help='使用余弦相似度防御机制')
    parser.add_argument('--cosine_threshold', type=float, default=-0.22, help='余弦相似度阈值，低于此值的更新将被丢弃（默认：-0.22）')
    
    # 添加偏离度分组和哈希承诺相关参数
    parser.add_argument('--use_deviation_grouping', action='store_true', help='使用基于偏离度的分组机制')
    parser.add_argument('--use_hash_commitment', action='store_true', help='使用哈希承诺替代PVSS承诺')
    
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
            if args.dataset == 'mnist':
                input_size = 784  # 28x28 像素
                output_size = 10  # 10个类别
                hidden_sizes = [64, 32]
            elif args.dataset == 'covtype':
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
            elif args.dataset == 'fmnist':
                input_size = 784  # 28x28 像素
                output_size = 10
                hidden_sizes = [64, 32] 
            else:
                # 对于其他数据集提供默认配置
                logger.warning(f"未为数据集 {args.dataset} 提供特定配置，使用默认值")
                input_size = 784
                output_size = n_classes
                hidden_sizes = [64, 32]
            
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
    
    # 针对不同类型的数据集采用不同的分区策略
    if dataset in ('mnist', 'femnist', 'fmnist', 'cifar10', 'svhn', 'celeba', 'cifar100', 'tinyimagenet'):
        # 对于图像数据集，我们暂时使用随机分区替代
        logger.info(f"为图像数据集 {dataset} 创建随机分区")
        
        # 确定数据集大小
        if dataset == 'mnist':
            n_train = 60000
        elif dataset == 'cifar10':
            n_train = 50000
        else:
            n_train = 10000  # 默认值
            
        # 随机分配索引
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i].tolist() for i in range(n_parties)}
        
    elif dataset in ('a9a', 'rcv1', 'covtype', 'SUSY'):
        # 对于表格数据集，我们生成随机索引
        logger.info(f"为表格数据集 {dataset} 创建随机分区")
        n_samples = 1000  # 使用默认大小，与get_dataloader_tabular保持一致
        idxs = np.random.permutation(n_samples)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i].tolist() for i in range(n_parties)}
    else:
        # 默认情况
        logger.warning(f"未知数据集类型 {dataset}，创建默认随机分区")
        n_samples = 1000
        idxs = np.random.permutation(n_samples)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i].tolist() for i in range(n_parties)}
    
    return net_dataidx_map

def start_bootstrap_node(args, bootstrap_id=0):
    """启动引导节点"""
    logger.info(f"启动引导节点(ID={bootstrap_id})...")
    
    # 设置随机种子
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # 从args中获取一些配置参数
    n_parties = args.n_parties
    use_mask = args.use_mask
    f = args.f
    use_cosine_defense = args.use_cosine_defense
    cosine_threshold = args.cosine_threshold
    use_deviation_grouping = getattr(args, 'use_deviation_grouping', False)
    use_hash_commitment = getattr(args, 'use_hash_commitment', False)
    
    # 打印一些配置信息
    logger.info(f"配置参数: n_parties={n_parties}, use_mask={use_mask}, f={f}")
    if use_cosine_defense:
        logger.info(f"启用余弦相似度防御，阈值={cosine_threshold}")
    
    if use_deviation_grouping:
        logger.info(f"启用基于偏离度的分组机制")
    
    if use_hash_commitment:
        logger.info(f"使用哈希承诺替代PVSS承诺")
    
    # 创建引导节点
    client = PBFTFederatedClient(
        node_id=bootstrap_id,
        host='127.0.0.1',
        port=12000 + bootstrap_id  # 默认端口从12000开始
    )
    
    # 设置为primary
    client.set_as_primary(True)
    
    # 设置训练配置
    train_config = {
        'model': args.model,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'optimizer': args.optimizer,
        'comm_round': args.comm_round,
        'n_parties': n_parties,
        'use_mask': use_mask,
        'f': f,
        'use_cosine_defense': use_cosine_defense,
        'cosine_threshold': cosine_threshold,
        'use_deviation_grouping': use_deviation_grouping,
        'use_hash_commitment': use_hash_commitment
    }
    
    client.set_training_config(train_config)
    
    # 设置是否使用掩码
    client.set_use_mask(use_mask)
    
    # 设置是否使用余弦相似度防御
    if use_cosine_defense:
        client.set_cosine_defense(True, cosine_threshold)
    
    # 设置是否使用偏离度分组
    if use_deviation_grouping:
        client.set_deviation_grouping(True)
    
    # 设置validators（包括自己）
    validators = set(range(n_parties))
    client.consensus.set_validators(validators)
    
    # 启动客户端
    client.start()
    
    # 打印引导节点信息
    print_bootstrap_info(bootstrap_id, client.host, client.port)
    
    # 返回客户端实例
    return client

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
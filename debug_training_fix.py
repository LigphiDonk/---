#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版训练调试脚本，特别处理args全局变量问题
"""

import os
import sys
import logging
import torch
import torch.optim as optim
import time
import numpy as np
import random
from model import SimpleCNNMNIST
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

# 设置日志级别，减少不必要的调试信息
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.colorbar").setLevel(logging.ERROR)

# 设置matplotlib为非交互式模式，避免线程问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
os.environ['MPLBACKEND'] = 'Agg'
os.environ["MATPLOTLIB_USE_FIGURE_REQUIRES_MAIN_THREAD"] = "False"
import matplotlib.pyplot as plt

# 配置详细的日志输出
logging.basicConfig(
    level=logging.INFO,  # 使用INFO级别减少详细日志输出
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到控制台
        logging.FileHandler('debug_training_fix.log')  # 同时保存到文件
    ]
)

logger = logging.getLogger("DebugTrainingFix")

# 定义全局args对象，确保在train_net中可以访问
class Args:
    def __init__(self):
        self.reg = 1e-5
        self.rho = 0.9
        self.dataset = 'mnist'
        self.model = 'simple-cnn'
        self.device = 'cpu'
        self.batch_size = 64
        self.epochs = 2
        self.lr = 0.01
        self.optimizer = 'sgd'
        self.datadir = '/Users/baifangning/Desktop/第二版/data'

# 设置全局args对象
args = Args()

def set_random_seed(seed=42):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_mnist_data(data_dir='/Users/baifangning/Desktop/第二版/data', batch_size=64):
    """加载MNIST数据集"""
    logger.info(f"从 {data_dir} 加载MNIST数据集，批量大小: {batch_size}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        logger.error(f"数据目录不存在: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"已创建数据目录: {data_dir}")
    
    try:
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        # 创建数据加载器
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        logger.info(f"成功加载MNIST数据集，训练集大小: {len(train_dataset)}，测试集大小: {len(test_dataset)}")
        return train_loader, test_loader
        
    except Exception as e:
        logger.error(f"加载MNIST数据集时出错: {str(e)}")
        raise e

def create_model():
    """创建一个简单的CNN模型用于MNIST"""
    logger.info("创建SimpleCNNMNIST模型")
    model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
    logger.info(f"模型创建成功，参数总数: {sum(p.numel() for p in model.parameters())}")
    return model

def compute_accuracy(net, dataloader, device="cpu", get_confusion_matrix=False, moon_model=False):
    """计算模型在数据集上的准确率"""
    net.to(device)
    net.eval()
    
    correct = 0
    total = 0
    
    if get_confusion_matrix:
        nb_classes = 10
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
    
    with torch.no_grad():
        if type(dataloader) == type([1]):
            # 如果是数据加载器列表
            for dl in dataloader:
                for batch_idx, (x, target) in enumerate(dl):
                    x, target = x.to(device), target.to(device)
                    out = net(x) if not moon_model else net(x)[-1]
                    _, predicted = torch.max(out.data, 1)
                    
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    
                    if get_confusion_matrix:
                        for t, p in zip(target.view(-1), predicted.view(-1)):
                            confusion_matrix[t.long(), p.long()] += 1
        else:
            # 单个数据加载器
            for batch_idx, (x, target) in enumerate(dataloader):
                x, target = x.to(device), target.to(device)
                out = net(x) if not moon_model else net(x)[-1]
                _, predicted = torch.max(out.data, 1)
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                if get_confusion_matrix:
                    for t, p in zip(target.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
    
    if get_confusion_matrix:
        return 100.0 * correct / float(total), confusion_matrix
    
    return 100.0 * correct / float(total)

def visualize_confusion_matrix(conf_matrix, classes=None, save_path='confusion_matrix.png', 
                               normalize=False, title='Confusion Matrix'):
    """
    绘制并保存混淆矩阵可视化
    
    Args:
        conf_matrix: 混淆矩阵张量
        classes: 类别名称列表，可选
        save_path: 保存路径
        normalize: 是否归一化混淆矩阵
        title: 图表标题
    """
    if classes is None:
        classes = [str(i) for i in range(conf_matrix.shape[0])]
    
    # 转换为numpy数组以便处理
    if isinstance(conf_matrix, torch.Tensor):
        conf_matrix = conf_matrix.cpu().numpy()
    
    # 归一化处理
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # 计算准确率
    accuracy = np.trace(conf_matrix) / float(np.sum(conf_matrix))
    misclass = 1 - accuracy
    
    # 创建图形，确保使用agg后端
    plt.figure(figsize=(10, 8))
    
    # 绘制混淆矩阵
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    plt.title(f'{title}\nAccuracy={accuracy:0.4f}; Error={misclass:0.4f}')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # 添加文本
    # 根据是否归一化选择正确的格式代码
    if normalize:
        fmt = '.2f'  # 浮点数格式
    else:
        fmt = 'd'    # 整数格式
    
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            # 根据数据类型使用不同的格式化方法
            if normalize or isinstance(conf_matrix[i, j], float):
                text = format(conf_matrix[i, j], '.2f')
            else:
                text = format(int(conf_matrix[i, j]), 'd')
            
            plt.text(j, i, text,
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # 保存图像
    try:
        plt.savefig(save_path)
        logger.info(f"混淆矩阵已保存到 {save_path}")
    except Exception as e:
        logger.error(f"保存混淆矩阵时出错: {str(e)}")
    
    # 确保关闭图形，避免内存泄漏
    plt.close()
    
    return save_path

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    """重新实现训练函数，确保所有必要的日志输出都可见"""
    logger.info(f'===== 开始训练神经网络 {str(net_id)} =====')
    logger.info(f'模型结构: {net.__class__.__name__}')
    logger.info(f'模型参数总数: {sum(p.numel() for p in net.parameters())}')
    logger.info(f'可训练参数总数: {sum(p.numel() for p in net.parameters() if p.requires_grad)}')
    
    # 将模型移到指定设备
    net.to(device)
    logger.info(f'模型已移至设备: {device}')

    # 计算并输出预训练的训练集和测试集准确率
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> 预训练训练集准确率: {:.4f}'.format(train_acc))
    logger.info('>> 预训练测试集准确率: {:.4f}'.format(test_acc))

    # 尝试可视化预训练的混淆矩阵
    try:
        confusion_matrix_path = f'client_{net_id}_pretrain_confusion_matrix.png'
        visualize_confusion_matrix(conf_matrix, save_path=confusion_matrix_path)
    except Exception as e:
        logger.error(f"可视化预训练混淆矩阵时出错: {str(e)}")

    # 根据输入的优化器类型选择合适的优化器
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
        logger.info(f'使用Adam优化器, 学习率={lr}, 权重衰减={args.reg}')
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
        logger.info(f'使用AMSGrad优化器, 学习率={lr}, 权重衰减={args.reg}')
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                               weight_decay=args.reg)
        logger.info(f'使用SGD优化器, 学习率={lr}, 动量={args.rho}, 权重衰减={args.reg}')

    # 使用交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss().to(device)
    logger.info('使用交叉熵损失函数')

    # 如果传入的训练集是一个列表，进行处理
    if type(train_dataloader) == type([1]):
        logger.info(f'训练数据加载器是列表类型，包含 {len(train_dataloader)} 个数据加载器')
    else:
        train_dataloader = [train_dataloader]
        logger.info('训练数据加载器已转换为列表类型')
    
    # 记录每个训练数据加载器的样本数
    for i, loader in enumerate(train_dataloader):
        logger.info(f'训练数据加载器 {i}: {len(loader.dataset)} 个样本, {len(loader)} 个批次')

    # 训练循环
    for epoch in range(epochs):
        logger.info(f'===== 开始训练 Epoch {epoch+1}/{epochs} =====')
        net.train()  # 设置为训练模式
        epoch_loss_collector = []
        batch_cnt = 0
        total_batches = sum(len(loader) for loader in train_dataloader)
        start_time = time.time()
        
        # 遍历训练集
        for loader_idx, tmp in enumerate(train_dataloader):
            logger.info(f'处理训练数据加载器 {loader_idx+1}/{len(train_dataloader)}')
            for batch_idx, (x, target) in enumerate(tmp):
                batch_cnt += 1
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                # 前向传播，计算损失
                out = net(x)
                loss = criterion(out, target)

                # 反向传播，优化模型参数
                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())
                
                # 每10个批次或最后一个批次输出一次进度
                if batch_idx % 10 == 0 or batch_idx == len(tmp) - 1:
                    _, predicted = out.max(1)
                    correct = predicted.eq(target).sum().item()
                    batch_acc = 100. * correct / target.size(0)
                    
                    progress = batch_cnt / total_batches * 100
                    logger.info(f'Epoch: {epoch+1}/{epochs} [{batch_cnt}/{total_batches} ({progress:.1f}%)] '
                               f'Loss: {loss.item():.4f} Batch Acc: {batch_acc:.2f}%')

        # 计算并输出每个epoch的平均损失和耗时
        epoch_time = time.time() - start_time
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info(f'Epoch: {epoch+1}/{epochs} 完成, 平均损失: {epoch_loss:.4f}, 耗时: {epoch_time:.2f}秒')
        
        # 每个epoch计算准确率
        logger.info(f'计算Epoch {epoch+1}的训练和测试准确率')
        current_train_acc = compute_accuracy(net, train_dataloader, device=device)
        current_test_acc = compute_accuracy(net, test_dataloader, get_confusion_matrix=False, device=device)
        logger.info(f'>> Epoch {epoch+1} - 训练准确率: {current_train_acc:.4f}, 测试准确率: {current_test_acc:.4f}')

    # 训练完成后计算并输出最终的训练集和测试集准确率
    logger.info('计算最终训练和测试准确率')
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> 最终训练集准确率: {:.4f}'.format(train_acc))
    logger.info('>> 最终测试集准确率: {:.4f}'.format(test_acc))

    # 可视化最终混淆矩阵
    try:
        confusion_matrix_path = f'client_{net_id}_final_confusion_matrix.png'
        visualize_confusion_matrix(conf_matrix, save_path=confusion_matrix_path)
    except Exception as e:
        logger.error(f"可视化最终混淆矩阵时出错: {str(e)}")

    # 将模型转移到CPU上
    net.to('cpu')
    logger.info('===== 训练完成 =====')

    # 返回训练集和测试集的准确率
    return train_acc, test_acc

def main():
    """主函数"""
    # 设置随机种子
    set_random_seed()
    logger.info("=== 开始训练调试 ===")
    
    # 获取设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 更新args对象的device
    args.device = device
    
    # 加载数据
    train_loader, test_loader = get_mnist_data(data_dir=args.datadir, batch_size=args.batch_size)
    
    # 创建模型
    model = create_model()
    
    # 打印模型结构
    logger.info(f"模型结构:\n{model}")
    
    # 计算初始准确率
    logger.info("计算初始准确率...")
    init_train_acc = compute_accuracy(model, train_loader, device=device)
    init_test_acc, _ = compute_accuracy(model, test_loader, get_confusion_matrix=True, device=device)
    logger.info(f"初始训练准确率: {init_train_acc:.4f}")
    logger.info(f"初始测试准确率: {init_test_acc:.4f}")
    
    # 使用train_net函数进行训练
    logger.info("开始使用train_net函数训练模型...")
    
    # 开始训练
    start_time = time.time()
    train_acc, test_acc = train_net(
        net_id=0,
        net=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        args_optimizer=args.optimizer,
        device=device
    )
    training_time = time.time() - start_time
    
    logger.info(f"训练完成，耗时: {training_time:.2f}秒")
    logger.info(f"最终训练准确率: {train_acc:.4f}")
    logger.info(f"最终测试准确率: {test_acc:.4f}")
    logger.info(f"准确率提升 - 训练: {train_acc - init_train_acc:.4f}, 测试: {test_acc - init_test_acc:.4f}")
    
    # 保存模型以便验证
    torch.save(model.state_dict(), "debug_model_fixed.pth")
    logger.info("模型已保存到 debug_model_fixed.pth")
    
    logger.info("=== 训练调试完成 ===")
    
if __name__ == "__main__":
    main() 
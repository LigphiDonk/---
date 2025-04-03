#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试已训练模型的脚本
"""

import os
import sys
import logging
import torch
import time
import numpy as np
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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_test.log')
    ]
)

logger = logging.getLogger("DebugTest")

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
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        # 创建数据加载器
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        logger.info(f"成功加载MNIST测试集，大小: {len(test_dataset)}")
        return test_loader
        
    except Exception as e:
        logger.error(f"加载MNIST数据集时出错: {str(e)}")
        raise e

def create_model():
    """创建一个简单的CNN模型用于MNIST"""
    logger.info("创建SimpleCNNMNIST模型")
    model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
    logger.info(f"模型创建成功，参数总数: {sum(p.numel() for p in model.parameters())}")
    return model

def compute_accuracy(net, dataloader, device="cpu", get_confusion_matrix=False):
    """计算模型在数据集上的准确率"""
    net.to(device)
    net.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    if get_confusion_matrix:
        nb_classes = 10
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
    
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            out = net(x)
            _, predicted = torch.max(out.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            if get_confusion_matrix:
                for t, p in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
    
    accuracy = 100.0 * correct / float(total)
    
    if get_confusion_matrix:
        return accuracy, confusion_matrix, all_preds, all_targets
    
    return accuracy, all_preds, all_targets

def plot_confusion_matrix(cm, classes, normalize=False, title='混淆矩阵', cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    """
    try:
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            logger.info("显示归一化的混淆矩阵")
        else:
            logger.info('显示未归一化的混淆矩阵')
    
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig('confusion_matrix.png')
        logger.info("混淆矩阵已保存到 confusion_matrix.png")
    except Exception as e:
        logger.error(f"绘制混淆矩阵时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 确保关闭图形，避免内存泄漏
        plt.close()

def visualize_predictions(dataloader, model, device="cpu", num_samples=10):
    """可视化模型的预测结果"""
    model.eval()
    images, labels = next(iter(dataloader))
    
    # 只选择指定数量的样本
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # 获取预测结果
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    images = images.cpu()
    preds = preds.cpu().numpy()
    labels = labels.numpy()
    
    try:
        # 创建图像展示
        fig = plt.figure(figsize=(12, 4))
        for idx in range(num_samples):
            ax = fig.add_subplot(1, num_samples, idx+1, xticks=[], yticks=[])
            img = images[idx].numpy().transpose((1, 2, 0))
            # 反归一化
            mean = np.array([0.1307])
            std = np.array([0.3081])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            ax.imshow(img.squeeze(), cmap='gray')
            ax.set_title(f"{preds[idx]} ({labels[idx]})",
                         color=("green" if preds[idx]==labels[idx] else "red"))
        
        plt.tight_layout()
        plt.savefig('prediction_visualization.png')
        logger.info("预测可视化已保存到 prediction_visualization.png")
    except Exception as e:
        logger.error(f"可视化预测时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 确保关闭图形，避免内存泄漏
        plt.close()

def main():
    """主函数"""
    logger.info("=== 开始测试模型 ===")
    
    # 获取设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 加载测试数据
    test_loader = get_mnist_data(data_dir='/Users/baifangning/Desktop/第二版/data', batch_size=64)
    
    # 创建模型并加载已训练的参数
    model = create_model()
    
    # 尝试加载已训练的模型参数
    try:
        model.load_state_dict(torch.load("debug_model_fixed.pth"))
        logger.info("成功加载已训练的模型参数")
    except Exception as e:
        logger.error(f"加载模型参数时出错: {str(e)}")
        logger.info("将使用未训练的模型进行测试")
    
    # 计算准确率和混淆矩阵
    logger.info("计算模型在测试集上的性能...")
    start_time = time.time()
    accuracy, conf_matrix, all_preds, all_targets = compute_accuracy(
        model, test_loader, device=device, get_confusion_matrix=True
    )
    test_time = time.time() - start_time
    
    logger.info(f"测试完成，耗时: {test_time:.2f}秒")
    logger.info(f"测试集准确率: {accuracy:.4f}%")
    
    # 绘制混淆矩阵
    class_names = [str(i) for i in range(10)]  # MNIST有10个类别，从0到9
    plot_confusion_matrix(conf_matrix.numpy(), classes=class_names, normalize=True)
    
    # 可视化一些预测样例
    visualize_predictions(test_loader, model, device=device, num_samples=10)
    
    logger.info("=== 测试完成 ===")
    
if __name__ == "__main__":
    main() 
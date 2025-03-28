import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random
from tqdm import tqdm
import datetime
# from torch.utils.tensorboard import SummaryWriter
from pprint import pprint
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
# torch.backends.cudnn.enabled = False
import socket
from Interact.server import Server

# 服务器配置
SERVER_HOST = '127.0.0.1'


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
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    parser.add_argument('--use_projection_head', type=bool, default=False,
                        help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1,
                        help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=0.001, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level',
                        help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    # add by czh, 2024.03.30
    parser.add_argument('--client_num', type=int, help='The number of clients')

    args = parser.parse_args()
    return args


def init_nets(net_configs, dropout_p, n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}

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
    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon(args.model + add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
    else:
        if args.alg == 'moon':
            add = ""
            if "mnist" in args.dataset and args.model == "simple-cnn":
                add = "-mnist"
            for net_i in range(n_parties):
                net = ModelFedCon_noheader(args.model + add, args.out_dim, n_classes, net_configs)
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
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
                    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                elif args.model == "vgg":
                    net = vgg11()
                elif args.model == "simple-cnn":
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                        net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset == 'celeba':
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
                elif args.model == "vgg-9":
                    if args.dataset in ("mnist", 'femnist'):
                        net = ModerateCNNMNIST()
                    elif args.dataset in ("cifar10", "cinic10", "svhn"):
                        # print("in moderate cnn")
                        net = ModerateCNN()
                    elif args.dataset == 'celeba':
                        net = ModerateCNN(output_dim=2)
                elif args.model == "resnet":
                    net = ResNet50_cifar10()
                elif args.model == "vgg16":
                    net = vgg16()
                else:
                    print("not supported yet")
                    exit(1)
                nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    # 输出正在训练的神经网络的信息
    logger.info('训练神经网络 %s' % str(net_id))

    # 计算并输出预训练的训练集和测试集准确率
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> 预训练训练集准确率: {}'.format(train_acc))
    logger.info('>> 预训练测试集准确率: {}'.format(test_acc))

    # 根据输入的优化器类型选择合适的优化器
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)

    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # 如果传入的训练集是一个列表，进行处理
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # 训练循环
    for epoch in range(epochs):
        epoch_loss_collector = []
        # 遍历训练集
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
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

                cnt += 1
                epoch_loss_collector.append(loss.item())

        # 计算并输出每个epoch的平均损失
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    # 训练完成后计算并输出最终的训练集和测试集准确率
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> 训练集准确率: %f' % train_acc)
    logger.info('>> 测试集准确率: %f' % test_acc)

    # 将模型转移到CPU上
    net.to('cpu')
    logger.info(' ** 训练完成 **')

    # 返回训练集和测试集的准确率
    return train_acc, test_acc


def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu,
                      device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        # if epoch % 10 == 0:
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr,
                       args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # writer = SummaryWriter()

    c_local.to(device)
    c_global.to(device)
    global_model.to(device)

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (
                cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, c_delta_para


def train_net_fednova(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer,
                      device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                          weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # writer = SummaryWriter()

    tau = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                tau = tau + 1

                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    global_model.to(device)
    a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
    global_model.to(device)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        # norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
        norm_grad[key] = torch.true_divide(global_model_para[key] - net_para[key], a_i)
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, a_i, norm_grad


def train_net_moon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr,
                   args_optimizer, mu, temperature, args,
                   round, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, moon_model=True, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=True,
                                             device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # conloss = ContrastiveLoss(temperature)

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    # global_net.to(device)

    if args.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to(device)
    global_w = global_net.state_dict()
    # oppsi_nets = copy.deepcopy(previous_nets)
    # for net_id, oppsi_net in enumerate(oppsi_nets):
    #     oppsi_w = oppsi_net.state_dict()
    #     prev_w = previous_nets[net_id].state_dict()
    #     for key in oppsi_w:
    #         oppsi_w[key] = 2*global_w[key] - prev_w[key]
    #     oppsi_nets.load_state_dict(oppsi_w)
    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1).to(device)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            if target.shape[0] == 1:
                continue

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)
            if args.loss == 'l2norm':
                loss2 = mu * torch.mean(torch.norm(pro2 - pro1, dim=1))

            elif args.loss == 'only_contrastive' or args.loss == 'contrastive':
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1, 1)

                for previous_net in previous_nets:
                    previous_net.to(device)
                    _, pro3, _ = previous_net(x)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                    # previous_net.to('cpu')

                logits /= temperature
                labels = torch.zeros(x.size(0)).to(device).long()

                # loss = criterion(out, target) + mu * ContraLoss(pro1, pro2, pro3)

                loss2 = mu * criterion(logits, labels)

            if args.loss == 'only_contrastive':
                loss = loss2
            else:
                loss1 = criterion(out, target)
                loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))

    if args.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to('cpu')
    train_acc = compute_accuracy(net, train_dataloader, moon_model=True, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=True,
                                             device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)


def local_train_net(nets, selected, args, net_dataidx_map, test_dl=None, device="cpu"):
    avg_acc = 0.0

    # 遍历所有神经网络
    for net_id, net in nets.items():
        # 如果当前神经网络不在选定的本地模型中，则跳过
        if net_id not in selected:
            continue
        # 获取当前神经网络对应的数据索引
        dataidxs = net_dataidx_map[net_id]

        # 输出当前神经网络的训练信息
        logger.info("训练网络 %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # 将模型移到指定的设备上
        net.to(device)

        # 根据神经网络的特定情况设置噪声水平
        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        # 根据噪声类型获取本地训练和测试数据集
        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        # 训练神经网络并获取训练和测试准确率
        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer,
                                      device=device)
        print(net.conv1)
        # 创建一个简单的Conv2d模型
        conv2d_model = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 打印Conv2d模型的状态字典
        pprint(net.conv1.__dict__)
        # model = SimpleCNNMNIST()
        torch.save(net.state_dict(), 'simple_cnn_mnist_model.pth' + str(net_id))
        target_model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
        target_model.load_state_dict(torch.load('simple_cnn_mnist_model.pth'))
        # 输出当前神经网络的最终测试准确率
        logger.info("神经网络 %d 最终测试准确率 %f" % (net_id, testacc))
        avg_acc += testacc
        # 保存训练好的模型
        # save_model(net, net_id, args)
        # 或者加载预训练模型
        # load_model(net, net_id, device=device)

    # 计算平均测试准确率
    avg_acc /= len(selected)
    # 如果联邦学习算法为局部训练（local_training），则输出平均测试准确率信息
    if args.alg == 'local_training':
        logger.info("平均测试准确率 %f" % avg_acc)

    # 将训练后的神经网络列表转换为列表形式并返回
    nets_list = list(nets.values())
    return nets_list


def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl=None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                              args.optimizer, args.mu, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl=None,
                             device="cpu"):
    avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        c_nets[net_id].to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global,
                                                             train_dl_local, test_dl, n_epoch, args.lr, args.optimizer,
                                                             device=device)

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]

        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    for key in total_delta:
        total_delta[key] /= args.n_parties
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            # print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl=None, device="cpu"):
    avg_acc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc, a_i, d_i = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl, n_epoch,
                                                        args.lr, args.optimizer, device=device)

        a_list.append(a_i)
        d_list.append(d_i)
        n_i = len(train_dl_local.dataset)
        n_list.append(n_i)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, a_list, d_list, n_list


def local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl=None, global_model=None, prev_model_pool=None,
                         round=None, device="cpu"):
    avg_acc = 0.0
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        prev_models = []
        for i in range(len(prev_model_pool)):
            prev_models.append(prev_model_pool[i][net_id])
        trainacc, testacc = train_net_moon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch,
                                           args.lr,
                                           args.optimizer, args.mu, args.temperature, args, round, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
    global_model.to('cpu')
    nets_list = list(nets.values())
    return nets_list


def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map


if __name__ == '__main__':
    # 如果当前脚本被直接执行而不是被导入到其他脚本中，则执行以下操作
    args = get_args()  # 获取命令行参数

    # 服务器初始化
    SERVER_PORT = 12345
    client_num = args.n_parties  # 默认2个客户端
    servers = [Server() for _ in range(client_num)]
    for i in range(client_num):
        servers[i].config(SERVER_HOST, SERVER_PORT + i)

    mkdirs(args.logdir)  # 创建日志目录
    mkdirs(args.modeldir)  # 创建模型保存目录
    if args.log_file_name is None:  # 如果未指定日志文件名
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    else:
        argument_path = args.log_file_name + '.json'  # 否则使用指定的日志文件名
    with open(os.path.join(args.logdir, argument_path), 'w') as f:  # 打开文件，准备写入参数
        json.dump(str(args), f)  # 将参数以JSON格式写入文件
    device = torch.device(args.device)  # 获取设备信息
    for handler in logging.root.handlers[:]:  # 清除所有已有的日志处理器
        logging.root.removeHandler(handler)
    if args.log_file_name is None:  # 如果未指定日志文件名
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    log_path = args.log_file_name + '.log'  # 设置日志文件路径
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),  # 设置日志文件保存路径
        format='%(asctime)s %(levelname)-8s %(message)s',  # 设置日志格式
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')  # 设置日志级别为DEBUG
    logger = logging.getLogger()  # 获取日志记录器
    logger.setLevel(logging.DEBUG)  # 设置日志记录器的级别为DEBUG
    logger.info(device)  # 记录设备信息
    seed = args.init_seed  # 设置随机数种子
    logger.info("#" * 100)  # 记录日志信息
    np.random.seed(seed)  # 设置numpy随机数种子
    torch.manual_seed(seed)  # 设置PyTorch随机数种子
    random.seed(seed)  # 设置Python内置random模块的随机数种子
    logger.info("Partitioning data")  # 记录日志信息，表示开始数据分区
    # 分割数据集并获取数据加载器
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)
    n_classes = len(np.unique(y_train))  # 获取训练集标签类别数量
    # 获取全局训练和测试数据加载器
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                      args.datadir,
                                                                                      args.batch_size,
                                                                                      32)
    print("len train_dl_global:", len(train_ds_global))  # 打印全局训练集的长度
    data_size = len(test_ds_global)  # 获取测试集数据长度
    train_all_in_list = []  # 创建一个空列表，用于存储所有局部训练集
    test_all_in_list = []  # 创建一个空列表，用于存储所有局部测试集
    if args.noise > 0:  # 如果存在噪音
        for party_id in range(args.n_parties):  # 遍历各方
            dataidxs = net_dataidx_map[party_id]  # 获取当前方的数据索引
            noise_level = args.noise  # 设置噪音水平
            if party_id == args.n_parties - 1:  # 如果是最后一个方
                noise_level = 0  # 将噪音水平设为0
            if args.noise_type == 'space':  # 如果是空间噪音类型
                # 获取局部训练和测试数据加载器
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset,
                                                                                              args.datadir,
                                                                                              args.batch_size, 32,
                                                                                              dataidxs, noise_level,
                                                                                              party_id,
                                                                                              args.n_parties - 1)
            else:
                # 计算当前方的噪音水平
                noise_level = args.noise / (args.n_parties - 1) * party_id
                # 获取局部训练和测试数据加载器
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset,
                                                                                              args.datadir,
                                                                                              args.batch_size, 32,
                                                                                              dataidxs, noise_level)
            # 将局部训练和测试数据集添加到列表中
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        # 将所有局部训练数据集合并为一个全局训练数据集
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        # 将所有局部测试数据集合并为一个全局测试数据集
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)

    # 判断选择的联邦学习算法是否为FedAvg
    if args.alg == 'fedavg':
        # 输出初始化神经网络的信息
        logger.info("初始化神经网络")
        # 初始化本地模型、本地模型元数据和层类型
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        # 初始化全局模型、全局模型元数据和全局层类型
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        # 获取全局模型
        global_model = global_models[0]

        # 获取全局模型的参数
        global_para = global_model.state_dict()
        # 如果所有本地模型要使用相同的初始化参数
        # # 修改-分发全局模型参数
        if args.is_same_initial:
            # 将所有本地模型加载相同的全局参数
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        # 循环执行通信轮次
        for round in tqdm(range(args.comm_round)):
            # 输出当前通信轮次的信息
            logger.info("在通信轮次：" + str(round))

            # 随机选择参与通信的本地模型
            arr = np.arange(args.n_parties)
            # np.random.shuffle(arr)
            # selected = arr[:int(args.n_parties * args.sample)]
            selected = arr

            # 获取全局模型的参数
            global_para = global_model.state_dict()
            # 如果是第一轮通信且所有本地模型使用相同的初始化参数
            if round == 0:
                if args.is_same_initial:
                    # 将选定的本地模型加载相同的全局参数
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                # 将选定的本地模型加载相同的全局参数
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            # 在选定的本地模型上进行本地训练
            # #修改 分发nets
            for i in range(client_num):
                servers[i].send_obj(nets)
                servers[i].send_obj(args)
                servers[i].send_obj(net_dataidx_map)
                servers[i].send_obj(test_dl_global)
                servers[i].send_obj(device)
            # local_train_net(nets, selected, args, net_dataidx_map, test_dl=test_dl_global, device=device)
            # 接受模型（已加掩码）
            new_nets = []
            for i in range(client_num):
                new_nets.append(servers[i].recv_obj())
            nets = {i: new_nets[i] for i in range(client_num)}
            # 更新全局模型

            fed_avg_freqs = [1 / client_num for _ in range(client_num)]
            print(fed_avg_freqs)
            # 遍历选定的本地模型，根据其在总体数据中的比例更新全局模型的权重：
            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            # 将更新后的全局模型参数加载到全局模型中：
            global_model.load_state_dict(global_para)

            # 输出全局模型的训练集和测试集信息
            logger.info('全局模型训练集大小: %d' % len(train_dl_global))
            logger.info('全局模型测试集大小: %d' % len(test_dl_global))


            # 将全局模型转移到指定设备上
            global_model.to(device)
            # 计算全局模型在训练集上的准确率
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            # 计算全局模型在测试集上的准确率和混淆矩阵
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            # 输出全局模型的训练准确率和测试准确率
            logger.info('>> 全局模型训练准确率: %f' % train_acc)
            logger.info('>> 全局模型测试准确率: %f' % test_acc)
            print('全局模型训练准确率: %f' % train_acc)
            print('全局模型测试准确率: %f' % test_acc)



    elif args.alg == 'fedprox':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl=test_dl_global,
                                    device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'scaffold':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        c_globals, _, _ = init_nets(args.net_config, 0, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map,
                                     test_dl=test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fednova':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)]
        d_total_round = copy.deepcopy(global_model.state_dict())
        for i in range(args.n_parties):
            for key in d_list[i]:
                d_list[i][key] = 0
        for key in d_total_round:
            d_total_round[key] = 0

        data_sum = 0
        for i in range(args.n_parties):
            data_sum += len(traindata_cls_counts[i])
        portion = []
        for i in range(args.n_parties):
            portion.append(len(traindata_cls_counts[i]) / data_sum)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            _, a_list, d_list, n_list = local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map,
                                                                test_dl=test_dl_global, device=device)
            total_n = sum(n_list)
            # print("total_n:", total_n)
            d_total_round = copy.deepcopy(global_model.state_dict())
            for key in d_total_round:
                d_total_round[key] = 0.0

            for i in range(len(selected)):
                d_para = d_list[i]
                for key in d_para:
                    # if d_total_round[key].type == 'torch.LongTensor':
                    #    d_total_round[key] += (d_para[key] * n_list[i] / total_n).type(torch.LongTensor)
                    # else:
                    d_total_round[key] += d_para[key] * n_list[i] / total_n

            # for i in range(len(selected)):
            #     d_total_round = d_total_round + d_list[i] * n_list[i] / total_n

            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            coeff = 0.0
            for i in range(len(selected)):
                coeff = coeff + a_list[i] * n_list[i] / total_n

            updated_model = global_model.state_dict()
            for key in updated_model:
                # print(updated_model[key])
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    # print(updated_model[key].type())
                    # print((coeff*d_total_round[key].type()))
                    updated_model[key] -= coeff * d_total_round[key]
            global_model.load_state_dict(updated_model)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'moon':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        old_nets_pool = []
        old_nets = copy.deepcopy(nets)
        for _, net in old_nets.items():
            net.eval()
            for param in net.parameters():
                param.requires_grad = False

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl=test_dl_global,
                                 global_model=global_model,
                                 prev_model_pool=old_nets_pool, round=round, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, moon_model=True, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     moon_model=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            old_nets = copy.deepcopy(nets)
            for _, net in old_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            if len(old_nets_pool) < 1:
                old_nets_pool.append(old_nets)
            else:
                old_nets_pool[0] = old_nets

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        arr = np.arange(args.n_parties)
        local_train_net(nets, arr, args, net_dataidx_map, test_dl=test_dl_global, device=device)

    elif args.alg == 'all_in':
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, 1, args)
        n_epoch = args.epochs
        nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl_global, n_epoch, args.lr, args.optimizer,
                                      device=device)

        logger.info("All in test acc: %f" % testacc)

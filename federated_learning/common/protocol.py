#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习通信协议定义
"""

import json
import torch
import base64
import numpy as np
import pickle

class MessageType:
    """定义联邦学习系统中的消息类型"""
    INIT = "init"            # 初始化消息
    MODEL_UPDATE = "update"  # 模型更新
    GRADIENT = "gradient"    # 梯度传输
    TRAIN = "train"          # 训练请求
    STOP = "stop"            # 停止训练
    FINISH = "finish"        # 完成联邦学习

class Message:
    """消息基类"""
    def __init__(self, msg_type, data=None):
        self.type = msg_type
        self.data = data if data is not None else {}
    
    def to_json(self):
        """将消息转换为JSON格式"""
        return json.dumps({
            "type": self.type,
            "data": self.data
        })
    
    @classmethod
    def from_json(cls, json_str):
        """从JSON字符串创建消息"""
        data = json.loads(json_str)
        return cls(data["type"], data["data"])

def serialize_model(model_state_dict):
    """序列化模型状态字典"""
    buffer = pickle.dumps(model_state_dict)
    return base64.b64encode(buffer).decode('ascii')

def deserialize_model(serialized_model):
    """反序列化模型状态字典"""
    buffer = base64.b64decode(serialized_model.encode('ascii'))
    return pickle.loads(buffer)

def serialize_gradient(gradient_dict):
    """序列化梯度字典"""
    serialized_gradients = {}
    for name, grad in gradient_dict.items():
        if grad is not None:
            # 将梯度张量转换为NumPy数组，然后序列化
            grad_np = grad.detach().cpu().numpy()
            serialized_gradients[name] = {
                "shape": grad_np.shape,
                "dtype": str(grad_np.dtype),
                "data": base64.b64encode(grad_np.tobytes()).decode('ascii')
            }
    return serialized_gradients

def deserialize_gradient(serialized_gradients):
    """反序列化梯度字典"""
    gradients = {}
    for name, grad_info in serialized_gradients.items():
        shape = grad_info["shape"]
        dtype = np.dtype(grad_info["dtype"])
        data = base64.b64decode(grad_info["data"].encode('ascii'))
        grad_np = np.frombuffer(data, dtype=dtype).reshape(shape)
        gradients[name] = torch.tensor(grad_np)
    return gradients 
import os
import sys
import socket


# 获取Interact文件夹路径，添加到sys.path中
def add_path():
    if os.path.exists('./Interact/'):
        sys.path.append(os.path.join(os.getcwd(), 'Interact'))
    else:
        print('Hint: There may be error when importing modules in server.py')


add_path()
from interact import Interact
from hashlib import sha1


class Client(Interact):
    def __init__(self):
        super().__init__()
        self.role = 'client'
        # 存放文件的名称
        self.filename = 'temp_' + self.role + '.txt'

    # 客户端发送消息
    def send(self, message: str):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.server_host, self.server_port))
        self.socket.send(message.encode())
        self.socket.close()

    # 客户端接收消息
    def recv(self) -> str:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.server_host, self.server_port))
        date = self.socket.recv(self.buffer_size)
        self.socket.close()
        return date.decode()

    # 客户端发送文件
    def send_file(self, filepath: str = None) -> str:
        if filepath is None:
            filepath = self.folder + self.filename
        # 计算文件的SHA1, 用于校验
        Hash = sha1(open(filepath, 'rb').read()).hexdigest()
        # 建立连接
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.server_host, self.server_port))
        # 发送文件
        with open(filepath, 'rb') as file:
            while True:
                data = file.read(self.buffer_size)
                if not data:
                    break
                self.socket.send(data)
        self.socket.close()
        return Hash

    # 客户端接收文件
    def recv_file(self) -> str:
        try:
            os.mkdir(self.folder)
        except FileExistsError:
            pass
        filepath = self.folder + self.filename
        # 建立连接
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.server_host, self.server_port))
        with open(filepath, 'wb') as file:
            while True:
                data = self.socket.recv(self.buffer_size)
                if not data:
                    break
                file.write(data)
        self.socket.close()
        return sha1(open(filepath, 'rb').read()).hexdigest()

import os
import sys


# 获取Interact文件夹路径，添加到sys.path中
def add_path():
    if os.path.exists('./Interact/'):
        sys.path.append(os.path.join(os.getcwd(), 'Interact'))
    else:
        print('Hint: There may be error when importing modules in server.py')


add_path()
from interact import Interact
from hashlib import sha1


class Server(Interact):
    def __init__(self):
        super().__init__()
        self.role = 'server'
        # 存放文件的名称
        self.filename = 'temp_' + self.role + '.txt'

    # 服务器发送消息
    def send(self, message: str):
        self.client_socket, _ = self.socket.accept()
        self.client_socket.send(message.encode())
        self.client_socket.close()

    # 服务器接收消息
    def recv(self) -> str:
        self.client_socket, _ = self.socket.accept()
        data = self.client_socket.recv(self.buffer_size)
        self.client_socket.close()
        return data.decode()

    # 服务器发送文件
    def send_file(self, filepath: str = None) -> str:
        if filepath is None:
            filepath = self.folder + self.filename
        # 计算文件的SHA1, 用于校验
        Hash = sha1(open(filepath, 'rb').read()).hexdigest()

        # 建立连接
        self.client_socket, _ = self.socket.accept()
        # 发送文件
        with open(filepath, 'rb') as file:
            while True:
                data = file.read(self.buffer_size)
                if not data:
                    break
                self.client_socket.send(data)
        self.client_socket.close()
        return Hash

    # 服务器接收文件
    def recv_file(self) -> str:
        # 创建存放文件的目录
        try:
            os.mkdir(self.folder)
        except FileExistsError:
            pass
        filepath = self.folder + self.filename

        # 建立连接
        self.client_socket, _ = self.socket.accept()
        # 接收文件
        with open(filepath, 'wb') as file:
            while True:
                data = self.client_socket.recv(self.buffer_size)
                if not data:
                    break
                file.write(data)
        self.client_socket.close()
        return sha1(open(filepath, 'rb').read()).hexdigest()

import os
import socket
import pickle


class Interact:
    def __init__(self):
        self.role = None
        self.server_host = None
        self.server_port = None
        self.socket = None
        # 接收消息的缓冲区大小
        self.buffer_size = 1024
        # 服务器当前连接的客户端
        self.client_socket = None
        # 存放文件的目录, 放在Interact文件夹下
        if os.path.exists('./Interact/'):
            self.folder = './Interact/storage/'
        else:
            self.folder = './storage/'
        # 存放文件的名称，不包含后缀
        self.filename = 'temp'

    # 配置服务器IP和端口
    def config(self, server_host: str, server_port: int, client_num: int = 5):
        self.server_host = server_host
        self.server_port = server_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.role == 'server':
            self.socket.bind((self.server_host, self.server_port))
            # 设置最大连接数，超过后排队
            self.socket.listen(client_num)

    # 发送文件
    # 在客户端和服务器中分别重写
    def send_file(self, filepath: str = None) -> str:
        pass

    # 接收文件
    # 在客户端和服务器中分别重写
    def recv_file(self) -> str:
        pass

    # 发送对象
    def send_obj(self, obj):
        self.obj2file(obj)
        self.send_file()

    # 接收对象
    def recv_obj(self) -> object:
        self.recv_file()
        return self.file2obj()

    # 对象 -> 文件
    def obj2file(self, obj, filepath: str = None):
        try:
            os.mkdir(self.folder)
        except FileExistsError:
            pass
        if filepath is None:
            filepath = self.folder + self.filename
        with open(filepath, 'wb') as file:
            file.write(pickle.dumps(obj))

    # 文件 -> 对象
    def file2obj(self, filepath: str = None) -> object:
        if filepath is None:
            filepath = self.folder + self.filename
        with open(filepath, 'rb') as file:
            return pickle.loads(file.read())

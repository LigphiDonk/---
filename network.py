import socket
import threading
import pickle
import time
import logging
import json
import random
from typing import Dict, List, Any, Tuple, Set

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("P2PNetwork")

class Message:
    """基本消息类，用于节点间通信"""
    
    def __init__(self, msg_type: str, sender_id: int, content: Any, timestamp: float = None):
        self.msg_type = msg_type  # 消息类型
        self.sender_id = sender_id  # 发送者ID
        self.content = content  # 消息内容
        self.timestamp = timestamp or time.time()  # 时间戳
    
    def __str__(self):
        return f"Message(type={self.msg_type}, sender={self.sender_id}, time={self.timestamp})"

class Node:
    """P2P网络中的节点基类"""
    
    def __init__(self, node_id: int, host: str = '127.0.0.1', port: int = None):
        """初始化节点
        
        Args:
            node_id: 节点唯一标识
            host: 节点主机地址
            port: 节点监听端口，如果为None则自动分配
        """
        self.node_id = node_id
        self.host = host
        self.port = port or (12000 + node_id)
        
        # 已知节点的地址映射 {node_id: (host, port)}
        self.peers: Dict[int, Tuple[str, int]] = {}
        
        # 连接的套接字 {node_id: socket}
        self.connections: Dict[int, socket.socket] = {}
        
        # 消息处理器映射 {msg_type: handler_function}
        self.message_handlers = {}
        
        # 节点运行状态
        self.running = False
        self.server_socket = None
        
        # 注册基本消息处理器
        self.register_handler("PING", self._handle_ping)
        self.register_handler("PONG", self._handle_pong)
        self.register_handler("DISCOVER", self._handle_discover)
        self.register_handler("PEERS", self._handle_peers)
        
        # 存储节点元数据，用于选举和信任评估
        self.metadata = {
            "online_time": 0,
            "start_time": 0,
            "connection_count": 0,
            "message_count": 0,
            "last_active": 0
        }

    def start(self):
        """启动节点，开始监听连接"""
        if self.running:
            return
            
        self.running = True
        self.metadata["start_time"] = time.time()
        
        # 创建服务器套接字
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)
        
        logger.info(f"节点 {self.node_id} 启动，监听 {self.host}:{self.port}")
        
        # 启动接收线程
        threading.Thread(target=self._listen_for_connections, daemon=True).start()
        
        # 启动心跳线程
        threading.Thread(target=self._heartbeat, daemon=True).start()

    def stop(self):
        """停止节点"""
        if not self.running:
            return
            
        self.running = False
        self.metadata["online_time"] += time.time() - self.metadata["start_time"]
        
        # 关闭所有连接
        for conn in self.connections.values():
            try:
                conn.close()
            except:
                pass
        
        # 关闭服务器套接字
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
                
        logger.info(f"节点 {self.node_id} 已停止")

    def connect_to_peer(self, peer_id: int, host: str, port: int) -> bool:
        """连接到对等节点
        
        Args:
            peer_id: 对等节点ID
            host: 对等节点主机地址
            port: 对等节点端口
            
        Returns:
            bool: 连接是否成功
        """
        if peer_id in self.connections:
            return True
            
        try:
            # 创建新连接
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            
            # 发送自我介绍
            intro_msg = Message("INTRO", self.node_id, {
                "node_id": self.node_id,
                "host": self.host,
                "port": self.port
            })
            self._send_to_socket(s, intro_msg)
            
            # 保存连接信息
            self.connections[peer_id] = s
            self.peers[peer_id] = (host, port)
            
            # 启动接收消息的线程
            threading.Thread(target=self._listen_to_peer, args=(peer_id, s), daemon=True).start()
            
            logger.info(f"已连接到节点 {peer_id} at {host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"连接节点 {peer_id} 失败: {str(e)}")
            return False

    def broadcast(self, msg_type: str, content: Any):
        """向所有连接的节点广播消息
        
        Args:
            msg_type: 消息类型
            content: 消息内容
        """
        message = Message(msg_type, self.node_id, content)
        for peer_id in list(self.connections.keys()):
            self.send_to_peer(peer_id, message)

    def send_to_peer(self, peer_id: int, message: Message) -> bool:
        """向特定节点发送消息
        
        Args:
            peer_id: 目标节点ID
            message: 消息对象
            
        Returns:
            bool: 发送是否成功
        """
        if peer_id not in self.connections:
            logger.warning(f"尝试发送消息到未连接的节点 {peer_id}")
            return False
            
        try:
            self._send_to_socket(self.connections[peer_id], message)
            self.metadata["message_count"] += 1
            return True
        except Exception as e:
            logger.error(f"发送消息到节点 {peer_id} 失败: {str(e)}")
            self._handle_disconnect(peer_id)
            return False

    def register_handler(self, msg_type: str, handler_func):
        """注册消息处理函数
        
        Args:
            msg_type: 消息类型
            handler_func: 处理函数，接收 (sender_id, content) 参数
        """
        self.message_handlers[msg_type] = handler_func

    def discover_peers(self, bootstrap_nodes: List[Tuple[int, str, int]] = None):
        """发现网络中的其他节点
        
        Args:
            bootstrap_nodes: 引导节点列表 [(node_id, host, port), ...]
        """
        # 首先连接到引导节点
        if bootstrap_nodes:
            for node_id, host, port in bootstrap_nodes:
                if node_id != self.node_id and node_id not in self.connections:
                    self.connect_to_peer(node_id, host, port)
        
        # 向连接的节点请求更多节点信息
        if self.connections:
            discover_msg = Message("DISCOVER", self.node_id, {
                "node_id": self.node_id,
                "host": self.host,
                "port": self.port
            })
            self.broadcast("DISCOVER", discover_msg.content)

    def _listen_for_connections(self):
        """监听新连接的线程"""
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                threading.Thread(target=self._handle_new_connection, 
                                args=(client_socket, address), 
                                daemon=True).start()
            except Exception as e:
                if self.running:
                    logger.error(f"接受连接时出错: {str(e)}")
                    time.sleep(1)

    def _handle_new_connection(self, client_socket, address):
        """处理新的连接请求"""
        try:
            # 接收自我介绍消息
            data = client_socket.recv(4096)
            if not data:
                client_socket.close()
                return
                
            message = pickle.loads(data)
            if message.msg_type != "INTRO":
                logger.warning(f"从 {address} 收到意外消息类型 {message.msg_type}")
                client_socket.close()
                return
                
            # 提取节点信息
            peer_id = message.content["node_id"]
            peer_host = message.content["host"]
            peer_port = message.content["port"]
            
            # 保存连接信息
            self.connections[peer_id] = client_socket
            self.peers[peer_id] = (peer_host, peer_port)
            
            # 启动接收消息的线程
            threading.Thread(target=self._listen_to_peer, args=(peer_id, client_socket), daemon=True).start()
            
            logger.info(f"接受来自节点 {peer_id} 的连接")
            self.metadata["connection_count"] += 1
            
        except Exception as e:
            logger.error(f"处理新连接时出错: {str(e)}")
            client_socket.close()

    def _listen_to_peer(self, peer_id: int, sock: socket.socket):
        """监听来自特定节点的消息"""
        try:
            while self.running and peer_id in self.connections:
                data = sock.recv(4096)
                if not data:
                    break
                    
                message = pickle.loads(data)
                self._process_message(peer_id, message)
                
        except Exception as e:
            if self.running:
                logger.error(f"从节点 {peer_id} 接收消息时出错: {str(e)}")
        finally:
            self._handle_disconnect(peer_id)

    def _process_message(self, sender_id: int, message: Message):
        """处理接收到的消息"""
        try:
            self.metadata["last_active"] = time.time()
            
            # 查找并调用对应的消息处理器
            if message.msg_type in self.message_handlers:
                self.message_handlers[message.msg_type](sender_id, message.content)
            else:
                logger.warning(f"未找到消息类型 {message.msg_type} 的处理器")
                
        except Exception as e:
            logger.error(f"处理来自节点 {sender_id} 的消息时出错: {str(e)}")

    def _send_to_socket(self, sock: socket.socket, message: Message):
        """将消息发送到套接字"""
        data = pickle.dumps(message)
        sock.sendall(data)

    def _handle_disconnect(self, peer_id: int):
        """处理节点断开连接"""
        if peer_id in self.connections:
            try:
                self.connections[peer_id].close()
            except:
                pass
            del self.connections[peer_id]
            logger.info(f"节点 {peer_id} 断开连接")

    def _heartbeat(self):
        """定期发送心跳包，维持连接和检测断开的节点"""
        while self.running:
            time.sleep(10)  # 每10秒发送一次心跳
            for peer_id in list(self.connections.keys()):
                self.send_to_peer(peer_id, Message("PING", self.node_id, {"timestamp": time.time()}))

    # -------- 基本消息处理器 --------
    
    def _handle_ping(self, sender_id: int, content: dict):
        """处理PING消息"""
        self.send_to_peer(sender_id, Message("PONG", self.node_id, {
            "echo_timestamp": content["timestamp"],
            "timestamp": time.time()
        }))

    def _handle_pong(self, sender_id: int, content: dict):
        """处理PONG消息"""
        # 可以用于计算网络延迟
        latency = time.time() - content["echo_timestamp"]
        logger.debug(f"与节点 {sender_id} 的往返延迟: {latency*1000:.2f}ms")

    def _handle_discover(self, sender_id: int, content: dict):
        """处理节点发现请求"""
        # 发送已知节点列表
        peer_list = []
        for peer_id, (host, port) in self.peers.items():
            if peer_id != sender_id and peer_id != self.node_id:
                peer_list.append({
                    "node_id": peer_id,
                    "host": host,
                    "port": port
                })
                
        self.send_to_peer(sender_id, Message("PEERS", self.node_id, peer_list))

    def _handle_peers(self, sender_id: int, peers_list: list):
        """处理接收到的节点列表"""
        new_connections = 0
        for peer_info in peers_list:
            peer_id = peer_info["node_id"]
            
            if peer_id != self.node_id and peer_id not in self.connections:
                success = self.connect_to_peer(
                    peer_id, 
                    peer_info["host"], 
                    peer_info["port"]
                )
                if success:
                    new_connections += 1
                    
        if new_connections > 0:
            logger.info(f"从节点 {sender_id} 发现并连接了 {new_connections} 个新节点")


class PBFTNode(Node):
    """支持PBFT共识的节点"""
    
    def __init__(self, node_id: int, host: str = '127.0.0.1', port: int = None):
        super().__init__(node_id, host, port)
        
        # PBFT状态
        self.is_validator = False  # 是否是验证节点
        self.is_primary = False    # 是否是主节点
        self.view_number = 0       # 当前视图编号
        self.sequence_number = 0   # 当前序列号
        
        # 已知验证节点集合
        self.validators: Set[int] = set()
        
        # 注册PBFT相关消息处理器
        self.register_handler("MODEL_UPDATE", self._handle_model_update)
        
        # 模型和训练相关状态
        self.current_model = None
        self.local_model = None
        self.training_args = None

    def set_as_validator(self, is_validator: bool):
        """设置节点是否为验证节点"""
        self.is_validator = is_validator
        logger.info(f"节点 {self.node_id} {'现在是' if is_validator else '不再是'}验证节点")
    
    def set_as_primary(self, is_primary: bool):
        """设置节点是否为主节点"""
        self.is_primary = is_primary
        logger.info(f"节点 {self.node_id} {'现在是' if is_primary else '不再是'}主节点")
    
    def broadcast_model_update(self, model):
        """广播模型更新"""
        if self.is_primary:
            logger.info(f"主节点 {self.node_id} 广播模型更新")
            self.broadcast("MODEL_UPDATE", {
                "model": model,
                "sequence": self.sequence_number,
                "view": self.view_number
            })
            self.sequence_number += 1
    
    def _handle_model_update(self, sender_id: int, content: dict):
        """处理模型更新消息"""
        # 这里只是一个基本实现，完整的PBFT处理逻辑会更复杂
        logger.info(f"收到来自节点 {sender_id} 的模型更新，序列号: {content['sequence']}")
        
        # 对于非验证节点，直接更新模型
        if not self.is_validator:
            self.current_model = content["model"]
            return
            
        # 验证节点需要参与共识流程
        # 这里将在PBFT共识部分实现 
"""
PVSS (公开可验证秘密分享) 工具模块

这个模块实现PVSS相关功能，用于在PBFT共识中生成和验证掩码种子。
注意：这是一个简化的模拟实现，并非完整的密码学实现。
在生产环境中，应该使用更安全的实现。
"""

import logging
import random
import hashlib
import torch
import json
from typing import Dict, List, Tuple, Set, Any, Optional
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
from Crypto.Util.number import getPrime, inverse
from Crypto.Random import get_random_bytes

logger = logging.getLogger("PVSS")

class PVSSHandler:
    """PVSS处理器，处理密钥生成、分享和重构"""
    
    def __init__(self, node_id: int, f: int = 1):
        """
        初始化PVSS处理器
        
        Args:
            node_id: 节点ID
            f: 系统容忍的最大恶意节点数
        """
        self.node_id = node_id
        self.f = f  # 容错数量
        self.t = f  # 分享阈值，通常t=f
        
        # 密钥相关
        self.private_key = None   # 自己的私钥
        self.public_key = None    # 自己的公钥
        self.public_keys = {}     # 所有节点的公钥 {node_id -> public_key}
        
        # 当前共识轮的密钥、份额等
        self.current_round = 0
        self.current_shares = {}  # 当前轮收到的份额 {node_id -> share}
        self.current_proofs = {}  # 当前轮收到的证明 {node_id -> proof}
        self.verified_shares = {} # 已验证的份额 {node_id -> share}
        
        # 生成初始密钥对
        self._generate_keys()
    
    def _generate_keys(self):
        """生成PVSS所需的RSA密钥对"""
        # 使用RSA作为基础密码系统
        key = RSA.generate(2048)
        self.private_key = key
        self.public_key = key.publickey()
        
        logger.info(f"节点 {self.node_id} 生成PVSS密钥对")
    
    def get_public_key(self) -> bytes:
        """获取公钥，用于分发给其他节点"""
        return self.public_key.export_key()
    
    def set_public_keys(self, public_keys: Dict[int, bytes]):
        """设置所有节点的公钥"""
        self.public_keys = {}
        for node_id, key_bytes in public_keys.items():
            try:
                self.public_keys[node_id] = RSA.import_key(key_bytes)
                logger.debug(f"导入节点 {node_id} 的公钥")
            except Exception as e:
                logger.error(f"导入节点 {node_id} 的公钥失败: {e}")
        
        logger.info(f"节点 {self.node_id} 设置 {len(public_keys)} 个节点的公钥")
    
    def set_public_key(self, node_id: int, key_bytes: bytes):
        """设置单个节点的公钥"""
        try:
            self.public_keys[node_id] = RSA.import_key(key_bytes)
            logger.debug(f"导入节点 {node_id} 的公钥")
        except Exception as e:
            logger.error(f"导入节点 {node_id} 的公钥失败: {e}")
    
    def _create_polynomial(self, secret: int, degree: int, modulus: int) -> List[int]:
        """
        创建多项式 f(x) = secret + a_1*x + a_2*x^2 + ... + a_degree*x^degree
        
        Args:
            secret: 多项式在x=0处的值 (秘密)
            degree: 多项式次数
            modulus: 模数
            
        Returns:
            多项式系数 [secret, a_1, a_2, ..., a_degree]
        """
        coefficients = [secret]
        for _ in range(degree):
            coefficient = random.randint(1, modulus - 1)  # 1到modulus-1之间的随机数
            coefficients.append(coefficient)
        return coefficients
    
    def _evaluate_polynomial(self, coefficients: List[int], x: int, modulus: int) -> int:
        """
        计算多项式在x处的值 f(x)
        
        Args:
            coefficients: 多项式系数 [a_0, a_1, a_2, ...]
            x: 自变量
            modulus: 模数
            
        Returns:
            f(x) mod modulus
        """
        value = 0
        for i, a in enumerate(coefficients):
            value = (value + a * pow(x, i, modulus)) % modulus
        return value
    
    def deal_secret(self, secret: int, node_ids: List[int]) -> Tuple[Dict[str, Dict], Dict[str, bytes]]:
        """
        使用PVSS分发秘密
        
        Args:
            secret: 要分发的秘密
            node_ids: 接收节点的ID列表
            
        Returns:
            Tuple[Dict[str, Dict], Dict[str, bytes]]: (份额, 证明)，注意键是字符串形式的节点ID
        """
        logger.info(f"节点 {self.node_id} 使用PVSS分发种子 {secret}")
        
        # 大素数作为有限域模数
        p = getPrime(512)  # 使用512位素数
        
        # 创建多项式，次数为门限t
        coefficients = self._create_polynomial(secret, self.t, p)
        
        shares = {}
        proofs = {}
        
        for node_id in node_ids:
            # 计算份额 f(node_id)
            share_value = self._evaluate_polynomial(coefficients, node_id, p)
            
            # 份额信息 - 确保使用字符串键
            share = {
                "dealer": self.node_id,  # 整数形式，由验证方处理
                "recipient": node_id,    # 整数形式，由验证方处理
                "value": share_value,
                "modulus": p,
                "timestamp": random.random()  # 添加随机性，确保每个份额的proof都不同
            }
            
            # 将份额转为字符串并加密
            import json
            share_str = json.dumps(share)
            
            # 使用接收者的公钥加密份额
            if node_id in self.public_keys:
                cipher = PKCS1_OAEP.new(self.public_keys[node_id])
                # 简化：在实际实现中，可能需要分块加密
                try:
                    # 签名份额
                    h = SHA256.new(share_str.encode())
                    signature = pkcs1_15.new(self.private_key).sign(h)
                    
                    # 将份额和签名一同保存，注意使用字符串键
                    shares[str(node_id)] = share
                    proofs[str(node_id)] = signature
                    
                    logger.info(f"为节点 {node_id} 创建份额: {share} 和证明(长度: {len(signature)})")
                except Exception as e:
                    logger.error(f"为节点 {node_id} 创建份额和证明失败: {e}")
            else:
                logger.warning(f"没有节点 {node_id} 的公钥，无法创建份额")
        
        return shares, proofs
    
    def verify_share(self, share: Dict, proof: bytes, dealer_id: int) -> bool:
        """
        验证份额的有效性
        
        Args:
            share: 份额
            proof: 证明
            dealer_id: 分发者节点ID
            
        Returns:
            bool: 份额是否有效
        """
        # 记录调试信息
        logger.info(f"节点 {self.node_id} 开始验证来自节点 {dealer_id} 的份额: {share}")
        logger.info(f"证明类型: {type(proof)}")
        
        # 检查基本字段
        if not isinstance(share, dict) or "dealer" not in share or "recipient" not in share:
            logger.warning("份额格式无效")
            return False
        
        # 检查份额是否分配给当前节点
        if isinstance(share["recipient"], str):
            recipient_id = int(share["recipient"])
        else:
            recipient_id = share["recipient"]
            
        if recipient_id != self.node_id:
            logger.warning(f"份额接收者 {recipient_id} 不是当前节点 {self.node_id}")
            return False
        
        # 检查分发者是否与提供的dealer_id一致
        if isinstance(share["dealer"], str):
            share_dealer_id = int(share["dealer"])
        else:
            share_dealer_id = share["dealer"]
            
        if share_dealer_id != dealer_id:
            logger.warning(f"份额的分发者 {share_dealer_id} 不一致 (期望 {dealer_id})")
            return False
        
        # 获取分发者的公钥
        if dealer_id not in self.public_keys:
            logger.warning(f"没有分发者 {dealer_id} 的公钥")
            return False
        
        dealer_public_key = self.public_keys[dealer_id]
        
        # 尝试转换proof为bytes（如果是字符串）
        if isinstance(proof, str):
            try:
                import base64
                proof = base64.b64decode(proof)
                logger.info("将字符串证明转换为字节")
            except:
                logger.error("无法将字符串证明转换为字节")
                return False
        
        # 验证签名
        try:
            # 将份额转为字符串
            import json
            share_str = json.dumps(share)
            logger.info(f"份额字符串: {share_str}")
            
            # 验证签名
            from Crypto.Hash import SHA256
            from Crypto.Signature import pkcs1_15
            
            h = SHA256.new(share_str.encode())
            pkcs1_15.new(dealer_public_key).verify(h, proof)
            
            logger.info(f"节点 {self.node_id} 成功验证来自节点 {dealer_id} 的份额")
            return True
        except Exception as e:
            logger.error(f"验证份额失败: {e}")
            return False
    
    def reconstruct_secret(self, shares: Dict[int, Dict]) -> Optional[int]:
        """
        重构秘密
        
        Args:
            shares: 份额字典 {node_id -> share}
            
        Returns:
            Optional[int]: 重构的秘密，如果重构失败则返回None
        """
        # 检查是否有足够的份额
        if len(shares) <= self.t:
            logger.warning(f"份额数量 ({len(shares)}) 不足以重构秘密 (需要至少 {self.t + 1} 个)")
            return None
        
        try:
            # 获取第一个份额以获取模数
            first_share = next(iter(shares.values()))
            modulus = first_share.get("modulus")
            
            if not modulus:
                logger.error("份额中没有模数信息")
                return None
            
            # 使用Lagrange插值重构秘密
            secret = 0
            x_values = list(shares.keys())
            
            for i, x_i in enumerate(x_values):
                # 计算拉格朗日基函数的分子
                numerator = 1
                # 计算拉格朗日基函数的分母
                denominator = 1
                
                for j, x_j in enumerate(x_values):
                    if i != j:
                        numerator = (numerator * (0 - x_j)) % modulus
                        denominator = (denominator * (x_i - x_j)) % modulus
                
                # 计算拉格朗日系数 lambda_i
                lambda_i = (numerator * inverse(denominator, modulus)) % modulus
                
                # 更新秘密
                share_value = shares[x_i].get("value")
                secret = (secret + lambda_i * share_value) % modulus
            
            # 确保结果为正
            secret = (secret + modulus) % modulus
            
            logger.info(f"节点 {self.node_id} 成功重构秘密: {secret}")
            return secret
        except Exception as e:
            logger.error(f"重构秘密失败: {e}")
            return None
    
    def set_round(self, round_number: int):
        """设置当前共识轮号并清除上一轮的状态"""
        if round_number != self.current_round:
            self.current_round = round_number
            self.current_shares = {}
            self.current_proofs = {}
            self.verified_shares = {}
            logger.info(f"节点 {self.node_id} 设置PVSS轮号: {round_number}")
    
    def process_share(self, share: Dict, proof: bytes, dealer_id: int) -> bool:
        """
        处理收到的份额和证明
        
        Args:
            share: 份额
            proof: 证明
            dealer_id: 分发者节点ID
            
        Returns:
            bool: 处理是否成功
        """
        # 验证份额
        if self.verify_share(share, proof, dealer_id):
            # 保存已验证的份额
            self.current_shares[dealer_id] = share
            self.current_proofs[dealer_id] = proof
            self.verified_shares[dealer_id] = share
            return True
        return False
    
    def get_reconstructed_secret(self) -> Optional[int]:
        """
        尝试从当前轮的已验证份额中重构秘密
        
        Returns:
            Optional[int]: 重构的秘密，如果重构失败则返回None
        """
        return self.reconstruct_secret(self.verified_shares) if self.verified_shares else None

def generate_hash_commitment(mask_seed, node_id):
    """
    使用哈希函数生成掩码承诺
    
    Args:
        mask_seed: 掩码种子
        node_id: 节点ID
        
    Returns:
        str: 哈希承诺
    """
    h = hashlib.sha256()
    h.update(str(mask_seed).encode())
    h.update(str(node_id).encode())
    return h.hexdigest()

def generate_sign_map_for_groups(groups: List[List[int]]) -> Dict[int, int]:
    """
    为分好组的节点生成符号映射，确保每组内的符号和为0
    
    Args:
        groups: 节点分组, 每组应有4个节点
        
    Returns:
        Dict[int, int]: 节点到符号(+1/-1)的映射
    """
    sign_map = {}
    
    # 对每个组，按照+1,+1,-1,-1的模式分配符号
    pattern = [1, 1, -1, -1]  # 确保每组符号和为0
    
    for group in groups:
        if len(group) != 4:
            logger.warning(f"组 {group} 中节点数不是4，可能导致掩码无法完全消除")
        
        # 为该组分配符号
        for i, node_id in enumerate(group):
            sign_map[node_id] = pattern[i % len(pattern)]
    
    # 验证总体符号和
    sign_sum = sum(sign_map.values())
    logger.info(f"生成符号映射: {sign_map}, 符号总和: {sign_sum}")
    assert sign_sum == 0, f"符号映射总和应为0，但实际为{sign_sum}"
    
    return sign_map

# 修改原有的generate_sign_map函数以兼容两种模式
def generate_sign_map(node_ids_or_groups, is_groups=False):
    """
    生成节点的符号映射，确保符号和为0（当节点数是4的倍数时）
    
    Args:
        node_ids_or_groups: 节点ID列表或已分组的节点列表
        is_groups: 是否已经是分好的组
        
    Returns:
        Dict[int, int]: 节点到符号(+1/-1)的映射
    """
    if is_groups:
        return generate_sign_map_for_groups(node_ids_or_groups)
    
    # 以下是原始实现（不基于分组）
    node_ids = node_ids_or_groups
    sign_map = {}
    
    # 使用固定模式1,1,-1,-1确保每4个节点的符号和为0
    pattern = [1, 1, -1, -1]
    
    for i, node_id in enumerate(node_ids):
        sign_map[node_id] = pattern[i % len(pattern)]
    
    # 验证总体符号和
    sign_sum = sum(sign_map.values())
    logger.info(f"生成符号映射: {sign_map}, 符号总和: {sign_sum}")
    
    # 如果节点数不是4的倍数，符号和可能不为0
    if len(node_ids) % 4 != 0 and sign_sum != 0:
        logger.warning(f"节点数 {len(node_ids)} 不是4的倍数，符号和为 {sign_sum}，掩码可能无法完全消除")
    
    return sign_map

def generate_mask_with_hash_commitment(mask_seed: Any, param_tensor: torch.Tensor, node_id: int, sign: int = 1) -> torch.Tensor:
    """
    使用哈希承诺生成确定性掩码
    
    Args:
        mask_seed: 掩码种子
        param_tensor: 参数张量，定义掩码的形状和类型
        node_id: 节点ID，用于哈希承诺
        sign: 符号 (+1 或 -1)，默认为 +1
        
    Returns:
        torch.Tensor: 生成的掩码
    """
    # 生成哈希承诺
    commitment = generate_hash_commitment(mask_seed, node_id)
    
    # 使用哈希承诺的前8位作为随机种子
    seed_int = int(commitment[:8], 16)
    
    # 设置随机种子
    torch.manual_seed(seed_int)
    
    # 存储原始数据类型
    original_dtype = param_tensor.dtype
    
    # 将参数转换为浮点类型以生成掩码
    float_tensor = param_tensor.float()
    
    # 生成与param_tensor形状相同、类型为浮点的随机掩码
    mask = torch.randn_like(float_tensor)
    
    # 将掩码转换回原始数据类型
    if original_dtype != torch.float32:
        mask = mask.to(original_dtype)
    
    # 根据符号调整掩码
    return sign * mask

def batch_generate_masks_with_hash_commitment(mask_seed: Any, model_state_dict: Dict[str, torch.Tensor], node_id: int, sign: int = 1) -> Dict[str, torch.Tensor]:
    """
    使用哈希承诺批量生成与模型状态字典形状匹配的掩码
    
    Args:
        mask_seed: 掩码种子
        model_state_dict: 模型状态字典
        node_id: 节点ID，用于哈希承诺
        sign: 符号 (+1 或 -1)，默认为 1
        
    Returns:
        Dict[str, torch.Tensor]: 掩码字典，与模型状态字典结构相同
    """
    masks = {}
    for key, param in model_state_dict.items():
        # 使用参数名作为额外的种子部分
        param_seed = f"{mask_seed}_{key}"
        masks[key] = generate_mask_with_hash_commitment(param_seed, param, node_id, sign=sign)
    
    return masks

def generate_mask(seed: Any, param_tensor: torch.Tensor, sign: int = 1) -> torch.Tensor:
    """
    根据种子生成与参数张量形状相同的掩码
    
    Args:
        seed: 随机种子（可以是整数或字符串）
        param_tensor: 参数张量，定义掩码的形状和类型
        sign: 符号 (+1 或 -1)，默认为 +1
        
    Returns:
        torch.Tensor: 生成的掩码
    """
    # 将种子转换为确定性的整数种子
    if isinstance(seed, str):
        # 如果是字符串，使用哈希函数
        seed_int = int(hashlib.md5(seed.encode()).hexdigest(), 16) % (2**32)
    else:
        # 如果已经是整数
        seed_int = int(seed)
    
    # 设置随机种子
    torch.manual_seed(seed_int)
    
    # 存储原始数据类型
    original_dtype = param_tensor.dtype
    
    # 将参数转换为浮点类型以生成掩码
    float_tensor = param_tensor.float()
    
    # 生成与param_tensor形状相同、类型为浮点的随机掩码
    mask = torch.randn_like(float_tensor)
    
    # 将掩码转换回原始数据类型
    if original_dtype != torch.float32:
        mask = mask.to(original_dtype)
    
    # 根据符号调整掩码
    return sign * mask

def apply_mask(model_update: torch.Tensor, mask: torch.Tensor, sign: int) -> torch.Tensor:
    """
    应用掩码到模型更新
    
    Args:
        model_update: 模型更新张量
        mask: 掩码张量
        sign: 符号 (+1 或 -1)
        
    Returns:
        torch.Tensor: 应用掩码后的模型更新
    """
    return model_update + sign * mask

def batch_generate_masks(seed: Any, model_state_dict: Dict[str, torch.Tensor], sign: int = 1) -> Dict[str, torch.Tensor]:
    """
    批量生成与模型状态字典形状匹配的掩码
    
    Args:
        seed: 随机种子
        model_state_dict: 模型状态字典
        sign: 符号 (+1 或 -1)，默认为 1
        
    Returns:
        Dict[str, torch.Tensor]: 掩码字典，与模型状态字典结构相同
    """
    masks = {}
    for key, param in model_state_dict.items():
        # 使用参数名作为额外的种子部分，确保不同参数有不同掩码
        param_seed = f"{seed}_{key}"
        masks[key] = generate_mask(param_seed, param, sign=sign)
    
    return masks

# 示例用法
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 参数
    num_nodes = 8
    f = 2  # 系统容忍的最大恶意节点数
    
    # 创建节点的PVSS处理器
    handlers = {}
    for i in range(num_nodes):
        handlers[i] = PVSSHandler(i, f)
    
    # 收集并分发所有节点的公钥
    public_keys = {}
    for i, handler in handlers.items():
        public_keys[i] = handler.get_public_key()
    
    for i, handler in handlers.items():
        handler.set_public_keys(public_keys)
    
    # 节点0作为dealer分发秘密 (模拟Primary节点)
    dealer_id = 0
    secret = random.randint(10000, 99999)  # 随机秘密
    node_ids = list(range(num_nodes))
    
    print(f"原始秘密: {secret}")
    
    # 分发秘密
    shares, proofs = handlers[dealer_id].deal_secret(secret, node_ids)
    
    # 各节点处理收到的份额
    verified_nodes = []
    for i, handler in handlers.items():
        if i in shares and i in proofs:
            if handler.process_share(shares[i], proofs[i], dealer_id):
                verified_nodes.append(i)
    
    print(f"成功验证份额的节点: {verified_nodes}")
    
    # 选择一些节点重构秘密 (至少需要 t+1 个)
    # 例如选择节点0,1,2,3
    reconstructing_nodes = [0, 1, 2, 3]
    reconstructed_shares = {}
    
    for i in reconstructing_nodes:
        if i in verified_nodes:
            reconstructed_shares[i] = handlers[i].current_shares[dealer_id]
    
    # 重构秘密
    if len(reconstructed_shares) > f:
        reconstructed_secret = handlers[0].reconstruct_secret(reconstructed_shares)
        print(f"重构的秘密: {reconstructed_secret}")
        
        # 验证重构是否成功
        if reconstructed_secret == secret:
            print("秘密重构成功！")
        else:
            print("秘密重构失败，结果不一致")
    else:
        print(f"没有足够的份额进行重构 (有 {len(reconstructed_shares)} 个，需要至少 {f+1} 个)")
    
    # 测试符号映射和掩码生成
    sign_map = generate_sign_map(node_ids)
    print(f"符号映射: {sign_map}")
    print(f"符号总和: {sum(sign_map.values())}")
    
    # 测试掩码生成
    if 'reconstructed_secret' in locals():
        # 创建一个测试模型更新
        test_param = torch.ones(3, 4)
        
        # 生成掩码
        mask = generate_mask(str(reconstructed_secret), test_param)
        print(f"掩码示例 (3x4):\n{mask}")
        
        # 测试掩码应用
        masked_updates = []
        for i in node_ids:
            sign = sign_map[i]
            # 直接使用带符号的掩码生成
            node_mask = generate_mask(str(reconstructed_secret), test_param, sign=sign)
            masked_update = test_param + node_mask
            masked_updates.append(masked_update)
            print(f"节点 {i} (符号 {sign}) 的掩码更新:\n{masked_update}")
        
        # 聚合掩码更新
        aggregated = torch.stack(masked_updates).sum(dim=0)
        print(f"聚合结果:\n{aggregated}")
        
        # 理论上的期望结果 (无掩码)
        expected = test_param * len(node_ids)
        print(f"期望结果 (无掩码):\n{expected}")
        
        # 检查掩码是否消除
        diff = torch.abs(aggregated - expected).sum().item()
        if diff < 1e-6:
            print("掩码成功消除!")
        else:
            print(f"掩码消除不完全，总差异: {diff}") 
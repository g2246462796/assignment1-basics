import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 1. 准备工厂函数
        factory_kwargs = {'device': device, 'dtype': dtype}

        # 2. 占坑：定义权重参数 W (形状： out x in)
        # 注意：这里不用 bias, 符合现代 LLM (如 LLama) 的做法
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # 3. 填土：执行阶段正态分布初始化
        # 注意：这里不用bias，符合现代 LLM（如 LLama）的做法
        # 如果用 std=1.0 初始化，经过多层线性变换后信号会爆炸或消失。所以用 std = sqrt(2/(d_in + d_out))（Xavier 初始化），让每层输出的方差大致保持稳定，跟输入输出维度相关。
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: [..., in_features], ...是batch_size等信息。
        # 使用 einsum 表达: "输入的最后一位 i 与 权重的最后一位 i 相乘, 输出 o"
        # 这种写法比 x @ self.weight.T 更具可读性, 且支持任意 Batch 维度
        """
        einsum 的语法是用字母标记每个维度，规则很简单：
        ... — 代表任意数量的未指定维度（通配符）
        逗号左边 ...i — 第一个张量 x 的维度标记：前面任意维度，最后一维叫 i
        逗号右边 oi — 第二个张量 self.weight 的维度标记：第一维叫 o, 第二维叫 i
        -> 右边 ...o — 输出张量的维度标记
        计算规则：
        相同字母的维度做求和（类似矩阵乘法中的内积）：这里 i 出现在两个输入中但不在输出里，所以对 i 维求和
        保留的维度直接传到输出：... 和 o 保留
        """
        return torch.einsum('...i, oi -> ...o', x, self.weight)

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        # 分配内存并包装为参数 (W 维度: vocab_size x d_model)
        # num_embeddings 就是词表大小 vocab_size
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))

        # 2. 按照作业要求执行初始化
        # mean=0, std=1.0, 截断在 [-3. 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids 形状: [B, S]
        # 直接通过索引从矩阵中“捞出”对应的向量
        # token_ids 形状是 [B, S]，里面每个值是 0 到 vocab_size-1 之间的整数。PyTorch 对每个整数去 self.weight 里取对应那一行，所以输出形状变成 [B, S, embedding_dim]
        return self.weight[token_ids] # 返回形状: [B, S, D]
    

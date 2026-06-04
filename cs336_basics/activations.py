import torch
import torch.nn as nn
from cs336_basics.Layers import Linear

def silu_fn(in_features):
    # Sigmoid：σ(x) = 1 / (1 + e^{-x})
    # SiLU / Swish：x * σ(x)
    return in_features * torch.sigmoid(in_features)

class SwigGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype= None):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        # W1 和 W3 是并行升维层: d_model -> d_ff
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        # W2 是降维层: d_ff -> d_model
        self.w2 = Linear(d_ff, d_model, device, dtype)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate = silu_fn(self.w1(x))
        signal = self.w3(x)

        return self.w2(gate * signal)

import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 数值稳定性：减去最大值，防止 exp 溢出
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    # 计算 exp 并归一化
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

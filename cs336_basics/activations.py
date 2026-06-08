import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 数值稳定性：减去最大值，防止 exp 溢出
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    # 计算 exp 并归一化
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

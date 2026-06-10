import math
import torch
from collections.abc import Callable
from typing import Optional

# -------------------- 自定义 SGD 优化器 (演示用) --------------------
class SGD(torch.optim.Optimizer):
    """
    一个简单的 SGD 实现，学习率按 1/sqrt(t+1) 自动衰减。
    注意：每个参数独立计数，实际使用中不推荐，仅用于教学示例。
    """
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)                     # 当前参数已更新次数
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad    # 更新参数
                state["t"] = t + 1
        return loss

# -------------------- 使用示例：最小化 weights 平方的均值 --------------------
if __name__ == "__main__":
    # 创建一个可训练的参数矩阵
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)

    for t in range(100):
        opt.zero_grad()                       # 清零梯度
        loss = (weights**2).mean()            # 计算损失
        print(f"step {t:3d}, loss = {loss.item():.6f}")
        loss.backward()                       # 反向传播计算梯度
        opt.step()                            # 更新参数
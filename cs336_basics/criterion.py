import torch
import math
from torch.optim import Optimizer
from collections.abc import Iterable

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 数值稳定性：减去最大值，防止 exp 溢出
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    # 计算 exp 并归一化
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算数值稳定的交叉熵损失。

    参数:
        logits: 形状为 (Batch, Seq, Vocab_size) 的预测分值
        targets: 形状为 (Batch, Seq) 的真实 Token ID
    """

    # 1. 计算每组 Logits 的最大值 M, 用于数值稳定
    # dim=-1 表示在词表维度搜索, keepdim=True 保证结果形状为 (Batch, Seq, 1)
    # 这样在后续执行 'logits - m' 时可以触发自动广播
    m = torch.max(logits, dim=-1, keepdim=True).values

    # 2. 提取目标位置对应的原始分值 o_y
    # 使用 gather 函数从词表维度中根据 targets 提取对应的分值
    # 由于 gather 要求 index 的维度与输入一致, 需将 targets 升维成 (Batch, Seq, 1)
    # 这里 相当于拿着tokenID取出来正确的那个概率。
    targets_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # 3. 计算 Log-Sum-Exp 项
    # shifted_logits 最大值为 0, exp 运算安全
    shifted_logits = logits - m

    # 公式: M + log(sum(exp(0 - M)))
    # 注意: m 提取出来后形状是 (B, S, 1), 求和项形状是 (B, S), 相加时需先 squeeze m
    log_sum_exp = m.squeeze(-1) + torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))

    # 4. 计算每个 Token 的独立损失值
    # shape:[B,S]
    loss = log_sum_exp - targets_logits

    # 5.按照作业要求, 对整个批次求平均, 返回一个标量
    # loss.backward()
    return torch.mean(loss)

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        # 1. 基本参数检查
        if lr < 0.0:
            raise ValueError(f"Invalid Learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        
        # 2. 将超参数存入 defaults 字典
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        """执行单步优化更新"""
        loss = None # loss 只是走个形式，并无实际意义，参考了pytorch官方的写法

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 3. 状态初始化 (第一次运行步时执行)
                if len(state) == 0:
                    state['step'] = 0
                    # m: 一阶矩 (梯度的指数移动平均)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # v: 二阶矩 (梯度平方的指数移动平均)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # 4. 更新矩估计 (Algorithm 1)
                # m = beta1 * m + (1 - beta1) * g
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v = betas * v + (1 - beta2) * g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 5. 计算偏差校正后的学习率 alpha_t
                # 这一步是为了消除初始值为 0 带来的偏移
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)

                # 6. 更新参数: theat = theta - alpha_t * m / (sqrt(v) + eps)
                denom = exp_avg_sq.sqrt().add(eps)
                # 这是一个专门为优化器设计的符合算子, 名字可以拆解为: add(加) + constant (常数) + div (除)。
                # p.addcdiv_(tensor1, tensor2, value=1.0)。 p=p+valuex( tensor1 / tensor2 )
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # 7. 应用解耦的权重衰减 (AdamW 的核心特性)
                # theta = theta - alpha * lambad * theta
                # p.add_(other, alpha=1.0) p=p+(alohaxother)
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)
        return loss
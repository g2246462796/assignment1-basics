import torch
import math
from torch.optim import Optimizer
from collections.abc import Iterable
import numpy  as np
import numpy.typing as npt

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
    
def get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int
) -> float:
    """
    计算第 it 次迭代时, 带预热的余弦退火学习率。

    参数:
        it: 当前迭代步数 (t)
        max_learning_rate: 学习率的峰值 (alpha_max)
        min_learning_rate: 学习率的底值 (alpha_min)
        warmup_iters: 预热阶段的总步数 (T_w)
        cosine_cycle_iters: 整个衰减周期结束的步数 (T_c)
    """

    # 1. 预热阶段: 线性增长周期
    if it < warmup_iters:
        # 从 0 匀速增长到 max_learning_rate
        return max_learning_rate * it / warmup_iters
    
    # 2. 衰减周期后: 维持最小值
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    # 3. 余弦退火核心逻辑
    # a. 计算当前处于退火阶段的进度百分比 (0.0 到 1.0)
    # it - warmup_iters: 距离预热结束走了多少步
    # cosine_cycle_iters - warmup_iters: 整个退火阶段的总长度
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)

    # b. 计算余弦系数
    # math.cos(math.pi  * decay_ratio):
    # 当前进度为 0 时, 结果为 cos(0) = 1
    # 当前进度为 1 时, 结果为 cos(pi) = -1
    # coeff = 0.5 * (1 + [-1, 1]) -> 范围 [0.0, 1.0]
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    # c. 最终计算
    # 学习率从 max 降向 min
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)

def clip_gradient_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float):
    """
    实现全局梯度裁剪(Global Norm Clipping)。
    
    参数:
        parameters: 模型的所有参数 (model.parameters())
        max_norm: 允许的最大梯度 L2 范数 (M)
    """
    # 1. 过滤掉没有梯度的参数 (防止对 None 对象操作)
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    
    # 2. 计算全局 L2 范数 (Global L2 Norm)
    total_norm = 0.0
    for p in params_with_grad:
        # 使用 .detach() 极其重要:
        # 梯度裁剪是在计算完导数后进行的数值操作, 我们不希望“计算范数”的过程也被记入计算图。
        # torch.norm(..., p=2) 算出当前层梯度的 L2 范数 L_i
        param_norm = torch.norm(p.grad.detach(), p=2)

        # 将各层范数的平方累加 (L_total = sqrt(sum(L_i^2)))
        total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5

    # 3. 检查是否触发裁剪
    eps = 1e-6 # 防止除零的稳定性常数
    if total_norm > max_norm:
        # 计算统一的缩放系数
        clip_coef = max_norm / (total_norm + eps)

        # 4. 原地 (in-place) 修改每个参数的梯度
        # 使用 mul_ 直接修改内存, 不产生临时副本, 节省显存
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef)

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    max_seq_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    随机采样一个训练批次。

    返回:
        x: 输入张量, 形状 [batch_size, max_seq_length]
        y: 目标张量, 形状 [batch_size, max_seq_length]
    """
    n = len(dataset)
    # 最后一个可用的起点, 必须流出 max_seq_length 的空间给 x, 再多留 1 位给 y
    max_idx = n - max_seq_length - 1

    # 随机选择 batch_size 个起始点
    ix = torch.randint(0, max_idx + 1, (batch_size,))

    # 提取序列并转为 Numpy 数组, 再转为 Tensor
    # 这样做比循环里逐个 to(device) 快得多
    x = torch.stack([torch.from_numpy(dataset[i : i + max_seq_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(dataset[i+1 : i + max_seq_length + 1].astype(np.int64)) for i in ix])

    # 一次性搬运到 GPU
    return x.to(device) , y.to(device)
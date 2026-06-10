import torch

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

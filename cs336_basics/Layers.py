import torch
import torch.nn as nn
from cs336_basics.criterion import softmax
import math
from einops import rearrange

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
    
class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype= None):
        """
        LayerNorm 的手动实现
        与 RMS Norm相比, 它同时处理了均值 (Mean) 和方差 (Variance) 。
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # 1. 学习参数初始化
        # weight (gamma): 缩放参数, 初始化为全 1
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        # bias (beta): 偏移参数, 初始化为全0
        # 这是 LayerNorm 独有的, RMSNorm 通常不使用 bias
        self.bias = nn.Parameter(torch.zeros(d_model, **factory_kwargs))

        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (batch_size, sequence_length, d_model)

        in_dtype = x.dtype
        # 2. 转换为 float32 以确保计算均值和方差时的数值稳定性 (防止溢出)
        x_float = x.to(torch.float32)

        # 3. 计算均值 (Mean)
        # 对最后一个维度 (特征维) 求平均, keepdim=True 用于后续减法广播
        # 公式: E[x]
        mean = x_float.mean(dim=-1, keepdim=True)

        # 4. 计算方差 (Variance)
        # 公式: Var(x) = E[(x - E[x])^2]
        # 注意: 这里使用 biased variance, 与 PyTorch 官方 nn.LayerNorm 对齐
        var = x_float.var(dim=-1, keepdim=True, unbiased=False)

        # 5. 归一化(Standardization)
        # 减去均值进行“中心化”，除以标准差进行缩放
        # 公式：(x - mean) / sqrt(var + eps)
        x_normed = (x_float - mean) / torch.sqrt(var + self.eps)

        # 6. 应用可学习的增益(weight)和偏置(bias)
        # 公式: y = x_normed * gamma + beta
        result = x_normed * self.weight + self.bias

        # 7. 转回输入时的原始数据类型 (如 bfloat16 或 float16)
        return result.to(in_dtype)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype= None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 1. 必须初始化为全 1 (ones)
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch_size, sequence_length, d_model)

        in_dtype = x.dtype
        # 2. 转换为 float32 以确保计算均值和方差时的数值稳定性 (防止溢出)
        x_float = x.to(torch.float32)

        # 3. 计算均方根 (Root Mean Square)
        # 公式: rms = sqrt( mean(x^2) + eps )
        # dim=-1 表示在隐藏层维度计算, keepdim=True 方便后续除法自动广播

        ms = x_float.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)

        # 4. 归一化并乘以可学习的增益函数 g
        result = (x_float / rms) * self.weight

        # 5. 转回原始类型
        return result.to(in_dtype)

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
    
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    参数:
        Q: [..., n, d_k] (n 为查询序列长度)
        K: [..., m, d_k] (m 为键值序列长度)
        V: [..., m, d_v]
        mask: [n, m] 布尔矩阵, True 为保留, False 为屏蔽
    """
    d_k = Q.size(-1)

    # 1.计算相似度分数 (Scores)
    # einsum 语义: 沿着 d_k 维度(k)进行点积, 保留 batch(...)、 query(n) 和 key(m) 维度
    # 结果形状: [..., n, m]
    scores = torch.einsum('...nk, ...mk -> ...nm', Q, K) / math.sqrt(d_k)

    # 2. 应用因果掩码 (Masking)
    if mask is not None:
        # 将 False 对应位置的分数设为负无穷, 使其在 Softmax 后概率为 0
        scores = scores.masked_fill(mask == False, float('-inf'))

    # 3. 计算注意力权重 (归一化)
    # dim=-1 对应的是每一个 Query 对所有 key 的分布
    probs = softmax(scores, dim=-1)

    # 4. 加权求和得到输出 (Output)
    # enisum 语义: 利用 probs(n, m) 对 V(m, k) 进行加权求和
    # 结果形状: [..., n, d_v]
    output = torch.einsum('...nm, ...mk -> ...nk', probs, V)

    return output

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化 RoPE 模块
        theta: 基准频率 (通常为 10000)
        d_k: 每个 Head 的维度 (必须是偶数)
        max_seq_len: 最大序列长度
        """
        super().__init__()
        self.d_k = d_k

        # 1. 计算频率 omega_k = theta^(-2k / d)
        # 我们只需要计算 d_k/2 个频率, 因为旋转是成对进行的
        # arange(0, d_k, 2) 产生 [0, 2, 4, ..., d_k-2], 对应公式中的2k-2(k从1开始)
        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        freqs = 1.0 / (theta ** powers) # 形状: (d_k/2,)

        # 创建位置序列 [0,1,..., max_seq_len - 1]
        t = torch.arange(max_seq_len, device=device).float() # 形状: (max_seq_len,)

        # 3. 计算所有位置的所有角度 (外积)
        # freqs_matrix 形状: (max_seq_len, d_k/2)
        freqs_matrix = torch.outer(t, freqs)

        # 4. 预计算 cos 和 sin 并作为 buffer 注册
        # 使用 persistent=False 确保这些缓存不会被保存在 state_dict 中 (因为可以随时重新生成)
        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 1. 提取 cos/sin (..., Seq, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 2. 维度对齐
        # 只有当 x 是 4D (含 Head 维) 且 cos 是 3D (含 Batch 维) 时, 才需要手动插入 Head 维。
        # 对于 test_rope 这种 3D x vs 2D cos 的情况, PyTorch 会自动左侧补 1, 无需操作。
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        
        # 确保类型一致
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # 3. 拆分
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos

        return output

class CausualSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        # 维度校验
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 1. Q, K, V 投影层: 将输入映射到三个不同的特征空间
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 2. 输出投影层: 整合所有头的信息
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 3.Rope 初始化: 仅在提供 theta 时启用
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        b, s, d = x.shape

        # 步骤 1 & 2: 线性投影并拆分多头
        # 使用 eniops.rearrange 替代 view + transpose
        # 语义: 将长度为 d 的特征维拆成 (h d_k), 并将 h 维移动到序列维 s 之前
        q = rearrange(self.q_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)
        k = rearrange(self.k_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)
        v = rearrange(self.v_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)

        # 步骤 3: 应用 RoPE 旋转位置编码
        if self.rope is not None:
            if token_positions is None:
                # 默认生成从 0 开始的顺序位置
                # expand 处理 Batch 维度, 不占用额外物理内存
                token_positions = torch.arange(s, device=x.device).expand(b, s)

            # 对 Q 和 K 进行旋转, V 保持不动
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        # 步骤 4: 生成因果掩码 (下三角矩阵)
        # 确保 Query 只能看到当前及以前的 Key
        mask = torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool))

        # 步骤 5: 核心注意力计算 (SDPA)
        # 结果形状: (Batch, Heads, Seq, d_k)
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)

        # 步骤 6: 合并多头
        # 语义: 将多头维度 h 重新并入特征维度
        attn_out = rearrange(attn_out, '... h s d -> ... s (h d)')

        # 步骤 7: 输出投影
        return self.output_proj(attn_out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int,
                 theta: float, device=None, dtype=None):
        super().__init__()
        # 初始化因果自注意力模块
        self.attn = CausualSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )
        # 初始化两个 RMSNorm 层, 分别服务于 Attention 和 FFN
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        # 初始化前馈网络 (SwiGLU)
        self.ffn = SwigGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # 步骤 1: Attention 子层 (Pre-norm 结构)
        # x 被分成两路: 一路直接传走 (残差), 一路进 Norm+Attention
        x = x + self.attn(self.ln1(x), token_positions=token_positions)

        # 步骤 2: FFN 子层 (Pre-norm 结构)
        # 再次分流: 一路直接传走, 一路进 Norm+FFN
        x = x + self.ffn(self.ln2(x))

        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int,
                 num_layers: int, num_heads: int, d_ff: int, rope_theta: float,
                 device=None, dtype=None,
                 # 新增实验参数
                 use_rms_norm: bool = True,
                 norm_mode: str = "pre",
                 ffn_type: str = "swiglu"):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.context_length = max_seq_len
        # 1. Token Embedding 层
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # 2. 堆叠 Transformer Blocks
        # 将实验参数透传给每一个 Block
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, max_seq_len, rope_theta,
                device=device,dtype=dtype,
                # use_rms_norm=use_rms_norm,
                # norm_mode=norm_mode,
                # ffn_type=ffn_type
            )
            for _ in range(num_layers)
        ])

        # 3. 最终的输出层
        # 如果全局禁用了 Norm, 这里的 Final Norm 也要变成 Identity
        if use_rms_norm:
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            """
            forward(input):
                return input
            """
            self.ln_final = nn.Identity()
        
        # 最后是一个 Linear 层映射回词表大小 (LM Head)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        b, s = token_ids.shape

        # 准备位置信息用于 RoPE, shape: [S] -> [1, S] -> [B, S]
        token_positions = torch.arange(s, device=token_ids.device).unsqueeze(0).expand(b, s)

        # 1. Embedding
        x = self.token_embeddings(token_ids)

        # 2. 逐层通过 Transformer Blocks
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        # 3. 最终归一化 (如果 use_rms_norm=False, 这里就是直通)
        x = self.ln_final(x)

        # 4. 投影到词表空间得到 logits
        return self.lm_head(x)
    
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int = None,
        temperature: float = 1.0,
        top_p: float = 1.0
    ) -> torch.Tensor:
        """
        从模型生成文本 ID 序列。

        参数: 
            prompt_ids: 提示词 ID (Batch, Seq_len)
            max_new_tokens: 最多生成的词数
            eos_token_id: 停止生成的 Token ID (如 <|endoftext|>)
            temperature: 温度系数 (越高越随机, 越低越稳定)
            top_p: 核采样阈值
        """
        # 设置为评估模式
        self.eval()

        # 将输入拷贝一份, 避免修改原始数据
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # 1. 裁剪输入: 模型只能处理 context_length 长度的内容
            # 如果生成的序列过长, 只取最后的 context_length 个词
            idx_cond = generated[:,-self.context_length:]

            # 2. 前向传播得到 Logits
            # 我们只关心最后一个时间步的预测
            logits = self.forward(idx_cond) # (Batch, T, Vocab)
            logits = logits[:, -1, :] # (Batch, Vocab)

            # 3. 应用温度 (Temperature)
            if temperature != 1.0:
                logits = logits / (temperature + 1e-8) # 加个 epsilon 防止除以 0
            
            # 4. 应用 Top-P (Nucleus Sampling) 过滤
            if top_p < 1.0:
                logits = self._top_p_filter(logits, top_p)
            
            # 5. 归一化并采样
            probs = softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (Batch, 1)
            
            # 6. 拼接新词
            generated = torch.cat((generated, next_token), dim=1)

            # 7. 如果遇到了 EOS, 提前结束生成
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated 


    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """内部工具函数: 执行 Top-P 截断"""
        # 对词表分值进行降序排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # 计算累计概率分布
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

        # 创建掩码: 我们要去掉累计概率超过 p 的 Token
        # 逻辑: 保留最小的集合 V(p), 使其概率之和 >= p
        # 我们把所有超过 p 的位置标记为 True (需要移除)
        sorted_indices_to_remove = cumulative_probs > p

        # 关键修正: 确保至少保留第一个词 (最高概率词),
        # 并且我们要保留第一个"使概率超过 p" 的那个词。
        # 做法是把标记位向右移动一格。
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # 将被移除的 Token 分数设为负无穷
        # 这里需要利用 scatter 将排序后的掩码映射回原始词表索引位置
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits
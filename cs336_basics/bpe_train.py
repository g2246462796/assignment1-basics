import os
from collections import defaultdict, Counter
import regex as re
import json

def read_text_in_chunks(file_path, chunk_size=1024*1024):
    """生成器：分块读取文本文件，按行切分，确保每行完整。"""
    with open(file_path, "r", encoding="utf-8") as f:
        buffer = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            lines = buffer.split("\n")
            # 除了可能不完整的最后一行，其余完整的行都 yield
            for line in lines[:-1]:
                yield line + "\n"
            buffer = lines[-1]   # 保留不完整的行
        if buffer:
            yield buffer

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练字节级 BPE 分词器（流式读取，避免内存爆炸）。
    """
    # --- 1. 初始化基础词表 ---
    vocab = {i: bytes([i]) for i in range(256)}
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges <= 0:
        # 若目标词表太小，直接返回基础词表+特殊token
        for s_tok in special_tokens:
            vocab[len(vocab)] = s_tok.encode("utf-8")
        return vocab, []

    # --- 2. 预编译正则（特殊 token 拆分 + GPT‑2 预分词）---
    if special_tokens:
        special_regex = "|".join(re.escape(t) for t in special_tokens)
    else:
        special_regex = None

    # GPT‑2 官方预分词正则（已验证）
    gpt2_pat = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    # --- 3. 流式统计 raw_counts（单词频率）---
    raw_counts = Counter()

    for chunk in read_text_in_chunks(input_path):
        # 3a. 根据特殊 token 切分当前 chunk
        if special_regex:
            parts = re.split(f"({special_regex})", chunk)
            train_segments = [p for p in parts if p not in special_tokens]
        else:
            train_segments = [chunk]

        # 3b. 对每个普通片段做预分词，统计单词频率
        for segment in train_segments:
            words = gpt2_pat.findall(segment)
            for word in words:
                # 将单词转为字节元组 (b'H', b'i')
                token_tuple = tuple(bytes([b]) for b in word.encode("utf-8"))
                raw_counts[token_tuple] += 1

    # 如果没有统计到任何单词（空文件），直接返回基础词表
    if not raw_counts:
        for s_tok in special_tokens:
            vocab[len(vocab)] = s_tok.encode("utf-8")
        return vocab, []

    # --- 4. 构建高效数据结构（words_list, counts_list, stats, indices）---
    words_list = []
    counts_list = []
    for word_tuple, freq in raw_counts.items():
        words_list.append(list(word_tuple))   # 转为可修改的 list
        counts_list.append(freq)

    stats = defaultdict(int)
    indices = defaultdict(set)          # pair -> set of word indices

    for idx, word in enumerate(words_list):
        freq = counts_list[idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            stats[pair] += freq
            indices[pair].add(idx)

    merges = []   # 记录合并顺序

    # --- 5. 迭代合并 ---
    for _ in range(num_merges):
        if not stats:
            break

        # 5a. 选择最佳 pair（频率最高，同频字典序最大）
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]
        if stats[best_pair] <= 0:
            break

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]

        # 5b. 获取所有包含该 pair 的单词索引（拷贝，因为循环中会修改 indices）
        relevant_indices = list(indices[best_pair])

        # 5c. 逐一更新受影响的单词
        for idx in relevant_indices:
            word = words_list[idx]
            freq = counts_list[idx]

            i = 0
            while i < len(word) - 1:
                if word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    # 1) 减少旧的相邻 pair 频率（左邻、右邻）
                    if i > 0:
                        prev_pair = (word[i-1], word[i])
                        stats[prev_pair] -= freq
                        if stats[prev_pair] == 0:
                            del stats[prev_pair]
                    if i < len(word) - 2:
                        next_pair = (word[i+1], word[i+2])
                        stats[next_pair] -= freq
                        if stats[next_pair] == 0:
                            del stats[next_pair]

                    # 2) 合并：替换第一个字节，删除第二个
                    word[i] = new_token
                    del word[i+1]

                    # 3) 添加新产生的相邻 pair
                    if i > 0:
                        new_prev = (word[i-1], word[i])
                        stats[new_prev] += freq
                        indices[new_prev].add(idx)
                    if i < len(word) - 1:
                        new_next = (word[i], word[i+1])
                        stats[new_next] += freq
                        indices[new_next].add(idx)

                    # 注意：合并后 i 不移动，因为当前位置已经是 new_token，
                    # 下一轮会检查 (new_token, word[i+1])
                else:
                    i += 1

        # 5d. 清理已完全合并的 best_pair（从 stats 和 indices 中删除）
        if best_pair in stats:
            del stats[best_pair]
        if best_pair in indices:
            del indices[best_pair]

    # --- 6. 构建最终词表 ---
    for pair in merges:
        new_id = len(vocab)
        vocab[new_id] = pair[0] + pair[1]

    for s_tok in special_tokens:
        vocab[len(vocab)] = s_tok.encode("utf-8")

    return vocab, merges
def bytes_to_unicode():
    """
    创建一个映射，将 0-255 字节映射为一组可见的 Unicode 字符。
    这是 GPT-2 源码的标准做法。
    """
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 初始化映射表
    byte_encoder = bytes_to_unicode()

    # 词表保存
    # 使用 byte_encoder 将 bytes 转换为可见字符串
    json_vocab = {
        k: "".join(byte_encoder[b] for b in v)
        for k, v in vocab.items()
    }
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)

    # 合并规则保存
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # 同样转换 p1 和 p2
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")

def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000 # 作业要求的词表大小
    # input_path = ""
    # input_path = ""
    # vocab_size = 1000 # 作业要求的词表大小

    special_tokens = ["<endoftext>"]
    output_dir = "data/TinyStoriesV2-GPT4-train"

    print(f"开始训练 BPE 分词器 (目标词表大小：{vocab_size})...")
    print("这可能需要几分钟，具体取决于你的 CPU 速度和倒排索引的效率")

    # 调用你之前写好的逻辑
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    # 保存结果
    save_tokenizer_files(vocab, merges, output_dir)

if __name__ == "__main__":
    main()
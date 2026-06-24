"""
预处理脚本：将 .txt 文本文件 tokenize 为 .bin（uint16）二进制文件
用法：python preprocess_data.py
"""

import os
import json
import numpy as np
from cs336_basics.bpe_train import bytes_to_unicode
from cs336_basics.BPETokenizer import BPETokenizer

# ===== 配置 =====
VOCAB_PATH = "data/TinyStoriesV2-GPT4-train/vocab.json"
MERGES_PATH = "data/TinyStoriesV2-GPT4-train/merges.txt"
SPECIAL_TOKENS = ["<endoftext>"]

FILES_TO_PROCESS = [
    ("data/TinyStoriesV2-GPT4-train.txt", "data/TinyStoriesV2-GPT4-train.bin"),
    ("data/TinyStoriesV2-GPT4-valid.txt", "data/TinyStoriesV2-GPT4-valid.bin"),
]

def load_tokenizer():
    """从 vocab.json 和 merges.txt 加载 tokenizer"""
    # 构建反向映射：可见字符 -> 字节
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    # 加载 vocab
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        json_vocab = json.load(f)

    vocab = {}
    for k, v in json_vocab.items():
        # 将可见字符串还原为 bytes
        vocab[int(k)] = bytes([byte_decoder[c] for c in v])

    # 加载 merges
    merges = []
    with open(MERGES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            a = bytes([byte_decoder[c] for c in parts[0]])
            b = bytes([byte_decoder[c] for c in parts[1]])
            merges.append((a, b))

    return BPETokenizer(vocab, merges, special_tokens=SPECIAL_TOKENS)

def tokenize_file_chunked(tokenizer, input_path, output_path, chunk_size=10_000_000):
    """分块 tokenize，避免内存爆炸，带进度条"""
    import os

    file_size = os.path.getsize(input_path)
    print(f"  分块编码: {input_path} ({file_size / 1024 / 1024:.0f} MB)")

    all_ids = []
    processed_bytes = 0

    with open(input_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            processed_bytes += len(chunk.encode("utf-8"))
            ids = tokenizer.encode(chunk)
            all_ids.extend(ids)

            # 进度条
            pct = processed_bytes / file_size * 100
            bar_len = 30
            filled = int(bar_len * processed_bytes / file_size)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"    [{bar}] {pct:.1f}% | {len(all_ids)} tokens", end="\r")

    print(f"\n  总计: {len(all_ids)} tokens")

    max_id = max(all_ids) if all_ids else 0
    assert max_id < 65536, f"Token ID {max_id} 超出 uint16 范围！"

    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(output_path)
    print(f"  保存: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")


def main():
    print("加载 tokenizer...")
    tokenizer = load_tokenizer()
    print(f"  词表大小: {len(tokenizer.vocab)}")

    for input_path, output_path in FILES_TO_PROCESS:
        print(f"\n处理: {input_path}")
        tokenize_file_chunked(tokenizer, input_path, output_path)

    print("\n完成！现在可以运行 sweep_lr.sh 了。")

if __name__ == "__main__":
    main()

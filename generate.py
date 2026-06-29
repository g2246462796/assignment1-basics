import argparse
import json
import torch
from cs336_basics.Layers import TransformerLM
from cs336_basics.criterion import load_checkpoint
from cs336_basics.BPETokenizer import BPETokenizer  # 按你实际文件名改

# ===== 配置区 =====
CKPT_PATH   = "model_result/TinyStories_baseline/ckpt_final.pt"
VOCAB_PATH  = "data/TinyStoriesV2-GPT4-train/vocab.json"
MERGES_PATH = "data/TinyStoriesV2-GPT4-train/merges.txt"
SPECIAL_TOKENS = ["<endoftext>"]   # 注意：和训练时一致，无竖线

# 必须和训练时一致
CFG = dict(vocab_size=10000, max_seq_len=256, d_model=512,
           num_layers=4, num_heads=16, d_ff=1344, rope_theta=10000.0)

CONTEXT_LENGTH = 256
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==================

def bytes_to_unicode():
    """和训练保存时用的同一个映射（GPT-2 标准做法）。"""
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

def load_tokenizer():
    # 反向映射：可见字符 -> 原始字节
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

    # 还原 vocab: {id_str: 可见字符串} -> {int: bytes}
    with open(VOCAB_PATH, encoding="utf-8") as f:
        json_vocab = json.load(f)
    vocab = {
        int(idx): bytes(byte_decoder[c] for c in token_str)
        for idx, token_str in json_vocab.items()
    }

    # 还原 merges: 每行 "s1 s2" -> (bytes, bytes)
    merges = []
    with open(MERGES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            s1, s2 = line.split(" ")
            b1 = bytes(byte_decoder[c] for c in s1)
            b2 = bytes(byte_decoder[c] for c in s2)
            merges.append((b1, b2))

    return BPETokenizer(vocab, merges, SPECIAL_TOKENS)

@torch.no_grad()
def generate(model, tokenizer, prompt):
    model.eval()
    eos_id = tokenizer.byte_to_id.get("<endoftext>".encode("utf-8"))

    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(MAX_NEW_TOKENS):
        logits = model(x[:, -CONTEXT_LENGTH:])[:, -1, :]
        logits = logits / (TEMPERATURE + 1e-8)
        probs = torch.softmax(logits, dim=-1)

        sp, si = torch.sort(probs, descending=True)
        mask = torch.cumsum(sp, dim=-1) - sp > TOP_P
        sp[mask] = 0.0
        sp = sp / sp.sum(dim=-1, keepdim=True)
        next_id = si.gather(-1, torch.multinomial(sp, 1))

        x = torch.cat([x, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break

    return tokenizer.decode(x[0].tolist())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="Once upon a time")
    args = p.parse_args()

    tokenizer = load_tokenizer()
    model = TransformerLM(device=DEVICE, **CFG).to(DEVICE)
    load_checkpoint(CKPT_PATH, model, optimizer=None)

    print("=" * 60)
    print(generate(model, tokenizer, args.prompt))
    print("=" * 60)

if __name__ == "__main__":
    main()

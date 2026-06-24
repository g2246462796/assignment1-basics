#!/bin/bash

# --- 1. 定义要跑试的学习率列表 ---
LR_LIST=(1e-3 3e-3 6e-3 1e-2 3e-2) # 已跑完：1e-4 3e-4 6e-4
# --- 2. 固定参数配置 ---
MAX_ITERS=7000
WARMUP_ITERS=700
MAX_NORM=1.0
BATCH_SIZE=32
CONTEXT_LEN=256
VOCAB_SIZE=10000

# 记录基础路径
OUTPUT_ROOT="model_result/sweep_lr"
WANDB_PROJECT="cs336-pretraining-TinyStories-LR"

# --- 3. 开始循环实验 ---
for LR in "${LR_LIST[@]}"; do

    MIN_LR=$(awk "BEGIN {print $LR * 0.1}")

    RUN_NAME="lr${LR}_min${MIN_LR}_step${MAX_ITERS}"
    OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

    echo "--------------------------------------------------------------"
    echo "📊 实验启动: max_lr=$LR, min_lr=$MIN_LR"
    echo "📛 监控名称: $RUN_NAME"
    echo "--------------------------------------------------------------"

    python cs336_basics/train.py \
        --train_data_path data/TinyStoriesV2-GPT4-train.bin \
        --valid_data_path data/TinyStoriesV2-GPT4-valid.bin \
        --run_name "$RUN_NAME" \
        --vocab_size "$VOCAB_SIZE" \
        --num_layers 4 --num_heads 16 --d_model 512 --d_ff 1344 \
        --max_iters "$MAX_ITERS" \
        --batch_size "$BATCH_SIZE" \
        --context_length "$CONTEXT_LEN" \
        --lr "$LR" \
        --min_lr "$MIN_LR" \
        --warmup_iters "$WARMUP_ITERS" \
        --max_norm "$MAX_NORM" \
        --out_dir "$OUT_DIR" \
        --device cuda \
        --wandb_project "$WANDB_PROJECT"

    if [ $? -ne 0 ]; then
        echo "❌ 警告: 学习率 $LR 导致训练中断, 跳过。"
    fi
done

echo "🎉 所有学习率扫参实验跑完! 请打开 WandB 观察曲线。"

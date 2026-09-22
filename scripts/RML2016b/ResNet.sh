#!/bin/bash

# 指定 GPU
export CUDA_VISIBLE_DEVICES=0

# 基础配置
model="ResNet"
dataset="RML2016b"
file_path="/root/autodl-tmp/dataset/RML2016.10b.dat"

# SNR 列表：训练时合并所有 SNR，测试时逐 SNR 单独评估
snr_list=($(seq -20 2 18))

# 任务列表：AMC(调制识别), WTC(技术识别), SS(频谱感知)
tasks=("AMC" "WTC" "SS")

# 1. 遍历任务
for task in "${tasks[@]}"
do
  batch_size=128
  epochs=100

  loss="cross_entropy"

  echo "=========================================================="
  echo "Task: $task | Model: $model"
  echo "SNR list: ${snr_list[*]}"
  echo "Optimizer: AdamW | Scheduler: ExponentialLR"
  echo "BS: $batch_size | LR: 0.001 | Epochs: $epochs"
  echo "=========================================================="

  # 执行训练与测试（一次性传入全部 SNR，训练合并、测试分离）
  python main.py \
    --task_name "$task" \
    --model "$model" \
    --dataset "$dataset" \
    --snr_list ${snr_list[@]} \
    --file_path "$file_path" \
    --mode "supervised" \
    --batch_size "$batch_size" \
    --num_epochs "$epochs" \
    --learning_rate 0.001 \
    --optimizer "adamw" \
    --weight_decay 1e-5 \
    --criterion "$loss" \
    --patience 10 \
    --warmup LinearLR \
    --warmup_epochs 0 \
    --split_ratio 0.6 \
    --seq_len 128 \
    --seed 42 \
    --dropout 0.1 \
    --d_model 64 \
    --d_ff 256 \
    --n_heads 8 \
    --n_layers 4 \
    --delta 0.0
done

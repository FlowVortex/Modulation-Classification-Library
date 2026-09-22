#!/bin/bash

# 指定 GPU
export CUDA_VISIBLE_DEVICES=0

# 基础配置
model="InceptionTime"                             # InceptionTime 用于异常检测
root_path="/root/autodl-tmp/dataset"             # MSL/PSM/SMAP/SMD 所在根目录
batch_size=128
epochs=100
lr=0.001

# AD 数据集列表
datasets=("MSL" "PSM" "SMAP" "SMD")

echo "=========================================================="
echo "  AD Benchmark: MSL / PSM / SMAP / SMD"
echo "  Model: $model | Loss: MSE"
echo "  BS: $batch_size | LR: $lr | Epochs: $epochs"
echo "=========================================================="

# 遍历 4 个 AD 数据集
for dataset in "${datasets[@]}"
do
  echo ""
  echo ">>> 开始训练: $dataset <<<"
  echo "=========================================================="

  python main.py \
    --task_name "AD" \
    --model "$model" \
    --dataset "$dataset" \
    --root_path "$root_path" \
    --mode "supervised" \
    --batch_size "$batch_size" \
    --num_epochs "$epochs" \
    --learning_rate "$lr" \
    --optimizer "adamw" \
    --weight_decay 1e-5 \
    --criterion "mse" \
    --patience 10 \
    --warmup LinearLR \
    --warmup_epochs 0 \
    --seq_len 100 \
    --seed 42 \
    --dropout 0.1 \
    --d_model 64 \
    --d_ff 256 \
    --n_heads 8 \
    --n_layers 4 \
    --delta 0.0

  echo ">>> $dataset 完成 <<<"
done

echo ""
echo "=========================================================="
echo "  All AD datasets completed!"
echo "=========================================================="

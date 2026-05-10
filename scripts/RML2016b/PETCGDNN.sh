#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

# 模型名称和训练数据集
model=PETCGDNN
dataset=RML2016b

# 训练集分割比例
split_ratio=0.6

# 遍历 SNR (信号比)
for snr in {-20..18..2}
do
  # 遍历 Batch Size
  for batch_size in 32 64 16 
  do
    # 遍历学习率
    for learning_rate in 0.001 0.0005 0.0001
    # for learning_rate in 0.0001 0.00005 0.000025 0.00001
    do
      clear
      echo "model: $model, dataset: $dataset, SNR: $snr, batch_size: $batch_size, learning_rate: $learning_rate"
      
      # 执行训练
      python main.py \
        --model $model \
        --dataset $dataset \
        --snr $snr \
        --file_path dataset/RML2016.10b.dat \
        --batch_size $batch_size \
        --num_epochs 64 \
        --learning_rate $learning_rate \
        --optimizer adam \
        --criterion cross_entropy \
        --split_ratio $split_ratio \
        --seq_len 128 \
        --d_model 128 \
        --patience 10 \
        --warmup_epochs 0
    done
  done
done
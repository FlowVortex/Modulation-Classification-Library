import torch
import numpy as np

# 1. 设置文件路径 (替换为你实际的 pth 路径)
file_path = './checkpoints/SS_CTNet_RML2016a_-2_supervised_sl128_bs400_lr0.001_dm64_df256_pat10_sd42_2026-07-20-18-43-06/results.pth'

# 2. 加载数据
# map_location='cpu' 确保即使是在 GPU 上保存的文件也能在没有 GPU 的机器上打开
data = torch.load(file_path, map_location='cpu')

# 3. 打印基本信息
print("="*30)
print(f"实验结果分析: {file_path}")
print("="*30)

# 读取准确率和时间
accuracy = data['accuracy'].item()
time_mean = data['time_mean'].item()
print(f"测试集准确率: {accuracy:.4f}")
print(f"平均推理时间: {time_mean:.6f} s")

# 4. 查看训练过程 (取最后一个 epoch 的值)
train_loss = data['train_loss']
val_acc = data['val_acc']
print(f"最终训练 Loss: {train_loss[-1]:.4f}")
print(f"最高验证 Acc: {max(val_acc):.4f}")

# 5. 查看混淆矩阵 (numpy 格式)
conf_matrix = data['confusion_matrix']
print("\n混淆矩阵:")
print(conf_matrix)
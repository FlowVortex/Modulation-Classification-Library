import sys
import torch
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("用法: python utils/check_result.py <results.pth 路径>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = torch.load(file_path, map_location='cpu')

    print("=" * 60)
    print(f"实验结果分析: {file_path}")
    print("=" * 60)

    # 基本信息
    accuracy = data['accuracy'].item()
    time_mean = data['time_mean'].item()
    print(f"\n测试集准确率: {accuracy:.4f}")
    print(f"平均推理时间:   {time_mean:.6f} s")

    # 训练过程
    train_loss = data['train_loss']
    val_acc = data['val_acc']
    print(f"\n最终训练 Loss:   {train_loss[-1]:.4f}")
    print(f"最高验证 Acc:     {max(val_acc):.4f}")

    # ===================== 逐 SNR 结果 =====================
    snr_accuracies = data.get('snr_accuracies', None)
    snr_conf_mats = data.get('snr_confusion_matrices', None)

    if snr_accuracies:
        snr_list = sorted(snr_accuracies.keys())
        print(f"\n{'─' * 60}")
        print(f"{'SNR (dB)':<12} {'Accuracy':<12}")
        print(f"{'─' * 60}")
        for snr in snr_list:
            print(f"  {snr:>5}      {snr_accuracies[snr]:.4f}")
        print(f"{'─' * 60}")

        # 逐 SNR 混淆矩阵
        if snr_conf_mats:
            for snr in snr_list:
                cm = snr_conf_mats.get(snr)
                if cm is not None:
                    print(f"\n--- SNR = {snr} dB 混淆矩阵 ---")
                    # print(np.array(cm))
    else:
        print("\n(未找到逐 SNR 准确率数据)")

    # 全局混淆矩阵
    conf_matrix = data.get('confusion_matrix', None)
    if conf_matrix is not None:
        print(f"\n{'=' * 60}")
        print("全局混淆矩阵 (所有 SNR 合并):")
        print(np.array(conf_matrix))


if __name__ == "__main__":
    main()

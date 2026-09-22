#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将全部实验结果汇总导出为 Excel 表格。

读取 `--checkpoints` 目录下所有 `results.pth`，按如下结构生成一个 Excel：
    - 每个任务 (AMC / WTC / SS / AD) 一个 sheet；
    - 横向 (列) 为模型名；
    - 纵向 (行) 为两级索引：第一层数据集名，第二层 SNR。

依赖：pandas / openpyxl / torch（后两者已随本项目安装）。

用法示例：
    python export_results.py
    python export_results.py --checkpoints ./checkpoints --output results_summary.xlsx
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

# 让中文输出在 Windows 终端 (Git Bash / cmd) 下不乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# ---------------------------------------------------------------------------
# 与 main.py 保持一致的已知集合（用于可靠地解析 setting 目录名）
# ---------------------------------------------------------------------------
TASKS = ["AMC", "WTC", "SS", "AD"]
MODELS = [
    "AMCNet", "CDAT", "CTNet", "DenseCNN", "DP_DRSN", "EMC2Net", "InceptionTime",
    "MCformer", "MCLDNN", "MTAMR", "PETCGDNN", "ModernTCN", "ResNet", "Conv_AE",
]
DATASETS = [
    "RML2016a", "RML2016b", "RML2018a", "HisarMod2019.1",
    "MSL", "PSM", "SMAP", "SMD",
]

# AD 任务无逐 SNR 结果，统一使用该标签作为“SNR”行名
OVERALL_SNR_LABEL = "overall"


def _alt(items):
    """把名称列表拼成正则分支（按长度降序，避免前缀误匹配）。"""
    return "|".join(sorted((re.escape(i) for i in items), key=len, reverse=True))


# setting 格式: {task}_{model}_{dataset}_{snr}_{mode}_sl{..}_bs{..}_..._{time}
SETTING_RE = re.compile(
    rf"^(?P<task>{_alt(TASKS)})_"
    rf"(?P<model>{_alt(MODELS)})_"
    rf"(?P<dataset>{_alt(DATASETS)})_"
    rf"(?P<snr>-?\d+)_"
    rf"(?P<mode>supervised|unsupervised)_"
)


def _to_float(value):
    """把 tensor / numpy 标量 / 数字统一转成 float。"""
    if value is None:
        return np.nan
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
    if isinstance(value, np.ndarray):
        return float(np.asarray(value).item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _order_unique(values, preferred):
    """按 preferred 给定顺序排列 values，其余按字母序追加。"""
    pref = [v for v in preferred if v in values]
    rest = sorted(set(values) - set(pref))
    return pref + rest


def collect_results(checkpoints_dir):
    """扫描 checkpoints 目录，返回 task -> [(model, dataset, snr_label, acc)]。

    其中 snr_label 为字符串（逐 SNR 时为 '-20' 之类，AD 任务为 'overall'）。
    同一 (model, dataset, snr) 若存在多次运行，取时间戳最新的一次。
    """
    pattern = os.path.join(checkpoints_dir, "*", "results.pth")
    paths = sorted(glob.glob(pattern))

    if not paths:
        print(f"[警告] 在 {checkpoints_dir} 下未找到任何 results.pth。")

    # 先收集所有记录（按路径排序 = 按时间戳排序），后写入时靠覆盖实现“最新优先”
    records = defaultdict(list)
    seen = set()
    for pth in paths:
        dirname = os.path.basename(os.path.dirname(pth))
        m = SETTING_RE.match(dirname)
        if not m:
            print(f"[警告] 无法解析目录名，已跳过: {dirname}")
            continue

        task = m.group("task")
        model = m.group("model")
        dataset = m.group("dataset")

        try:
            # weights_only=False：results.pth 内含普通 float/dict，需完整反序列化
            data = torch.load(pth, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 加载失败，已跳过 {pth}: {exc}")
            continue

        snr_accs = data.get("snr_accuracies", None)
        if snr_accs:
            for snr, acc in snr_accs.items():
                label = str(int(snr))
                key = (model, dataset, label)
                records[task].append((model, dataset, label, _to_float(acc), dirname))
                seen.add((task, key))
        else:
            # AD 等无逐 SNR 结果的任务，退化为整体准确率
            acc = _to_float(data.get("accuracy", None))
            key = (model, dataset, OVERALL_SNR_LABEL)
            records[task].append((model, dataset, OVERALL_SNR_LABEL, acc, dirname))
            seen.add((task, key))

    # 每组只保留最后一条（时间戳最新）
    deduped = defaultdict(dict)
    for task, recs in records.items():
        for model, dataset, snr, acc, _dirname in recs:
            deduped[task][(model, dataset, snr)] = acc

    return deduped


def build_sheet(task_records):
    """把某个任务的 (model, dataset, snr) -> acc 记录构建成透视 DataFrame。"""
    models = _order_unique({r[0] for r in task_records}, MODELS)
    datasets = _order_unique({r[1] for r in task_records}, DATASETS)

    snr_labels = [r[2] for r in task_records]
    # 数值 SNR 按数值升序，'overall' 排最后
    int_snrs = sorted({int(s) for s in snr_labels if s != OVERALL_SNR_LABEL})
    ordered_snrs = [str(s) for s in int_snrs]
    if OVERALL_SNR_LABEL in snr_labels:
        ordered_snrs.append(OVERALL_SNR_LABEL)

    index = pd.MultiIndex.from_product(
        [datasets, ordered_snrs], names=["Dataset", "SNR"]
    )

    df = pd.DataFrame(np.nan, index=index, columns=models)
    df.index.name = "Dataset \\ SNR"

    for (model, dataset, snr), acc in task_records.items():
        if dataset in datasets and snr in ordered_snrs:
            df.loc[(dataset, snr), model] = acc

    return df


def _style_sheet(ws):
    """轻量美化：冻结表头与索引列、表头加粗、调整列宽。"""
    from openpyxl.styles import Alignment, Font, PatternFill

    ws.freeze_panes = "C2"  # 冻结第一行 + 前两列（Dataset / SNR）

    header_fill = PatternFill("solid", fgColor="DDEBF7")
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 10
    for col in ws.iter_cols(min_col=3, max_col=ws.max_column):
        ws.column_dimensions[col[0].column_letter].width = 13


def export_excel(task_data, output_path):
    """把 task -> {(model,dataset,snr): acc} 写成 Excel，每个任务一个 sheet。"""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        written = 0
        for task in TASKS:
            if task not in task_data:
                continue
            df = build_sheet(task_data[task])
            df.to_excel(writer, sheet_name=task, float_format="%.4f", na_rep="")
            _style_sheet(writer.sheets[task])
            written += 1

            print(f"[完成] 任务 {task}: "
                  f"{df.shape[0]} 行 (数据集×SNR) × {df.shape[1]} 个模型")
        if written == 0:
            print("[提示] 没有可导出的数据，未生成 Excel。")
    return written


def main():
    parser = argparse.ArgumentParser(description="汇总实验结果并导出 Excel")
    parser.add_argument(
        "--checkpoints", type=str, default="./checkpoints",
        help="checkpoints 根目录（其中每个子目录含 results.pth），默认 ./checkpoints",
    )
    parser.add_argument(
        "--output", type=str, default="./results_summary.xlsx",
        help="输出 Excel 路径，默认 ./results_summary.xlsx",
    )
    args = parser.parse_args()

    task_data = collect_results(args.checkpoints)
    if not task_data:
        return

    export_excel(task_data, args.output)
    print(f"\n已导出: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

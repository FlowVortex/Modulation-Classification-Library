#!/bin/bash
echo "批量执行 scripts/RML2016a 和 scripts/AD 目录下所有sh脚本"

# 定义需要遍历的目录列表
dirs=(
    "scripts/RML2018a"
)

# 循环每个目标目录
for dir in "${dirs[@]}"; do
    # 检查目录是否存在
    if [ ! -d "$dir" ]; then
        echo "警告：目录 $dir 不存在，跳过"
        continue
    fi

    # 遍历当前目录下所有.sh文件
    for file in "$dir"/*.sh; do
        # 判断是普通文件
        if [ -f "$file" ]; then
            # 跳过自身脚本，防止递归执行
            if [ "$file" != "./$0" ]; then
                echo -e "\n========== 执行 $file =========="
                # 添加执行权限并运行
                chmod +x "$file"
                ./"$file"
            fi
        fi
    done
done

echo -e "\n所有sh脚本运行结束"
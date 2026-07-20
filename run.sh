#!/bin/bash
echo "批量执行当前目录所有sh脚本"
for file in scripts/RML2018a/*.sh; do
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
echo -e "\n所有sh脚本运行结束"
#!/bin/bash

# 设置参数
tp=2
bs_list=(1 2 4 8)
model_list=("GPT-125M" "GPT-350M" "GPT-760M" "GPT-1.3B")

# 遍历 bs 和 model_name 列表
for model_name in "${model_list[@]}"; do
    for bs in "${bs_list[@]}"; do
        # 调用 Python 脚本，并传入参数
        PYTHONPATH=$PYTHON_PATH:./megatron torchrun --nproc-per-node 2 ../codes/test_profile_m.py --tp $tp --bs $bs --model_name $model_name
    done
done

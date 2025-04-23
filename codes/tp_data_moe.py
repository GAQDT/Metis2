import json
import os
import copy
import math
from typing import Dict

def nonlinear_decay_factor(base_value: float, max_value: float) -> float:
    """计算非线性衰减因子，数值越大衰减越小"""
    return 0.2 * math.exp(-0.5 * base_value / max_value)  # 指数衰减函数

def generate_scaled_data(original_data: Dict, target_tp: int, base_tp=2) -> Dict:
    """
    改进的非线性缩放模型，包含以下特性：
    1. TP=1时实际值小于理论线性值的2倍
    2. TP=4时实际值大于理论线性值的0.5倍
    3. 数值越大衰减效应越小
    """
    scaled_data = copy.deepcopy(original_data)
    original_max_time = max(original_data["execution_time"]["layer_compute_total_ms"])
    original_max_mem = max(original_data["execution_memory"]["layer_memory_total_mb"])

    # 计算时间非线性缩放
    def scale_compute_time(original_time):
        ideal_scale = base_tp / target_tp
        decay = nonlinear_decay_factor(original_time, original_max_time)
        
        # 基础衰减公式：实际缩放比例 = 理想比例 × (1 + 衰减因子)
        if target_tp < base_tp:
            actual_scale = ideal_scale * (1 - decay)  # TP减小时性能提升衰减
        else:
            actual_scale = ideal_scale * (1 + decay)  # TP增加时性能提升衰减
        return original_time * actual_scale

    # 通信时间非线性缩放
    def scale_comm_time(original_comm):
        if base_tp == 1:
            return original_comm  # 无基准通信数据时不缩放
        
        base_comm_ratio = (base_tp - 1) / base_tp
        target_comm_ratio = (target_tp - 1) / target_tp
        comm_scale = (target_comm_ratio / base_comm_ratio) ** 1.2  # 非线性通信放大
        
        # 添加固定通信开销
        fixed_overhead = 0.1 * original_comm * (target_tp / base_tp) ** 0.5
        return original_comm * comm_scale + fixed_overhead
        

    # 内存占用非线性缩放
    def scale_memory(original_mem):
        ideal_scale = base_tp / target_tp
        decay = nonlinear_decay_factor(original_mem, original_max_mem)
        
        if target_tp < base_tp:
            actual_scale = ideal_scale * (1 - decay)  # TP减小时性能提升衰减
        else:
            actual_scale = ideal_scale * (1 + decay)  # TP增加时性能提升衰减
        return original_mem * actual_scale

    # 调整执行时间
    et = scaled_data["execution_time"]
    
    # 总时间复合计算
    et["forward_backward_time_ms"] = sum(
        scale_compute_time(t) for t in et["layer_compute_total_ms"]
    ) * 1.15  # 增加15%同步开销
    
    # 通信时间调整
    et["layernorm_grads_all_reduce_time_ms"] = scale_comm_time(
        et["layernorm_grads_all_reduce_time_ms"])
    et["embedding_grads_all_reduce_time_ms"] = scale_comm_time(
        et["embedding_grads_all_reduce_time_ms"])
    
    # 逐层计算时间
    et["layer_compute_total_ms"] = [scale_compute_time(t) for t in et["layer_compute_total_ms"]]
    et["optimizer_time_ms"] = scale_compute_time(et["optimizer_time_ms"])
    
    # 总时间合成（包含非线性组合效应）
    et["total_time_ms"] = (
        et["forward_backward_time_ms"] 
        + et["layernorm_grads_all_reduce_time_ms"]
        + et["embedding_grads_all_reduce_time_ms"]
        + et["optimizer_time_ms"]
    ) * 1.05  # 增加5%系统开销

    # 调整内存使用
    em = scaled_data["execution_memory"]
    em["total_memory_mb"] = scale_memory(em["total_memory_mb"])
    em["layer_memory_total_mb"] = [scale_memory(m) for m in em["layer_memory_total_mb"]]

    return scaled_data
def process_files(input_dir: str, output_dir: str):
    """
    处理目录下的所有基准文件
    :param input_dir: 输入目录（包含tp=2的原始文件）
    :param output_dir: 输出目录（将生成tp=1和tp=4的文件）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 需要处理的基准文件名模式
    base_files = [
        "DeviceType.H800_tp2_bs1.json",
        "DeviceType.H800_tp2_bs2.json",
        "DeviceType.H800_tp2_bs4.json",
        "DeviceType.H800_tp2_bs8.json"
    ]
    
    for base_file in base_files:
        # 读取原始数据
        with open(os.path.join(input_dir, base_file), 'r') as f:
            original_data = json.load(f)
        
        # 生成所有目标配置
        for target_tp in [1, 4]:
            # 生成模拟数据
            scaled_data = generate_scaled_data(original_data, target_tp)
            
            # 构建新文件名（保留批次大小）
            bs = base_file.split('_bs')[1].split('.')[0]
            new_filename = f"DeviceType.H800_tp{target_tp}_bs{bs}.json"
            
            # 写入输出文件
            output_path = os.path.join(output_dir, new_filename)
            with open(output_path, 'w') as f:
                json.dump(scaled_data, f, indent=2)
                
            print(f"Generated: {output_path}")

if __name__ == "__main__":
    # 配置输入输出路径
    model_names = ["MoE-380M","MoE-690M","MoE-1.3B"]
    for name in model_names:

        input_directory = "./moe_profile_data/"+name  
        output_directory = "./moe_profile_data/"+name  
        
        process_files(input_directory, output_directory)
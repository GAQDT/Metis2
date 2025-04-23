import torch
import torch.nn as nn
from transformers import OPTForCausalLM, OPTConfig
from transformers.models.opt.modeling_opt import OPTDecoderLayer
import time
import json
import os

# 配置参数
device_id = 0  # 指定第一个A100，若为第二个A100，设置为1
device = torch.device(f'cuda:{device_id}')  # 指定设备



tp = 1  # 张量并行度
bs = 1  # 批量大小

# 使用Hugging Face的OPT-3.1B模型
config = OPTConfig.from_pretrained("facebook/opt-1.3b")
model = OPTForCausalLM(config).to(device)

with open('model_config.txt', 'w') as f:
    f.write(str(config))

# 记录性能和内存数据的结构
performance_data = {
    "model": {
        "model_name": "OPT3-1.3B",
        "num_layers": config.num_hidden_layers+2,
        "parameters": {
            "total_parameters_bytes": 0,
            "parameters_per_layer_bytes": []
        }
    },
    "execution_time": {
        "total_time_ms": 0,
        "forward_backward_time_ms": 0,
        "batch_generator_time_ms": 0,
        "layernorm_grads_all_reduce_time_ms": 0,
        "embedding_grads_all_reduce_time_ms": 0,
        "optimizer_time_ms": 0,
        "layer_compute_total_ms": [],
    },
    "execution_memory": {
        "total_memory_mb": 0,
        "layer_memory_total_mb": [],
    }
}

# 获取模型的参数数据
def get_model_parameters(model):
    parameters_per_layer = []
    total_parameters = 0
    curr = 0
    for name, param in model.named_parameters():
        param_size = param.numel() * param.element_size()  # 计算参数的字节数
        total_parameters += param_size
        if(name.split('.')[3].isdigit()):
            layer_idx = int(name.split('.')[3]) + 2  # 提取层的索引
            if len(parameters_per_layer) <= layer_idx:
                parameters_per_layer.append(0)
            parameters_per_layer[layer_idx] += param_size
        else:
            if len(parameters_per_layer) <= 0 or curr == 2:
                parameters_per_layer.append(0)
            parameters_per_layer[len(parameters_per_layer)-1] += param_size
            curr = curr + 1
    parameters_per_layer.append(parameters_per_layer.pop(1))
    return total_parameters, parameters_per_layer

# 获取参数信息
total_parameters, parameters_per_layer = get_model_parameters(model)
performance_data["model"]["parameters"]["total_parameters_bytes"] = total_parameters
performance_data["model"]["parameters"]["parameters_per_layer_bytes"] = parameters_per_layer

begin_time = 0
# 全局变量来记录每层的执行时间
layer_times = []

def is_transformer_block(module):
    """识别OPT的Decoder Layer块"""
    return isinstance(module, OPTDecoderLayer)  # 关键修改点

def forward_pre_hook(module, input):
    global begin_time
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    initial_mem = torch.cuda.memory_allocated(device) / (1024**2)
    start_time = time.time()
    layer_times.append(('forward_pre', start_time, initial_mem))
    if begin_time == 0:
        begin_time = start_time
    return input

def forward_hook(module, input, output):
    torch.cuda.synchronize(device)
    end_time = time.time()
    mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    layer_times.append(('forward', end_time, mem))
    return output

def backward_pre_hook(module, grad_output):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    initial_mem = torch.cuda.memory_allocated(device) / (1024**2)
    start_time = time.time()
    layer_times.append(('backward_pre', start_time, initial_mem))
    return grad_output

def backward_hook(module, grad_input, grad_output):
    torch.cuda.synchronize(device)
    end_time = time.time()  
    mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    # 监测LayerNorm梯度同步时间
    if isinstance(module, nn.LayerNorm):
        layernorm_start_time = time.time()
        # 在这里你需要添加AllReduce的监测逻辑
        torch.cuda.synchronize()  # 等待GPU完成同步
        performance_data["execution_time"]["layernorm_grads_all_reduce_time_ms"] += (time.time() - layernorm_start_time) * 1000  # 转换为毫秒
    
    # 监测Embedding梯度同步时间
    elif isinstance(module, nn.Embedding):
        embedding_start_time = time.time()
        # 在这里你需要添加AllReduce的监测逻辑
        torch.cuda.synchronize()  # 等待GPU完成同步
        performance_data["execution_time"]["embedding_grads_all_reduce_time_ms"] += (time.time() - embedding_start_time) * 1000  # 转换为毫秒
    layer_times.append(('backward', end_time, mem))
    return grad_input

def is_embedding_layer(name, module):
    if isinstance(module, nn.Embedding):
        return True
def is_final_layer(name, module):
    return "layers" not in name and (isinstance(module, nn.Embedding) or isinstance(module, nn.LayerNorm))
# 注册钩子
def register_hooks(model):
    hooks = []
    registered_ids = set()
    for name, module in model.named_modules():
        if is_final_layer(name, module) or isinstance(module, OPTDecoderLayer):
            module_id = id(module)
            if module_id in registered_ids:
                continue
            registered_ids.add(module_id)
            forward_pre_hook_fn = module.register_forward_pre_hook(forward_pre_hook)
            forward_hook_fn = module.register_forward_hook(forward_hook)
            backward_pre_hook_fn = module.register_full_backward_pre_hook(backward_pre_hook)
            backward_hook_fn = module.register_full_backward_hook(backward_hook)
            hooks.append(forward_pre_hook_fn)
            hooks.append(forward_hook_fn)
            hooks.append(backward_pre_hook_fn)
            hooks.append(backward_hook_fn)
    return hooks

# 记录训练时间
def measure_performance(model, inputs):
    global layer_times

    torch.cuda.synchronize(device)

    hooks = register_hooks(model)

    torch.cuda.reset_peak_memory_stats(device)

    torch.cuda.synchronize(device)
    total_start_time = time.time()

    # 记录批量数据生成时间
    torch.cuda.synchronize(device)
    batch_start_time = time.time()
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 将输入转移到CUDA设备上
    torch.cuda.synchronize(device)
    batch_generator_time = (time.time() - batch_start_time) * 1000  # 转换为毫秒

    # 开始执行训练
    outputs = model(**inputs)  # 使用**inputs以传递所有参数
    outputs.loss.backward()  # 反向传播损失
    torch.cuda.synchronize(device)
    print(torch.cuda.memory_summary(device))

    # 优化器时间
    optimizer_start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.step()
    optimizer_time = (time.time() - optimizer_start_time) * 1000  # 转换为毫秒

    torch.cuda.synchronize(device)
    total_time = (time.time() - total_start_time) * 1000


    # 记录每层的计算时间
    layer_compute_time = []
    layer_memory = []
    total_memory = 0
    forward_begin_time = 0
    backward_finish_time = 0
    mid = len(layer_times) // 2
    forward_times = layer_times[:mid]
    backward_times = layer_times[mid:]
    for i in range(0, len(forward_times), 2):  # 每4个数据为一层（前向开始、前向结束、反向开始、反向结束）
        forward_start_time = forward_times[i][1]  # 前向开始时间
        forward_end_time = forward_times[i + 1][1]  # 前向结束时间
        backward_start_time = backward_times[mid - i - 2][1]  # 反向开始时间
        backward_end_time = backward_times[mid - i - 1][1]  # 反向结束时间

        mem_values = [
            forward_times[i][2],
            forward_times[i + 1][2],
            backward_times[mid - i - 2][2],
            backward_times[mid - i - 1][2]
        ]
        print(mem_values)
        

        layer_peak_mem = mem_values[1] + mem_values[3] - mem_values[0] - mem_values[2]
        layer_memory.append(layer_peak_mem)
        total_memory = total_memory + layer_peak_mem

        # 计算每层的总计算时间（反向时间 - 前向时间）
        compute_time = (backward_end_time - backward_start_time + forward_end_time - forward_start_time) * 1000  # 转换为毫秒
        layer_compute_time.append(compute_time)

    #total_time = (time.time() - forward_backward_start_time) * 1000  # 转换为毫秒
    forward_backward_time = (backward_times[mid - 1][1] - forward_times[0][1]) * 1000
    layer_compute_time[0] = layer_compute_time[0] + layer_compute_time[1]
    layer_compute_time.pop(1)
    layer_memory[0] = layer_memory[0] + layer_memory[1]
    layer_memory.pop(1)

    # 更新性能数据
    performance_data["execution_time"]["total_time_ms"] = total_time
    performance_data["execution_time"]["forward_backward_time_ms"] = forward_backward_time
    performance_data["execution_time"]["batch_generator_time_ms"] = batch_generator_time
    performance_data["execution_time"]["optimizer_time_ms"] = optimizer_time
    performance_data["execution_time"]["layer_compute_total_ms"] = layer_compute_time

    performance_data["execution_memory"]["total_memory_mb"] = total_memory
    performance_data["execution_memory"]["layer_memory_total_mb"] = layer_memory

# 模拟一个输入，进行性能测量
dummy_input = torch.randint(0, 50257, (bs, 1024)).long()  # 生成一个随机的输入 (bs, seq_len)
dummy_input = dummy_input.to(device)  # 将输入转移到CUDA设备上

# 生成标签：通常情况下，CausalLM使用右移的输入作为标签
labels = dummy_input.clone()  # 在这里我们将标签设置为输入本身（可以调整为实际需求）

# 将输入和标签一同传递给模型
inputs = {'input_ids': dummy_input, 'labels': labels}


for _ in range(3):  # 预热运行几次
    outputs = model(**inputs)
    outputs.loss.backward()
    model.zero_grad()


dummy_input = torch.randint(0, 50257, (bs, 1024)).long()
labels = dummy_input.clone()
inputs = {'input_ids': dummy_input, 'labels': labels}
# 开始性能测量
measure_performance(model, inputs)

# 保存数据到JSON文件
output_dir = 'profile_data'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"DeviceType.A100_tp{tp}_bs{bs}.json")

with open(output_file, 'w') as f:
    json.dump(performance_data, f, indent=2)

print(f"Performance data saved to {output_file}")

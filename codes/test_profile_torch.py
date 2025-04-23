import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Model, GPT2Config
import torch.profiler

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 GPT-125M（GPT-2 Small）的配置
config = GPT2Config.from_pretrained("gpt2")
model = GPT2Model(config).to(device)

# 生成随机输入（假设 batch_size=2，序列长度=32）
batch_size = 2
seq_length = 32
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)

# 定义优化器和损失函数
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.MSELoss()

# 目标输出（模拟任务）
target = torch.randn(batch_size, seq_length, config.hidden_size).to(device)

# 使用 profiler 进行性能分析
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:

    for step in range(5):
        optimizer.zero_grad()
        
        # 记录前向传播
        with torch.profiler.record_function("forward"):
            outputs = model(input_ids).last_hidden_state
        
        # 计算损失
        loss = criterion(outputs, target)

        # 记录反向传播
        with torch.profiler.record_function("backward"):
            loss.backward()

        optimizer.step()

        # 记录 Profiler
        prof.step()

# 输出 Profile 结果
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

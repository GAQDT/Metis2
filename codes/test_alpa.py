import alpa
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# 定义一个简单的模型
class SimpleModel(nn.Module):
    hidden_size: int
    output_size: int

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.output_size)

    def __call__(self, x):
        x = self.dense1(x)
        x = jax.nn.relu(x)
        x = self.dense2(x)
        return x

# 定义并行策略
tensor_parallel_degree = 2  # 张量并行度
data_parallel_degree = 2   # 数据并行度

parallel_strategy = alpa.ParallelStrategy(
    tensor_parallel_degree=tensor_parallel_degree,
    data_parallel_degree=data_parallel_degree
)

# 模型参数
hidden_size = 128
output_size = 10
batch_size = 32

# 创建模型实例
model = SimpleModel(hidden_size=hidden_size, output_size=output_size)

# 生成一个假的数据加载器
def generate_fake_data(batch_size):
    return jnp.ones((batch_size, hidden_size))

# 定义训练函数
def train_step(model, params, batch):
    def loss_fn(params, batch):
        logits = model.apply(params, batch)
        loss = jnp.mean((logits - jnp.ones_like(logits))**2)  # 简单的MSE损失
        return loss

    grads = jax.grad(loss_fn)(params, batch)
    return grads

# 初始化模型参数
rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((batch_size, hidden_size)))

# 设置优化器
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# 数据并行训练循环
num_epochs = 5

for epoch in range(num_epochs):
    # 模拟一个训练批次
    batch = generate_fake_data(batch_size)

    # 自动并行化模型
    model_parallel = alpa.auto_parallel(model, parallel_strategy)

    # 执行训练步骤
    grads = train_step(model_parallel, params, batch)

    # 更新模型参数
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    print(f"Epoch {epoch+1}/{num_epochs} completed")

# 测试模型的输出
test_batch = generate_fake_data(batch_size)
output = model.apply(params, test_batch)
print("Test output:", output)

from functools import partial

import alpa
from alpa.testing import assert_allclose
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax

class MLPModel(nn.Module):
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            if i % 2 == 0:
                x = nn.Dense(features=self.hidden_dim * 4)(x)
            else:
                x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)
        return x

dim = 2048
batch_size = 2048
num_layers = 10

# Generate ground truth W and b
rngkey = jax.random.PRNGKey(0)
k1, k2 = random.split(rngkey)
W = random.normal(k1, (dim, dim))
b = random.normal(k2, (dim,))

# Generate the training data
ksample, knoise = random.split(k1)
x = random.normal(ksample, (batch_size, dim))
y = (x @ W + b) + 0.1 * random.normal(knoise, (batch_size, dim))

# Initialize a train state, which includes the model paramter and optimizer state.
model = MLPModel(hidden_dim=dim, num_layers=num_layers)
params = model.init(rngkey, x)
tx = optax.adam(learning_rate=1e-3)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def train_step(state, batch):
    def loss_func(params):
        out = state.apply_fn(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    grads = jax.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

batch = {"x": x, "y": y}
expected_state = train_step(state, batch)

# Define the training function and execute one step
@alpa.parallelize
def alpa_train_step(state, batch):
    def loss_func(params):
        out = state.apply_fn(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    grads = jax.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

actual_state = alpa_train_step(state, batch)
assert_allclose(expected_state.params, actual_state.params, atol=5e-3)

print("Input parameter type:", type(state.params["params"]["Dense_0"]["kernel"]))
print("Output parameter type:", type(actual_state.params["params"]["Dense_0"]["kernel"]))

# We can use `np.array` to convert a distributed array back to a numpy array.
kernel_np = np.array(actual_state.params["params"]["Dense_0"]["kernel"])

state = actual_state  # We need this assignment because the original `state` is "donated" and freed.
from alpa.util import benchmark_func

# Benchmark serial execution with jax.jit
jit_train_step = jax.jit(train_step, donate_argnums=(0,))

def sync_func():
    jax.local_devices()[0].synchronize_all_activity()

def serial_execution():
    global state
    state = jit_train_step(state, batch)

costs = benchmark_func(serial_execution, sync_func, warmup=5, number=10, repeat=5) * 1e3
print(f"Serial execution time. Mean: {np.mean(costs):.2f} ms, Std: {np.std(costs):.2f} ms")

# Benchmark parallel execution with alpa
# We distribute arguments in advance for the benchmarking purpose.
state, batch = alpa_train_step.preshard_dynamic_args(state, batch)

def alpa_execution():
    global state
    state = alpa_train_step(state, batch)

alpa_costs = benchmark_func(alpa_execution, sync_func, warmup=5, number=10, repeat=5) * 1e3
print(f"Alpa execution time.   Mean: {np.mean(alpa_costs):.2f} ms, Std: {np.std(alpa_costs):.2f} ms")
GB = 1024 ** 3

executable = jit_train_step.lower(state, batch).compile().runtime_executable()
print(f"Serial execution per GPU memory usage: {executable.total_allocation_size() / GB:.2f} GB")

alpa_executable = alpa_train_step.get_executable(state, batch)
print(f"Alpa execution per GPU memory usage:   {alpa_executable.get_total_allocation_size() / GB:.2f} GB")

@partial(jax.pmap, axis_name="batch")
def pmap_train_step(state, batch):
    def loss_func(params):
        out = model.apply(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    grads = jax.grad(loss_func)(state.params)
    # all-reduce gradients
    grads = jax.lax.pmean(grads, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)
    return new_state

# Replicate model and distribute batch
devices = jax.local_devices()
state = jax.device_put_replicated(state, devices)
def shard_batch(x):
    x = x.reshape((len(devices), -1) + x.shape[1:])
    return jax.device_put_sharded(list(x), devices)
batch = jax.tree_map(shard_batch, batch)

# Benchmark data parallel execution
def data_parallel_execution():
    global state
    state = pmap_train_step(state, batch)

costs = benchmark_func(data_parallel_execution, sync_func, warmup=5, number=10, repeat=5) * 1e3
print(f"Data parallel execution time. Mean: {np.mean(costs):.2f} ms, Std: {np.std(costs):.2f} ms")
print(f"Alpa execution time.          Mean: {np.mean(alpa_costs):.2f} ms, Std: {np.std(alpa_costs):.2f} ms\n")

executable = pmap_train_step.lower(state, batch).compile().runtime_executable()
print(f"Data parallel execution per GPU memory usage: {executable.total_allocation_size() / GB:.2f} GB")
print(f"Alpa execution per GPU memory usage:          {alpa_executable.get_total_allocation_size() / GB:.2f} GB")
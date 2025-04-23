"""Pretrain OPT3"""
import argparse
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path

import time
import json
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

from megatron.core.timers import Timers

from megatron.core import parallel_state
from megatron.core import dist_checkpointing
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.datasets.utils import compile_helpers 
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.training.tokenizer.tokenizer import _NullTokenizer

from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_args_from_checkpoint
from megatron.training.async_utils import init_persistent_async_worker


_SEQUENCE_LENGTH = 1024
bs = 0
model_name = None

##################################################################################

performance_data = {
    "model": {
        "model_name": "",
        "num_layers": 0,
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

layer_times = []
timers = Timers(2,'all')
fbt = None
bet = None

def forward_pre_hook(module, input):
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated() / (1024**2)
    torch.cuda.synchronize()
    timers('forward').start()  # 正确启动计时器
    global fbt 
    if fbt == None:
        fbt = time.time()

    if torch.distributed.get_rank() == 0:
        layer_times.append(('forward_pre', 0, initial_mem))
        layer_name = module.__class__.__name__
        print(f"forward_pre: Layer: {layer_name}")

    return input

def forward_hook(module, input, output):
    torch.cuda.synchronize()
    
    timers('forward').stop()  # 停止前向传播计时
    end_time = timers('forward').elapsed()  # 获取时间

    mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    if torch.distributed.get_rank() == 0:
        layer_times.append(('forward', end_time, mem))
        layer_name = module.__class__.__name__
        print(f"forward: {end_time :.6f}, Layer: {layer_name}")

    return output

def backward_pre_hook(module, grad_output):
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated() / (1024**2)
    torch.cuda.synchronize()
    timers('backward').start()  # 正确启动计时器

    if torch.distributed.get_rank() == 0:
        layer_times.append(('backward_pre', 0, initial_mem))
        layer_name = module.__class__.__name__
        print(f"backward_pre: Layer: {layer_name}")

    return grad_output

def backward_hook(module, grad_input, grad_output):
    torch.cuda.synchronize()

    timers('backward').stop()  # 停止反向传播计时
    end_time = timers('backward').elapsed()
    global bet
    if bet == None:
        bet = time.time()

    mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    if isinstance(module, LanguageModelEmbedding):
        timers('embedding_grads_all_reduce').start()
        #torch.distributed.all_reduce(grad_output[0])
        torch.cuda.synchronize()
        timers('embedding_grads_all_reduce').stop()
        performance_data["execution_time"]["embedding_grads_all_reduce_time_ms"] += timers('embedding_grads_all_reduce').elapsed() * 1000

    if torch.distributed.get_rank() == 0:
        layer_times.append(('backward', end_time, mem))
        print(f"backward: {end_time :.6f}")

    return grad_input

cpl = []
def forward_pre_hook_cpl(module, input):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated() / (1024**2)
    
    timers('forward_cpl').start()  # 正确启动计时器

    if torch.distributed.get_rank() == 0:
        cpl.append(('forward_pre', 0, initial_mem))
        layer_name = module.__class__.__name__
        print(f"forward_pre: Layer: {layer_name}")

    return input

def forward_hook_cpl(module, input, output):
    torch.cuda.synchronize()
    
    timers('forward_cpl').stop()  # 停止前向传播计时
    end_time = timers('forward_cpl').elapsed()  # 获取时间

    mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    if torch.distributed.get_rank() == 0:
        cpl.append(('forward', end_time, mem))
        layer_name = module.__class__.__name__
        print(f"forward: {end_time :.6f}, Layer: {layer_name}")

    return output

def backward_pre_hook_layernorm(module, grad_output):
    torch.cuda.synchronize()
    timers('backward_layernorm').start()  # 正确启动计时器
    return grad_output
def backward_hook_layernorm(module, grad_input, grad_output):
    torch.cuda.synchronize()
    timers('backward_layernorm').stop()  # 停止反向传播计时
    end_time = timers('backward_layernorm').elapsed()
    performance_data["execution_time"]["layernorm_grads_all_reduce_time_ms"] += end_time*1000
    return grad_input

# 注册钩子
def register_hooks(model):
    hooks = []
    registered_ids = set()
    max_estimate = True
    for name, module in model.named_modules():
        print(name)
        if isinstance(module, LanguageModelEmbedding) or isinstance(module, TransformerLayer):
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
        elif name == "output_layer":
            forward_pre_hook_fn = module.register_forward_pre_hook(forward_pre_hook_cpl)
            forward_hook_fn = module.register_forward_hook(forward_hook_cpl)
            hooks.append(forward_pre_hook_fn)
            hooks.append(forward_hook_fn)
        elif "pre_mlp_layernorm" in name:
            backward_pre_hook_fn = module.register_full_backward_pre_hook(backward_pre_hook_layernorm)
            backward_hook_fn = module.register_full_backward_hook(backward_hook_layernorm)
            hooks.append(backward_pre_hook_fn)
            hooks.append(backward_hook_fn)
    return hooks



######################################################################################

def initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=1):
    parallel_state.destroy_model_parallel()

    # Torch setup for distributed training
    rank = int(os.environ['LOCAL_RANK'])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)


    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)

def model_provider():
    """Build the model."""
    global bs,model_name

    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    args = parse_args(None, False)
    if args.ckpt_convert_format is not None:
        assert args.ckpt_convert_save is not None
        assert args.load is not None
        args.exit_on_missing_checkpoint = True

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        assert args.load is not None, "--use-checkpoint-args requires --load argument"
        assert args.non_persistent_ckpt_type != "local", (
            "--use-checkpoint-args is not supported with --non_persistent_ckpt_type=local. "
            "Two-stage checkpoint loading is not implemented, and all arguments must be defined "
            "before initializing LocalCheckpointManager."
        )
        load_args_from_checkpoint(args)

    if args.async_save and args.use_persistent_ckpt_worker:
        init_persistent_async_worker()

    validate_args(args, args_defaults)
    args.gradient_accumulation_fusion = False
    bs = args.micro_batch_size
    if args.num_layers == 8 and args.num_experts == 8:
        model_name = "MoE-380M"
    elif args.num_layers == 8 and args.num_experts == 16:
        model_name = "MoE-690M"
    elif args.num_layers == 16 and args.num_experts == 16:
        model_name = "MoE-1.3B"

    from megatron.training.arguments import core_transformer_config_from_args
    transformer_config = core_transformer_config_from_args(args)

    gpt_model = GPTModel(
        config=transformer_config, 
        transformer_layer_spec=get_gpt_decoder_block_spec(transformer_config, use_transformer_engine=True, normalization='RMSNorm'), 
        vocab_size=32000, 
        max_sequence_length=1024,
        pre_process=True,
        post_process=True,
        fp16_lm_cross_entropy=False,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type='rope',
        rotary_percent=1.0,
        rotary_base=10000,
        rope_scaling=False
    )

    return gpt_model

def get_train_data_iterator(batch_size):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_NullTokenizer(vocab_size=_SEQUENCE_LENGTH),
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()

    train_dataloader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True)

    train_iterator = iter(train_dataloader)

    return train_iterator

def forward_step_func(data_iterator, model):

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups.
        # If pipeline parallel, loss computation is done only in last stage.

        return loss, {'lm loss': loss}
    
    torch.cuda.synchronize()
    batch_start_time = time.time()
    data = next(data_iterator)
    torch.cuda.synchronize()
    batch_end_time = time.time()
    if torch.distributed.get_rank() == 0:
        performance_data["execution_time"]["batch_generator_time_ms"] = (batch_end_time - batch_start_time) * 1000

    tokens = data['tokens'].to(device)
    attention_mask = data['attention_mask'].to(device)
    position_ids = data['position_ids'].to(device)
    labels = data['labels'].to(device)
    loss_mask = data['loss_mask'].to(device)

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)

def save_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix='')
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)

def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict=gpt_model.sharded_state_dict(prefix='')
    checkpoint = dist_checkpointing.load(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)
    gpt_model.load_state_dict(checkpoint)
    return gpt_model


def get_model_parameters(model):
    import re
    parameters_per_layer = []
    total_parameters = 0
    curr = 0
    state = -1
    index = -1
    for name, param in model.named_parameters():
        if state == -1:
            curr = param.numel() * param.element_size()
            parameters_per_layer.append(curr)
            curr = 0
            total_parameters += parameters_per_layer[-1]
            state = 0
        elif state == 0 and "layers" in name:
            match = re.search(r'\d+', name)
            index = int(match.group())
            curr += param.numel() * param.element_size()
            state = 1
        elif state == 1 and "layers" in name:
            match = re.search(r'\d+', name)
            if index == int(match.group()):
                curr += param.numel() * param.element_size()
            else:
                parameters_per_layer.append(curr)
                total_parameters += parameters_per_layer[-1]
                curr = param.numel() * param.element_size()
                index = int(match.group())
        elif state == 1 and "layers" not in name:
            parameters_per_layer.append(curr)
            total_parameters += parameters_per_layer[-1]
            curr = param.numel() * param.element_size()
            state = 2
        elif state == 2:
            curr += param.numel() * param.element_size()
    parameters_per_layer.append(curr)
    total_parameters += parameters_per_layer[-1]
    performance_data["model"]["num_layers"] = index+1

    return total_parameters, parameters_per_layer

if __name__ == "__main__":
    tp = 2
    initialize_distributed(tensor_model_parallel_size=tp, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    gpt_model = model_provider()
    device = torch.device("cuda")
    gpt_model.to(device)

    performance_data["model"]["model_name"]=model_name

    total_parameters, parameters_per_layer = get_model_parameters(gpt_model)
    performance_data["model"]["parameters"]["total_parameters_bytes"] = total_parameters * tp
    for i in range(0,len(parameters_per_layer),1):
        parameters_per_layer[i] *= tp
    performance_data["model"]["parameters"]["parameters_per_layer_bytes"] = parameters_per_layer

    optim = Adam(gpt_model.parameters())

    train_iterator = get_train_data_iterator(batch_size = bs)
 
    forward_backward_func = get_forward_backward_func()
    for name, param in gpt_model.named_parameters():
        print(f"{name} is on device {param.device}, shape: {param.shape}")

    def print_module_details(module, indent=0):
        # 输出当前模块的名称和类型
        print(" " * indent + f"Module: {module.__class__.__name__}")
        
        # 遍历子模块
        for name, child in module.named_children():
            print(" " * (indent + 2) + f"Submodule: {name} ({child.__class__.__name__})")
            print_module_details(child, indent + 4)  # 递归遍历子模块'''

    # 示例调用
    print_module_details(gpt_model)

    for name, param in gpt_model.named_parameters():
        print(name,param.numel() * param.element_size())
    # Running the model for 5 iterations
    fb_begin_time = None
    fb_end_time = None
    iter_begin_time = None
    iter_end_time = None
    for i in range(0,5,1):
        if i == 3 and torch.distributed.get_rank() == 0:
            iter_begin_time = time.time()
        optim.zero_grad()

        if i == 3 and torch.distributed.get_rank() == 0:
            fb_begin_time=time.time()
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=1,
            seq_length=_SEQUENCE_LENGTH,
            micro_batch_size=bs,
            decoder_seq_length=_SEQUENCE_LENGTH,
            forward_only=False)
        if i == 3 and torch.distributed.get_rank() == 0:
            fb_end_time=time.time()
            timers('optim').start()
        optim.step()
        if i == 3 and torch.distributed.get_rank() == 0:
            timers('optim').stop()
            performance_data["execution_time"]["optimizer_time_ms"] += (timers('optim').elapsed()) * 1000  # 转换为毫秒
            iter_end_time = time.time()


        print(f'Losses reduced :  {losses_reduced}')
        if i == 3 and torch.distributed.get_rank() == 0:
            hooks = register_hooks(gpt_model)

    # Saving the model
    #ckpt_path = os.getcwd() + '/ckpt'
    #Path(ckpt_path).mkdir(exist_ok=True)
    #save_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)

    # Loading the model
    '''gpt_model = load_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)
    gpt_model.to(device)
    print('Successfully loaded the model')'''
    print(len(layer_times))

#######################################################################################
    if torch.distributed.get_rank() == 0: 
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

        total_time = (iter_end_time - iter_begin_time) * 1000  # 转换为毫秒
        #forward_backward_time = (fb_end_time - fb_begin_time) * 1000
        forward_backward_time = (bet - fbt) * 1000

        layer_compute_time.append((cpl[1][1]-cpl[0][1] + (backward_times[mid-1][1]-backward_times[mid-2][1])/(forward_times[1][1]-forward_times[0][1])*(cpl[1][1]-cpl[0][1])) * 1000)
        layer_memory.append((cpl[1][2]-cpl[0][2] + (backward_times[mid-1][2]-backward_times[mid-2][2])/(forward_times[1][2]-forward_times[0][2])*(cpl[1][2]-cpl[0][2])))

        performance_data["execution_time"]["total_time_ms"] = total_time
        performance_data["execution_time"]["forward_backward_time_ms"] = forward_backward_time
        performance_data["execution_time"]["layer_compute_total_ms"] = layer_compute_time
        performance_data["execution_memory"]["total_memory_mb"] = total_memory
        performance_data["execution_memory"]["layer_memory_total_mb"] = layer_memory

        output_dir = "/workspace/codes/moe_profile_data/"+model_name
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"DeviceType.H800_tp{tp}_bs{bs}.json")
        with open(output_file, 'w') as f:
            json.dump(performance_data, f, indent=2)
        print(f"Performance data saved to {output_file}")


#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/workspace #<Specify path>
TENSORBOARD_LOGS_PATH=/workspace #<Specify path>
VOCAB_FILE=/workspace #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=/workspace #<Specify path to file>/gpt2-merges.txt
DATA_PATH=/workspace #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2 
	--pipeline-model-parallel-size 1 
)

model_names=("125M" "350M" "760M" "1.3B")
num_layers_list=(12 24 24 24)
num_attention_heads_list=(12 16 16 32)
hidden_size_list=(768 1024 1536 2048)

batch_sizes=(1 2 4 8)

for i in "${!model_names[@]}"; do
    model_name="${model_names[i]}"
    num_layers="${num_layers_list[i]}"
    num_attention_heads="${num_attention_heads_list[i]}"
    hidden_size="${hidden_size_list[i]}"

    for bs in "${batch_sizes[@]}"; do

        GPT_MODEL_ARGS=(
            --num-layers "$num_layers"
            --hidden-size "$hidden_size" 
            --num-attention-heads "$num_attention_heads"
            --seq-length 1024
            --max-position-embeddings 1024 
            --attention-backend auto # Can use (flash/fused/unfused/local)
        )

        TRAINING_ARGS=(
            --micro-batch-size "$bs"
            --global-batch-size "$bs"
            --train-iters 500000 
            --weight-decay 0.1 
            --adam-beta1 0.9 
            --adam-beta2 0.95 
            --init-method-std 0.006 
            --clip-grad 1.0 
            --fp16
            --lr 6.0e-5 
            --lr-decay-style cosine 
            --min-lr 6.0e-6
            --lr-warmup-fraction .001 
            --lr-decay-iters 430000 
        )

        echo "Training Model: $model_name, Batch Size: $bs"
        PYTHONPATH=$PYTHONPATH:./megatron torchrun ${DISTRIBUTED_ARGS[@]} ../codes/test_profile_gpt.py \
            ${GPT_MODEL_ARGS[@]} \
            ${TRAINING_ARGS[@]} \
            ${MODEL_PARALLEL_ARGS[@]}
    done
done


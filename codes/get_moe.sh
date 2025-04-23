#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=1
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/workspace
TOKENIZER_MODEL=/workspace
DATA_PATH=/workspace

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --data-path "$DATA_PATH"
    --split 99990,8,2
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
)

# Model configurations
model_names=("380M" "690M" "1.3B")
num_layers_list=(8 8 16)
num_experts_list=(8 16 16)

# Batch sizes to loop through
batch_sizes=(1 2 4 8)

for i in "${!model_names[@]}"; do
    model_name="${model_names[i]}"
    num_layers="${num_layers_list[i]}"
    num_experts="${num_experts_list[i]}"
    ffn_hidden_size="${ffn_hidden_size_list[i]}"

    for bs in "${batch_sizes[@]}"; do

        # Dynamic model arguments
        MODEL_ARGS=(
            --disable-bias-linear
            --seq-length 1024
            --max-position-embeddings 1024
            --num-layers "$num_layers"
            --hidden-size 768
            --ffn-hidden-size 6144
            --num-attention-heads 16
            --init-method-std 0.01
            --attention-dropout 0.0
            --hidden-dropout 0.0
            --normalization RMSNorm
            --position-embedding-type rope
            --swiglu
            --untie-embeddings-and-output-weights
            --group-query-attention
            --num-query-groups 8
            --no-masked-softmax-fusion
            --no-position-embedding
        )

        # MoE arguments
        MOE_ARGS=(
            --num-experts "$num_experts"
            --expert-model-parallel-size 1
            --moe-router-load-balancing-type aux_loss
            --moe-router-topk 2
            --moe-aux-loss-coeff 1e-2
            #--moe-grouped-gemm
        )

        # Training arguments with dynamic batch size
        TRAINING_ARGS=(
            --micro-batch-size "$bs"
            --global-batch-size "$bs"
            --lr 1e-4
            --train-iters 1        # Adjust this for actual training
            --lr-decay-iters 1
            --lr-decay-style cosine
            --min-lr 1.0e-5
            --weight-decay 0.1
            --lr-warmup-iters 3
            --clip-grad 1.0
            --fp16
            #--bf16
            --overlap-grad-reduce
            --overlap-param-gather
        )

        # Logging and checkpointing
        LOGGING_ARGS=(
            --log-interval 1
            --save-interval 10000
            --eval-interval 1000
            --eval-iters 10
            --save "$CHECKPOINT_PATH"
            --load "$CHECKPOINT_PATH"
            --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
            --no-load-optim
            --no-load-rng
        )

        # Wandb configuration
        if [ -n "${WANDB_API_KEY}" ]; then
            LOGGING_ARGS+=(
                --wandb-project "${WANDB_PROJECT:-"Mixtral-Finetuning"}"
                --wandb-exp-name "Mixtral_${model_name}_bs${bs}"
            )
        fi

        echo "Training Model: $model_name, Batch Size: $bs"
        PYTHONPATH=$PYTHONPATH:./megatron torchrun "${DISTRIBUTED_ARGS[@]}" ../codes/test_profile_moe.py \
            "${MODEL_ARGS[@]}" \
            "${MOE_ARGS[@]}" \
            "${DATA_ARGS[@]}" \
            "${TRAINING_ARGS[@]}" \
            "${MODEL_PARALLEL_ARGS[@]}" \
            "${LOGGING_ARGS[@]}"
    done
done
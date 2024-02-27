#!/bin/bash

# > Default arguments - can be overriden by environment variables:
# architecture to train, must be compatible with the Llama architecture
arch=${ARCH:-princeton-nlp/Sheared-LLaMA-1.3b}
# total batch size across all devices with gradient accumulation
bsz=${BSZ:-2048}
# number of sequences per device
seq=${SEQ:-16}
# peak learning rate
lr=${LR:-5e-4}
# number of epochs
epochs=${EPOCHS:-1}
# warmup ratio
warmup=${WARMUP:-0.05}
# save model every n steps
save_steps=${SAVE:-1000}
# path to dataset to train on
dataset=${DATASET:-""}
# suffix to append to run name
suffix=${SUFFIX:-""}

run_name="lm_$(basename $arch)_bsz${bsz}_lr${lr}to10pc_epochs${epochs}_warmup${warmup}_dataset$(basename $dataset)${suffix}"
out_dir="checkpoints/$run_name"
mkdir -p $out_dir

nvidia-smi

# Determine available number of GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")
fi
num_gpus=${NUM_GPUS:-$num_gpus}

# Determine number of nodes if run inside slurm job job
num_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
if [ $num_nodes == 0 ]; then
    num_nodes=1
fi
num_nodes=${NUM_NODES:-$num_nodes}

if [ $num_nodes -gt 1 ]; then
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    master_addr=${MASTER_ADDR:-$master_addr}
    master_port=${MASTER_PORT:-56321}

    header="srun torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$master_addr:$master_port \
    --nnodes=$num_nodes \
    --nproc-per-node=$num_gpus \
    -m training.train_language_model"
else
    # Find a free port at random
    master_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
    master_port=${master_port:-56321}
    master_port=${MASTER_PORT:-$master_port}

    header="torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:$master_port \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    -m training.train_language_model"
fi

export OMP_NUM_THREADS=$num_gpus

export WANDB_PROJECT="qurating"
export WANDB_DIR=$out_dir
export WANDB_MODE="offline"

export FSDP_SHARDING_STRATEGY="5" # 5 corresponds to _hybrid_shard_zero2
export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"

base_arguments=(
    --report_to wandb

    --do_train
    --config_name $arch
    --config_overrides ""
    --tokenizer_name $arch

    --run_name $run_name
    --output_dir $out_dir
    --log_level info
    --logging_steps 1
    --disable_tqdm true
    --save_steps $save_steps
    --cache_dir .cache
    --overwrite_output_dir
    --dataloader_num_workers 8

    --num_train_epochs $epochs
    --per_device_train_batch_size $seq
    --gradient_accumulation_steps $(($bsz / $seq / $num_gpus / $num_nodes))
    --learning_rate $lr
    --lr_scheduler_type cosine
    --min_lr_ratio 0.1
    --max_grad_norm 1.0
    --adam_beta1 0.9
    --adam_beta2 0.95
    --weight_decay 0.1
    --warmup_ratio $warmup
    #--optim adamw_torch_fused

    --bf16
    --bf16_full_eval
    --fsdp auto_wrap
    --ddp_find_unused_parameters false

    # Depending on model size and sequence length, gradient checkpointing might result in higher throughput
    #--gradient_checkpointing

    --tokenized_train_dataset $dataset

    $@
)

echo command: "${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out

#!/bin/bash

# > Default arguments - can be overriden by environment variables:
# architecture to train, must be compatible with the Llama architecture
model=${MODEL:-princeton-nlp/Sheared-LLaMA-1.3b}
# total batch size across all devices with gradient accumulation
bsz=${BSZ:-512}
# number of sequences per device
seq=${SEQ:-16}
# peak learning rate
lr=${LR:-5e-5}
# number of epochs
epochs=${EPOCHS:-2}
# warmup ratio
warmup=${WARMUP:-0.1}
# save model every n steps
save_steps=${SAVE:-200}
# suffix to append to run name
suffix=${SUFFIX:-""}
# only predict labels with certain confidence
confidence=${CONFIDENCE:-0.5}
# temperature applied to labels
labeltemp=${LABELTEMP:-1.0}
# which labels to predict
label_index=${LABELINDEX:-"all"}

run_name="qurater_$(basename $model)_bsz${bsz}_lr${lr}_epochs${epochs}_warmup${warmup}_conf${confidence}_labeltemp${labeltemp}${suffix}"
out_dir="checkpoints-preferences/$run_name"
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
    --rdzv-endpoint=$master_addr:56321 \
    --nnodes=$num_nodes \
    --nproc-per-node=$num_gpus \
    -m training.train_preference_model"
else
    master_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
    master_port=${master_port:-56321}
    master_port=${MASTER_PORT:-$master_port}

    header="torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:$master_port \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    -m training.train_preference_model"
fi

export OMP_NUM_THREADS=$num_gpus

export WANDB_PROJECT="lm-data-selection"
export WANDB_DIR=$out_dir
export WANDB_MODE="offline"

export FSDP_SHARDING_STRATEGY="5" # 5 corresponds to _hybrid_shard_zero2
export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"

labels=(
    writing_style_average
    required_expertise_average
    facts_and_trivia_average
    educational_value_average
)
if [[ $label_index == "all" ]]; then
    label_field="${labels[@]}"
else
    label_field="${labels[$label_index]}"
fi

base_arguments=(
    --report_to wandb

    --do_eval
    --do_train
    --model_name_or_path $model
    --config_name $model
    --config_overrides ""
    --tokenizer_name $model

    --run_name $run_name
    --output_dir $out_dir
    --log_level info
    --logging_steps 1
    --disable_tqdm true
    --save_steps $save_steps
    --evaluation_strategy steps
    --eval_steps $save_steps
    --load_best_model_at_end true
    --metric_for_best_mode eval_validation_acc
    --greater_is_better true
    --dataloader_num_workers 2
    --cache_dir .cache
    --overwrite_output_dir
    --remove_unused_columns false
    --use_fast_tokenizer false

    --num_train_epochs $epochs
    --max_length 2048
    --per_device_train_batch_size $seq
    --gradient_accumulation_steps $(($bsz / $seq / $num_gpus / $num_nodes))
    --learning_rate $lr
    --max_grad_norm 1.0
    --weight_decay 0.1
    --warmup_ratio $warmup

    --bf16_full_eval
    --bf16
    --ddp_find_unused_parameters false
    --fsdp auto_wrap

    # Depending on model size and sequence length, gradient checkpointing might result in higher throughput
    # --gradient_checkpointing

    --label_field $label_field
    --confidence_threshold $confidence
    --label_temperature $labeltemp

    --train_datasets princeton-nlp/QuRating-GPT3.5-Judgments
    --eval_split_size_train 0.1

    --eval_datasets princeton-nlp/QuRating-GPT3.5-Judgments-Test
    --eval_split_size 1.0

    $@
)

echo command: "${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out

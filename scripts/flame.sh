#!/bin/bash

MODE="flame"
DATASET="cifar100"
# DATASET="imagenet-r"
# DATASET="tinyimagenet"

N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
WANDB="--wandb"

if [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=0; ONLINE_ITER=3
    MODEL_NAME="vit"; EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3; OPT_NAME="sgd"; SCHED_NAME="default"
    LORA_RANK=16; LORA_ALPHA=32; ADAPTER_TARGETS="qkv,proj,fc1,fc2"; COSINE_SCALE=20.0
    CA_LR=0.005; CA_EPOCHS=10; CA_SAMPLES=256
    SHRINK_K=10.0
elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=0; ONLINE_ITER=3
    MODEL_NAME="vit"; EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3; OPT_NAME="sgd"; SCHED_NAME="default"
    LORA_RANK=16; LORA_ALPHA=32; ADAPTER_TARGETS="qkv,proj,fc1,fc2"; COSINE_SCALE=20.0
    CA_LR=0.005; CA_EPOCHS=10; CA_SAMPLES=256
    SHRINK_K=10.0
elif [ "$DATASET" == "imagenet-r" ]; then
    MEM_SIZE=0; ONLINE_ITER=3
    MODEL_NAME="vit"; EVAL_PERIOD=1000
    BATCHSIZE=64; LR=1e-2; OPT_NAME="sgd"; SCHED_NAME="default"
    LORA_RANK=16; LORA_ALPHA=32; ADAPTER_TARGETS="qkv,proj,fc1,fc2"; COSINE_SCALE=20.0
    CA_LR=0.005; CA_EPOCHS=10; CA_SAMPLES=256
    SHRINK_K=10.0
else
    echo "Undefined setting"
    exit 1
fi

NOTE="FLAME"

for seed in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES="0" python main.py --mode $MODE \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --rnd_seed $seed \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM $USE_AMP --online_iter $ONLINE_ITER --data_dir local_datasets \
    --note $NOTE --eval_period $EVAL_PERIOD --n_worker 4 --rnd_NM \
    --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA --adapter_targets "$ADAPTER_TARGETS" --cosine_scale $COSINE_SCALE \
    --ca_lr $CA_LR --ca_epochs $CA_EPOCHS --ca_samples $CA_SAMPLES \
    --shrink_k $SHRINK_K $WANDB \
    --wandb_project "cifar100-flame"
done
wait

#!/bin/bash
# Run InfLoRA baseline (faithful re-implementation adapted to Si-Blurry online
# setting) on CIFAR-100 / ImageNet-R / Tiny-ImageNet.
#
# Usage:
#   ./scripts/inflora.sh                       # defaults: cifar100, seeds 1..5
#   DATASET=imagenet-r SEEDS="1 2 3" ./scripts/inflora.sh

MODE="inflora"
DATASET="${DATASET:-cifar100}"

N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
WANDB="--wandb"

SEEDS="${SEEDS:-1 2 3 4 5}"

if [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=0; ONLINE_ITER=3
    MODEL_NAME="vit"; EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3; OPT_NAME="sgd"; SCHED_NAME="default"
    LORA_RANK=16; LORA_ALPHA=32; ADAPTER_TARGETS="qkv,proj,fc1,fc2"; COSINE_SCALE=20.0
elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=0; ONLINE_ITER=3
    MODEL_NAME="vit"; EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3; OPT_NAME="sgd"; SCHED_NAME="default"
    LORA_RANK=16; LORA_ALPHA=32; ADAPTER_TARGETS="qkv,proj,fc1,fc2"; COSINE_SCALE=20.0
elif [ "$DATASET" == "imagenet-r" ]; then
    MEM_SIZE=0; ONLINE_ITER=3
    MODEL_NAME="vit"; EVAL_PERIOD=1000
    BATCHSIZE=64; LR=1e-2; OPT_NAME="sgd"; SCHED_NAME="default"
    LORA_RANK=16; LORA_ALPHA=32; ADAPTER_TARGETS="qkv,proj,fc1,fc2"; COSINE_SCALE=20.0
else
    echo "Undefined dataset: $DATASET"
    exit 1
fi

INFLORA_WARMUP=256
INFLORA_SVD_SAMPLES=2048
INFLORA_M_ENERGY=0.99
INFLORA_CALIB_CAP=2048
INFLORA_CALIB_PER_BATCH=16

NOTE="InfLoRA"

for seed in $SEEDS
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
    --inflora_warmup $INFLORA_WARMUP \
    --inflora_svd_samples $INFLORA_SVD_SAMPLES \
    --inflora_m_energy $INFLORA_M_ENERGY \
    --inflora_calib_cap $INFLORA_CALIB_CAP \
    --inflora_calib_per_batch $INFLORA_CALIB_PER_BATCH \
    $WANDB --wandb_project "${DATASET}-inflora"
done
wait

#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=AllegroThickScrewDriver headless=True seed=${SEED} \
experiment=rl \
train.algo=PPO \
task.env.reset_dist_threshold=0.1 \
wandb_activate=False \
train.ppo.output_name=AllegroThickScrewDriver_teacher/${CACHE} \
${EXTRA_ARGS}
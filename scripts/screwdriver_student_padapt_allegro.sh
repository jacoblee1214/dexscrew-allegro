#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=AllegroHoraScrewDriver headless=True seed=${SEED} \
train.algo=ProprioAdapt \
train.ppo.proprio_adapt=True train.ppo.output_name=AllegroHoraScrewDriver_student_padapt/${CACHE} \
experiment=student_sim \
task.env.randomization.obs_noise_t_scale=0.01 \
task.env.randomization.obs_noise_e_scale=0.02 \
task.env.numEnvs=48 \
wandb_activate=False \
task.env.reset_dist_threshold=0.12 \
checkpoint=last.pth \
${EXTRA_ARGS}
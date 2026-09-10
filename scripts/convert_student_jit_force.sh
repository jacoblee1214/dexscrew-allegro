#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3
# JIT output file name
JIT_NAME=${4:-screwdriver.pt} # default to screwdriver.pt

array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

JIT_OUTPUT_NAME=${JIT_NAME} CUDA_VISIBLE_DEVICES=${GPUS} \
python student_eval.py task=AllegroForceScrewDriver headless=True seed=${SEED} \
task.env.numEnvs=3 test=True \
task.env.object.type=screw_driver \
train.algo=ProprioAdapt \
task.env.rotation_axis=-z \
train.ppo.proprio_adapt=True \
wandb_activate=False \
task.env.reset_dist_threshold=0.12 \
train.load_path=model_best.ckpt \
${EXTRA_ARGS}
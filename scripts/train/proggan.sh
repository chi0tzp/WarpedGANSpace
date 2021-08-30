#!/bin/bash

GPUS="1,2"

# === Experiment configuration ===
gan_type="ProgGAN"
learn_alphas=false
learn_gammas=true
num_support_sets=128
num_support_dipoles=32
eps=0.1
max_steps=10
reconstructor_type="ResNet"
batch_size=8
max_iter=200000
tensorboard=true


# === Run training script ===
learn_a=""
if $learn_alphas ; then
  learn_a="--learn-alphas"
fi

learn_g=""
if $learn_gammas ; then
  learn_g="--learn-gammas"
fi

tb=""
if $tensorboard ; then
  tb="--tensorboard"
fi

CUDA_VISIBLE_DEVICES=${GPUS} python train.py $tb \
                                             --gan-type=${gan_type} \
                                             --reconstructor-type=${reconstructor_type} \
                                             $learn_a \
                                             $learn_g \
                                             --num-support-sets=${num_support_sets} \
                                             --num-support-dipoles=${num_support_dipoles} \
                                             --eps=${eps} \
                                             --max-steps=${max_steps} \
                                             --batch-size=${batch_size} \
                                             --max-iter=${max_iter} \
                                             --log-freq=10 \
                                             --ckp-freq=100

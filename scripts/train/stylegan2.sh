#!/bin/bash

GPUS="2,3"

# === Experiment configuration ===
gan_type="StyleGAN2"
stylegan2_resolution=1024
learn_alphas=false
learn_gammas=true
num_support_sets=200
num_support_dipoles=512
min_shift_magnitude=0.25
max_shift_magnitude=0.45
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
                                             --stylegan2-resolution=${stylegan2_resolution} \
                                             --stylegan2-w-shift \
                                             --reconstructor-type=${reconstructor_type} \
                                             $learn_a \
                                             $learn_g \
                                             --num-support-sets=${num_support_sets} \
                                             --num-support-dipoles=${num_support_dipoles} \
                                             --min-shift-magnitude=${min_shift_magnitude} \
                                             --max-shift-magnitude=${max_shift_magnitude} \
                                             --batch-size=${batch_size} \
                                             --max-iter=${max_iter} \
                                             --log-freq=10 \
                                             --ckp-freq=100

#!/bin/bash


# === Experiment configuration ===
gan_type="StyleGAN2"
stylegan2_resolution=1024
z_truncation=0.7
w_space=true
learn_alphas=false
learn_gammas=true
num_support_sets=200
num_support_dipoles=512
min_shift_magnitude=0.1
max_shift_magnitude=0.2
reconstructor_type="ResNet"
batch_size=14
max_iter=150000
tensorboard=true
# ================================


# Run training script
stylegan2_w_space=""
if $w_space ; then
  stylegan2_w_space="--stylegan2-w-space"
fi

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

python train.py $tb \
                --gan-type=${gan_type} \
                --z-truncation=${z_truncation} \
                --stylegan2-resolution=${stylegan2_resolution} \
                $stylegan2_w_space \
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

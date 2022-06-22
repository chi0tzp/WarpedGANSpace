#!/bin/bash

# === Experiment configuration ===
gan="stylegan2_ffhq1024"
stylegan_space="W"
stylegan_layer=11
truncation=0.7
learn_alphas=false
learn_gammas=true
beta=0.5
num_support_sets=32
num_support_dipoles=16
min_shift_magnitude=0.1
max_shift_magnitude=0.2
batch_size=12
max_iter=150000
# ================================


# Run training script
learn_a=""
if $learn_alphas ; then
  learn_a="--learn-alphas"
fi

learn_g=""
if $learn_gammas ; then
  learn_g="--learn-gammas"
fi


python train.py --gan=${gan} \
                --truncation=${truncation} \
                --stylegan-space=${stylegan_space} \
                --stylegan-layer=${stylegan_layer} \
                $learn_a \
                $learn_g \
                --beta=${beta} \
                --num-support-sets=${num_support_sets} \
                --num-support-dipoles=${num_support_dipoles} \
                --min-shift-magnitude=${min_shift_magnitude} \
                --max-shift-magnitude=${max_shift_magnitude} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=10 \
                --ckp-freq=100

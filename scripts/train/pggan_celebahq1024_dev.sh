#!/bin/bash

# === Experiment configuration ===
gan="pggan_celebahq1024"
learn_alphas=false
learn_gammas=true
beta=0.5
num_support_sets=8
num_support_dipoles=1
min_shift_magnitude=0.1
max_shift_magnitude=0.2
batch_size=2
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
                --support-set-lr=1e-3 \
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

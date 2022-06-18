#!/bin/bash

# === Experiment configuration ===
gan="pggan_celebahq1024"
learn_alphas=false
learn_gammas=true
num_support_sets=100
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

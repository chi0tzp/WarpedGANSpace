#!/bin/bash

# === Configuration ===
pool="SNGAN_MNIST_10"
eps=0.2
shift_steps=16
shift_leap=1
# =====================

declare -a EXPERIMENTS=("experiments/complete/SNGAN_MNIST-LeNet-K64-D128-LearnGammas-eps0.15_0.25")

for exp in "${EXPERIMENTS[@]}"
do
  # Traverse latent space
  python traverse_latent_space.py -v --gif \
                                  --exp="${exp}" \
                                  --pool=${pool} \
                                  --eps=${eps} \
                                  --shift-steps=${shift_steps} \
                                  --shift-leap=${shift_leap}
done

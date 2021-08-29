#!/bin/bash

pool="StyleGAN2_6"
eps=0.15
shift_steps=16
batch_size=5

declare -a EXPERIMENTS=("experiments/complete/StyleGAN2-1024-W-ResNet-K200-N512-LearnGammas-eps0.1_0.2")

for exp in "${EXPERIMENTS[@]}"
do
  # Traverse latent space
  python traverse_latent_space.py -v --gif --exp="${exp}" --pool=${pool} --eps=${eps} --shift-steps=${shift_steps} --batch-size=${batch_size}
  # Traverse attribute space
  python traverse_attribute_space.py -v --exp="${exp}" --pool=${pool} --eps=${eps} --shift-steps=${shift_steps}
  # Rank interpretable paths
  python rank_interpretable_paths.py -v --exp="${exp}" --pool=${pool} --eps=${eps} --shift-steps=${shift_steps}
done

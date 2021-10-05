#!/bin/bash

pool="ProgGAN_1"
eps=0.15
shift_steps=24
shift_leap=5
batch_size=8

declare -a EXPERIMENTS=("experiments/complete/ProgGAN-ResNet-K200-D512-LearnGammas-eps0.1_0.2")

for exp in "${EXPERIMENTS[@]}"
do
  # Traverse latent space
  # python traverse_latent_space.py -v --gif --exp="${exp}" --pool=${pool} --eps=${eps} --shift-steps=${shift_steps} --shift-leap=${shift_leap} --batch-size=${batch_size}
  # Traverse attribute space
  python traverse_attribute_space.py -v --exp="${exp}" --pool=${pool} --eps=${eps} --shift-steps=${shift_steps}
  # Rank interpretable paths
  # python rank_interpretable_paths_bkp.py -v --exp="${exp}" --pool=${pool} --eps=${eps} --shift-steps=${shift_steps}
done

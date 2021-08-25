#!/bin/bash

pool="ProgGAN_6"
eps=0.2
shift_steps=24
batch_size=4

declare -a EXPERIMENTS=("experiments/complete/ProgGAN-ResNet-K128-N1-LearnGammas-eps0.15_0.25"
                        "experiments/complete/ProgGAN-ResNet-K128-N32-LearnGammas-eps0.15_0.25"
                        "experiments/complete/ProgGAN-ResNet-K128-N64-LearnGammas-eps0.15_0.25"
                        "experiments/complete/ProgGAN-ResNet-K128-N128-LearnGammas-eps0.15_0.25"
                        "experiments/complete/ProgGAN-ResNet-K128-N256-LearnGammas-eps0.15_0.25"
                        "experiments/complete/ProgGAN-ResNet-K128-N512-LearnGammas-eps0.15_0.25"
                        "experiments/complete/ProgGAN-ResNet-K200-N512-LearnGammas-eps0.15_0.25")

for exp in "${EXPERIMENTS[@]}"
do
  # Traverse latent space
  python traverse_latent_space.py -v --gif --exp="${exp}" --pool=${pool} --eps=${eps} --shift-steps=${shift_steps} --batch-size=${batch_size}
  # Traverse attribute space
  python traverse_attribute_space.py -v --exp="${exp}" --pool=${pool} --eps=${eps} --shift-steps=${shift_steps}
  # Rank interpretable paths
  python rank_interpretable_paths.py -v --exp="${exp}" --pool=${pool} --eps=${eps} --shift-steps=${shift_steps}
done

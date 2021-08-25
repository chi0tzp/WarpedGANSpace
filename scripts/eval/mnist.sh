#!/bin/bash

pool="SNGAN_MNIST_10"
eps=0.2
shift_steps=16

declare -a EXPERIMENTS=("experiments/complete/SNGAN_MNIST-LeNet-K64-N1-LearnGammas-eps0.15_0.25"
                        "experiments/complete/SNGAN_MNIST-LeNet-K64-N16-LearnGammas-eps0.15_0.25"
                        "experiments/complete/SNGAN_MNIST-LeNet-K64-N32-LearnGammas-eps0.15_0.25"
                        "experiments/complete/SNGAN_MNIST-LeNet-K64-N64-LearnGammas-eps0.15_0.25"
                        "experiments/complete/SNGAN_MNIST-LeNet-K64-N128-LearnGammas-eps0.15_0.25")

for exp in "${EXPERIMENTS[@]}"
do
  # Traverse latent space
  python traverse_latent_space.py -v --gif --exp="${exp}" --pool=${pool} --eps=${eps} --shift-steps=${shift_steps}
done

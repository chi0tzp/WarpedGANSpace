#!/bin/bash

# === Configuration ===
pool="stylegan2_ffhq1024-1"
eps=0.15
shift_steps=80
shift_leap=5
batch_size=6
gif_size=256
num_imgs=5
metric="corr+corr_l1"
# =====================

# Define experiment directories list
declare -a EXPERIMENTS=("experiments/complete/WarpedGANSpace_stylegan2_ffhq1024-W+-K100-D16-LearnGammas-eps0.1_0.2")

# Define attribute groups (see `rank_interpretable_paths.py`)
declare -a ATTRIBUTE_GROUPS=("Age-FareFace"
                             "Age-CelebA"
                             "Gender"
                             "Rotation"
                             "Smiling-AU12"
                             "Smiling-CelebA")

# Traverse latent and attribute spaces, and rank interpretable paths for the given experiments
for exp in "${EXPERIMENTS[@]}"
do
  # --- Traverse latent space ---------------------------------------------------------------------------------------- #
#  python traverse_latent_space.py -v --gif \
#                                  --exp="${exp}" \
#                                  --pool=${pool} \
#                                  --eps=${eps} \
#                                  --shift-steps=${shift_steps} \
#                                  --shift-leap=${shift_leap} \
#                                  --batch-size=${batch_size}
  # ------------------------------------------------------------------------------------------------------------------ #

  # --- Traverse attribute space ------------------------------------------------------------------------------------- #
#  python traverse_attribute_space.py -v \
#                                     --exp="${exp}" \
#                                     --pool=${pool} \
#                                     --eps=${eps} \
#                                     --shift-steps=${shift_steps}
  # ------------------------------------------------------------------------------------------------------------------ #

  # --- Rank interpretable paths for all given attribute groups ------------------------------------------------------ #
  for attr_group in "${ATTRIBUTE_GROUPS[@]}"
  do
    python rank_interpretable_paths.py -v --exp="${exp}" \
                                          --pool=${pool} \
                                          --eps=${eps} \
                                          --shift-steps=${shift_steps} \
                                          --num-imgs=${num_imgs} \
                                          --gif-size=${gif_size} \
                                          --attr-group="${attr_group}" \
                                          --metric=${metric}
  done
  # ------------------------------------------------------------------------------------------------------------------ #
done

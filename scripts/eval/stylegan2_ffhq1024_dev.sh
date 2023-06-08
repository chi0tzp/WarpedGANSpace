#!/bin/bash

# === Configuration ===
pool="stylegan2_ffhq1024-1"
eps=0.25
shift_steps=10
shift_leap=5
batch_size=4
gif_size=256
num_imgs=5
metric="corr+corr_l1"
# =====================

# Define experiment directories list
declare -a EXPERIMENTS=(
                        "experiments/complete/WarpedGANSpace_stylegan2_ffhq1024-W-K160-D16-LearnGammas-beta_0.5-eps0.25_0.45"
                        )

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
  python traverse_latent_space.py -v --gif \
                                  --exp="${exp}" \
                                  --pool=${pool} \
                                  --eps=${eps} \
                                  --shift-steps=${shift_steps} \
                                  --shift-leap=${shift_leap} \
                                  --batch-size=${batch_size}
  # ------------------------------------------------------------------------------------------------------------------ #

  # --- Traverse attribute space ------------------------------------------------------------------------------------- #
  python traverse_attribute_space.py -v \
                                     --exp="${exp}" \
                                     --pool=${pool} \
                                     --eps=${eps} \
                                     --shift-steps=${shift_steps}
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

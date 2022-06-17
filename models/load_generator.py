import sys
from models.genforce.models import MODEL_ZOO
from models.genforce.models import build_generator
import os
import os.path as osp
import subprocess
import torch


def load_generator(model_name, latent_is_w=False, verbose=False, CHECKPOINT_DIR='models/pretrained/genforce/'):

    if verbose:
        print("  \\__Building generator for model {}...".format(model_name), end="")

    model_config = MODEL_ZOO[model_name].copy()
    url = model_config.pop('url')  # URL to download model if needed.
    model_config.update({'latent_is_w': latent_is_w})

    # Build generator
    generator = build_generator(**model_config)
    if verbose:
        print("Done!")

    # Load pre-trained weights.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = osp.join(CHECKPOINT_DIR, model_name + '.pth')

    if verbose:
        print("  \\__Loading checkpoint from {}...".format(checkpoint_path), end="")

    if not osp.exists(checkpoint_path):
        subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    if verbose:
        print("Done!")

    generator.dim_z = generator.z_space_dim

    return generator

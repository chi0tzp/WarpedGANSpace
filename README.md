# WarpedGANSpace: Finding non-linear RBF paths in GAN latent space

Authors official PyTorch implementation of the **WarpedGANSpace: Finding non-linear RBF paths in GAN latent space (ICCV 2021)**. If you use this code for your research, please [cite](#citation) our paper.


## Overview

<p align="center">
<img src="./figs/overview.svg" alt="WarpedGANSpace Overview"/>
</p>


<p align="center">
<img src="./figs/interpretable_path.svg" alt="Non-linear interpretable path"/>
</p>


## Installation

We recommend installing the required packages using python's native virtual environment. For Python 3.4+, this can be done as follows:

```bash
$ python -m venv warped-gan-space
$ source warped-gan-space/bin/activate
(warped-gan-space) $ pip install --upgrade pip
(warped-gan-space) $ pip install -r requirements.txt
```



## Prerequisite pretrained models

Download the prerequisite pretrained models (i.e., GAN generators, face detector, pose estimator, etc.) as follows:

```bash
$ python download.py	
```

This will create a directory `models/pretrained` with the following sub-directories (~3.2GiB):

```
./models/pretrained/
├── generators/
├── arcface/
├── fairface/
├── hopenet/
└── sfd/
```



## Training

For training a *WarpedGANSpace* model you need to use `train.py` (check its basic usage by running `python train.py -h`).

For example, in order to train a *WarpedGANSpace* model on the `ProgGAN` pre-trained (on CelebA) generator for discovering `K=128` interpretable paths (latent warping functions) with `N=32`  support dipoles each (i.e., 32 pairs of bipolar RBFs) run the following command:

```bash
python train.py -v --gan-type=ProgGAN --reconstructor-type=ResNet --learn-gammas --num-support-sets=128 --num-support-dipoles=32 --min-shift-magnitude=0.15 --max-shift-magnitude=0.25 --batch-size=8 --max-iter=200000
```

In the example above, batch size is set to `8` and the training will be conducted for `200000` iterations. Minimum and maximum shift magnitudes are set to `0.15` and `0.25`, respectively (please see Sect. 3.2 in the paper for more details).  A set of auxiliary training scripts (for all available GAN generators) can be found under `scripts/train/`.

The training script will create a directory with the following name format:

```
<gan_type>(-<stylegan2_resolution>)-<reconstructor_type>-K<num_support_sets>-N<num_support_dipoles>(-LearnAlphas)(-LearnGammas)-eps<min_shift_magnitude>_<max_shift_magnitude>
```

E.g., `ProgGAN-ResNet-K128-N128-LearnGammas-eps0.15_0.25`, under `experiments/wip/` while training is in progress, which after training completion, will be copied under `experiments/complete/`. This directory has the following structure:

```
├── models/
├── tensorboard/
├── args.json
├── stats.json
└── command.sh
```

where `models/` contains the weights for the reconstructor (`reconstructor.pt`) and the support sets (`support_sets.pt`). While training is in progress (i.e., while this directory is found under `experiments/wip/`), the corresponding `models/` directory contains a checkpoint file (`checkpoint.pt`) containing the last iteration, and the weights for the reconstructor and the support sets, so as to resume training. Re-run the same command, and if the last iteration is less than the given maximum number of iterations, training will resume from the last iteration. This directory will be referred to as `EXP_DIR` for the rest of this document. 



## Evaluation

After a *WarpedGANSpace* is trained, the corresponding experiment's directory (i.e., `EXP_DIR`) can be found under `experiments/complete/`. The evaluation of the model includes the following steps:

-  **Latent space traversals** For a given set of latent codes, we first generate images for all `K` paths (warping functions) and save the traversals (path latent codes and generated image sequences).
- **Attribute space traversals** In the case of facial images (i.e., `ProgGAN` and `StyleGAN2`), for the latent traversals above, we calculate the corresponding attribute paths (i.e., facial expressions, pose, etc.).
- **Interpretable paths discovery and ranking** [*To Appear Soon*]

Before calculating latent space traversals, you need to create a pool of latent codes/images for the corresponding GAN type. This can be done using `sample_gan.py`. The name of the pool can be passed using `--pool`; if left empty `<gan_type><num_samples>` will be used instead. The pool of latent codes/images will be stored under `experiments/latent_codes/<gan_type>/`.  We will be referring to it as a `POOL` for the rest of this document. 

For example, the following command will create a pool named `ProgGAN_4` under `experiments/latent_codes/ProgGAN/`:

```
python sample_gan.py -v --gan-type=ProgGAN --num-samples=4
```



### Latent space traversals

Latent space traversals can be calculated using the script `traverse_latent_space.py` (please check its basic usage by running `traverse_latent_space.py -h`) for a given model and a given `POOL`. 

### Attribute space traversals

[*To Appear Soon*]

### Interpretable paths discovery and ranking

[*To Appear Soon*]



<!-- ## Results -->

<!--### SNGAN (MNIST, AnimeFaces)-->

<!--### BigGAN (ImageNet)-->

<!--### ProgGAN (CelebA)-->

<!--### StyleGAN2 (FFHQ)-->



## Citation

[1] Christos Tzelepis, Georgios Tzimiropoulos, and Ioannis Patras. WarpedGANSpace: Finding non-linear rbf paths in gan latent space. IEEE International Conference on Computer Vision (ICCV), 2021.

Bibtex entry:

```bibtex
@inproceedings{warpedganspace,
  title={{WarpedGANSpace}: Finding non-linear {RBF} paths in {GAN} latent space},
  author={Tzelepis, Christos and Tzimiropoulos, Georgios and Patras, Ioannis},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2021}
}
```



## Acknowledgment

This research was supported by the EU's Horizon 2020 programme H2020-951911 [AI4Media](https://www.ai4media.eu/) project.


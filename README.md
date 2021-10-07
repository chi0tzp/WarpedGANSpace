# WarpedGANSpace: Finding non-linear RBF paths in GAN latent space

Authors official PyTorch implementation of the **[WarpedGANSpace: Finding non-linear RBF paths in GAN latent space (ICCV 2021)](https://arxiv.org/abs/2109.13357)**. If you use this code for your research, please [**cite**](#citation) our paper.

## Overview

In this work, we try to discover non-linear interpretable paths in GAN latent space. For doing so, we model non-linear paths using RBF-based *warping functions*, which by warping the latent space, endow it with vector fields (their gradients).  We use the latter to traverse the latent space across the paths determined by the aforementioned vector fields for any given latent code.

<p align="center">
<img src="./figs/latent_space_warping.svg" alt="WarpedGANSpace Overview"/>
</p>


Each warping function is defined by a set of *N* support vectors (a "support set") and its gradient is given analytically as shown above. For a given warping function *f<sup>k</sup>* and a given latent code **z**,  we traverse the latent space as illustrated below: 



<p align="center">
<img src="./figs/interpretable_path.svg" alt="Non-linear interpretable path"/>
</p>


Each warping function gives rise to a family of non-linear paths. We learn a set of such warping functions (implemented by the *Warping Network*), i.e., a set of such non-linear path families, so as they are distinguishable to each other; that is, the image transformations that they produce should be easily distinguishable be a discriminator network (the *Reconstructor*). An overview of the method is given below.



<p align="center">
<img src="./figs/overview.svg" alt="WarpedGANSpace Overview"/>
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

Download the prerequisite pretrained models (i.e., GAN generators, face detector, pose estimator, and other attribute detectors), as well as pre-trained WarpedGANSpace models (optionally, by passing `-m`), as follows:

```bash
$ python download.py	
```

This will create a directory `models/pretrained` with the following sub-directories (~2.0 GiB):

```
./models/pretrained/
├── au_detector/
├── generators/
├── arcface/
├── fairface/
├── hopenet/
└── sfd/
```

as well as, a directory `experiments/complete/` (if not already created by the user upon an experiment's completion) for downloading the WarpedGANSpace pretrained models (if selected) with the following sub-directories (~??? GiB):

```
.experiments/complete/
├── SNGAN_AnimeFaces-LeNet-K64-D128-LearnGammas-eps0.25_0.35/
├── SNGAN_MNIST-LeNet-K64-D128-LearnGammas-eps0.15_0.25/
├── BigGAN-239-ResNet-K120-D256-LearnGammas-eps0.15_0.25/
└── ProgGAN-ResNet-K200-D512-LearnGammas-eps0.1_0.2/
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

For instance, `ProgGAN-ResNet-K128-N128-LearnGammas-eps0.15_0.25`, under `experiments/wip/` while training is in progress, which after training completion, will be copied under `experiments/complete/`. This directory has the following structure:

```
├── models/
├── tensorboard/
├── args.json
├── stats.json
└── command.sh
```

where `models/` contains the weights for the reconstructor (`reconstructor.pt`) and the support sets (`support_sets.pt`). While training is in progress (i.e., while this directory is found under `experiments/wip/`), the corresponding `models/` directory contains a checkpoint file (`checkpoint.pt`) containing the last iteration, and the weights for the reconstructor and the support sets, so as to resume training. Re-run the same command, and if the last iteration is less than the given maximum number of iterations, training will resume from the last iteration. This directory will be referred to as `EXP_DIR` for the rest of this document. 



## Evaluation

After a *WarpedGANSpace* model is trained, the corresponding experiment's directory (i.e., `EXP_DIR`) can be found under `experiments/complete/`. The evaluation of the model includes the following steps:

-  **Latent space traversals** For a given set of latent codes, we first generate images for all `K` paths (warping functions) and save the traversals (path latent codes and generated image sequences).
- **Attribute space traversals** In the case of facial images (i.e., `ProgGAN` and `StyleGAN2`), for the latent traversals above, we calculate the corresponding attribute paths (i.e., pose, id scores, action units intensities, pose, etc.).
- **Interpretable paths discovery and ranking** [*To Appear Soon*]

Before calculating latent space traversals, we need to create a pool of latent codes/images for the corresponding GAN type. This can be done using `sample_gan.py`. The name of the pool can be passed using `--pool`; if left empty `<gan_type>_<num_samples>` will be used instead. The pool of latent codes/images will be stored under `experiments/latent_codes/<gan_type>/`.  We will be referring to it as  `POOL` for the rest of this document. 

For example, the following command will create a pool named `ProgGAN_4` under `experiments/latent_codes/ProgGAN/`:

```
python sample_gan.py -v --gan-type=ProgGAN --num-samples=4
```



### Latent space traversals

Latent space traversals can be calculated using the script `traverse_latent_space.py` (please check its basic usage by running `traverse_latent_space.py -h`) for a given model and a given `POOL`. Upon completion, results (i.e., latent traversals) will be stored under the following directory:

`experiments/complete/EXP_DIR/results/POOL/<2*shift_steps>_<eps>_<total_length>`

where `eps`,  `shift_steps`, and `total_length` denote respectively the shift magnitude (of a single step on the path), the number of such steps, and the total traversal length. We will be referring to a directory `<2*shift_steps>_<eps>_<total_length>` as `TRAVERSAL_CONFIG` for the rest of this document.

If `--gif` is set, a directory `experiments/complete/EXP_DIR/results/POOL/TRAVERSAL_CONFIG/paths_gifs/` will be created and populated by GIF images for all latent codes (original images) in `POOL` and for all discovered paths. 



### Attribute space traversals

Granted that the GAN at hand generates facial images (i.e., `ProgGAN` or `StyleGAN2`), we can traverse an *attribute space* by running `traverse_attribute_space.py` These attributes include the facial bounding box (in terms of width and height), an identity score, age, race, and gender estimation, pose estimation in terms of yaw, pitch, and roll angles, and 12 facial action units predictions. For more details, please check its basic usage by running `traverse_attribute_space.py -h`.

This script needs a `TRAVERSAL_CONFIG` found under `experiments/complete/EXP_DIR/results/POOL/`. Upon completion, the corresponding attribute paths will be stored under the same directory.



### Interpretable paths discovery and ranking

[*To Appear Soon*]



## Qualitative results

### [SNGAN (MNIST, AnimeFaces)](demo/SNGAN.md)

### [BigGAN (ImageNet)](demo/BigGAN.md)

### [ProgGAN (CelebA)](demo/ProgGAN.md)

### [StyleGAN2 (FFHQ)](demo/StyleGAN2.md)



## Demo

[*To Appear soon*]



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


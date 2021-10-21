# WarpedGANSpace: Finding non-linear RBF paths in GAN latent space

Authors official PyTorch implementation of the **[WarpedGANSpace: Finding non-linear RBF paths in GAN latent space (ICCV 2021)](https://arxiv.org/abs/2109.13357)**. If you use this code for your research, please [**cite**](#citation) our paper.

<p align="center">
<img src="demo/banner/celeba_age_1_6_689f0ab17f84b9cd36c0a5bdfb0469aadce4c4ba.gif" style="width: 75vw" title="Age"/>
<img src="demo/banner/gender_1_0_50441c6d696cdd52c8435da33ed7bb94f550013d.gif " style="width: 75vw" title="Gender"/>
<img src="demo/banner/race_2_109_5e9893b9508f383603de655a34750820723e1f95.gif" style="width: 75vw" title="Race"/>
<img src="demo/banner/gender_1_195_b4e18c876322aedcef6d2b36b273f4b2c0b642bb.gif " style="width: 75vw" title="Gender"/>
</p>


## Overview

In this work, we try to discover *non-linear* interpretable paths in GAN latent space in an *unsupervised* and *model-agnostic* manner. For doing so, we model non-linear paths using RBF-based *warping functions*, which by warping the latent space, endow it with vector fields (i.e., their gradients).  We use the latter to traverse the latent space across the paths determined by the aforementioned vector fields for any given latent code.

<p align="center">
<img src="./figs/latent_space_warping.svg" alt="WarpedGANSpace Overview"/>
</p>


Each warping function is defined by a set of *N* support vectors (which form a "support set") and its gradient is given analytically as shown above. For a given warping function *f<sup>k</sup>* and a given latent code **z**,  we traverse the latent space as illustrated below: 



<p align="center">
<img src="./figs/interpretable_path.svg" alt="Non-linear interpretable path"/>
</p>


Each warping function gives rise to a family of non-linear paths. We learn a set of such warping functions (implemented by the *Warping Network*), i.e., a set of such non-linear path families, so as the image transformations that they produce are distinguishable to each other by a discriminator network (the *Reconstructor*). An overview of the method is given below.

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
- **Attribute space traversals** In the case of facial images (i.e., `ProgGAN` and `StyleGAN2`), for the latent traversals above, we calculate the corresponding attribute paths (i.e., pose, id scores, action units intensities, etc.).
- **Interpretable paths discovery and ranking** For the attribute traversals above, we rank the discovered paths based on the how much correlated each path is with each attribute path.

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

Granted that the GAN at hand generates facial images (i.e., `ProgGAN` or `StyleGAN2`), we can traverse an *attribute space* by running `traverse_attribute_space.py` These attributes include the facial bounding box (in terms of width and height), an identity score, age, race, and gender estimation, pose estimation in terms of yaw, pitch, and roll angles, 12 facial action units predictions, and 5 CelebA attributes. For more details, please check its basic usage by running `traverse_attribute_space.py -h`.

This script needs a `TRAVERSAL_CONFIG` found under `experiments/complete/EXP_DIR/results/POOL/`. Upon completion, the corresponding attribute paths will be stored under the same directory.



### Interpretable paths discovery and ranking

After generating the attribute traversals, as described above, the discovered latent paths can be ranked based on the correlation they exhibit with respect to a set of attributes. In other words, based on how certain attributes change when traversing the discovered latent paths. 

This can be done by using  `rank_interpretable_paths.py` for a given group of attributes. An attribute group, to which we will be referring to it as  `ATTR_GROUP` for the rest of this document, is a subset of the set of all available attributes (see `ATTRIBUTE_GROUPS` dictionary in `rank_interpretable_paths.py` and/or run `rank_interpretable_paths.py -h` for more details). The ranking results will be stored under `experiments/complete/EXP_DIR/results/POOL/interpretable_paths/ATTR_GROUP/` for the chosen group of attributes `ATTR_GROUP`. A markdown file the summarizes the results will be created under `experiments/complete/EXP_DIR/results/POOL/interpretable_paths/ATTR_GROUP/ATTRIBUTE` for each `ATTRIBUTE` in `ATTR_GROUP`. An example of such summarizing file is given [here](TODO).



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
@InProceedings{Tzelepis_2021_ICCV,
    author    = {Tzelepis, Christos and Tzimiropoulos, Georgios and Patras, Ioannis},
    title     = {{WarpedGANSpace}: Finding Non-Linear RBF Paths in {GAN} Latent Space},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {6393-6402}
}
```



## Acknowledgment

This research was supported by the EU's Horizon 2020 programme H2020-951911 [AI4Media](https://www.ai4media.eu/) project.


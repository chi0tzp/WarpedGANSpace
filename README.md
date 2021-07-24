# WarpedGANSpace: Finding non-linear RBF paths in GAN latent space

Authors official PyTorch implementation of the **WarpedGANSpace: Finding non-linear RBF paths in GAN latent space (ICCV 2021)**. If you use this code for your research, please [cite](#citation) our paper.



## Overview

<p align="center">
<img src="overview.svg" alt="WarpedGANSpace Overview"/>
</p>


## Installation

We recommend installing the required packages using python's native virtual environment (Python 3.4+) as follows:

```bash
$ python -m venv warped-gan-space
$ source warped-gan-space/bin/activate
(warped-gan-space) $ pip install --upgrade pip
(warped-gan-space) $ pip install -r requirements.txt
```



## Download pre-trained models

Download pre-trained models (i.e., GAN generators, face detector, etc.) as follows:

```bash
$ python download.py	
```

This will create a directory `./models/pretrained` with the following sub-directories (~3.2GiB):

```
./models/pretrained/
├── arcface
├── deformators
├── fairface
├── fanet
├── generators
├── hopenet
└── sfd
```



## Training

TODO



## Evaluation

TODO



## Examples

TODO



## Citation

```
@inproceedings{warpedganspace,
  title={{WarpedGANSpace}: Finding non-linear RBF paths in GAN latent space},
  author={Tzelepis, Christos and Tzimiropoulos, Georgios and Patras, Ioannis},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2021}
}
```





## Acknowledgment

This research was supported by the EU's Horizon 2020 programme H2020-951911 [AI4Media](https://www.ai4media.eu/) project.


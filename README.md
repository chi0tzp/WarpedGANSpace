# WarpedGANSpace: Finding non-linear RBF paths in GAN latent space

Authors official PyTorch implementation of the **WarpedGANSpace: Finding non-linear RBF paths in GAN latent space (ICCV 2021)**. If you use this code for your research, please [cite](#citation) our paper.



## Overview

<p align="center">
<img src="overview.svg" alt="WarpedGANSpace Overview"/>
</p>
<!--A latent code <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}\sim\mathcal{N}\left(\mathbf{0},I_d\right)"> is shifted by a vector induced by a warping function <img src="https://render.githubusercontent.com/render/math?math=f^k"> implemented by the warping network <img src="https://render.githubusercontent.com/render/math?math=\mathcal{W}"> after choosing the corresponding support set <img src="https://render.githubusercontent.com/render/math?math=\mathcal{S}^k">, weights <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A}^k"> and parameters <img src="https://render.githubusercontent.com/render/math?math=\mathcal{G}^k">. The pair of latent codes, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}"> and <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}+\epsilon_k\frac{\nabla f^k(\mathbf{z})}{\lVert\nabla f^k(\mathbf{z})\rVert}">, are then fed into the generator <img src="https://render.githubusercontent.com/render/math?math=\mathfrak{G}"> in order to produce two images. The reconstructor <img src="https://render.githubusercontent.com/render/math?math=\mathfrak{R}"> is optimized to reproduce the signed shift magnitude <img src="https://render.githubusercontent.com/render/math?math=\epsilon_k"> and predict the index <img src="https://render.githubusercontent.com/render/math?math=k"> of the support set used.-->



## Installation

We recommend installing the required packages using python's native virtual environment (Python 3.4+) as follows:

```bash
$ python -m venv warped-gan-space
$ source warped-gan-space/bin/activate
(warped-gan-space) $ pip install --upgrade pip
(warped-gan-space) $ pip install -r requirements.txt
```



## Training



## Evaluation



## Examples



## Citation

```
@inproceedings{warpedganspace,
  title={{WarpedGANSpace}: Finding non-linear RBF paths in GAN latent space},
  author={Tzelepis, Christos and Tzimiropoulos, Georgios and Patras, Ioannis},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2021}
}
```



## References

[1] Voynov, Andrey, and Artem Babenko. "Unsupervised discovery of interpretable directions in the gan latent space." *International Conference on Machine Learning*. PMLR, 2020. [[Paper](https://arxiv.org/abs/2002.03754)] [[Code](https://github.com/anvoynov/GANLatentDiscovery)]









## Acknowledgment

This research was supported by the EU's Horizon 2020 programme H2020-951911 [AI4Media](https://www.ai4media.eu/) project.


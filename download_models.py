import sys
import os
import os.path as osp
import hashlib
import tarfile
import time
import urllib.request
from lib import GAN_WEIGHTS, LINEAR_DIRECTIONS, SFD, ARCFACE, FAIRFACE, HOPENET, FANET


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r      \\__%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))

    sys.stdout.flush()


def download(src, sha256sum, dest):
    tmp_tar = osp.join(dest, ".tmp.tar")
    try:
        urllib.request.urlretrieve(src, tmp_tar, reporthook)
    except:
        raise ConnectionError("Error: {}".format(src))

    sha256_hash = hashlib.sha256()
    with open(tmp_tar, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

        sha256_check = sha256_hash.hexdigest() == sha256sum
        print()
        print("      \\__Check sha256: {}".format("OK!" if sha256_check else "Error"))
        if not sha256_check:
            raise Exception("Error: Invalid sha256 sum: {}".format(sha256_hash.hexdigest()))

    tar_file = tarfile.open(tmp_tar, mode='r')
    tar_file.extractall(dest)
    os.remove(tmp_tar)


def main():
    """Download pre-trained models:
        -- GAN generators
            -- Spectral Norm GAN [1] for MNIST [2] and AnimeFaces [3] datasets.
            -- BigGAN [4] for ILSVRC dataset [5]
            -- ProgGAN [6] for CelebA-HQ dataset [7]
            -- StyleGAN2 for FFHQ dataset [8, 9]
        -- Linear deformators [10]
        -- SFD face detector [11]
        -- ArcFace [12]
        -- FairFace [13]
        -- Hopenet [14]

    [1] Miyato, T., Kataoka, T., Koyama, M., and Yoshida, Y. Spectral normalization for generative adversarial networks.
        In International Conference on Learning Representations, 2018.

    [2] LeCun, Y. The mnist database of handwritten digits. http://yann. lecun. com/exdb/mnist/, 1989.

    [3] Jin, Y., Zhang, J., Li, M., Tian, Y., and Zhu, H. Towards the high-quality anime characters generation with
        generative adversarial networks. In Proceedings of the Machine Learning for Creativity and Design Workshop at
        NIPS, 2017.

    [4] Brock, A., Donahue, J., and Simonyan, K. Large scale GAN training for high fidelity natural image synthesis.
        In International Conference on Learning Representations, 2019.

    [5] Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. ImageNet: A Large-Scale Hierarchical Image
        Database. In CVPR09, 2009.

    [6] Karras, T., Aila, T., Laine, S., and Lehtinen, J. Progressive growing of gans for improved quality,
        stability, and variation. Proceedings of the International Conference on Learning Representations (ICLR), 2018.

    [7] Liu, Z., Luo, P., Wang, X., and Tang, X. Deep learning face attributes in the wild. In Proceedings of
        International Conference on Computer Vision (ICCV), December 2015.

    [8] Karras, Tero, Samuli Laine, and Timo Aila. "A style-based generator architecture for generative adversarial
        networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

    [9] Karras, Tero, et al. "Analyzing and improving the image quality of stylegan." Proceedings of the IEEE/CVF
        Conference on Computer Vision and Pattern Recognition. 2020.

    [10] Andrey Voynov, Artem Babenko, "Unsupervised Discovery of Interpretable Directions in the GAN Latent Space",
         ICML 2020: 9786-9796, https://github.com/anvoynov/GANLatentDiscovery

    [11] Zhang, Shifeng, et al. "S3FD: Single shot scale-invariant face detector." Proceedings of the IEEE
         international conference on computer vision. 2017.

    [12] Deng, Jiankang, et al. "ArcFace: Additive angular margin loss for deep face recognition." Proceedings of the
         IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

    [13] Karkkainen, Kimmo, and Jungseock Joo. "FairFace: Face attribute dataset for balanced race, gender, and age."
         arXiv preprint arXiv:1908.04913 (2019).

    [14] Doosti, Bardia, et al. "Hope-net: A graph-based model for hand-object pose estimation." Proceedings of the
         IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

    """
    # Create pre-trained models root directory
    pretrained_root = osp.join('models', 'pretrained')
    os.makedirs(pretrained_root, exist_ok=True)

    # Download pre-trained GAN generators
    print("#. Download pre-trained GAN generators models:")
    for m in ('SNGAN_MNIST', 'SNGAN_AnimeFaces', 'BigGAN', 'ProgGAN', 'StyleGAN2'):
        print("  \\__.{} ".format(m))
        model_dir = osp.join(pretrained_root, 'generators', m)
        print("      \\__Create dir: {}".format(model_dir))
        os.makedirs(model_dir, exist_ok=True)
        download(src=GAN_WEIGHTS[m]['url'], sha256sum=GAN_WEIGHTS[m]['sha256sum'], dest=model_dir)

    print("#. Download pre-trained linear directions (deformators)...")
    print("  \\__.Deformators")
    download(src=LINEAR_DIRECTIONS[0], sha256sum=LINEAR_DIRECTIONS[1], dest=pretrained_root)

    print("#. Download pre-trained SFD face detector model...")
    print("  \\__.Face detector (SFD)")
    download(src=SFD[0], sha256sum=SFD[1], dest=pretrained_root)

    print("#. Download pre-trained ArcFace model...")
    print("  \\__.ArcFace")
    download(src=ARCFACE[0], sha256sum=ARCFACE[1], dest=pretrained_root)

    print("#. Download pre-trained FairFace model...")
    print("  \\__.FairFace")
    download(src=FAIRFACE[0], sha256sum=FAIRFACE[1], dest=pretrained_root)

    print("#. Download pre-trained Hopenet model...")
    print("  \\__.Hopenet")
    download(src=HOPENET[0], sha256sum=HOPENET[1], dest=pretrained_root)

    print("#. Download pre-trained FANet model...")
    print("  \\__.FANet")
    download(src=FANET[0], sha256sum=FANET[1], dest=pretrained_root)


if __name__ == '__main__':
    main()

import sys
import os
import os.path as osp
import argparse
import hashlib
import tarfile
import time
import urllib.request
from lib import GENFORCE, GENFORCE_MODELS, SFD, ARCFACE, FAIRFACE, HOPENET, AUDET, CELEBA_ATTRIBUTES


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
    """Download pre-trained attribute detectors models:
        -- GAN generators
            -- Spectral Norm GAN [1] for MNIST [2] and AnimeFaces [3] datasets.
            -- BigGAN [4] for ILSVRC dataset [5]
            -- ProgGAN [6] for CelebA-HQ dataset [7]
            -- StyleGAN2 for FFHQ dataset [8, 9]
        -- SFD face detector [10]
        -- ArcFace [11]
        -- FairFace [12]
        -- Hopenet [13]
        -- AU detector [14] for 12 DISFA [15] Action Units
        -- Facial attributes detector [16] for 5 CelebA [17] attributes

    and (optionally) the following pre-trained WarpedGANSpace [18] models:
        -- TODO: +++

    References:

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
    [10] Zhang, Shifeng, et al. "S3FD: Single shot scale-invariant face detector." Proceedings of the IEEE
         international conference on computer vision. 2017.
    [11] Deng, Jiankang, et al. "ArcFace: Additive angular margin loss for deep face recognition." Proceedings of the
         IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
    [12] Karkkainen, Kimmo, and Jungseock Joo. "FairFace: Face attribute dataset for balanced race, gender, and age."
         arXiv preprint arXiv:1908.04913 (2019).
    [13] Doosti, Bardia, et al. "Hope-net: A graph-based model for hand-object pose estimation." Proceedings of the
         IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
    [14] Ntinou, Ioanna, et al. "A transfer learning approach to heatmap regression for action unit intensity
        estimation." IEEE Transactions on Affective Computing (2021).
    [15] Mavadati, S. Mohammad, et al. "DISFA: A spontaneous facial action intensity database." IEEE Transactions on
         Affective Computing 4.2 (2013): 151-160.
    [16] Jiang, Yuming, et al. "Talk-to-Edit: Fine-Grained Facial Editing via Dialog." Proceedings of the IEEE/CVF
         International Conference on Computer Vision. 2021.
    [17] Liu, Ziwei, et al. "Deep learning face attributes in the wild." Proceedings of the IEEE international
         conference on computer vision. 2015.
    [18] Tzelepis, Christos, Georgios Tzimiropoulos, and Ioannis Patras. "WarpedGANSpace: Finding non-linear RBF paths
         in GAN latent space." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
    """
    parser = argparse.ArgumentParser("Download pre-trained attribute detectors and (optionally) WarpedGANSpace models")
    parser.add_argument('-m', '--models', action='store_true',
                        help="download pre-trained WarpedGANSpace models under experiments/complete/")
    # Parse given arguments
    args = parser.parse_args()

    # Create pre-trained models root directory
    pretrained_models_root = osp.join('models', 'pretrained')
    os.makedirs(pretrained_models_root, exist_ok=True)

    # Download the following pre-trained GAN generators (under models/pretrained/)
    print("#. Download pre-trained GAN generators...")
    print("  \\__.GenForce")
    download_genforce_models = False
    for k, v in GENFORCE_MODELS.items():
        if not osp.exists(osp.join(pretrained_models_root, 'genforce', v[0])):
            download_genforce_models = True
            break
    if download_genforce_models:
        download(src=GENFORCE[0], sha256sum=GENFORCE[1], dest=pretrained_models_root)
    else:
        print("      \\__Already exists.")

    print("#. Download pre-trained ArcFace model...")
    print("  \\__.ArcFace")
    if osp.exists(osp.join(pretrained_models_root, 'arcface', 'model_ir_se50.pth')):
        print("      \\__Already exists.")
    else:
        download(src=ARCFACE[0], sha256sum=ARCFACE[1], dest=pretrained_models_root)

    print("#. Download pre-trained SFD face detector model...")
    print("  \\__.Face detector (SFD)")
    if osp.exists(osp.join(pretrained_models_root, 'sfd', 's3fd-619a316812.pth')):
        print("      \\__Already exists.")
    else:
        download(src=SFD[0], sha256sum=SFD[1], dest=pretrained_models_root)

    print("#. Download pre-trained FairFace model...")
    print("  \\__.FairFace")
    if osp.exists(osp.join(pretrained_models_root, 'fairface', 'fairface_alldata_4race_20191111.pt')) and \
            osp.exists(osp.join(pretrained_models_root, 'fairface', 'res34_fair_align_multi_7_20190809.pt')):
        print("      \\__Already exists.")
    else:
        download(src=FAIRFACE[0], sha256sum=FAIRFACE[1], dest=pretrained_models_root)

    print("#. Download pre-trained Hopenet model...")
    print("  \\__.Hopenet")
    if osp.exists(osp.join(pretrained_models_root, 'hopenet', 'hopenet_alpha1.pkl')) and \
            osp.exists(osp.join(pretrained_models_root, 'hopenet', 'hopenet_alpha2.pkl')) and \
            osp.exists(osp.join(pretrained_models_root, 'hopenet', 'hopenet_robust_alpha1.pkl')):
        print("      \\__Already exists.")
    else:
        download(src=HOPENET[0], sha256sum=HOPENET[1], dest=pretrained_models_root)

    print("#. Download pre-trained AU detector model...")
    print("  \\__.FANet")
    if osp.exists(osp.join(pretrained_models_root, 'au_detector', 'disfa_adaptation_f0.pth')):
        print("      \\__Already exists.")
    else:
        download(src=AUDET[0], sha256sum=AUDET[1], dest=pretrained_models_root)

    print("#. Download pre-trained CelebA attributes predictors models...")
    print("  \\__.CelebA")
    if osp.exists(osp.join(pretrained_models_root, 'celeba_attributes', 'eval_predictor.pth.tar')):
        print("      \\__Already exists.")
    else:
        download(src=CELEBA_ATTRIBUTES[0], sha256sum=CELEBA_ATTRIBUTES[1], dest=pretrained_models_root)

    ####################################################################################################################

    # Download pre-trained WarpedGANSpace models
    # if args.models:
    #     # Create WarpedGANSpace pre-trained models root directory
    #     pretrained_warpedganspace_root = osp.join('experiments', 'complete')
    #     os.makedirs(pretrained_warpedganspace_root, exist_ok=True)
    #
    #     print("#. Download pre-trained WarpedGANSpace models...")
    #
    #     print("  \\__.ProgGAN_ResNet_K200_D512_LearnGammas_eps0.1_0.2")
    #     download(src=ProgGAN_ResNet_K200_D512_LearnGammas_eps0d1_0d2[0],
    #              sha256sum=ProgGAN_ResNet_K200_D512_LearnGammas_eps0d1_0d2[1],
    #              dest=pretrained_warpedganspace_root)
    #
    #     print("  \\__.StyleGAN2_1024_W_ResNet_K200_D512_LearnGammas_eps0d1_0d2")
    #     download(src=StyleGAN2_1024_W_ResNet_K200_D512_LearnGammas_eps0d1_0d2[0],
    #              sha256sum=StyleGAN2_1024_W_ResNet_K200_D512_LearnGammas_eps0d1_0d2[1],
    #              dest=pretrained_warpedganspace_root)


if __name__ == '__main__':
    main()

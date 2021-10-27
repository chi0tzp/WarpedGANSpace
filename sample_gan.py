import os
import os.path as osp
import argparse
import torch
import json
from torch import nn
from hashlib import sha1
from torchvision.transforms import ToPILImage
from lib import *
from models.gan_load import build_biggan, build_proggan, build_stylegan2, build_sngan


def tensor2image(tensor, adaptive=False):
    tensor = tensor.squeeze(dim=0)
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
    else:
        tensor = (tensor + 1) / 2
        tensor.clamp(0, 1)
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def main():
    """A script for sampling from a pre-trained GAN latent space and generating images. The generated images, along with
    the corresponding latent codes (in torch.Tensor format), will be stored under
        `experiments/latent_codes/<gan_type>/<pool>/`.
    If no pool name is given, then `<gan_type>_<num_samples>/` will be used instead.

    Options:
        -v, --verbose           : set verbose mode on
        -g, --gan-type          : set GAN type (SNGAN_MNIST, SNGAN_AnimeFaces, BigGAN, ProgGAN, or StyleGAN2)
        --z-truncation          : set latent code sampling truncation parameter. If set, latent codes will be sampled
                                  from a standard Gaussian distribution truncated to the range [-args.z_truncation,
                                  +args.z_truncation]
        --biggan-target-classes : set list of classes to use for conditional BigGAN (see BIGGAN_CLASSES in
                                  lib/config.py). E.g., --biggan-target-classes 14 239.
        --stylegan2-resolution  : set StyleGAN2 generator output images resolution (256 or 1024)
        --num-samples           : set the number of latent codes to sample for generating images
        --pool                  : set name of the latent codes/images pool.
        --cuda                  : use CUDA (default)
        --no-cuda               : do not use CUDA
    """
    parser = argparse.ArgumentParser(description="Sample a pre-trained GAN latent space and generate images")
    parser.add_argument('-v', '--verbose', action='store_true', help="set verbose mode on")
    parser.add_argument('-g', '--gan-type', type=str, required=True, choices=GAN_WEIGHTS.keys(),
                        help='GAN generator model type')
    parser.add_argument('--z-truncation', type=float, help="set latent code sampling truncation parameter")
    parser.add_argument('--biggan-target-classes', nargs='+', type=int, help="list of classes for conditional BigGAN")
    parser.add_argument('--stylegan2-resolution', type=int, default=1024, choices=(256, 1024),
                        help="StyleGAN2 image resolution")
    parser.add_argument('--num-samples', type=int, default=4, help="number of latent codes to sample")
    parser.add_argument('--pool', type=str, help="name of latent codes/images pool")
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # Create output dir for generated images
    out_dir = osp.join('experiments', 'latent_codes', args.gan_type)
    biggan_classes = None
    if args.gan_type == 'BigGAN':
        # Get BigGAN classes
        if args.biggan_target_classes is None:
            raise parser.error("In case of BigGAN, a list of classes needs to be determined.")
        biggan_classes = ''
        for c in args.biggan_target_classes:
            biggan_classes += '-{}'.format(c)
        out_dir += biggan_classes
    if args.pool:
        out_dir = osp.join(out_dir, args.pool)
    else:
        out_dir = osp.join(out_dir, '{}_{}'.format(args.gan_type + biggan_classes if args.gan_type == 'BigGAN'
                                                   else args.gan_type, args.num_samples))
    os.makedirs(out_dir, exist_ok=True)

    # Save argument in json file
    with open(osp.join(out_dir, 'args.json'), 'w') as args_json_file:
        json.dump(args.__dict__, args_json_file)

    # Set default tensor type
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    use_cuda = args.cuda and torch.cuda.is_available()

    # Build GAN generator model and load with pre-trained weights
    if args.verbose:
        print("#. Build GAN generator model G and load with pre-trained weights...")
        print("  \\__GAN type: {}".format(args.gan_type))
        if args.gan_type == 'BigGAN':
            print("      \\__Target classes: {}".format(args.biggan_target_classes))
        print("  \\__Pre-trained weights: {}".format(
            GAN_WEIGHTS[args.gan_type]['weights'][args.stylegan2_resolution] if args.gan_type == 'StyleGAN2' else
            GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]]))

    # -- BigGAN
    if args.gan_type == 'BigGAN':
        G = build_biggan(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]],
                         target_classes=args.biggan_target_classes)
    # -- ProgGAN
    elif args.gan_type == 'ProgGAN':
        G = build_proggan(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]])
    # -- StyleGAN2
    elif args.gan_type == 'StyleGAN2':
        G = build_stylegan2(resolution=args.stylegan2_resolution,
                            pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][args.stylegan2_resolution])
    # -- Spectrally Normalised GAN (SNGAN)
    else:
        G = build_sngan(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]],
                        gan_type=args.gan_type)

    # Upload GAN generator model to GPU
    if use_cuda:
        G = G.cuda()

    # Set generator to evaluation mode
    G.eval()

    # Latent codes sampling
    if args.verbose:
        print("#. Sample {} {}-dimensional latent codes...".format(args.num_samples, G.dim_z))
        if args.z_truncation:
            print("  \\__Truncate standard Gaussian to range [{}, +{}]".format(-args.z_truncation, args.z_truncation))

    # zs = torch.randn(args.num_samples, G.dim_z)
    zs = sample_z(batch_size=args.num_samples, dim_z=G.dim_z, truncation=args.z_truncation)

    if use_cuda:
        zs = zs.cuda()

    if args.verbose:
        print("#. Generate images...")
        print("  \\__{}".format(out_dir))

    # Iterate over given latent codes
    for i in range(args.num_samples):
        # Un-squeeze current latent code in shape [1, dim] and create hash code for it
        z = zs[i, :].unsqueeze(0)
        latent_code_hash = sha1(z.cpu().numpy()).hexdigest()

        if args.verbose:
            update_progress(
                "  \\__.Latent code hash: {} [{:03d}/{:03d}] ".format(latent_code_hash, i + 1, args.num_samples),
                args.num_samples, i)

        # Create directory for current latent code
        latent_code_dir = osp.join(out_dir, '{}'.format(latent_code_hash))
        os.makedirs(latent_code_dir, exist_ok=True)

        # Save latent code tensor under `latent_code_dir`
        torch.save(z.cpu(), osp.join(latent_code_dir, 'latent_code.pt'))

        # Generate image for the given latent code z
        with torch.no_grad():
            img = G(z).cpu()

        # Convert image's tensor into an RGB image and save it
        img_pil = tensor2image(img, adaptive=True)
        img_pil.save(osp.join(latent_code_dir, 'image.jpg'), "JPEG", quality=95, optimize=True, progressive=True)

    if args.verbose:
        update_stdout(1)
        print()
        print()


if __name__ == '__main__':
    main()

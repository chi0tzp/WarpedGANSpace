import os
import os.path as osp
import argparse
import torch
import json
from hashlib import sha1
from torchvision.transforms import ToPILImage
from lib import GENFORCE_MODELS, update_progress, update_stdout
from models.load_generator import load_generator


def tensor2image(tensor, img_size=None, adaptive=False):
    # Squeeze tensor image
    tensor = tensor.squeeze(dim=0)
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        if img_size:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8)).resize((img_size, img_size))
        else:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
    else:
        tensor = (tensor + 1) / 2
        tensor.clamp(0, 1)
        if img_size:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8)).resize((img_size, img_size))
        else:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))


def main():
    """A script for sampling from a pre-trained GAN's latent space and generating images. The generated images, along
    with the corresponding latent codes, will be stored under `experiments/latent_codes/<gan>/`.

    Options:
        -v, --verbose    : set verbose mode on
        --gan            : set GAN generator (see GENFORCE_MODELS in lib/config.py)
        --stylegan-space : set StyleGAN latent space (Z, W, W+) -- sampling is always done in Z space but in case this
                           is set in W/W+-space, all latent codes will be stored
        --truncation     : set W-space truncation parameter. If set, W-space codes will be truncated
        --num-samples    : set the number of latent codes to sample for generating images
        --cuda           : use CUDA (default)
        --no-cuda        : do not use CUDA
    """
    parser = argparse.ArgumentParser(description="Sample a pre-trained GAN latent space and generate images")
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose mode on")
    parser.add_argument('--gan', type=str, required=True, choices=GENFORCE_MODELS.keys(), help='GAN generator')
    parser.add_argument('--truncation', type=float, default=1.0, help="W-space truncation parameter")
    parser.add_argument('--num-samples', type=int, default=4, help="set number of latent codes to sample")
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # Create output dir for generated images
    out_dir = osp.join('experiments', 'latent_codes', args.gan)
    out_dir = osp.join(out_dir, '{}-{}'.format(args.gan, args.num_samples))
    os.makedirs(out_dir, exist_ok=True)

    # Save argument in json file
    with open(osp.join(out_dir, 'args.json'), 'w') as args_json_file:
        json.dump(args.__dict__, args_json_file)

    # CUDA
    use_cuda = False
    if torch.cuda.is_available():
        if args.cuda:
            use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Build GAN generator model and load with pre-trained weights
    if args.verbose:
        print("#. Build GAN generator model G and load with pre-trained weights...")
        print("  \\__GAN generator : {} (res: {})".format(args.gan, GENFORCE_MODELS[args.gan][1]))
        print("  \\__Pre-trained weights: {}".format(GENFORCE_MODELS[args.gan][0]))

    G = load_generator(model_name=args.gan,
                       latent_is_w='stylegan' in args.gan,
                       verbose=args.verbose).eval()

    # Upload GAN generator model to GPU
    if use_cuda:
        G = G.cuda()

    # Latent codes sampling
    if args.verbose:
        print("#. Sample {} {}-dimensional latent codes...".format(args.num_samples, G.dim_z))
    zs = torch.randn(args.num_samples, G.dim_z)

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

        if 'stylegan' in args.gan:
            # Get the w+ and w codes for the given z code, save them, and the generated image based on the w code
            # Note that w+ has torch.Size([1, 512]) and w torch.Size([18, 512]) -- the latter is just a repetition of
            # the w code for all 18 layers
            w_plus = G.get_w(z, truncation=args.truncation)[0, :, :]
            w = w_plus[0, :].unsqueeze(0)
            torch.save(z.cpu(), osp.join(latent_code_dir, 'latent_code_z.pt'))
            torch.save(w.cpu(), osp.join(latent_code_dir, 'latent_code_w.pt'))
            torch.save(w_plus.cpu(), osp.join(latent_code_dir, 'latent_code_w+.pt'))

            img_w = G(w).cpu()
            tensor2image(img_w, adaptive=True).save(osp.join(latent_code_dir, 'image_w.jpg'),
                                                    "JPEG", quality=95, optimize=True, progressive=True)
        else:
            # Save latent code (Z-space), generate image for this code, and save the generated image
            torch.save(z.cpu(), osp.join(latent_code_dir, 'latent_code_z.pt'))
            img_z = G(z).cpu()
            tensor2image(img_z, adaptive=True).save(osp.join(latent_code_dir, 'image_z.jpg'),
                                                    "JPEG", quality=95, optimize=True, progressive=True)

    if args.verbose:
        update_stdout(1)
        print()
        print()


if __name__ == '__main__':
    main()

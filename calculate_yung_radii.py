import sys
import argparse
import numpy as np
import os.path as osp
import torch
from lib import GENFORCE_MODELS
from models.load_generator import load_generator
from sklearn import linear_model
from collections import defaultdict
from tqdm import tqdm
import json


def make_dict():
    return defaultdict(make_dict)


def main():
    """A script for calculating the radii of minimal enclosing balls for the latent space of a (i.e., in Z/W/W+ space),
    given a truncation parameter. When applicable, a linear model is trained in order to predict the radii of the latent
    codes, given a truncation parameter.

    The parameters of the linear model (i.e., the weight w and the bias b) are stored for each GAN type and each latent
    space in a json file (i.e., models/jung_radii.json) as a dictionary with the following format:
        {
            ...
            <gan>:
                {
                    'Z': (<w>, <b>),
                    'W':
                        {
                            ...
                            <stylegan-layer>: (<w>, <b>),
                            ...
                        },
                },
            ...
        }
    so as, given a truncation parameter t, the radius is given as `w * t + b`.

    Options:
        -v, --verbose    : set verbose mode on
        --num-samples    : set the number of latent codes to sample for generating images
        --cuda           : use CUDA (default)
        --no-cuda        : do not use CUDA
    """
    parser = argparse.ArgumentParser(description="Fit a linear model for the jung radius of GAN's latent code given "
                                                 "a truncation parameter")
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose mode on")
    parser.add_argument('--num-samples', type=int, default=1000, help="set number of latent codes to sample")
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

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

    # Build jung radii dictionary and populate it
    nested_dict = lambda: defaultdict(nested_dict)
    jung_radii_dict = nested_dict()
    for gan in GENFORCE_MODELS.keys():
        ################################################################################################################
        ##                                                                                                            ##
        ##                                               [ StyleGANs ]                                                ##
        ##                                                                                                            ##
        ################################################################################################################
        if 'stylegan' in gan:
            ############################################################################################################
            ##                                                                                                        ##
            ##                                         [ StyleGAN / Z-space ]                                         ##
            ##                                                                                                        ##
            ############################################################################################################
            # Build GAN generator model and load with pre-trained weights
            if args.verbose:
                print("  \\__Build GAN generator model G and load with pre-trained weights...")
                print("      \\__GAN generator : {} (res: {})".format(gan, GENFORCE_MODELS[gan][1]))
                print("      \\__Pre-trained weights: {}".format(GENFORCE_MODELS[gan][0]))

            G = load_generator(model_name=gan, latent_is_w=False, verbose=args.verbose).eval()

            # Upload GAN generator model to GPU
            if use_cuda:
                G = G.cuda()

            # Latent codes sampling
            if args.verbose:
                print("  \\__Sample {} {}-dimensional latent codes...".format(args.num_samples, G.dim_z))
            zs = torch.randn(args.num_samples, G.dim_z)

            if use_cuda:
                zs = zs.cuda()

            # Calculate expected latent norm
            if args.verbose:
                print("  \\__Calculate Jung radius...")
            jung_radius = torch.cdist(zs, zs).max() * np.sqrt(G.dim_z / (2 * (G.dim_z + 1)))
            jung_radii_dict[gan]['Z'] = (0.0, jung_radius.cpu().detach().item())

            ############################################################################################################
            ##                                                                                                        ##
            ##                                       [ StyleGAN / W/W+-space ]                                        ##
            ##                                                                                                        ##
            ############################################################################################################
            # Build GAN generator model and load with pre-trained weights
            if args.verbose:
                print("  \\__Build GAN generator model G and load with pre-trained weights...")
                print("      \\__GAN generator : {} (res: {})".format(gan, GENFORCE_MODELS[gan][1]))
                print("      \\__Pre-trained weights: {}".format(GENFORCE_MODELS[gan][0]))

            G = load_generator(model_name=gan, latent_is_w=True, verbose=args.verbose).eval()

            # Upload GAN generator model to GPU
            if use_cuda:
                G = G.cuda()

            # Latent codes sampling
            if args.verbose:
                print("  \\__Sample {} {}-dimensional latent codes...".format(args.num_samples, G.dim_z))
            zs = torch.randn(args.num_samples, G.dim_z)

            if use_cuda:
                zs = zs.cuda()

            # Get number of W layers for the given StyleGAN
            stylegan_num_layers = G.get_w(zs, truncation=1.0).shape[1]

            # Calculate expected latent norm and fit a linear model for each version of the W+ space
            if args.verbose:
                print("  \\__Calculate Jung radii and fit linear models...")
            data_per_layer = dict()
            tmp = []
            for truncation in tqdm(np.linspace(0.1, 1.0, 100), desc="  \\__Calculate radii (W space): "):
                ws = G.get_w(zs, truncation=truncation)[:, 0, :]
                jung_radius = torch.cdist(ws, ws).max() * np.sqrt(ws.shape[1] / (2 * (ws.shape[1] + 1)))
                tmp.append([truncation, jung_radius.cpu().detach().item()])
            data_per_layer.update({0: tmp})

            for ll in tqdm(range(1, stylegan_num_layers), desc="  \\__Calculate radii (W+ space): "):
                tmp = []
                for truncation in np.linspace(0.1, 1.0, 100):
                    ws_plus = G.get_w(zs, truncation=truncation)[:, :ll + 1, :]
                    ws_plus = ws_plus.reshape(ws_plus.shape[0], -1)
                    jung_radius = torch.cdist(ws_plus, ws_plus).max() * \
                        np.sqrt(ws_plus.shape[1] / (2 * (ws_plus.shape[1] + 1)))
                    tmp.append([truncation, jung_radius.cpu().detach().item()])
                data_per_layer.update({ll: tmp})

            for ll, v in tqdm(data_per_layer.items(), desc="  \\__Fit linear models"):
                v = np.array(v)
                lm = linear_model.LinearRegression()
                lm.fit(v[:, 0].reshape(-1, 1), v[:, 1].reshape(-1, 1))
                jung_radii_dict[gan]['W'][ll] = (float(lm.coef_[0, 0]), float(lm.intercept_[0]))

        ################################################################################################################
        ##                                                                                                            ##
        ##                                                [ ProgGAN ]                                                 ##
        ##                                                                                                            ##
        ################################################################################################################
        else:
            # Build GAN generator model and load with pre-trained weights
            if args.verbose:
                print("  \\__Build GAN generator model G and load with pre-trained weights...")
                print("      \\__GAN generator : {} (res: {})".format(gan, GENFORCE_MODELS[gan][1]))
                print("      \\__Pre-trained weights: {}".format(GENFORCE_MODELS[gan][0]))

            G = load_generator(model_name=gan, latent_is_w=False, verbose=args.verbose).eval()

            # Upload GAN generator model to GPU
            if use_cuda:
                G = G.cuda()

            # Latent codes sampling
            if args.verbose:
                print("  \\__Sample {} {}-dimensional latent codes...".format(args.num_samples, G.dim_z))
            zs = torch.randn(args.num_samples, G.dim_z)

            if use_cuda:
                zs = zs.cuda()

            # Calculate expected latent norm
            if args.verbose:
                print("  \\__Calculate Jung radius...")
            jung_radius = torch.cdist(zs, zs).max() * np.sqrt(G.dim_z / (2 * (G.dim_z + 1)))

            print("jung_radius")
            print(jung_radius)
            print(type(jung_radius))

            jung_radii_dict[gan]['Z'] = (0.0, jung_radius.cpu().detach().item())

    # Save expected latent norms dictionary
    with open(osp.join('models', 'jung_radii.json'), 'w') as fp:
        json.dump(jung_radii_dict, fp)


if __name__ == '__main__':
    main()

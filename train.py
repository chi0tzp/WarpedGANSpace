import argparse
import torch
import json
import os.path as osp
from lib import *
from lib import GENFORCE_MODELS, STYLEGAN_LAYERS
from models.load_generator import load_generator


def main():
    """WarpedGANSpace -- Training script.

    Options:
        ===[ Pre-trained GAN Generator (G) ]============================================================================
        --gan                     : set pre-trained GAN generator (see GENFORCE_MODELS in lib/config.py)
        --stylegan-space          : set StyleGAN's latent space (Z, W, W+) to look for interpretable paths
        --stylegan-layer          : choose up to which StyleGAN's layer to use for learning latent paths
                                    E.g., if --stylegan-layer=11, then interpretable paths will be learnt in a
                                    (12 * 512)-dimensional latent space.
        --truncation              : set W-space truncation parameter. If set, W-space codes will be truncated

        ===[ Support Sets (S) ]=========================================================================================
        -K, --num-support-sets    : set number of support sets; i.e., number of warping functions -- number of
                                    interpretable paths
        -D, --num-support-dipoles : set number of support dipoles per support set
        --learn-alphas            : learn RBF alpha params
        --learn-gammas            : learn RBF gamma params
        --gamma                   : set RBF gamma param (by default, gamma will be set to the inverse of the latent
                                    space dimensionality)
        --support-set-lr          : set learning rate for learning support sets

        ===[ Reconstructor (R) ]========================================================================================
        --reconstructor-type      : set reconstructor network type
        --min-shift-magnitude     : set minimum shift magnitude
        --max-shift-magnitude     : set maximum shift magnitude
        --reconstructor-lr        : set learning rate for reconstructor R optimization

        ===[ Training ]=================================================================================================
        --max-iter                : set maximum number of training iterations
        --batch-size              : set training batch size
        --lambda-cls              : classification loss weight
        --lambda-reg              : regression loss weight
        --log-freq                : set number iterations per log
        --ckp-freq                : set number iterations per checkpoint model saving

        ===[ CUDA ]=====================================================================================================
        --cuda                    : use CUDA during training (default)
        --no-cuda                 : do NOT use CUDA during training
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="WarpedGANSpace training script")

    # === Pre-trained GAN Generator (G) ============================================================================== #
    parser.add_argument('--gan', type=str, choices=GENFORCE_MODELS.keys(), help='GAN generator model')
    parser.add_argument('--stylegan-space', type=str, default='Z', choices=('Z', 'W', 'W+'),
                        help="StyleGAN's latent space")
    parser.add_argument('--stylegan-layer', type=int, default=11, choices=range(18),
                        help="choose up to which StyleGAN's layer to use for learning latent paths")
    parser.add_argument('--truncation', type=float, help="latent code sampling truncation parameter")

    # === Support Sets (S) ======================================================================== #
    parser.add_argument('-K', '--num-support-sets', type=int, help="set number of support sets (warping functions)")
    parser.add_argument('-D', '--num-support-dipoles', type=int, help="set number of support dipoles per support set")
    parser.add_argument('--learn-alphas', action='store_true', help='learn RBF alpha params')
    parser.add_argument('--learn-gammas', action='store_true', help='learn RBF gamma params')
    parser.add_argument('-b', '--beta', type=float, default=0.1, help="set RBF beta param; when --learn-gammas is set, "
                                                                      "this will be used to initialise the RBF gammas")
    parser.add_argument('--support-set-lr', type=float, default=1e-4, help="set learning rate")

    # === Reconstructor (R) ========================================================================================== #
    parser.add_argument('--min-shift-magnitude', type=float, default=0.25, help="set minimum shift magnitude")
    parser.add_argument('--max-shift-magnitude', type=float, default=0.45, help="set shifts magnitude scale")
    parser.add_argument('--reconstructor-lr', type=float, default=1e-4,
                        help="set learning rate for reconstructor R optimization")

    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=100000, help="set maximum number of training iterations")
    parser.add_argument('--batch-size', type=int, default=32, help="set batch size")
    parser.add_argument('--lambda-cls', type=float, default=1.00, help="classification loss weight")
    parser.add_argument('--lambda-reg', type=float, default=0.25, help="regression loss weight")
    parser.add_argument('--log-freq', default=10, type=int, help='set number iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='set number iterations per checkpoint model saving')

    # === CUDA ======================================================================================================= #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # Check StyleGAN's layer
    if 'stylegan' in args.gan:
        if (args.stylegan_layer < 0) or (args.stylegan_layer > STYLEGAN_LAYERS[args.gan] - 1):
            raise ValueError("Invalid stylegan_layer for given GAN ({}). Choose between 0 and {}".format(
                args.gan, STYLEGAN_LAYERS[args.gan] - 1))

    # Create output dir and save current arguments
    exp_dir = create_exp_dir(args)

    # CUDA
    use_cuda = False
    multi_gpu = False
    if torch.cuda.is_available():
        if args.cuda:
            use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if torch.cuda.device_count() > 1:
                multi_gpu = True
        else:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Build GAN generator model and load with pre-trained weights
    print("#. Build GAN generator model G and load with pre-trained weights...")
    print("  \\__GAN generator : {} (res: {})".format(args.gan, GENFORCE_MODELS[args.gan][1]))
    print("  \\__Pre-trained weights: {}".format(GENFORCE_MODELS[args.gan][0]))
    G = load_generator(model_name=args.gan,
                       latent_is_w=('stylegan' in args.gan) and ('W' in args.stylegan_space),
                       verbose=True)

    # Set support vector dimensionality and initial gamma param
    support_vectors_dim = G.dim_z
    if ('stylegan' in args.gan) and (args.stylegan_space == 'W+'):
        support_vectors_dim *= (args.stylegan_layer + 1)

    # Get Jung radii
    with open(osp.join('models', 'jung_radii.json'), 'r') as f:
        jung_radii_dict = json.load(f)

    if 'stylegan' in args.gan:
        if 'W+' in args.stylegan_space:
            lm = jung_radii_dict[args.gan]['W']['{}'.format(args.stylegan_layer)]
        elif 'W' in args.stylegan_space:
            lm = jung_radii_dict[args.gan]['W']['0']
        else:
            lm = jung_radii_dict[args.gan]['Z']
        jung_radius = lm[0] * args.truncation + lm[1]
    else:
        jung_radius = jung_radii_dict[args.gan]['Z'][1]

    # Build Support Sets model S
    print("#. Build Support Sets S...")
    print("  \\__Number of Support Sets    : {}".format(args.num_support_sets))
    print("  \\__Number of Support Dipoles : {}".format(args.num_support_dipoles))
    print("  \\__Support Vectors dim       : {}".format(support_vectors_dim))
    print("  \\__Learn RBF alphas          : {}".format(args.learn_alphas))
    print("  \\__Learn RBF gammas          : {}".format(args.learn_gammas))
    print("  \\__RBF beta param            : {}".format(args.beta))
    print("  \\_Jung radius                : {}".format(jung_radius))

    S = SupportSets(num_support_sets=args.num_support_sets,
                    num_support_dipoles=args.num_support_dipoles,
                    support_vectors_dim=support_vectors_dim,
                    learn_alphas=args.learn_alphas,
                    learn_gammas=args.learn_gammas,
                    beta=args.beta,
                    jung_radius=jung_radius)

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in S.parameters() if p.requires_grad)))

    # Build reconstructor model R
    print("#. Build reconstructor model R...")

    R = Reconstructor(dim=S.num_support_sets)

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in R.parameters() if p.requires_grad)))

    # Set up trainer
    print("#. Experiment: {}".format(exp_dir))
    trn = Trainer(params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu)

    # Train
    trn.train(generator=G, support_sets=S, reconstructor=R)


if __name__ == '__main__':
    main()

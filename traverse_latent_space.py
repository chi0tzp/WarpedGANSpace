import argparse
import os
import os.path as osp
import torch
from torch import nn
from PIL import Image, ImageDraw
import json
from torchvision.transforms import ToPILImage
from lib import *
from models.gan_load import build_biggan, build_proggan, build_stylegan2, build_sngan


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class ModelArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def tensor2image(tensor, adaptive=False):
    # Squeeze tensor image
    tensor = tensor.squeeze(dim=0)

    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
    else:
        tensor = (tensor + 1) / 2
        tensor.clamp(0, 1)
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))


def build_gan(gan_type, target_classes, stylegan2_resolution, stylegan2_w_shift, use_cuda, multi_gpu):
    # -- BigGAN
    if gan_type == 'BigGAN':
        G = build_biggan(pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]],
                         target_classes=target_classes)
    # -- ProgGAN
    elif gan_type == 'ProgGAN':
        G = build_proggan(pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]])
    # -- StyleGAN2
    elif gan_type == 'StyleGAN2':
        G = build_stylegan2(pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][stylegan2_resolution],
                            resolution=stylegan2_resolution,
                            shift_in_w=stylegan2_w_shift)
    # -- Spectrally Normalised GAN (SNGAN)
    else:
        G = build_sngan(pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]],
                        gan_type=gan_type)

    # Upload GAN generator model to GPU
    if use_cuda:
        G = G.cuda()

    # Parallelize GAN generator model into multiple GPUs, if available and `args.multi_gpu=True`.
    if multi_gpu:
        G = DataParallelPassthrough(G)

    return G


def one_hot(dims, value, idx):
    vec = torch.zeros(dims)
    vec[idx] = value
    return vec


def get_concat_h(img_file_orig,
                 shifted_img_file,
                 size,
                 img_id,
                 s,
                 shift_steps,
                 path_id,
                 draw_header=False,
                 draw_progress_bar=True):
    img_orig = Image.open(img_file_orig).resize((size, size))
    img_orig_w = img_orig.width
    img_orig_h = img_orig.height

    img_shifted = Image.open(shifted_img_file).resize((size, size))
    img_shifted_w = img_shifted.width

    dst = Image.new('RGB', (img_orig_w + img_shifted_w, img_orig_h))
    dst.paste(img_orig, (0, 0))
    dst.paste(img_shifted, (img_orig_w, 0))

    # Add header with img_id and path_id
    if draw_header:
        draw = ImageDraw.Draw(dst)
        offset_w = 6
        offset_h = 6
        t_w = 270
        t_h = 13
        draw.rectangle(xy=[(offset_w, offset_h), (offset_w + t_w, offset_h + t_h)], fill=(0, 0, 0))
        draw.text((offset_w + 2, offset_h + 2), "{}/{:03d}".format(img_id, path_id), fill=(255, 255, 255))

    # Draw progress bar
    if draw_progress_bar:
        draw = ImageDraw.Draw(dst)
        bar_h = 7
        bar_color = (252, 186, 3)
        draw.rectangle(xy=[(size, size - bar_h), ((1 + s / shift_steps) * size, size)], fill=bar_color)

    return dst


def main():
    """A script for traversing the latent space of a pre-trained GAN generator through paths defined by the warpings of
    a set of support vectors, or by linear pre-trained [1] or random directions. Latent codes are either sampled
    randomly or they are pooled from a pre-defined collection via the `--pool` argument. The generated images are stored
    under `results/` directory.

    Options:
        ================================================================================================================
        -v, --verbose : set verbose mode on
        ================================================================================================================
        --exp         : set experiment's model dir, as created by `train.py`, i.e., it should contain a sub-directory
                        `models/` with two files, namely `reconstructor.pt` and `support_sets.pt`, which
                        contain the weights for the reconstructor and the support sets, respectively, and an `args.json`
                        file that contains the arguments the model has been trained with.
        --pool        : directory of pre-defined pool of latent codes (created by `sample_gan.py`)
        ================================================================================================================
        --shift-steps : number of shifts to be applied to each latent code at each direction (positive/negative).
                        That is, the total number of shifts applied to each latent code will be equal to
                        2 * args.shift_steps.
        --eps         : shift step magnitude for generating G(z'), where z' = z + eps * direction.
        --path-step   : path step (after how many steps to generate images)
        --batch-size  : generator batch size
        --img-size    : image size
        --img-quality : JPEG image quality (max 95)
        ================================================================================================================
        --cuda        : use CUDA (default)
        --no-cuda     : do not use CUDA
        --multi-gpu   : use data parallelism (multiple GPUs) when possible
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="Generate transformed images given a pre-trained GAN generator")
    parser.add_argument('-v', '--verbose', action='store_true', help="set verbose mode on")
    # ================================================================================================================ #
    parser.add_argument('--exp', type=str, required=True, help="set experiment's model dir (created by `train.py`)")
    parser.add_argument('--pool', type=str, required=True, help="directory of pre-defined pool of latent codes"
                                                                "(created by `sample_gan.py`)")
    # ---------------------------------------------------------------------------------------------------------------- #
    parser.add_argument('--shift-steps', type=int, default=16, help="set number of shifts +++")
    parser.add_argument('--eps', type=float, default=0.5, help="set shift magnitude")
    parser.add_argument('--path-step', type=int, default=1, help="")
    parser.add_argument('--batch-size', type=int, help="set generator batch size")
    parser.add_argument('--img-size', type=int, default=256, help="set image resolution")
    parser.add_argument('--img-quality', type=int, default=75, help="set JPEG image quality")
    # ---------------------------------------------------------------------------------------------------------------- #
    parser.add_argument('--gif', action='store_true', help="Create GIF traversals")
    parser.add_argument('--gif-size', type=int, default=196, help="set gif resolution")
    parser.add_argument('--gif-fps', type=int, default=30, help="set gif frame rate")
    # ================================================================================================================ #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    parser.add_argument('--multi-gpu', action='store_true', help="use data parallelism (multiple GPUs)")
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # Check structure of `args.exp`
    if not osp.isdir(args.exp):
        raise NotADirectoryError("Invalid given directory: {}".format(args.exp))
    args_json_file = osp.join(args.exp, 'args.json')

    if not osp.isfile(args_json_file):
        raise FileNotFoundError("File not found: {}".format(args_json_file))
    models_dir = osp.join(args.exp, 'models')
    if not osp.isdir(models_dir):
        raise NotADirectoryError("Invalid models directory: {}".format(models_dir))
    support_sets_model = osp.join(models_dir, 'support_sets.pt')
    if not osp.isfile(support_sets_model):
        raise FileNotFoundError("Support sets model file not found: {}".format(support_sets_model))
    reconstructor_model = osp.join(models_dir, 'reconstructor.pt')
    if not osp.isfile(reconstructor_model):
        raise FileNotFoundError("Reconstructor model file not found: {}".format(reconstructor_model))

    # Read pre-trained model arguments
    args_json = ModelArgs(**json.load(open(args_json_file)))
    gan_type = args_json.__dict__["gan_type"]

    # Check given pool directory
    pool = osp.join('experiments', 'latent_codes', gan_type, args.pool)
    if not osp.isdir(pool):
        raise NotADirectoryError("Invalid pool directory: {} -- not found under experiments/latent_codes/{}/. "
                                 "Please run sample_gan.py to create it.".format(args.pool, gan_type))

    # Set default tensor type
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if args.multi_gpu and torch.cuda.device_count() < 2:
                print("*** WARNING ***: Multiple GPUs (--multi-gpu) cannot be used in this machine due to insufficient "
                      "number of available GPU devices.")
                args.multi_gpu = False
        if not args.cuda:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
            args.multi_gpu = False
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        args.multi_gpu = False
    use_cuda = args.cuda and torch.cuda.is_available()

    # Build GAN generator model and load with pre-trained weights
    if args.verbose:
        print("#. Build GAN generator model G and load with pre-trained weights...")
        print("  \\__GAN type: {}".format(gan_type))
        print("  \\__Pre-trained weights: {}".format(
            GAN_WEIGHTS[gan_type]['weights'][args_json.__dict__["stylegan2_resolution"]]
            if gan_type == 'StyleGAN2' else GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]]))

    G = build_gan(gan_type=gan_type,
                  target_classes=args_json.__dict__["biggan_target_classes"],
                  stylegan2_resolution=args_json.__dict__["stylegan2_resolution"],
                  stylegan2_w_shift=args_json.__dict__["stylegan2_w_shift"],
                  use_cuda=use_cuda,
                  multi_gpu=args.multi_gpu).eval()

    # Build support sets model S
    if args.verbose:
        print("#. Build support sets model S...")

    S = SupportSets(num_support_sets=args_json.__dict__["num_support_sets"],
                    num_support_dipoles=args_json.__dict__["num_support_dipoles"],
                    support_vectors_dim=G.dim_z,
                    learn_alphas=args_json.__dict__["learn_alphas"],
                    learn_gammas=args_json.__dict__["learn_gammas"],
                    gamma=1.0 / G.dim_z if args_json.__dict__["gamma"] is None else args_json.__dict__["gamma"])

    # Load pre-trained weights and set to evaluation mode
    if args.verbose:
        print("  \\__Pre-trained weights: {}".format(support_sets_model))
    S.load_state_dict(torch.load(support_sets_model, map_location=lambda storage, loc: storage))
    if args.verbose:
        print("  \\__Set to evaluation mode")
    S.eval()

    # Upload support sets model to GPU
    if use_cuda:
        S = S.cuda()

    # Set number of generative paths
    num_gen_paths = S.num_support_sets

    # Create output dir for generated images
    out_dir = osp.join(args.exp, 'results', args.pool, '{}'.format(args.eps))
    os.makedirs(out_dir, exist_ok=True)

    # Set default batch size
    if args.batch_size is None:
        args.batch_size = 2 * args.shift_steps + 1

    ## ============================================================================================================== ##
    ##                                                                                                                ##
    ##                                              [Latent Codes Pool]                                               ##
    ##                                                                                                                ##
    ## ============================================================================================================== ##
    # Get latent codes from the given pool
    if args.verbose:
        print("#. Use latent codes from pool {}...".format(args.pool))
    latent_codes_dirs = [dI for dI in os.listdir(pool) if os.path.isdir(os.path.join(pool, dI))]
    latent_codes_dirs.sort()
    latent_codes = []
    for subdir in latent_codes_dirs:
        latent_codes.append(torch.load(osp.join(pool, subdir, 'latent_code.pt'),
                                       map_location=lambda storage, loc: storage))
    zs = torch.cat(latent_codes)
    num_of_latent_codes = zs.size()[0]

    if use_cuda:
        zs = zs.cuda()

    ## ============================================================================================================== ##
    ##                                                                                                                ##
    ##                                            [Latent space traversal]                                            ##
    ##                                                                                                                ##
    ## ============================================================================================================== ##
    if args.verbose:
        print("#. Traverse latent space...")
        print("  \\__{}".format(out_dir))

    # Iterate over given latent codes
    for i in range(num_of_latent_codes):
        # Un-squeeze current latent code in shape [1, dim] and create hash code for it
        z_ = zs[i, :].unsqueeze(0)

        latent_code_hash = latent_codes_dirs[i]
        if args.verbose:
            update_progress("  \\__.Latent code hash: {} [{:03d}/{:03d}] ".format(latent_code_hash,
                                                                                  i+1,
                                                                                  num_of_latent_codes),
                            num_of_latent_codes, i)

        # Create directory for current latent code
        latent_code_dir = osp.join(out_dir, '{}'.format(latent_code_hash))
        os.makedirs(latent_code_dir, exist_ok=True)

        # Create directory for storing path images
        transformed_images_root_dir = osp.join(latent_code_dir, 'paths_images')
        os.makedirs(transformed_images_root_dir, exist_ok=True)

        # Keep all latent paths the current latent code (sample)
        paths_latent_codes = []

        ## ========================================================================================================== ##
        ##                                                                                                            ##
        ##                                             [ Path Traversal ]                                             ##
        ##                                                                                                            ##
        ## ========================================================================================================== ##
        # Iterate over (interpretable) directions
        for dim in range(num_gen_paths):
            if args.verbose:
                print()
                update_progress("      \\__path: {:03d}/{:03d} ".format(dim + 1, num_gen_paths), num_gen_paths, dim + 1)

            # Create shifted latent codes (for the given latent code z) and generate transformed images
            transformed_images = []

            # Current path's latent codes and shifts lists
            current_path_latent_codes = [z_]
            current_path_latent_shifts = [torch.zeros_like(z_).cuda() if use_cuda else torch.zeros_like(z_)]

            ## ====================================================================================================== ##
            ##                                                                                                        ##
            ##                    [ Traverse through current path (positive/negative directions) ]                    ##
            ##                                                                                                        ##
            ## ====================================================================================================== ##
            # == Positive direction ==
            z = z_.clone()
            cnt = 0
            for _ in range(args.shift_steps):
                cnt += 1
                # Calculate shift vector based on current z
                support_sets_mask = torch.zeros(1, S.num_support_sets)
                support_sets_mask[0, dim] = 1.0
                if use_cuda:
                    support_sets_mask.cuda()
                # Get latent space shift vector
                with torch.no_grad():
                    shift = args.eps * S(support_sets_mask, z)

                # Update z
                z = z + shift

                # Store latent codes and shifts
                if cnt == args.path_step:
                    current_path_latent_shifts.append(shift)
                    current_path_latent_codes.append(z)
                    cnt = 0

            # == Negative direction ==
            z = z_.clone()
            cnt = 0
            for _ in range(args.shift_steps):
                cnt += 1
                # Calculate shift vector based on current z
                support_sets_mask = torch.zeros(1, S.num_support_sets)
                support_sets_mask[0, dim] = 1.0
                if use_cuda:
                    support_sets_mask.cuda()
                # Get latent space shift vector
                with torch.no_grad():
                    shift = -args.eps * S(support_sets_mask, z)

                # Update z
                z = z + shift
                # Store latent codes and shifts
                if cnt == args.path_step:
                    current_path_latent_shifts = [shift] + current_path_latent_shifts
                    current_path_latent_codes = [z] + current_path_latent_codes
                    cnt = 0

            # Generate transformed images
            # Split latent codes and shifts in batches
            current_path_latent_codes = torch.cat(current_path_latent_codes)
            current_path_latent_codes_batches = torch.split(current_path_latent_codes, args.batch_size)
            current_path_latent_shifts = torch.cat(current_path_latent_shifts)
            current_path_latent_shifts_batches = torch.split(current_path_latent_shifts, args.batch_size)

            if len(current_path_latent_codes_batches) != len(current_path_latent_shifts_batches):
                raise AssertionError()
            else:
                num_batches = len(current_path_latent_codes_batches)

            transformed_img = []
            for t in range(num_batches):
                with torch.no_grad():
                    transformed_img.append(G(z=current_path_latent_codes_batches[t],
                                             shift=current_path_latent_shifts_batches[t]))
            transformed_img = torch.cat(transformed_img)

            # Convert tensors (transformed images) into PIL images
            for t in range(transformed_img.size()[0]):
                transformed_images.append(tensor2image(transformed_img[t, :].cpu(),
                                                       adaptive=True).resize((args.img_size, args.img_size)))

            # Save all images in `transformed_images` list under `transformed_images_root_dir/<path_<dim>/`
            transformed_images_dir = osp.join(transformed_images_root_dir, 'path_{:03d}'.format(dim))
            os.makedirs(transformed_images_dir, exist_ok=True)
            for t in range(len(transformed_images)):
                transformed_images[t].save(osp.join(transformed_images_dir, '{:06d}.jpg'.format(t)),
                                           "JPEG", quality=args.img_quality, optimize=True, progressive=True)
                # Save original image
                if (t == len(transformed_images) // 2) and (dim == 0):
                    transformed_images[t].save(osp.join(latent_code_dir, 'original_image.jpg'),
                                               "JPEG", quality=95, optimize=True, progressive=True)

            # Append latent paths
            paths_latent_codes.append(current_path_latent_codes.unsqueeze(0))

            if args.verbose:
                update_stdout(1)
        # ============================================================================================================ #

        # Save all latent paths and shifts for the current latent code (sample) in a tensor of size:
        #   paths_latent_codes : torch.Size([num_gen_paths, 2 * args.shift_steps + 1, G.dim_z])
        torch.save(torch.cat(paths_latent_codes), osp.join(latent_code_dir, 'paths_latent_codes.pt'))

        if args.verbose:
            update_stdout(1)
            print()
            print()

    # Collate traversal GIFs
    if args.gif:
        # Build results file structure
        structure = dict()
        generated_img_subdirs = [dI for dI in os.listdir(out_dir) if os.path.isdir(osp.join(out_dir, dI)) and
                                 dI not in ('paths_gifs', 'validation_results')]
        generated_img_subdirs.sort()
        for img_id in generated_img_subdirs:
            structure.update({img_id: {}})
            path_images_dir = osp.join(out_dir, '{}'.format(img_id), 'paths_images')
            path_images_subdirs = [dI for dI in os.listdir(path_images_dir)
                                   if os.path.isdir(os.path.join(path_images_dir, dI))]
            path_images_subdirs.sort()
            for item in path_images_subdirs:
                structure[img_id].update({item: [dI for dI in os.listdir(osp.join(path_images_dir, item))
                                                 if osp.isfile(os.path.join(path_images_dir, item, dI))]})

        # Create directory for storing traversal GIFs
        os.makedirs(osp.join(out_dir, 'paths_gifs'), exist_ok=True)

        # For each interpretable path (warping function), collect the generated image sequences for each original latent
        # code and collate them into a GIF file
        print("#. Collate GIFs...")
        num_of_frames = list()
        for dim in range(num_gen_paths):
            if args.verbose:
                update_progress("  \\__path: {:03d}/{:03d} ".format(dim + 1, num_gen_paths), num_gen_paths, dim + 1)

            gif_frames = []
            for img_id in structure.keys():
                original_img_file = osp.join(out_dir, '{}'.format(img_id), 'original_image.jpg')
                shifted_images_dir = osp.join(out_dir, '{}'.format(img_id), 'paths_images', 'path_{:03d}'.format(dim))

                row_frames = []
                img_id_num_of_frames = 0
                for t in range(len(structure[img_id]['path_{:03d}'.format(dim)])):
                    img_id_num_of_frames += 1
                for t in range(len(structure[img_id]['path_{:03d}'.format(dim)])):
                    shifted_img_file = osp.join(shifted_images_dir, '{:06d}.jpg'.format(t))

                    # Concatenate `original_img_file` and `shifted_img_file`
                    row_frames.append(get_concat_h(img_file_orig=original_img_file,
                                                   shifted_img_file=shifted_img_file,
                                                   size=args.gif_size,
                                                   img_id=img_id,
                                                   s=t,
                                                   shift_steps=img_id_num_of_frames,
                                                   path_id=dim))
                num_of_frames.append(img_id_num_of_frames)
                gif_frames.append(row_frames)

            if len(set(num_of_frames)) > 1:
                print("#. Warning: Inconsistent number of frames for image sequences: {}".format(num_of_frames))

            # Create full GIF frames
            full_gif_frames = []
            for f in range(int(num_of_frames[0])):
                gif_f = Image.new('RGB', (2 * args.gif_size, len(structure) * args.gif_size))
                for i in range(len(structure)):
                    gif_f.paste(gif_frames[i][f], (0, i * args.gif_size))
                full_gif_frames.append(gif_f)

            # Save gif
            im = Image.new(mode='RGB', size=(2 * args.gif_size, len(structure) * args.gif_size))
            im.save(
                fp=osp.join(out_dir, 'paths_gifs', 'path_{:03d}.gif'.format(dim)),
                append_images=full_gif_frames,
                save_all=True,
                optimize=True,
                loop=0,
                duration=1000 // args.gif_fps)


if __name__ == '__main__':
    main()

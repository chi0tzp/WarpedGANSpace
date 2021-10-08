import os
import argparse
import os.path as osp
import math
from PIL import Image, ImageDraw


def get_gif_size(img_paths):
    img_widths = []
    img_heights = []
    for img in img_paths:
        img_ = Image.open(img)
        img_widths.append(img_.width)
        img_heights.append(img_.height)

    if len(set(img_widths)) == len(set(img_heights)) == 1:
        return img_widths[0], img_heights[1]
    else:
        raise ValueError("Inconsistent image dimensions.")


def main():
    """Create GIF image (static and animated image sequence).

    Options:
        ================================================================================================================
        --dir      : set directory of latent code hash for a given experiment, pool, and traversal config
        --path-id  : set path id
        --num-imgs : set number of static images per sequence; by default all available images in the path will be used
        --gif-size : set GIF image size (otherwise use dimensions of given images)
        --gif-fps  : set number of frames per second for the generated GIF image
        ================================================================================================================

    Example:
        python create_gif.py --gif-size=196 --num-imgs=7 --dir=experiments/complete/ProgGAN-ResNet-K200-D512-LearnGammas-eps0.1_0.2/results/ProgGAN_4_F/56_0.15_8.4/435c92ab04f994fd192526b9107396747caf283a/ --path-id=96

    """
    parser = argparse.ArgumentParser("Create GIF image (static and animated image sequence) for the given latent path.")
    parser.add_argument('--dir', type=str, required=True, help="set ???")

    parser.add_argument('--path-id', type=int, required=True, help="set path id")
    parser.add_argument('--num-imgs', type=int, help="set number of static images per sequence; by default all "
                                                     "available images in the path will be used")
    parser.add_argument('--gif-size', type=int, default=128, help="GIF image size")
    parser.add_argument('--gif-fps', type=int, default=30, help="set GIF frame rate")

    # Parse given arguments
    args = parser.parse_args()

    # GIF Config
    frame_line_w = 6
    frame_outline_color = 'black'
    gap = 15
    progress_bar_h = 15
    progress_bar_color = (252, 186, 3)

    # Check if given path directory exists
    img_path_dir = osp.join(args.dir, 'paths_images', 'path_{:03d}'.format(args.path_id))

    print("#. Create GIF...")
    if not osp.isdir(img_path_dir):
        raise NotADirectoryError("Invalid directory: {}".format(img_path_dir))
    else:
        print("  \\__Image path dir: {}".format(img_path_dir))

    path_images = [osp.join(img_path_dir, dI) for dI in os.listdir(img_path_dir)
                   if osp.isfile(osp.join(img_path_dir, dI))]
    path_images.sort()

    num_images = len(path_images)
    print("  \\__Number of images in path: {}".format(num_images))

    if args.num_imgs is None:
        args.num_imgs = num_images
    elif args.num_imgs > num_images:
        args.num_imgs = num_images
    print("  \\__Number of images used for static image: {}".format(args.num_imgs))

    # Get paths of static images
    static_imgs = []
    for i in range(0, len(path_images), math.ceil(len(path_images) / args.num_imgs)):
        static_imgs.append(osp.join(img_path_dir, '{:06}.jpg'.format(i)))
    args.num_imgs = len(static_imgs)

    # Get GIF image resolution
    if args.gif_size:
        gif_w = gif_h = args.gif_size
    else:
        gif_w, gif_h = get_gif_size(static_imgs)

    latent_code_gifs_dir = osp.join(args.dir, 'paths_gifs')
    os.makedirs(latent_code_gifs_dir, exist_ok=True)

    # Create PIL static image
    static_img_pil = Image.new('RGB', size=(len(static_imgs) * gif_w, gif_h))
    for i in range(len(static_imgs)):
        static_img_pil.paste(Image.open(static_imgs[i]).resize((gif_w, gif_h)), (i * gif_w, 0))

    # Create PIL GIF frames
    gif_frames = []
    for i in range(len(path_images)):
        # Create new PIL frame
        gif_frame_pil = Image.new('RGB', size=((args.num_imgs + 1) * gif_w + gap, gif_h), color=(255, 255, 255))

        # Paste static image
        gif_frame_pil.paste(static_img_pil, (0, 0))

        # Paste current image
        gif_frame_pil.paste(Image.open(path_images[i]).resize((gif_w, gif_h)), (args.num_imgs * gif_w + gap, 0))

        # Draw frames and progress bar
        gif_frame_pil_drawing = ImageDraw.Draw(gif_frame_pil)

        gif_frame_pil_drawing.rectangle(xy=[0, 0, args.num_imgs * gif_w, gif_h],
                                        width=frame_line_w,
                                        outline=frame_outline_color)

        progress = (i / len(path_images)) * gif_w
        gif_frame_pil_drawing.rectangle(xy=[args.num_imgs * gif_w + gap, gif_h - progress_bar_h,
                                            args.num_imgs * gif_w + gap + progress, gif_h],
                                        fill=progress_bar_color)
        gif_frame_pil_drawing.rectangle(xy=[args.num_imgs * gif_w + gap, 0,
                                            args.num_imgs * gif_w + gap + gif_w, gif_h],
                                        width=frame_line_w,
                                        outline=frame_outline_color)
        # Append to GIF frames list
        gif_frames.append(gif_frame_pil)

    # Save GIF file
    gif_pil = Image.new(mode='RGB', size=((args.num_imgs + 1) * gif_w + gap, gif_h))
    gif_pil.save(
        fp=osp.join(latent_code_gifs_dir, 'path_{:03d}.gif'.format(args.path_id)),
        append_images=gif_frames,
        save_all=True,
        optimize=True,
        loop=0,
        duration=1000 // args.gif_fps)


if __name__ == '__main__':
    main()

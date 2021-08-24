import sys
import os
import os.path as osp
import json
import argparse
import numpy as np
import torch
from scipy.stats import truncnorm


class TrainingStatTracker(object):
    def __init__(self):
        self.stat_tracker = {
            'accuracy': [],
            'classification_loss': [],
            'regression_loss': [],
            'total_loss': []
        }

    def update(self, accuracy, classification_loss, regression_loss, total_loss):
        self.stat_tracker['accuracy'].append(float(accuracy))
        self.stat_tracker['classification_loss'].append(float(classification_loss))
        self.stat_tracker['regression_loss'].append(float(regression_loss))
        self.stat_tracker['total_loss'].append(float(total_loss))

    def get_means(self):
        stat_means = dict()
        for key, value in self.stat_tracker.items():
            stat_means.update({key: np.mean(value)})
        return stat_means

    def flush(self):
        for key in self.stat_tracker.keys():
            self.stat_tracker[key] = []


def sample_z(batch_size, dim_z, truncation=None):
    """Sample a random latent code from multi-variate standard Gaussian distribution with/without truncation.

    Args:
        batch_size (int)   : batch size (number of latent codes)
        dim_z (int)        : latent space dimensionality
        truncation (float) : TODO: +++

    Returns:
        z (torch.Tensor)   : batch of latent codes
    """
    if truncation is None or truncation == 1.0:
        return torch.randn(batch_size, dim_z)
    else:
        return torch.from_numpy(truncnorm.rvs(-truncation, truncation, size=(batch_size, dim_z))).to(torch.float)


def create_exp_dir(args):
    """Create output directory for current experiment under experiments/wip/ and save given the arguments (json) and
    the given command (bash script).

    Experiment's directory name format:

        <gan_type>(-<stylegan2_resolution>)-<reconstructor_type>-K<num_support_sets>-N<num_support_dipoles>
            (-LearnAlphas)(-LearnGammas)-eps<min_shift_magnitude>_<max_shift_magnitude>
    E.g.:

        experiments/wip/ProgGAN-ResNet-K200-N32-LearnGammas-eps0.35_0.5

    Args:
        args (argparse.Namespace): the namespace object returned by `parse_args()` for the current run

    """
    exp_dir = "{}".format(args.gan_type)
    if args.gan_type == 'StyleGAN2':
        exp_dir += '-{}'.format(args.stylegan2_resolution)
        if args.stylegan2_w_shift:
            exp_dir += '-W'
        else:
            exp_dir += '-Z'
    if args.gan_type == 'BigGAN':
        biggan_classes = '-'
        for c in args.biggan_target_classes:
            biggan_classes += '{}'.format(c)
        exp_dir += '{}'.format(biggan_classes)
    exp_dir += "-{}".format(args.reconstructor_type)
    exp_dir += "-K{}-N{}".format(args.num_support_sets, args.num_support_dipoles)
    if args.learn_alphas:
        exp_dir += '-LearnAlphas'
    if args.learn_gammas:
        exp_dir += '-LearnGammas'
    exp_dir += "-eps{}_{}".format(args.min_shift_magnitude, args.max_shift_magnitude)

    # Create output directory (wip)
    wip_dir = osp.join("experiments", "wip", exp_dir)
    os.makedirs(wip_dir, exist_ok=True)
    # Save args namespace object in json format
    with open(osp.join(wip_dir, 'args.json'), 'w') as args_json_file:
        json.dump(args.__dict__, args_json_file)

    # Save the given command in a bash script file
    with open(osp.join(wip_dir, 'command.sh'), 'w') as command_file:
        command_file.write('#!/usr/bin/bash\n')
        command_file.write(' '.join(sys.argv) + '\n')

    return exp_dir


def update_progress(msg, total, progress):
    bar_length, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(bar_length * progress))
    block_symbol = u"\u2588"
    empty_symbol = u"\u2591"
    text = "\r{}{} {:.0f}% {}".format(msg, block_symbol * block + empty_symbol * (bar_length - block),
                                      round(progress * 100, 0), status)
    sys.stdout.write(text)
    sys.stdout.flush()


def update_stdout(num_lines):
    """Update stdout by moving cursor up and erasing line for given number of lines.

    Args:
        num_lines (int): number of lines

    """
    cursor_up = '\x1b[1A'
    erase_line = '\x1b[1A'
    for _ in range(num_lines):
        print(cursor_up + erase_line)


def sec2dhms(t):
    """Convert time into days, hours, minutes, and seconds string format.

    Args:
        t (float): time in seconds

    Returns (string):
        "<days> days, <hours> hours, <minutes> minutes, and <seconds> seconds"

    """
    day = t // (24 * 3600)
    t = t % (24 * 3600)
    hour = t // 3600
    t %= 3600
    minutes = t // 60
    t %= 60
    seconds = t
    return "%02d days, %02d hours, %02d minutes, and %02d seconds" % (day, hour, minutes, seconds)

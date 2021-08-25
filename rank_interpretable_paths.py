import argparse
import os
import os.path as osp
import torch
import numpy as np
import pandas as pd
import json
import shutil
from PIL import Image
import matplotlib.pyplot as plt


ALL_ATTRIBUTES = ('identity', 'yaw', 'pitch', 'roll', 'race', 'age', 'gender', 'au_1', 'au_2', 'au_4', 'au_5',
                  'au_6', 'au_9', 'au_12', 'au_17', 'au_25', 'arousal', 'valence', 'smile', 'bald', 'gray_hair',
                  'brown_hair', 'black_hair', 'blond_hair', 'straight_hair', 'wavy_hair', 'wearing_hat',
                  'arched_eyebrows', 'big_nose', 'pointy_nose', 'no_beard', 'narrow_eyes', 'eyeglasses',
                  'face_width', 'face_height')

AUs = {
    "au_1": "Inner Brow Raiser",
    "au_2": "Outer Brow Raiser",
    "au_4": "Brow Lowerer",
    "au_5": "Upper Lid_Raiser",
    "au_6": "Cheek Raiser",
    "au_9": "Nose Wrinkler",
    "au_12": "Lip Corner Puller",
    "au_17": "Chin Raiser",
    "au_25": "Lips part"
}

# Set attributes min-max ranges
yaw_lim = 1.1
pitch_lim = 0.5
roll_lim = 0.3

attribute_ranges = {
    # === Identity comparison (ArcFace) ===
    'identity': np.array([0, 1.0]),
    # === Pose estimation (HopeNet) ===
    'yaw': np.array([-yaw_lim, +yaw_lim]),
    'pitch': np.array([-pitch_lim, +pitch_lim]),
    'roll': np.array([-roll_lim, +roll_lim]),
    # === FairFace ===
    'race': np.array([0.0, 1.0]),
    'age': np.array([0.0, 1.0]),
    'gender': np.array([0.0, 1.0]),
    # === AUs (EmotioNet) ===
    'au_1_Inner_Brow_Raiser': np.array([0.0, 0.5]),
    'au_2_Outer_Brow_Raiser': np.array([0.0, 0.5]),
    'au_4_Brow_Lowerer': np.array([0.0, 0.5]),
    'au_5_Upper_Lid_Raiser': np.array([0.0, 0.5]),
    'au_6_Cheek_Raiser': np.array([0.0, 0.5]),
    'au_9_Nose_Wrinkler': np.array([0.0, 0.5]),
    'au_12_Lip_Corner_Puller': np.array([0.0, 0.5]),
    'au_17_Chin_Raiser': np.array([0.0, 0.5]),
    'au_25_Lips_part': np.array([0.0, 0.5]),
    # === AffectNet ===
    'arousal': np.array([-0.30, 0.30]),
    'valence': np.array([-0.30, 0.30]),
    # === CelebA Attributes ===
    'smile': np.array([0.0, 0.4]),
    'bald': np.array([0.0, 1.0]),
    'gray_hair': np.array([0.0, 1.0]),
    'brown_hair': np.array([0.0, 1.0]),
    'black_hair': np.array([0.0, 1.0]),
    'blond_hair': np.array([0.0, 1.0]),
    'straight_hair': np.array([0.0, 1.0]),
    'wavy_hair': np.array([0.0, 0.5]),
    'wearing_hat': np.array([0.0, 1.0]),
    'arched_eyebrows': np.array([0.0, 0.5]),
    'big_nose': np.array([0.0, 0.5]),
    'pointy_nose': np.array([0.0, 1.0]),
    'no_beard': np.array([0.0, 1.0]),
    'narrow_eyes': np.array([0.0, 1.0]),
    'eyeglasses': np.array([0.0, 0.5]),
    # === Face detection (SFD) ===
    'face_width': np.array([0.0, 1.0]),
    'face_height': np.array([0.0, 1.0])
}


def l1(x):
    """Perform L1-normalization."""
    x_ = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_[i, j] = x[i, j] / np.abs(x[i]).sum()
    return x_


def concat_images_h(img_dir, trimmed_frames=1, concat_step=3):
    img_list = [osp.join(img_dir, dI) for dI in os.listdir(img_dir) if osp.isfile(osp.join(img_dir, dI))]
    img_list.sort()

    # Trim list of frames
    img_list = [img_list[i] for i in range(trimmed_frames, len(img_list) - trimmed_frames, concat_step)]

    images = [Image.open(x) for x in img_list]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_img = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_img.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_img


def save_results(X, PHI, R, metric, header, hashes_dir, hashes, sort_by_attribute=True, top_k=3,
                 trimmed_frames=6, concat_step=3):
    """

    Args:
        X (np.ndarray)           :
        PHI (np.ndarray)         :
        R (np.ndarray)           :
        metric (str)             :
        hashes ()                :
        header (list)            :
        hashes_dir (str)         :
        sort_by_attribute (bool) :
        top_k (int)              :
        trimmed_frames (int)     :
        concat_step (int)        :

    Returns:

    """
    # TODO: add comment ================================================================================================
    dest_paths_dir = osp.join(hashes_dir, 'validation_results', metric)
    os.makedirs(dest_paths_dir, exist_ok=True)

    df = pd.DataFrame(X)
    df.to_csv(path_or_buf=osp.join(dest_paths_dir, 'validation_results_{}.csv'.format(metric)),
              header=header,
              index_label='path_id',
              float_format='%.3f')

    if sort_by_attribute:
        indices_dict = dict()
        for i in range(top_k):
            indices_dict.update({i: []})
        first_rows = []
        first_phis = []
        first_ranges = []
        for t in range(df.shape[1]):
            df_sorted_by_t = df.abs().sort_values(by=t, ascending=False)
            PHI_sorted_by_t = PHI[df_sorted_by_t.index.tolist()]
            first_phis.append(PHI_sorted_by_t[0])
            R_sorted_by_t = R[df_sorted_by_t.index.tolist()]
            first_ranges.append(R_sorted_by_t[0, t])
            # REVIEW
            first_rows.append(df_sorted_by_t.to_numpy()[0, :])
            # first_rows.append(df.reindex(df_sorted_by_t.index).to_numpy()[0, :])
            for i in range(top_k):
                indices_dict[i].append(df_sorted_by_t.index.tolist()[i])

            df_sorted_by_t.to_csv(
                path_or_buf=osp.join(dest_paths_dir, 'validation_results_{}_sorted_by_{}.csv'.format(metric, header[t])),
                header=header,
                float_format='%.3f')

        pd.DataFrame(np.stack(first_rows)).to_csv(
            path_or_buf=osp.join(dest_paths_dir, 'validation_results_{}_diag.csv'.format(metric)),
            header=header,
            float_format='%.2f')

        # Copy paths images
        for k, indices in indices_dict.items():
            for t in range(len(indices)):
                dest_paths_hashes_dir = osp.join(dest_paths_dir, '{}_path_{}_{:03d}'.format(header[t], k+1, indices[t]))
                os.makedirs(dest_paths_hashes_dir, exist_ok=True)

                # Copy full gif, if available
                full_gif_file_src = osp.join(hashes_dir, 'paths_gifs', 'path_{:03d}.gif'.format(indices[t]))
                full_gif_file_dest = osp.join(dest_paths_hashes_dir, '{}_path_{}_{:03d}.gif'.format(header[t], k+1, indices[t]))
                if osp.exists(full_gif_file_src):
                    shutil.copy(full_gif_file_src, full_gif_file_dest)

                for h in hashes:
                    # TODO: add comment
                    # shutil.copytree(osp.join(hashes_dir, h, 'paths_images', 'path_{:03d}'.format(indices[t])),
                    #                 osp.join(dest_paths_hashes_dir, h), dirs_exist_ok=True)

                    # concatenate images in `osp.join(dest_paths_hashes_dir, h)` and save image under
                    #  `dest_paths_hashes_dir` as osp.join(dest_paths_hashes_dir, '{}.jpeg'.format(h))
                    # ---
                    # img = concat_images_h(img_dir=osp.join(dest_paths_hashes_dir, h),
                    #                       trimmed_frames=trimmed_frames,
                    #                       concat_step=concat_step)
                    img = concat_images_h(img_dir=osp.join(hashes_dir, h, 'paths_images', 'path_{:03d}'.format(indices[t])),
                                          trimmed_frames=trimmed_frames,
                                          concat_step=concat_step)
                    img.save(osp.join(dest_paths_hashes_dir, '{}.jpeg'.format(h)), "JPEG", quality=95, optimize=True,
                             progressive=True)

    # TODO: add comment ================================================================================================
    dest_paths_dir = osp.join(hashes_dir, 'validation_results', '{}_l1'.format(metric))
    os.makedirs(dest_paths_dir, exist_ok=True)

    df = pd.DataFrame(l1(np.abs(X)))
    df.to_csv(path_or_buf=osp.join(dest_paths_dir, 'validation_results_{}_l1.csv'.format(metric)),
              header=header,
              index_label='path_id',
              float_format='%.3f')

    if sort_by_attribute:
        indices_dict = dict()
        for i in range(top_k):
            indices_dict.update({i: []})
        first_rows = []
        first_phis = []
        first_ranges = []
        for t in range(df.shape[1]):
            df_sorted_by_t = df.sort_values(by=t, ascending=False)
            PHI_sorted_by_t = PHI[df_sorted_by_t.index.tolist()]
            first_phis.append(PHI_sorted_by_t[0])
            R_sorted_by_t = R[df_sorted_by_t.index.tolist()]
            # first_ranges.append(R_sorted_by_t[0, t])
            first_ranges.append(R_sorted_by_t[:3, t].mean())

            # REVIEW
            first_rows.append(df_sorted_by_t.to_numpy()[0, :])
            # first_rows.append(df.reindex(df_sorted_by_t.index).to_numpy()[0, :])
            for i in range(top_k):
                indices_dict[i].append(df_sorted_by_t.index.tolist()[i])
            df_sorted_by_t.to_csv(
                path_or_buf=osp.join(dest_paths_dir,
                                     'validation_results_{}_l1_sorted_by_{}.csv'.format(metric, header[t])),
                header=header,
                float_format='%.3f')

        pd.DataFrame(np.stack(first_rows)).to_csv(
            path_or_buf=osp.join(dest_paths_dir, 'validation_results_{}_l1_diag.csv'.format(metric)),
            header=header,
            float_format='%.2f')

        # Copy paths images
        for k, indices in indices_dict.items():
            for t in range(len(indices)):
                dest_paths_hashes_dir = osp.join(dest_paths_dir, '{}_path_{}_{:03d}'.format(header[t], k+1, indices[t]))
                os.makedirs(dest_paths_hashes_dir, exist_ok=True)

                # Copy full gif, if available
                full_gif_file_src = osp.join(hashes_dir, 'paths_gifs', 'path_{:03d}.gif'.format(indices[t]))
                full_gif_file_dest = osp.join(dest_paths_hashes_dir, '{}_path_{}_{:03d}.gif'.format(header[t], k+1, indices[t]))
                if osp.exists(full_gif_file_src):
                    shutil.copy(full_gif_file_src, full_gif_file_dest)

                for h in hashes:
                    # shutil.copytree(osp.join(hashes_dir, h, 'paths_images', 'path_{:03d}'.format(indices[t])),
                    #                 osp.join(dest_paths_hashes_dir, h),
                    #                 dirs_exist_ok=True)
                    # Concatenate images in `osp.join(dest_paths_hashes_dir, h)` and save image under
                    # `dest_paths_hashes_dir` as osp.join(dest_paths_hashes_dir, '{}.jpeg'.format(h))
                    img = concat_images_h(img_dir=osp.join(hashes_dir, h, 'paths_images', 'path_{:03d}'.format(indices[t])),
                                          trimmed_frames=trimmed_frames,
                                          concat_step=concat_step)
                    img.save(osp.join(dest_paths_hashes_dir, '{}.jpeg'.format(h)), "JPEG", quality=95, optimize=True,
                             progressive=True)


def main():
    """A script for validating the interpretability of the learned paths using the the evaluation results produced by
    `evaluate.py`.

    TODO: +++


    Options:
        -v, --verbose : set verbose mode on
        --traversal   : set traversal directory as generated by `traverse.py`, e.g.,
                            results/rbf/ProgGAN-ResNet-LearnSV-LearnAlphas-LearnGammas/
                        A traversal directory should contain one or more sub-directories of the form <eps>/, where
                        eps is a shift magnitude step (e.g., 0.5/).
        --pool       : set pool of latent codes
        TODO: If the following two arguments are specified, evaluation (attribute space traversals) will be performed only for
        the given configuration (subdir of experiments/<exp>/results/<gan_type>/<pool>/):
        --shift-steps : number of shift steps (per positive/negative path direction)
        --eps         : shift magnitude
    """
    parser = argparse.ArgumentParser(description="Post-process evaluation results")
    parser.add_argument('-v', '--verbose', action='store_true', help="set verbose mode on")
    # ================================================================================================================ #
    parser.add_argument('--exp', type=str, required=True, help="set experiment's model dir (created by `train.py`)")
    parser.add_argument('--pool', type=str, required=True, help="choose pool of pre-defined latent codes and their "
                                                                "attribute traversals (found under <exp>/results/)")
    parser.add_argument('--shift-steps', type=int, default=16, help="number of shifts per positive/negative path "
                                                                    "direction")
    parser.add_argument('--eps', type=float, help="shift magnitude")
    # ================================================================================================================ #
    parser.add_argument('-a', '--attributes', nargs='+', type=str,
                        # default=['identity', 'yaw', 'pitch', 'smile', 'race', 'gender', 'age', 'au_4', 'wavy_hair'],
                        # default=['identity', 'yaw', 'pitch', 'smile', 'race', 'wavy_hair'],
                        default=['identity', 'yaw', 'pitch', 'roll', 'race', 'age', 'gender', 'au_1_Inner_Brow_Raiser',
                                 'au_2_Outer_Brow_Raiser', 'au_4_Brow_Lowerer', 'au_5_Upper_Lid_Raiser',
                                 'au_6_Cheek_Raiser', 'au_9_Nose_Wrinkler', 'au_12_Lip_Corner_Puller',
                                 'au_17_Chin_Raiser', 'au_25_Lips_part', 'arousal', 'valence', 'smile', 'bald',
                                 'gray_hair', 'brown_hair', 'black_hair', 'blond_hair', 'straight_hair', 'wavy_hair',
                                 'wearing_hat', 'arched_eyebrows', 'big_nose', 'pointy_nose', 'no_beard',
                                 'narrow_eyes', 'eyeglasses', 'face_width', 'face_height'],
                        help="list of evaluation attributes")
    # ================================================================================================================ #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # Check given experiment's directory
    latent_traversal_dir = osp.join(args.exp, 'results', args.pool)
    if not osp.isdir(args.exp):
        raise NotADirectoryError("Error: invalid experiment's directory: {}".format(args.exp))
    else:
        if not osp.isdir(latent_traversal_dir):
            raise NotADirectoryError("Error: pool directory {} not found under {}".format(
                args.pool, osp.join(args.exp, 'results')))

    # TODO: Get shift magnitude steps under the given latent traversals directory
    if (args.shift_steps is None) and (args.eps is None):
        latent_space_traversal_configs = [dI for dI in os.listdir(latent_traversal_dir) if
                                          osp.isdir(os.path.join(latent_traversal_dir, dI))]
    else:
        latent_space_traversal_configs = ['{}_{}_{}'.format(2 * args.shift_steps,
                                                            args.eps,
                                                            round(2 * args.shift_steps * args.eps, 3))]

    if args.verbose:
        print("#. Rank interpretable paths in {}".format(latent_traversal_dir))
        print("  \\__.Latent space traversal configs: {}".format(latent_space_traversal_configs))

    # Build attributes ranges matrix
    # att_ranges = np.stack([att_1_range, att_2_range, att_3_range])
    att_ranges_list = []
    for a in args.attributes:
        att_ranges_list.append(attribute_ranges[a])
    att_ranges = np.stack(att_ranges_list)

    for l_config in latent_space_traversal_configs:
        if args.verbose:
            print("       \\__.Latent space traversal config: {}".format(l_config))

        # Get shift magnitude, number of shift steps, and traversal length
        eps, shift_steps, traversal_length = l_config.split('_')

        # Get samples hashes for current eps
        hashes_dir = osp.join(latent_traversal_dir, '{}_{}_{}'.format(2 * args.shift_steps, args.eps,
                                                                      round(2 * args.shift_steps * args.eps, 3)))
        hashes = [dI for dI in os.listdir(hashes_dir)
                  if osp.isdir(os.path.join(hashes_dir, dI)) and dI not in ('paths_gifs', 'validation_results')]

        # Get attributes for all samples (hashes) for the given eps and store in numpy array of shape:
        #   [num_samples, num_of_attributes, num_of_paths, num_of_path_steps]
        ATTRIBUTES = []
        LATENT_CODES = []
        for i in range(len(hashes)):
            if args.verbose:
                print("           \\__.hash: {} [{}/{}]".format(hashes[i], i + 1, len(hashes)))
            h_dir = osp.join(hashes_dir, hashes[i])
            h_eval_np_dir = osp.join(h_dir, 'eval_np')
            SAMPLE_ATTRIBUTES = []
            for a in args.attributes:
                SAMPLE_ATTRIBUTES.append(np.load(osp.join(h_eval_np_dir, '{}.npy'.format(a))))
            ATTRIBUTES.append(SAMPLE_ATTRIBUTES)
            LATENT_CODES.append(torch.load(osp.join(h_dir, 'paths_latent_codes.pt')))
        ATTRIBUTES = np.array(ATTRIBUTES)
        LATENT_CODES = torch.stack(LATENT_CODES, dim=0)

        # Reshape as [num_samples, num_of_paths, num_of_attributes, num_of_path_steps]
        ATTRIBUTES = np.transpose(ATTRIBUTES, axes=(0, 2, 1, 3))

        ATTRIBUTES_min = np.min(ATTRIBUTES, axis=3)
        ATTRIBUTES_max = np.max(ATTRIBUTES, axis=3)
        # ATTRIBUTES_range = np.abs(ATTRIBUTES_max - ATTRIBUTES_min)
        ATTRIBUTES_range = ATTRIBUTES_max - ATTRIBUTES_min
        ATTRIBUTES_range = ATTRIBUTES_range.mean(0)

        num_of_samples = ATTRIBUTES.shape[0]
        num_of_paths = ATTRIBUTES.shape[1]
        num_of_attributes = ATTRIBUTES.shape[2]
        num_of_points_per_path = ATTRIBUTES.shape[3]
        if args.verbose:
            print("           \\__.ATTRIBUTES: {}".format(ATTRIBUTES.shape))
            print("               \\__Number of samples         : {}".format(num_of_samples))
            print("               \\__Number of paths           : {}".format(num_of_paths))
            print("               \\__Number of attributes      : {}".format(num_of_attributes))
            print("               \\__Number of points per path : {}".format(num_of_points_per_path))

        with open(osp.join(hashes_dir, 'traversal_details.json'), 'w') as fp:
            json.dump({
                'eps': eps,
                'num_of_samples': num_of_samples,
                'num_of_paths': num_of_paths,
                'num_of_attributes': num_of_attributes,
                'num_of_points_per_path': num_of_points_per_path
            }, fp)

        # REVIEW
        LATENT_CODE_STARTS = LATENT_CODES[:, :, 0, :]
        LATENT_CODE_ENDS = LATENT_CODES[:, :, -1, :]
        LATENT_CODE_STRINGS = LATENT_CODE_ENDS - LATENT_CODE_STARTS
        LATENT_CODE_STRING_LENGTHS = torch.norm(LATENT_CODE_STRINGS, dim=2)
        LATENT_CODE_PATH_LENGTHS = torch.ones_like(LATENT_CODE_STRING_LENGTHS) * (num_of_points_per_path - 1) * float(eps)
        PHI = LATENT_CODE_STRING_LENGTHS / LATENT_CODE_PATH_LENGTHS
        PHI = PHI.mean(0).cpu().numpy()

        # Calculate attribute STDs
        ATTRIBUTES_STD = np.std(ATTRIBUTES, axis=3)
        ATTRIBUTES_STD_MEAN = np.mean(ATTRIBUTES_STD, axis=0)

        # Sum of attribute changes
        ATTRIBUTES_SUM_DELTAS = np.zeros((num_of_samples, num_of_paths, num_of_attributes))
        ATTRIBUTES_SUM_ABS_DELTAS = np.zeros((num_of_samples, num_of_paths, num_of_attributes))

        # TODO: add comment
        ATTRIBUTES_IDX_CORR_UNNORM = np.zeros((num_of_samples, num_of_paths, num_of_attributes))

        for s in range(num_of_samples):
            for k in range(num_of_paths):
                # For the s-th samples, k-th path, A is a matrix with shape [num_attributes, num_of_path_steps].
                # Each row of A gives the values of the corresponding attribute across the sequence of images for the
                # given path. E.g., A: (24, 33)
                A = ATTRIBUTES[s, k, :, :]

                # Scale attributes in given ranges
                A_scaled = (2.0 * (A.transpose() - att_ranges[:, 0]) / (
                            att_ranges[:, 1] - att_ranges[:, 0]) - 1.0).transpose()
                A_scaled[A_scaled < -1.0] = -1.0
                A_scaled[A_scaled > +1.0] = +1.0
                A = A_scaled.copy()

                # Sum of attribute changes
                delta_A = np.zeros((A.shape[0], A.shape[1] - 1))
                abs_delta_A = np.zeros((A.shape[0], A.shape[1] - 1))
                for j in range(1, A.shape[1]):
                    abs_delta_A[:, j - 1] = np.abs(A[:, j] - A[:, j - 1])
                    delta_A[:, j - 1] = A[:, j] - A[:, j - 1]
                ATTRIBUTES_SUM_ABS_DELTAS[s, k] = abs_delta_A.sum(1)
                ATTRIBUTES_SUM_DELTAS[s, k] = delta_A.sum(1)

                for t in range(A.shape[0]):
                    A_t = A[t, :]
                    if t == args.attributes.index('identity'):
                        A_t_idx = np.concatenate([-1 * np.arange(-A_t.shape[0] // 2 + 1, 0),
                                                  np.arange(A_t.shape[0] // 2 + 1)])
                    else:
                        A_t_idx = np.arange(A_t.shape[0])
                    # REVIEW: This gives warnings when A_t is constant
                    # ATTRIBUTES_IDX_CORR_UNNORM[s, k, t] = np.nan_to_num(np.corrcoef(A_t, A_t_idx), nan=0.0)[0, 1] * \
                    #     np.sqrt(np.cov(A_t))
                    ATTRIBUTES_IDX_CORR_UNNORM[s, k, t] = np.cov(A_t, A_t_idx)[0, 1] / np.sqrt(np.cov(A_t_idx))

        # Average over samples
        ATTRIBUTES_SUM_ABS_DELTAS_MEAN = ATTRIBUTES_SUM_ABS_DELTAS.mean(0)
        ATTRIBUTES_SUM_DELTAS_MEAN = ATTRIBUTES_SUM_DELTAS.mean(0)
        ATTRIBUTES_IDX_CORR_UNNORM_MEAN = ATTRIBUTES_IDX_CORR_UNNORM.mean(0)

        # Save results
        if args.verbose:
            print("           \\__.Save results...")

        # Create output directory for validation results
        validation_results_dir = osp.join(hashes_dir, 'validation_results')
        if osp.exists(validation_results_dir):
            shutil.rmtree(validation_results_dir)
        os.makedirs(validation_results_dir, exist_ok=True)

        # Save phi coefficients
        plt.figure(figsize=(10, 2))
        plt.plot(np.sort(PHI))
        plt.savefig(fname=osp.join(validation_results_dir, 'phi.svg'), dpi=300, format='svg')

        # Save attribute correlation results
        if args.verbose:
            print("               \\__.Correlation...")
        save_results(X=ATTRIBUTES_IDX_CORR_UNNORM_MEAN,
                     PHI=PHI,
                     R=ATTRIBUTES_range,
                     metric='corr',
                     header=args.attributes,
                     hashes_dir=hashes_dir,
                     hashes=hashes,
                     trimmed_frames=1,
                     concat_step=3)


if __name__ == '__main__':
    main()

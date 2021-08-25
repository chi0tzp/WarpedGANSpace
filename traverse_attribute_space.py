import argparse
import os
import os.path as osp
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import json
import torchvision
from torchvision import transforms
from lib import update_progress, update_stdout
from lib import PathImages, IDComparator, SFDDetector, Hopenet, FANet

# CelebA attributes
CELEBA_ATTRIBUTES = ('bald', 'gray_hair', 'brown_hair', 'black_hair', 'blond_hair', 'straight_hair', 'wavy_hair',
                     'wearing_hat', 'arched_eyebrows', 'big_nose', 'pointy_nose', 'no_beard', 'narrow_eyes',
                     'eyeglasses')

# AUs dictionary
AUs = {
    "au_1": "Inner_Brow_Raiser",
    "au_2": "Outer_Brow_Raiser",
    "au_4": "Brow_Lowerer",
    "au_5": "Upper_Lid_Raiser",
    "au_6": "Cheek_Raiser",
    "au_9": "Nose_Wrinkler",
    "au_12": "Lip_Corner_Puller",
    "au_17": "Chin_Raiser",
    "au_25": "Lips_part"
}


def crop_face(images, idx, bbox, padding=0.0):
    """Crop faces from given images for the given bounding boxes and padding."""
    x_min = int((1.0 - padding) * bbox[0])
    y_min = int((1.0 - padding) * bbox[1])
    x_max = int((1.0 + padding) * bbox[2])
    y_max = int((1.0 + padding) * bbox[3])

    x_min -= 50
    x_max += 50
    y_min -= 50
    y_max += 30
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)

    x_max = min(images.shape[2], x_max)
    y_max = min(images.shape[3], y_max)
    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_max)
    y_max = int(y_max)

    return images[idx, :, x_min:x_max, y_min:y_max].unsqueeze(0)


def json_exists(json_file):
    """Check if given json file exists and contains a non-empty json object (an empty json object takes 2 bytes)."""
    return osp.exists(json_file) and os.stat(json_file).st_size > 2


def main():
    """A script for traversing the attribute space of the generated paths (as calculated by `traverse_latent_space.py`).
    Attribute space includes the following:
        -- Face bounding box (in terms of face width and height), using SFD [1] face detector
        -- Age, race, and gender ("femaleness") using FairFace [2]
        -- Pose (in terms of the Euler angles, yaw, pitch, and roll) using Hopenet [3]
        -- Identity score (compare to the original image) using ArcFace [4]
        -- Smile using an in-house model pre-trained on CelebA [5]
        -- Nine AUs using an in-house model pre-trained on EmotioNet [6]
        -- Arousal/Valence using an in-house model pre-trained on AffectNet [7]

    Options:
        ================================================================================================================
        -v, --verbose : set verbose mode on
        ================================================================================================================
        --exp         : set experiment's model dir, as created by `train.py` and used by `traverse_latent_space.py`.
                        That is, it should contain the traversals for at least one latent codes/images pool, under the
                        results/ directory.
        --pool        : set pool of latent codes (should be a subdirectory of experiments/<exp>/results/<gan_type>, as
                        created by traverse_latent_space.py)
        If the following two arguments are specified, evaluation (attribute space traversals) will be performed only for
        the given configuration (subdir of experiments/<exp>/results/<gan_type>/<pool>/):
        --shift-steps : number of shift steps (per positive/negative path direction)
        --eps         : shift magnitude
        ================================================================================================================
        --cuda      : use CUDA during training (default).
        --no-cuda   : do NOT use CUDA during training.
        ================================================================================================================

    References:
        [1] Zhang, Shifeng, et al. "S3FD: Single shot scale-invariant face detector." Proceedings of the IEEE
            international conference on computer vision. 2017.
        [2] Karkkainen, Kimmo, and Jungseock Joo. "FairFace: Face attribute dataset for balanced race, gender, and age."
            arXiv preprint arXiv:1908.04913 (2019).
        [3] Doosti, Bardia, et al. "Hope-net: A graph-based model for hand-object pose estimation." Proceedings of the
            IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
        [4] Deng, Jiankang, et al. "ArcFace: Additive angular margin loss for deep face recognition." Proceedings of the
            IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
        [5] Liu, Z., Luo, P., Wang, X., and Tang, X. Deep learning face attributes in the wild. In Proceedings of
            International Conference on Computer Vision (ICCV), December 2015.
        [6] Fabian Benitez-Quiroz, C., Ramprakash  Srinivasan, and Aleix M. Martinez. "Emotionet: An accurate, real-time
            algorithm for the automatic annotation of a million facial expressions  in the wild." *Proceedings of the
            IEEE conference on computer vision and pattern recognition*. 2016.
        [7] Mollahosseini, Ali, Behzad Hasani, and Mohammad H. Mahoor. "Affectnet: A database for facial expression,
            valence, and arousal computing in the wild." *IEEE Transactions on Affective Computing* 10.1 (2017): 18-31.

    """
    parser = argparse.ArgumentParser(description="Traversals evaluation script")
    # ================================================================================================================ #
    parser.add_argument('-v', '--verbose', action='store_true', help="set verbose mode on")
    # ================================================================================================================ #
    parser.add_argument('--exp', type=str, required=True, help="set experiment's model dir (created by `train.py`)")
    parser.add_argument('--pool', type=str, required=True, help="choose pool of pre-defined latent codes and their "
                                                                "latent traversals (should be a subdirectory of "
                                                                "experiments/<exp>/results/, as created by "
                                                                "traverse_latent_space.py)")
    parser.add_argument('--shift-steps', type=int, default=16, help="number of shifts per positive/negative path "
                                                                    "direction")
    parser.add_argument('--eps', type=float, help="shift magnitude")
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
        print("#. Calculate attribute traversals in {}".format(latent_traversal_dir))
        print("  \\__.Latent space traversal configs: {}".format(latent_space_traversal_configs))

    # Use CUDA boolean
    use_cuda = args.cuda and torch.cuda.is_available()

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                             [ Pre-trained Models ]                                             ##
    ##                                                                                                                ##
    ####################################################################################################################

    # Define SFD face detector model
    face_detector = SFDDetector(path_to_detector='models/pretrained/sfd/s3fd-619a316812.pth',
                                device="cuda" if use_cuda else "cpu")

    # Define ID comparator based on ArcFace
    id_comp = IDComparator()
    id_comp.eval()
    if use_cuda:
        id_comp.cuda()

    # Define FairFace model for predicting gender, age, and race
    fairface = torchvision.models.resnet34(pretrained=True)
    fairface.fc = nn.Linear(fairface.fc.in_features, 18)
    fairface.load_state_dict(torch.load('models/pretrained/fairface/fairface_alldata_4race_20191111.pt'))
    fairface.eval()
    if use_cuda:
        fairface.cuda()

    # Define Hopenet pose estimator
    hopenet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    hopenet.load_state_dict(torch.load('models/pretrained/hopenet/hopenet_alpha2.pkl'))
    hopenet.eval()
    if use_cuda:
        hopenet.cuda()

    hopenet_softmax = nn.Softmax(dim=1)
    if use_cuda:
        hopenet_softmax.cuda()

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor)
    if use_cuda:
        idx_tensor = idx_tensor.cuda()

    # Define face transformation required by Hopenet and FairFace
    face_transformations = transforms.Compose([transforms.Resize(224),
                                               transforms.CenterCrop(224),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    # Define a Facial Analysis Network (FANet) for predicting smile (trained on CelebA), 9 Action Units (trained on
    # EmotioNet), and Arousal/Valence (trained on AffectNet)
    fanet_1_pretrained = 'models/pretrained/fanet/fanet-96-resnet50-1024-celeba_smile+emotionet+affectnet-sgd.pth'
    fanet_1_model = torch.load(fanet_1_pretrained, map_location=lambda storage, loc: storage)
    fanet_1_model_state_dict = fanet_1_model['state_dict']
    fanet_1_model_classification_head_ids = fanet_1_model['classification_head_ids']
    fanet_1_model_regression_head_ids = fanet_1_model['regression_head_ids']
    fanet_1_model_args = fanet_1_model['args']
    fanet_1_model_basenet = fanet_1_model_args.basenet
    fanet_1_model_face_dim = fanet_1_model_args.face_dim

    fanet_1_model_fc_cfg = {
        'cls': {'fc_1': {'units': fanet_1_model_args.fc_units, 'dropout': fanet_1_model_args.fc_dropout},
                'fc_2': {'units': fanet_1_model_args.fc_units, 'dropout': fanet_1_model_args.fc_dropout}},
        'reg': {'fc_1': {'units': fanet_1_model_args.fc_units, 'dropout': fanet_1_model_args.fc_dropout},
                'fc_2': {'units': fanet_1_model_args.fc_units, 'dropout': fanet_1_model_args.fc_dropout}}}

    fanet_1 = FANet(basenet=fanet_1_model_basenet,
                    classification_head_ids=fanet_1_model_classification_head_ids,
                    regression_head_ids=fanet_1_model_regression_head_ids,
                    face_dim=fanet_1_model_face_dim,
                    fc_cfg=fanet_1_model_fc_cfg,
                    mode='eval')
    fanet_1.load_state_dict(fanet_1_model_state_dict)
    fanet_1.eval()

    if use_cuda:
        fanet_1 = fanet_1.cuda()

    # Face transformations for CelebA's smile, EmotioNet's AUs, and AffectNet's Arousal/Valence
    fanet_1_transformations = transforms.Compose([transforms.Resize(fanet_1_model_args.face_dim),
                                                  transforms.CenterCrop(fanet_1_model_args.face_dim),
                                                  transforms.Normalize(mean=[0.384, 0.463, 0.627],
                                                                       std=[0.102, 0.099, 0.109])])

    # Define a Facial Analysis Network (FANet) for predicting CelebA attributes
    fanet_2_pretrained = 'models/pretrained/fanet/fanet-128-vgg16-512-celeba-sgd-CelebAFullImages.pth'
    fanet_2_model = torch.load(fanet_2_pretrained, map_location=lambda storage, loc: storage)
    fanet_2_model_state_dict = fanet_2_model['state_dict']
    fanet_2_model_classification_head_ids = fanet_2_model['classification_head_ids']
    fanet_2_model_regression_head_ids = fanet_2_model['regression_head_ids']
    fanet_2_model_args = fanet_2_model['args']
    fanet_2_model_basenet = fanet_2_model_args.basenet
    fanet_2_model_face_dim = fanet_2_model_args.face_dim

    fanet_2_model_fc_cfg = {
        'cls': {'fc_1': {'units': fanet_2_model_args.fc_units, 'dropout': fanet_2_model_args.fc_dropout},
                'fc_2': {'units': fanet_2_model_args.fc_units, 'dropout': fanet_2_model_args.fc_dropout}},
        'reg': {'fc_1': {'units': fanet_2_model_args.fc_units, 'dropout': fanet_2_model_args.fc_dropout},
                'fc_2': {'units': fanet_2_model_args.fc_units, 'dropout': fanet_2_model_args.fc_dropout}}}

    fanet_2 = FANet(basenet=fanet_2_model_basenet,
                    classification_head_ids=fanet_2_model_classification_head_ids,
                    regression_head_ids=fanet_2_model_regression_head_ids,
                    face_dim=fanet_2_model_face_dim,
                    fc_cfg=fanet_2_model_fc_cfg,
                    mode='eval')
    fanet_2.load_state_dict(fanet_2_model_state_dict)
    fanet_2.eval()

    if use_cuda:
        fanet_2 = fanet_2.cuda()

    # Face transformations for CelebA attributes
    fanet_2_transformations = transforms.Compose([transforms.Resize(fanet_2_model_args.face_dim),
                                                  transforms.CenterCrop(fanet_2_model_args.face_dim),
                                                  transforms.Normalize(mean=[0.384, 0.463, 0.627],
                                                                       std=[0.102, 0.099, 0.109])])

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                         [ Attribute Space Traversals ]                                         ##
    ##                                                                                                                ##
    ####################################################################################################################
    for l_config in latent_space_traversal_configs:
        if args.verbose:
            print("       \\__.Latent space traversal config: {}".format(l_config))

        # Navigate all hashes directories (i.e., sample latent codes)
        hashes_dir = osp.join(latent_traversal_dir, '{}_{}_{}'.format(2 * args.shift_steps, args.eps,
                                                                      round(2 * args.shift_steps * args.eps, 3)))
        hashes = [dI for dI in os.listdir(hashes_dir)
                  if osp.isdir(os.path.join(hashes_dir, dI)) and dI not in ('paths_gifs', 'validation_results')]

        # For each hash (i.e., for each latent code) and for all paths (warping functions), find the attributes
        # predictions across all image sequences
        hash_cnt = 0
        for h in hashes:
            hash_cnt += 1
            if args.verbose:
                print("           \\__.hash: {} [{}/{}]".format(h, hash_cnt, len(hashes)))

            h_dir = osp.join(hashes_dir, h)

            # Load path latent codes and get number of paths and points per path
            paths_latent_codes = torch.load(osp.join(h_dir, 'paths_latent_codes.pt'),
                                            map_location=lambda storage, loc: storage)
            num_of_paths = paths_latent_codes.size()[0]
            num_of_img_per_path = paths_latent_codes.size()[1]
            path_images_dir = osp.join(h_dir, 'paths_images')

            # Define json dictionaries for the various evaluation predictions
            face_bbox_dict = dict()
            id_dict = dict()
            gender_dict = dict()
            age_dict = dict()
            race_dict = dict()
            pose_dict = dict()
            smile_dict = dict()
            au_dict = dict()
            av_dict = dict()
            celeba_dict = dict()

            # Define numpy arrays for the various evaluation predictions
            face_width_np = np.zeros((num_of_paths, num_of_img_per_path))
            face_height_np = np.zeros((num_of_paths, num_of_img_per_path))
            id_np = np.zeros((num_of_paths, num_of_img_per_path))
            gender_np = np.zeros((num_of_paths, num_of_img_per_path))
            age_np = np.zeros((num_of_paths, num_of_img_per_path))
            race_np = np.zeros((num_of_paths, num_of_img_per_path))
            yaw_np = np.zeros((num_of_paths, num_of_img_per_path))
            pitch_np = np.zeros((num_of_paths, num_of_img_per_path))
            roll_np = np.zeros((num_of_paths, num_of_img_per_path))
            smile_np = np.zeros((num_of_paths, num_of_img_per_path))
            aus_np = np.zeros((len(AUs), num_of_paths, num_of_img_per_path))
            arousal_np = np.zeros((num_of_paths, num_of_img_per_path))
            valence_np = np.zeros((num_of_paths, num_of_img_per_path))
            celeba_np = np.zeros((len(CELEBA_ATTRIBUTES), num_of_paths, num_of_img_per_path))

            for d in range(num_of_paths):
                if args.verbose:
                    update_progress("               \\__path: {:03d}/{:03d} ".format(d + 1, num_of_paths),
                                    num_of_paths, d + 1)

                ########################################################################################################
                ##                                                                                                    ##
                ##                                        [ Face Detection ]                                          ##
                ##                                                                                                    ##
                ########################################################################################################
                data_loader = data.DataLoader(dataset=PathImages(root_path=osp.join(path_images_dir,
                                                                                    'path_{:03d}'.format(d))),
                                              batch_size=num_of_img_per_path,
                                              num_workers=0,
                                              shuffle=False,
                                              pin_memory=use_cuda)

                path_images_tensor = next(iter(data_loader))
                if use_cuda:
                    path_images_tensor = path_images_tensor.cuda()

                # Detect faces in path images (B x 3 x 256 x 256)
                with torch.no_grad():
                    detected_faces, _, _ = face_detector.detect_from_batch(path_images_tensor)

                ########################################################################################################
                ##                                                                                                    ##
                ##                                     [ Face Bounding Boxes ]                                        ##
                ##                                                                                                    ##
                ########################################################################################################
                face_bbox_list = []
                face_w = []
                face_h = []
                for t in range(len(detected_faces)):
                    if len(detected_faces[t]) > 0:
                        face_bbox = detected_faces[t][0].tolist()
                        face_bbox_list.append(face_bbox)
                        face_w.append((face_bbox[2] - face_bbox[0]) / 256.0)
                        face_h.append((face_bbox[3] - face_bbox[1]) / 256.0)
                    else:
                        face_w.append(256.0)
                        face_h.append(256.0)
                face_bbox_dict.update({d: face_bbox_list})
                face_width_np[d] = np.array(face_w)
                face_height_np[d] = np.array(face_h)
                ########################################################################################################

                ########################################################################################################
                ##                                                                                                    ##
                ##                                              [ ID ]                                                ##
                ##                                                                                                    ##
                ########################################################################################################
                original_img = path_images_tensor[num_of_img_per_path // 2, :].unsqueeze(0)
                id_scores = [id_comp(original_img.div(255.0).mul(2.0).add(-1.0),
                                     original_img.div(255.0).mul(2.0).add(-1.0)).item()]
                for t in range((num_of_img_per_path - 1) // 2):
                    transformed_img = path_images_tensor[num_of_img_per_path // 2 + t + 1, :].unsqueeze(0)
                    with torch.no_grad():
                        id_sim = id_comp(original_img.div(255.0).mul(2.0).add(-1.0),
                                         transformed_img.div(255.0).mul(2.0).add(-1.0))
                        id_scores.append(id_sim.item())
                for t in range((num_of_img_per_path - 1) // 2):
                    transformed_img = path_images_tensor[num_of_img_per_path // 2 - t - 1, :].unsqueeze(0)
                    with torch.no_grad():
                        id_sim = id_comp(original_img.div(255.0).mul(2.0).add(-1.0),
                                         transformed_img.div(255.0).mul(2.0).add(-1.0))
                        id_scores = [id_sim.item()] + id_scores
                # Update `id_dict` with the id scores of the current path
                id_dict.update({d: id_scores})
                id_np[d] = np.array(id_scores)
                ########################################################################################################

                ########################################################################################################
                ##                                                                                                    ##
                ##                                    [ Gender / Age  / Race ]                                        ##
                ##                                                                                                    ##
                ########################################################################################################
                cropped_faces = torch.zeros(len(detected_faces), 3, 224, 224)
                for t in range(len(detected_faces)):
                    cropped_faces[t] = face_transformations(crop_face(images=path_images_tensor,
                                                                      idx=t,
                                                                      bbox=detected_faces[t][0][:-1]
                                                                      if len(detected_faces[t]) > 0
                                                                      else [0, 0, 256, 256],
                                                                      padding=0.25).div(255.0))
                if use_cuda:
                    cropped_faces = cropped_faces.cuda()

                with torch.no_grad():
                    outputs = fairface(cropped_faces).cpu().detach().numpy()

                # Gender predictions
                gender_outputs = outputs[:, 7:9]
                gender_scores = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs), axis=1, keepdims=True)
                femaleness_scores = gender_scores[:, 1].tolist()
                gender_dict.update({d: femaleness_scores})
                gender_np[d] = np.array(femaleness_scores)

                # Age predictions
                age_outputs = outputs[:, 9:18]
                age_scores = np.exp(age_outputs) / np.sum(np.exp(age_outputs), axis=1, keepdims=True)
                age_predictions = np.argmax(age_scores, axis=1).tolist()
                age_dict.update({d: age_predictions})
                age_np[d] = np.array(age_predictions) / 8.0

                # Race Prediction
                # 0: 'white'
                # 1: 'East Asian'
                # 2: 'Latino_Hispanic'
                # 3: 'Southeast Asian'
                # 4: 'Indian'
                # 5: 'Middle Eastern'
                # 6: 'Black'
                race_outputs = outputs[:, :7]
                race_scores = np.exp(race_outputs) / np.sum(np.exp(race_outputs), axis=1, keepdims=True)
                race_predictions = np.argmax(race_scores, axis=1).tolist()
                race_dict.update({d: race_predictions})
                race_np[d] = np.array(race_predictions) / 6.0
                ########################################################################################################

                ########################################################################################################
                ##                                                                                                    ##
                ##                                              [ Pose ]                                              ##
                ##                                                                                                    ##
                ########################################################################################################
                cropped_faces = torch.zeros(len(detected_faces), 3, 224, 224)
                for t in range(len(detected_faces)):
                    cropped_faces[t] = face_transformations(crop_face(images=path_images_tensor,
                                                                      idx=t,
                                                                      bbox=detected_faces[t][0][:-1]
                                                                      if len(detected_faces[t]) > 0
                                                                      else [0, 0, 256, 256]).div(255.0))
                if use_cuda:
                    cropped_faces = cropped_faces.cuda()

                # Predict yaw, pitch, roll in degrees
                with torch.no_grad():
                    yaw, pitch, roll = hopenet(cropped_faces)
                yaw_predicted = hopenet_softmax(yaw)
                pitch_predicted = hopenet_softmax(pitch)
                roll_predicted = hopenet_softmax(roll)
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
                roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

                # Save yaw, pitch, roll lists in json dict (degrees)
                yaw_list_deg = yaw_predicted.cpu().detach().numpy().tolist()
                pitch_list_deg = pitch_predicted.cpu().detach().numpy().tolist()
                roll_list_deg = roll_predicted.cpu().detach().numpy().tolist()
                pose_dict.update({d: [yaw_list_deg, pitch_list_deg, roll_list_deg]})

                # Save yaw, pitch, roll numpy arrays (radians)
                yaw_np[d] = np.array(yaw_list_deg) * np.pi / 180
                pitch_np[d] = np.array(pitch_list_deg) * np.pi / 180
                roll_np[d] = np.array(roll_list_deg) * np.pi / 180
                ########################################################################################################

                ########################################################################################################
                ##                                                                                                    ##
                ##                [ CelebA-Smile / EmotioNet-ActionUnits / AffectNet Arousal-Valence ]                ##
                ##                                                                                                    ##
                ########################################################################################################
                cropped_faces = torch.zeros(len(detected_faces), 3,
                                            fanet_1_model_args.face_dim,
                                            fanet_1_model_args.face_dim)
                for t in range(len(detected_faces)):
                    cropped_faces[t] = fanet_1_transformations(crop_face(images=path_images_tensor,
                                                                         idx=t,
                                                                         bbox=detected_faces[t][0][:-1]
                                                                         if len(detected_faces[t]) > 0
                                                                         else [0, 0, 256, 256]).div(255.0))
                if use_cuda:
                    cropped_faces = cropped_faces.cuda()

                with torch.no_grad():
                    outputs = fanet_1(x=cropped_faces,
                                      head_ids=fanet_1_model_classification_head_ids + fanet_1_model_regression_head_ids)

                # Smile (CelebA)
                scores = outputs[0].data.cpu().numpy().tolist()
                smiling_scores = [item[0] for item in scores]
                smile_dict.update({d: smiling_scores})
                smile_np[d] = np.array(smiling_scores)

                # Action Units (EmotioNet)
                aus_list = []
                for c in range(1, len(fanet_1_model_classification_head_ids)):
                    scores = outputs[c].data.cpu().numpy().tolist()
                    au_scores = [item[0] for item in scores]
                    aus_list.append(au_scores)
                    aus_np[c - 1, d, :] = np.array(au_scores)
                au_dict.update({d: aus_list})

                # Arousal/Valence (AffectNet)
                scores = outputs[len(fanet_1_model_classification_head_ids) + 0].data.cpu().numpy().tolist()
                arousal_scores = [item[0] for item in scores]
                arousal_np[d] = np.array(arousal_scores)
                scores = outputs[len(fanet_1_model_classification_head_ids) + 1].data.cpu().numpy().tolist()
                valence_scores = [item[0] for item in scores]
                valence_np[d] = np.array(valence_scores)
                av_dict.update({d: [arousal_scores, valence_scores]})
                ########################################################################################################

                ########################################################################################################
                ##                                                                                                    ##
                ##                                        [ CelebA Attributes ]                                       ##
                ##                                                                                                    ##
                ########################################################################################################
                with torch.no_grad():
                    outputs = fanet_2(x=fanet_2_transformations(path_images_tensor).div(255.0),
                                      head_ids=fanet_2_model_classification_head_ids + fanet_2_model_regression_head_ids)

                # CelebA attributes
                celeba_attributes_list = []
                for c in range(len(fanet_2_model_classification_head_ids)):
                    scores = outputs[c].data.cpu().numpy().tolist()
                    celeba_attributes_scores = [item[0] for item in scores]
                    celeba_attributes_list.append(celeba_attributes_scores)
                    celeba_np[c - 1, d, :] = np.array(celeba_attributes_scores)
                celeba_dict.update({d: celeba_attributes_list})
                ########################################################################################################

                # Empty CUDA cache
                if use_cuda:
                    torch.cuda.empty_cache()

            # --- Create directory for storing evaluation results in json format
            json_files_dir = osp.join(h_dir, 'eval_json')
            os.makedirs(json_files_dir, exist_ok=True)

            # --- Create directory for storing evaluation results in numpy arrays
            np_files_dir = osp.join(h_dir, 'eval_np')
            os.makedirs(np_files_dir, exist_ok=True)

            # --- Save json and np files
            # Save `face_bbox_dict` in json format, `face_width_np` and `face_height_np` in numpy array format
            with open(osp.join(json_files_dir, 'face_bbox.json'), 'w') as out:
                json.dump(face_bbox_dict, out)
            np.save(osp.join(np_files_dir, 'face_width.npy'), face_width_np)
            np.save(osp.join(np_files_dir, 'face_height.npy'), face_height_np)

            # Save `age_dict` in json format and `age_np` in numpy array format
            with open(osp.join(json_files_dir, 'age.json'), 'w') as out:
                json.dump(age_dict, out)
            np.save(osp.join(np_files_dir, 'age.npy'), age_np)

            # Save `race_dict` in json format and `race_np` in numpy array format
            with open(osp.join(json_files_dir, 'race.json'), 'w') as out:
                json.dump(race_dict, out)
            np.save(osp.join(np_files_dir, 'race.npy'), race_np)

            # Save `gender_dict` in json format and `gender_np` in numpy array format
            with open(osp.join(json_files_dir, 'gender.json'), 'w') as out:
                json.dump(gender_dict, out)
            np.save(osp.join(np_files_dir, 'gender.npy'), gender_np)

            # Save `pose_dict` in json format and `yaw_np`, `pitch_np`, and `roll_np` in numpy array format
            with open(osp.join(json_files_dir, 'pose.json'), 'w') as out:
                json.dump(pose_dict, out)
            np.save(osp.join(np_files_dir, 'yaw.npy'), yaw_np)
            np.save(osp.join(np_files_dir, 'pitch.npy'), pitch_np)
            np.save(osp.join(np_files_dir, 'roll.npy'), roll_np)

            # Save `id_dict` in json format and `id_np` in numpy array format
            with open(osp.join(json_files_dir, 'identity.json'), 'w') as out:
                json.dump(id_dict, out)
            np.save(osp.join(np_files_dir, 'identity.npy'), id_np)

            # Save `smile_dict` in json format and `smile_np` in numpy array format
            with open(osp.join(json_files_dir, 'smile.json'), 'w') as out:
                json.dump(smile_dict, out)
            np.save(osp.join(np_files_dir, 'smile.npy'), smile_np)

            # Save `au_dict` in json format and `aus_np` in numpy array format
            with open(osp.join(json_files_dir, 'au.json'), 'w') as out:
                json.dump(au_dict, out)
            for t, k in enumerate(AUs.keys()):
                np.save(osp.join(np_files_dir, '{}_{}.npy'.format(k, AUs[k])), aus_np[t, :])

            # Save `av_dict` in json format and `arousal_np` and `valence_np` in numpy array format
            with open(osp.join(json_files_dir, 'av.json'), 'w') as out:
                json.dump(av_dict, out)
            np.save(osp.join(np_files_dir, 'arousal.npy'), arousal_np)
            np.save(osp.join(np_files_dir, 'valence.npy'), valence_np)

            # Save `celeba_dict` in json format and celeba attributes in numpy array format
            with open(osp.join(json_files_dir, 'celeba.json'), 'w') as out:
                json.dump(celeba_dict, out)
            for t in range(len(CELEBA_ATTRIBUTES)):
                np.save(osp.join(np_files_dir, '{}.npy'.format(CELEBA_ATTRIBUTES[t])), celeba_np[t, :])

    if args.verbose:
        update_stdout(1)
        print()


if __name__ == '__main__':
    main()

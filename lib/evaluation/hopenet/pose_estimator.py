"""
Calculate Euler angles (yaw, pitch, roll) using deep network HopeNet: https://github.com/natanielruiz/deep-head-pose

The face detector used is SFD (taken from face-alignment FAN) https://github.com/1adrianb/face-alignment

"""
import os
import numpy as np
import sys
from matplotlib import pyplot as plt
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

from .hopenet import Hopenet
from ..sfd.sfd_detector import SFDDetector as FaceDetector


class PoseEstimator:
    def __init__(self, device='cuda'):
        self.device = device
        if self.device == 'cuda':
            cudnn.benchmark = True
            self.is_cuda = True
        else:
            self.is_cuda = False
        # Load all needed models - Face detector and Pose detector

        # Pose model HopeNet
        self.model_hopenet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        self.model_hopenet.load_state_dict(torch.load('models/pretrained/hopenet/hopenet_alpha2.pkl'))
        self.model_hopenet.eval()
        if self.is_cuda:
            self.model_hopenet.cuda()

        # SFD face detection
        self.face_detector = FaceDetector(device=self.device,
                                          verbose=False,
                                          path_to_detector='models/pretrained/sfd/s3fd-619a316812.pth')

        self.transformations = transforms.Compose([transforms.Resize(224),
                                                   transforms.CenterCrop(224),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])

    def calculate_pose(self, face, batch_index, image):
        d = face
        x_min = int(d[0])
        y_min = int(d[1])
        x_max = int(d[2])
        y_max = int(d[3])

        x_min -= 50
        x_max += 50
        y_min -= 50
        y_max += 30
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)

        x_max = min(image.shape[2], x_max)
        y_max = min(image.shape[3], y_max)
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)

        # Crop image
        img = image[batch_index, :, x_min:x_max, y_min:y_max].unsqueeze(0)

        img = img.div(255.0)
        img = self.transformations(img)
        img = img.cuda()

        yaw, pitch, roll = self.model_hopenet(img)

        return yaw, pitch, roll

    def detect_pose_batch(self, image):

        # image tensor B x 3 x 256 x 256
        detected_faces, error, error_index = self.face_detector.detect_from_batch(image)
        batch_index = 0
        for face in detected_faces:
            yaw, pitch, roll = self.calculate_pose(face[0], batch_index, image)
            batch_index += 1

        return yaw, pitch, roll


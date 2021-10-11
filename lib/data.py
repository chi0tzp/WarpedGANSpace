import torch
import os.path as osp
import glob
import cv2
import numpy as np
from torch.utils import data


class PathImages(data.Dataset):
    def __init__(self, root_path):
        self.images_files = glob.glob(osp.join(root_path, '*.jpg'))
        self.images_files.sort()

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, index):
        return self.image2tensor(self.images_files[index])

    @staticmethod
    def image2tensor(image_file):
        # Open image in BGR order and convert to RBG order
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8')
        return torch.tensor(np.transpose(img, (2, 0, 1))).float()

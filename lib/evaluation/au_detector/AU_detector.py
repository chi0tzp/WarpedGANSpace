import torch
from .hourglass import FANAU


class Model:
    def __init__(self, npts=12, corenet='pretrained_models/disfa_adaptation_f0.pth', use_cuda=True):
        self.FAN = FANAU(num_modules=1, n_points=npts)
        self.FAN.load_state_dict(torch.load(corenet, map_location='cpu')['state_dict'])
        self.FAN.eval()
        if use_cuda:
            self.FAN.cuda()

    def __call__(self, x):
        H = self.FAN(x)
        H = H if H.__class__.__name__ == 'Tensor' else H[-1]
        return H

    def _forward_FAN(self, images):
        with torch.no_grad():
            self.FAN.eval()
            H = self.FAN(images)
        return H

    def forward_FAN(self, images):
        H = self.FAN(images)
        return H


class AUdetector:
    def __init__(self, au_model_path='models/pretrained/au_detector/disfa_adaptation_f0.pth', use_cuda=True):
        self.naus = 12
        self.AUdetector = Model(npts=self.naus, corenet=au_model_path, use_cuda=use_cuda)
        self.use_cuda = use_cuda

    def detect_AU(self, img):
        img_normalized = (img - img.min()) / (img.max() - img.min())
        if self.use_cuda:
            img_normalized = img_normalized.cuda()

        if img_normalized.ndim == 3:
            img_normalized = img_normalized.unsqueeze(0)

        heatmaps = self.AUdetector.forward_FAN(img_normalized)
        intensities = torch.nn.MaxPool2d((64, 64))(heatmaps).squeeze(2).squeeze(2)

        return intensities

import torch
from torch import nn
from torchvision.models import resnet18


def save_hook(module, input, output):
    setattr(module, 'output', output)


class Reconstructor(nn.Module):
    def __init__(self, dim):
        super(Reconstructor, self).__init__()

        # Define ResNet18 backbone for feature extraction
        self.features_extractor = resnet18(pretrained=False)

        # Modify ResNet18 first conv layer to get 2 rgb images (concatenated as a 6-channel tensor)
        self.features_extractor.conv1 = nn.Conv2d(in_channels=6,
                                                  out_channels=64,
                                                  kernel_size=(7, 7),
                                                  stride=(2, 2),
                                                  padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)

        # Define classification head (for predicting warping functions (paths) indices)
        self.path_indices = nn.Linear(512, dim)

        # Define regression head (for predicting shift magnitudes)
        self.shift_magnitudes = nn.Linear(512, 1)

    def forward(self, x1, x2):
        self.features_extractor(torch.cat([x1, x2], dim=1))
        features = self.features.output.view([x1.shape[0], -1])
        return self.path_indices(features), self.shift_magnitudes(features).squeeze()

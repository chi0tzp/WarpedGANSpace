import torch
from torch import nn
from torchvision.models import resnet18


def save_hook(module, input, output):
    setattr(module, 'output', output)


class Reconstructor(nn.Module):
    def __init__(self, reconstructor_type, dim, channels=3):
        super(Reconstructor, self).__init__()
        self.reconstructor_type = reconstructor_type
        self.dim = dim
        self.channels = channels

        # === LeNet ===
        if self.reconstructor_type == 'LeNet':
            # Define LeNet backbone for feature extraction
            self.lenet_width = 2
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(self.channels * 2, 3 * self.lenet_width, kernel_size=(5, 5)),
                nn.BatchNorm2d(3 * self.lenet_width),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(3 * self.lenet_width, 8 * self.lenet_width, kernel_size=(5, 5)),
                nn.BatchNorm2d(8 * self.lenet_width),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(8 * self.lenet_width, 60 * self.lenet_width, kernel_size=(5, 5)),
                nn.BatchNorm2d(60 * self.lenet_width),
                nn.ReLU()
            )

            # Define classification head (for predicting warping functions (paths) indices)
            self.path_indices = nn.Sequential(
                nn.Linear(60 * self.lenet_width, 42 * self.lenet_width),
                nn.BatchNorm1d(42 * self.lenet_width),
                nn.ReLU(),
                nn.Linear(42 * self.lenet_width, self.dim)
            )

            # Define regression head (for predicting shift magnitudes)
            self.shift_magnitudes = nn.Sequential(
                nn.Linear(60 * self.lenet_width, 42 * self.lenet_width),
                nn.BatchNorm1d(42 * self.lenet_width),
                nn.ReLU(),
                nn.Linear(42 * self.lenet_width, 1)
            )

        # === ResNet ===
        elif self.reconstructor_type == 'ResNet':
            # Define ResNet18 backbone for feature extraction
            self.features_extractor = resnet18(pretrained=False)
            # Modify ResNet18 first conv layer so as to get 2 rgb images (concatenated as a 6-channel tensor)
            self.features_extractor.conv1 = nn.Conv2d(in_channels=6,
                                                      out_channels=64,
                                                      kernel_size=(7, 7),
                                                      stride=(2, 2),
                                                      padding=(3, 3), bias=False)
            nn.init.kaiming_normal_(self.features_extractor.conv1.weight, mode='fan_out', nonlinearity='relu')
            self.features = self.features_extractor.avgpool
            self.features.register_forward_hook(save_hook)

            # Define classification head (for predicting warping functions (paths) indices)
            self.path_indices = nn.Linear(512, self.dim)

            # Define regression head (for predicting shift magnitudes)
            self.shift_magnitudes = nn.Linear(512, 1)

    def forward(self, x1, x2):
        if self.reconstructor_type == 'LeNet':
            features = self.feature_extractor(torch.cat([x1, x2], dim=1))
            features = features.mean(dim=[-1, -2]).view(x1.shape[0], -1)
            return self.path_indices(features), self.shift_magnitudes(features).squeeze()
        elif self.reconstructor_type == 'ResNet':
            self.features_extractor(torch.cat([x1, x2], dim=1))
            features = self.features.output.view([x1.shape[0], -1])
            return self.path_indices(features), self.shift_magnitudes(features).squeeze()

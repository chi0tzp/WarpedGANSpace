import torch
import torch.nn as nn
from collections import OrderedDict

face_basenet_feat_map_dim_vgg = {
    48: 1 ** 2 * 512,
    72: 2 ** 2 * 512,
    96: 3 ** 2 * 512,
    128: 4 ** 2 * 512,
    160: 5 ** 2 * 512,
    192: 6 ** 2 * 512,
    224: 7 ** 2 * 512
}


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = nn.ModuleDict(OrderedDict({
            'conv_1_1': nn.Conv2d(3, 64, 3, stride=1, padding=1),
            'relu_1_1': nn.ReLU(inplace=True),
            'conv_1_2': nn.Conv2d(64, 64, 3, stride=1, padding=1),
            'relu_1_2': nn.ReLU(inplace=True),
            'maxp_1_2': nn.MaxPool2d(kernel_size=2, stride=2),
            'conv_2_1': nn.Conv2d(64, 128, 3, stride=1, padding=1),
            'relu_2_1': nn.ReLU(inplace=True),
            'conv_2_2': nn.Conv2d(128, 128, 3, stride=1, padding=1),
            'relu_2_2': nn.ReLU(inplace=True),
            'maxp_2_2': nn.MaxPool2d(kernel_size=2, stride=2),
            'conv_3_1': nn.Conv2d(128, 256, 3, stride=1, padding=1),
            'relu_3_1': nn.ReLU(inplace=True),
            'conv_3_2': nn.Conv2d(256, 256, 3, stride=1, padding=1),
            'relu_3_2': nn.ReLU(inplace=True),
            'conv_3_3': nn.Conv2d(256, 256, 3, stride=1, padding=1),
            'relu_3_3': nn.ReLU(inplace=True),
            'maxp_3_3': nn.MaxPool2d(kernel_size=2, stride=2),
            'conv_4_1': nn.Conv2d(256, 512, 3, stride=1, padding=1),
            'relu_4_1': nn.ReLU(inplace=True),
            'conv_4_2': nn.Conv2d(512, 512, 3, stride=1, padding=1),
            'relu_4_2': nn.ReLU(inplace=True),
            'conv_4_3': nn.Conv2d(512, 512, 3, stride=1, padding=1),
            'relu_4_3': nn.ReLU(inplace=True),
            'maxp_4_3': nn.MaxPool2d(kernel_size=2, stride=2),
            'conv_5_1': nn.Conv2d(512, 512, 3, stride=1, padding=1),
            'relu_5_1': nn.ReLU(inplace=True),
            'conv_5_2': nn.Conv2d(512, 512, 3, stride=1, padding=1),
            'relu_5_2': nn.ReLU(inplace=True),
            'conv_5_3': nn.Conv2d(512, 512, 3, stride=1, padding=1),
            'relu_5_3': nn.ReLU(inplace=True),
            'maxp_5_3': nn.MaxPool2d(kernel_size=2, stride=2)
        }))

        # Initialize network weights
        self.init()

    def init(self):
        for layer in self.modules():
            if hasattr(layer, 'weight'):
                if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.normal_(layer.weight, mean=1., std=0.02)
                else:
                    nn.init.xavier_normal_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0.)

    def load_vggface_basenet(self):
        vggface_weights_file = 'models/pretrained/fanet/vggface_basenet.pth'
        vggface_weights_state_dict = torch.load(vggface_weights_file, map_location=lambda storage, loc: storage)
        self.layers.load_state_dict(vggface_weights_state_dict)

    def forward(self, x):
        for k, v in self.layers.items():
            x = v(x)
        return x.view(x.size(0), -1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init


def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride), padding=padding, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, padding=0, bias=False):
    """1x1 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(stride, stride), padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, lightweight = False):
        super(ConvBlock, self).__init__()

        if lightweight:
            self.conv1 = conv1x1(in_planes, int(out_planes / 2))
            self.conv2 = conv1x1(int(out_planes / 2), int(out_planes / 4))
            self.conv3 = conv1x1(int(out_planes / 4), int(out_planes / 4))
        else: 
            self.conv1 = conv3x3(in_planes, int(out_planes / 2))
            self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
            self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        self.bn1 = nn.BatchNorm2d(int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU6(True),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = F.relu6(out1, True)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = F.relu6(out2, True)
 
        out3 = self.conv3(out2)
        out3 = self.bn3(out3)
        out3 = F.relu6(out3, True)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, lightweight = False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.lightweight = lightweight
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, lightweight=self.lightweight))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.max_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest') 

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class QFAN(nn.Module):
    def __init__(self, num_modules=1, num_in=3, num_features = 128, num_out=68, return_features=False):
        super(QFAN, self).__init__()
        self.num_modules = num_modules
        self.num_in = num_in
        self.num_features = num_features
        self.num_out = num_out
        self.return_features = return_features

        # Base part
        self.conv1 = nn.Conv2d(self.num_in, int(self.num_features / 2), kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(int(self.num_features / 2))
        self.conv2 = ConvBlock(int(self.num_features / 2), int(self.num_features / 2))
        self.conv3 = ConvBlock(int(self.num_features / 2), self.num_features)
        self.conv4 = ConvBlock(self.num_features, self.num_features)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, self.num_features))
            self.add_module('top_m_' + str(hg_module), ConvBlock(self.num_features, self.num_features))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(self.num_features, self.num_features, kernel_size=(1, 1), stride=(1, 1),
                                      padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(self.num_features))
            self.add_module('l' + str(hg_module), nn.Conv2d(self.num_features,
                                                            self.num_out, kernel_size=(1, 1), stride=(1, 1), padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(self.num_features, self.num_features, kernel_size=(1, 1),
                                                     stride=(1, 1), padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(self.num_out, self.num_features, kernel_size=(1, 1),
                                                                 stride=(1, 1), padding=0))

    def forward(self, x):
        features = []
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.max_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)
        if self.return_features:
            features.append(x)
        
        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        if self.return_features:
            return outputs, features
        else:
            return outputs


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                weight_init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                weight_init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                weight_init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                weight_init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                weight_init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            weight_init.normal_(m.weight.data, 1.0, gain)
            weight_init.constant_(m.bias.data, 0.0)

    net.apply(init_func)



class FANAU(nn.Module):
    def __init__(self, num_modules=1, num_features = 128, n_points=66, block=ConvBlock):
        super(FANAU, self).__init__()
        self.num_modules = 1
        self.num_features = num_features
        self.fan = QFAN(num_modules = self.num_modules, return_features=True)
        block = eval(block) if isinstance(block,str) else block
        
        # input features
        self.conv1 = nn.Sequential(nn.Conv2d(68, self.num_features, 1, 1), nn.BatchNorm2d(self.num_features), nn.ReLU6())
        self.conv2 = nn.Sequential(nn.Conv2d(self.num_features, self.num_features, 1, 1),
                                   nn.BatchNorm2d(self.num_features), nn.ReLU6())

        self.net = HourGlass(1,4, self.num_features, lightweight=True)
        self.conv_last = nn.Sequential(nn.Conv2d(self.num_features, self.num_features, 1, 1),
                                       nn.BatchNorm2d(self.num_features), nn.ReLU6())
        self.l = nn.Conv2d(self.num_features, n_points, 1, 1)

        init_weights(self)
        
    def forward(self, x):
        self.fan.eval()
        # with torch.no_grad():
        output, features = self.fan(x)
        # print(len(output), len(features))
        # print(output[0].shape, features[0].shape)
        
        out = output[-1]
        x = self.conv1(out) + self.conv2(features[0]) 
        x = self.net(x)
        x = self.conv_last(x)
        x = self.l(x)
        # print(x.shape)
        # quit()
        return x

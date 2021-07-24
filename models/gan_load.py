import json
import numpy as np
import torch
from torch import nn
from models.BigGAN import BigGAN, utils
from models.ProgGAN.model import Generator as ProgGANGenerator
from models.SNGAN.sn_gen_resnet import SN_RES_GEN_CONFIGS, make_resnet_generator
from models.SNGAN.distribution import NormalDistribution

try:
    from models.StyleGAN2.model import Generator as StyleGAN2Generator
except Exception as e:
    print("StyleGAN2 load fail: {}".format(e))


########################################################################################################################
##                                                                                                                    ##
##                                                     [ SNGAN ]                                                      ##
##                                                                                                                    ##
########################################################################################################################
class SNGANWrapper(nn.Module):
    def __init__(self, G):
        super(SNGANWrapper, self).__init__()
        self.G = G.model
        self.dim_z = G.distribution.dim

    def forward(self, z, shift=None):
        return self.G(z if shift is None else z + shift)


def build_sngan(pretrained_gan_weights, gan_type):
    # SNGAN configuration for MNIST and AnimeFaces datasets
    SNGAN_CONFIG = {
        'SNGAN_MNIST': {
            'image_channels': 1,
            'latent_dim': 128,
            'model': 'sn_resnet32',
            'img_size': 32
        },
        'SNGAN_AnimeFaces': {
            'image_channels': 3,
            'latent_dim': 128,
            'model': 'sn_resnet64',
            'img_size': 64
        }
    }

    # Build SNGAN generator (for the given dataset)
    G = make_resnet_generator(resnet_gen_config=SN_RES_GEN_CONFIGS[SNGAN_CONFIG[gan_type]['model']],
                              img_size=SNGAN_CONFIG[gan_type]['img_size'],
                              channels=SNGAN_CONFIG[gan_type]['image_channels'],
                              distribution=NormalDistribution(SNGAN_CONFIG[gan_type]['latent_dim']))

    # REVIEW: Load pre-trained weights
    G.load_state_dict(torch.load(pretrained_gan_weights, map_location=torch.device('cpu')), strict=False)

    return SNGANWrapper(G)


########################################################################################################################
##                                                                                                                    ##
##                                                   [ BigGAN ]                                                       ##
##                                                                                                                    ##
########################################################################################################################
class BigGANWrapper(nn.Module):
    def __init__(self, G, target_classes=(239, )):
        super(BigGANWrapper, self).__init__()
        self.G = G
        self.target_classes = nn.Parameter(data=torch.tensor(target_classes, dtype=torch.int64),
                                           requires_grad=False)
        self.dim_z = self.G.dim_z

    def mixed_classes(self, batch_size):
        if len(self.target_classes.data.shape) == 0:
            return self.target_classes.repeat(batch_size).cuda()
        else:
            return torch.from_numpy(np.random.choice(self.target_classes.cpu(), [batch_size])).cuda()

    def forward(self, z, shift=None):
        target_classes = self.mixed_classes(z.shape[0]).to(z.device)
        return self.G(z if shift is None else z + shift, self.G.shared(target_classes))


def build_biggan(pretrained_gan_weights, target_classes):
    # Get BigGAN configuration
    with open('models/BigGAN/generator_config.json') as f:
        config = json.load(f)

    # Build BigGAN generator for the given configuration
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config['skip_init'] = True
    config['no_optim'] = True
    G = BigGAN.Generator(**config)

    # Load pre-trained weights
    G.load_state_dict(torch.load(pretrained_gan_weights, map_location=torch.device('cpu')), strict=True)

    return BigGANWrapper(G, target_classes)


########################################################################################################################
##                                                                                                                    ##
##                                                    [ ProgGAN ]                                                     ##
##                                                                                                                    ##
########################################################################################################################
class ProgGANWrapper(nn.Module):
    def __init__(self, G):
        super(ProgGANWrapper, self).__init__()
        self.G = G
        self.dim_z = 512

    @staticmethod
    def _reshape(z):
        return z.reshape(z.size()[0], z.size()[1], 1, 1)

    def forward(self, z, shift=None):
        return self.G(self._reshape(z) if shift is None else self._reshape(z + shift))


def build_proggan(pretrained_gan_weights):
    # Build ProgGAN generator model
    G = ProgGANGenerator()
    # Load pre-trained generator model
    G.load_state_dict(torch.load(pretrained_gan_weights, map_location='cpu'))

    return ProgGANWrapper(G)


########################################################################################################################
##                                                                                                                    ##
##                                                  [ StyleGAN2 ]                                                     ##
##                                                                                                                    ##
########################################################################################################################
class StyleGAN2Wrapper(nn.Module):
    def __init__(self, G, shift_in_w):
        super(StyleGAN2Wrapper, self).__init__()
        self.G = G
        self.shift_in_w = shift_in_w
        self.dim_z = 512
        self.dim_w = self.G.style_dim if shift_in_w else self.dim_z

    def forward(self, z, shift=None):
        if self.shift_in_w:
            w = self.G.get_latent(z)
            return self.G([w if shift is None else w + shift], input_is_latent=True)[0]
        else:
            return self.G([z if shift is None else z + shift], input_is_latent=False)[0]


def build_stylegan2(pretrained_gan_weights, resolution, shift_in_w=False):
    # Build StyleGAN2 generator model
    G = StyleGAN2Generator(resolution, 512, 8)
    # Load pre-trained weights
    G.load_state_dict(torch.load(pretrained_gan_weights)['g_ema'], strict=False)

    return StyleGAN2Wrapper(G, shift_in_w=shift_in_w)

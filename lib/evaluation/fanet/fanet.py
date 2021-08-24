import sys
import torch
import torch.nn as nn
from collections import OrderedDict
from .fanet_basenet_vgg import VGG16, face_basenet_feat_map_dim_vgg
from .fanet_basenet_resnet50 import ResNet50, face_basenet_feat_map_dim_resnet50
from .fanet_basenet_seresnet50 import SEResNet50, face_basenet_feat_map_dim_seresnet50


class FANet(nn.Module):
    def __init__(self,
                 basenet='vgg16',
                 classification_head_ids=None,
                 regression_head_ids=None,
                 face_dim=96,
                 fc_cfg=None,
                 mode='train'):
        """FANet - Facial Analysis Network for Facial Analysis in the Wild.

        Args:
            basenet (str)                  : Basenet architecture (vgg16, resnet50, seresnet50)
            classification_head_ids (list) : List of classification head ids
            regression_head_ids (list)     : List of regression head ids
            face_dim (int)                 : Input face image dimension
            fc_cfg (dict)                  : Fully connected layers configuration dict (number of units and dropout)
            mode (str)                     : network mode ('train'/'eval')
        """
        super(FANet, self).__init__()
        self.basenet = basenet
        if (classification_head_ids is None) and (regression_head_ids is None):
            print("[Error] No classification or regression classes given. Abort.")
            sys.exit()
        self.classification_head_ids = [] if classification_head_ids is None else classification_head_ids
        self.regression_head_ids = [] if regression_head_ids is None else regression_head_ids
        self.face_dim = face_dim
        self.fc_cfg = fc_cfg
        self.mode = mode

        # Set basenet (backbone)
        if self.basenet == 'vgg16':
            self.face_basenet_feat_map_dim = face_basenet_feat_map_dim_vgg
            self.face_basenet = VGG16()

        elif self.basenet == 'resnet50':
            self.face_basenet_feat_map_dim = face_basenet_feat_map_dim_resnet50
            self.face_basenet = ResNet50()
            self.face_basenet.load_state_dict(torch.load('models/pretrained/fanet/resnet50_basenet.pth',
                                                         map_location=lambda storage, loc: storage))

        elif self.basenet == 'seresnet50':
            self.face_basenet_feat_map_dim = face_basenet_feat_map_dim_seresnet50
            self.face_basenet = SEResNet50()
            self.face_basenet.load_state_dict(torch.load('models/pretrained/fanet/seresnet50_basenet.pth',
                                                         map_location=lambda storage, loc: storage))

        # Classification heads
        fc_1_cls_hidden_units = self.fc_cfg['cls']['fc_1']['units']
        fc_1_cls_dropout_p = self.fc_cfg['cls']['fc_1']['dropout']
        fc_2_cls_hidden_units = self.fc_cfg['cls']['fc_2']['units']
        fc_2_cls_dropout_p = self.fc_cfg['cls']['fc_2']['dropout']

        # Regression heads
        fc_1_reg_hidden_units = self.fc_cfg['reg']['fc_1']['units']
        fc_1_reg_dropout_p = self.fc_cfg['reg']['fc_1']['dropout']
        fc_2_reg_hidden_units = self.fc_cfg['reg']['fc_2']['units']
        fc_2_reg_dropout_p = self.fc_cfg['reg']['fc_2']['dropout']

        # Define classification heads
        for cls_head_id in self.classification_head_ids:
            setattr(self, cls_head_id, nn.ModuleDict(
                OrderedDict({
                    'fc_1': nn.Linear(self.face_basenet_feat_map_dim[face_dim], fc_1_cls_hidden_units),
                    'fc_1-relu': nn.ReLU(inplace=True),
                    'fc_1-dropout': nn.Dropout(fc_1_cls_dropout_p),
                    'fc_2': nn.Linear(fc_1_cls_hidden_units, fc_2_cls_hidden_units),
                    'fc_2-relu': nn.ReLU(inplace=True),
                    'fc_2-dropout': nn.Dropout(fc_2_cls_dropout_p),
                    'fc_3': nn.Linear(fc_2_cls_hidden_units, 1),
                    'fc_3-sigmoid': nn.Sigmoid()
                })
            ))

        # Define regression heads
        for reg_head_id in self.regression_head_ids:
            setattr(self, reg_head_id, nn.ModuleDict(
                OrderedDict({
                    'fc_1': nn.Linear(self.face_basenet_feat_map_dim[face_dim], fc_1_reg_hidden_units),
                    'fc_1-relu': nn.ReLU(inplace=True),
                    'fc_1-dropout': nn.Dropout(fc_1_reg_dropout_p),
                    'fc_2': nn.Linear(fc_1_reg_hidden_units, fc_2_reg_hidden_units),
                    'fc_2-relu': nn.ReLU(inplace=True),
                    'fc_2-dropout': nn.Dropout(fc_2_reg_dropout_p),
                    'fc_3': nn.Linear(fc_2_reg_hidden_units, 1)
                })
            ))

        # Initialize classification/regression FC heads
        self.init_heads()

    def init_heads(self):
        # Classification heads
        for cls_head_id in self.classification_head_ids:
            for layer in getattr(self, cls_head_id).values():
                if hasattr(layer, 'weight'):
                    nn.init.xavier_normal_(layer.weight)
                if hasattr(layer, 'bias'):
                    nn.init.constant_(layer.bias, 0.)
        # Regression heads
        for reg_head_id in self.regression_head_ids:
            for layer in getattr(self, reg_head_id).values():
                if hasattr(layer, 'weight'):
                    nn.init.xavier_normal_(layer.weight)
                if hasattr(layer, 'bias'):
                    nn.init.constant_(layer.bias, 0.)

    def forward(self, x, head_ids=None):
        """FANet forward function.

        Args:
            x (torch.Tensor): input face image
            head_ids (tuple): classification/regression head ids

        Returns:
            outputs (list): Classification/regression predictions for the given head_ids.

        """
        # Forward through face basenet
        x = self.face_basenet(x)

        # Forward through classification/regression heads
        outputs = []
        for head_id in head_ids:
            y = x.clone()
            for v in getattr(self, head_id).values():
                y = v(y)
            outputs.append(y)

        return outputs


import sys
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50

from src.networks.common.conv_layers import WSConv2d


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class EncoderResnet(nn.Module):
    def __init__(self, 
                 pretrained_encoder_basis_path='', 
                 norm_type='bn', 
                 num_basis=10,
                 head_init_gain=1e-3):
        super(EncoderResnet, self).__init__()
        if norm_type == 'gn+ws':
            self.backbone = resnet50(num_classes=159, norm_layer=lambda x: nn.GroupNorm(32, x))
            self.backbone = patch_conv_to_wsconv(self.backbone)

        elif norm_type == 'bn':
            self.backbone = resnet50(num_classes=159)

        self.backbone.fc = nn.Linear(in_features=2048, out_features=num_basis, bias=True)
        nn.init.zeros_(self.backbone.fc.bias)
        nn.init.xavier_normal_(self.backbone.fc.weight, gain=head_init_gain)

        if pretrained_encoder_basis_path:
            print('Load checkpoint in Encoder Resnet!')
            state_dict = torch.load(pretrained_encoder_basis_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
        
    def forward(self, x):
        y = self.backbone(x)
        return y


def patch_conv_to_wsconv(module):
    new_module = module
    
    if isinstance(module, nn.Conv2d):
        # Split affine part of instance norm into separable 1x1 conv
        new_module = WSConv2d(module.in_channels, module.out_channels, module.kernel_size,
                              module.stride, module.padding, module.dilation, module.groups, module.bias)
        
        new_module.weight.data = module.weight.data.clone()
        if module.bias:
            new_module.bias.data = module.bias.data.clone()
        
    else:
        for name, child in module.named_children():
            new_module.add_module(name, patch_conv_to_wsconv(child))

    return new_module


class EncoderVertex(nn.Module):
    def __init__(self, 
                 path_to_deca_lib='DECA/decalib',
                 pretrained_vertex_basis_path='', 
                 norm_type='bn', 
                 num_basis=10,
                 num_vertex=5023,
                 basis_init='pca'): 
        super(EncoderVertex, self).__init__()
        self.num_vertex = num_vertex
                                    
        if basis_init == 'pca':
            path = os.path.join(path_to_deca_lib, 'data', 'generic_model.pkl')
            with open(path, 'rb') as f:
                ss = pickle.load(f, encoding='latin1')
                flame_model = Struct(**ss)
            shapedirs = to_tensor(to_np(flame_model.shapedirs[:, :, :num_basis]), dtype=torch.float32)
            self.vertex = torch.nn.parameter.Parameter(shapedirs)
            del flame_model
        else:
            self.vertex = torch.nn.parameter.Parameter(torch.normal(mean=3.7647e-12, std=0.0003, size=[self.num_vertex, 3, num_basis]))
                                    
        if pretrained_vertex_basis_path:
            state_dict = torch.load(pretrained_vertex_basis_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)

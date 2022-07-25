import torch
from torch import nn

import math
import itertools
from typing import Union



class NormParamsPredictor(nn.Module):
    def __init__(self, 
                 net_or_nets: Union[list, nn.Module], 
                 embed_channels: int) -> None:
        super(NormParamsPredictor, self).__init__()        
        self.mappers = nn.ModuleList()

        if isinstance(net_or_nets, list):
            modules = itertools.chain(*[net.modules() for net in net_or_nets])
        else:
            modules = net_or_nets.modules()

        for m in modules:
            m_name = m.__class__.__name__
            if (m_name == 'AdaptiveBatchNorm' or m_name == 'AdaptiveSyncBatchNorm' or 
                m_name == 'AdaptiveInstanceNorm' or m_name == 'AdaptiveGroupNorm'):

                self.mappers.append(nn.Linear(
                    in_features=embed_channels, 
                    out_features=m.num_features * 2,
                    bias=False))

    def forward(self, embed):
        params = []

        for mapper in self.mappers:
            param = mapper(embed)
            weight, bias = param.split(param.shape[1] // 2, dim=1)
            params += [(weight, bias)]

        return params


class SPADEParamsPredictor(nn.Module):
    def __init__(self, 
                 net_or_nets: Union[list, nn.Module], 
                 embed_channels: int,
                 spatial_dims=2) -> None:
        super(SPADEParamsPredictor, self).__init__()        
        self.mappers = nn.ModuleList()

        if isinstance(net_or_nets, list):
            modules = itertools.chain(*[net.modules() for net in net_or_nets])
        else:
            modules = net_or_nets.modules()

        for m in modules:
            m_name = m.__class__.__name__
            if (m_name == 'AdaptiveBatchNorm' or m_name == 'AdaptiveSyncBatchNorm' or 
                m_name == 'AdaptiveInstanceNorm' or m_name == 'AdaptiveGroupNorm'):

                if spatial_dims == 2:
                    self.mappers.append(nn.Conv2d(
                        in_channels=embed_channels, 
                        out_channels=m.num_features * 2,
                        kernel_size=1,
                        bias=False))
                else:
                    self.mappers.append(nn.Conv3d(
                        in_channels=embed_channels, 
                        out_channels=m.num_features * 2,
                        kernel_size=1,
                        bias=False))

    def forward(self, embed):
        params = []

        for mapper in self.mappers:
            param = mapper(embed)
            weight, bias = param.split(param.shape[1] // 2, dim=1)
            params += [(weight, bias)]

        return params


class ConvParamsPredictor(nn.Module):
    def __init__(self, 
                 net_or_nets: Union[list, nn.Module], 
                 embed_channels: int) -> None:
        super(ConvParamsPredictor, self).__init__()        
        # Matrices that perform a lowrank matrix decomposition W = U E V
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()
        self.kernel_size = []

        if isinstance(net_or_nets, list):
            modules = itertools.chain(*[net.modules() for net in net_or_nets])
        else:
            modules = net_or_nets.modules()

        for m in modules:
            if m.__class__.__name__ == 'AdaptiveConv' or m.__class__.__name__ == 'AdaptiveConv3d':
                # Assumes that adaptive conv layers have no bias
                kernel_numel = m.kernel_size[0] * m.kernel_size[1]
                if len(m.kernel_size) == 3:
                    kernel_numel *= m.kernel_size[2]

                if kernel_numel == 1:
                    self.u += [nn.Parameter(torch.empty(m.out_channels, embed_channels))]
                    self.v += [nn.Parameter(torch.empty(embed_channels, m.in_channels))]

                elif kernel_numel > 1:
                    self.u += [nn.Parameter(torch.empty(m.out_channels, embed_channels))]
                    self.v += [nn.Parameter(torch.empty(m.in_channels, embed_channels))]

                self.kernel_size += [m.kernel_size]

                nn.init.xavier_normal_(self.u[-1], gain=0.02)
                nn.init.xavier_normal_(self.v[-1], gain=0.02)

    def forward(self, embed):       
        params = []

        for u, v, kernel_size in zip(self.u, self.v, self.kernel_size):
            kernel_numel = kernel_size[0] * kernel_size[1]
            if len(kernel_size) == 3:
                kernel_numel *= kernel_size[2]

            embed_ = embed

            if kernel_numel == 1:
                # AdaptiveConv with kernel size = 1
                weight = u[None].matmul(embed_).matmul(v[None])
                weight = weight.view(*weight.shape, *kernel_size) # B x C_out x C_in x 1 ...

            else:
                embed_ = embed_[..., None]

                kernel_numel_ = 1
                kernel_size_ = (1,)*len(kernel_size)

                param = embed_.view(*embed_.shape[:2], -1)
                param = u[None].matmul(param) # B x C_out x C_emb/2
                b, c_out = param.shape[:2]
                param = param.view(b, c_out, -1, kernel_numel_)
                param = v[None].matmul(param) # B x C_out x C_in x kernel_numel
                weight = param.view(*param.shape[:3], *kernel_size_)

            params += [weight]

        return params


def assign_adaptive_norm_params(net_or_nets, params, alpha=1.0):
    if isinstance(net_or_nets, list):
        modules = itertools.chain(*[net.modules() for net in net_or_nets])
    else:
        modules = net_or_nets.modules()

    for m in modules:
        m_name = m.__class__.__name__
        if (m_name == 'AdaptiveBatchNorm' or m_name == 'AdaptiveSyncBatchNorm' or 
            m_name == 'AdaptiveInstanceNorm' or m_name == 'AdaptiveGroupNorm'):
            ada_weight, ada_bias = params.pop(0)

            if len(ada_weight.shape) == 2:
                m.ada_weight = m.weight[None] + ada_weight * alpha
                m.ada_bias = m.bias[None] + ada_bias * alpha
            elif len(ada_weight.shape) == 4:
                m.ada_weight = m.weight[None, :, None, None] + ada_weight * alpha
                m.ada_bias = m.bias[None, :, None, None] + ada_bias + alpha
            elif len(ada_weight.shape) == 5:
                m.ada_weight = m.weight[None, :, None, None, None] + ada_weight * alpha
                m.ada_bias = m.bias[None, :, None, None, None] + ada_bias + alpha

def assign_adaptive_conv_params(net_or_nets, params, alpha=1.0):
    if isinstance(net_or_nets, list):
        modules = itertools.chain(*[net.modules() for net in net_or_nets])
    else:
        modules = net_or_nets.modules()

    for m in modules:
        m_name = m.__class__.__name__
        if m_name == 'AdaptiveConv' or m_name == 'AdaptiveConv3d':
            attr_name = 'weight_orig' if hasattr(m, 'weight_orig') else 'weight'

            weight = getattr(m, attr_name)
            ada_weight = params.pop(0)

            ada_weight = weight[None] + ada_weight * alpha
            setattr(m, 'ada_' + attr_name, ada_weight)
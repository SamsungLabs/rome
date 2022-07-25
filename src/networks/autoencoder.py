import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

from src.networks.common import layers


class Autoencoder(nn.Module):
    def __init__(self,
                 num_channels: int,
                 max_channels: int,
                 num_groups: int,
                 num_bottleneck_groups: int,
                 num_blocks: int,
                 num_layers: int,
                 block_type: str,
                 input_channels: int,
                 input_size: int,
                 output_channels: int,
                 norm_layer_type: str,
                 activation_type: str,
                 conv_layer_type: str,
                 use_psp: bool,
                 pretrained_model_path: str = '',
                 pretrained_model_name: str = 'autoencoder'):
        super(Autoencoder, self).__init__()
        # Encoder from inputs to latents
        expansion_factor = 4 if block_type == 'bottleneck' else 1

        layers_ = [nn.Conv2d(input_channels, num_channels * expansion_factor, 7, 1, 3, bias=False)]
        in_channels = num_channels
        out_channels = num_channels

        for i in range(num_groups):
            for j in range(num_blocks):
                layers_.append(layers.blocks[block_type](
                    in_channels=in_channels if j == 0 else out_channels,
                    out_channels=out_channels,
                    mid_channels=out_channels,
                    expansion_factor=expansion_factor,
                    num_layers=num_layers,
                    kernel_size=3,
                    stride=1,
                    norm_layer_type=norm_layer_type,
                    activation_type=activation_type,
                    conv_layer_type=conv_layer_type))

            in_channels = out_channels
            out_channels = min(num_channels * 2 ** (i + 1), max_channels)

            if i < num_groups - 1:
                layers_.append(nn.MaxPool2d(kernel_size=2))

        for i in range(num_bottleneck_groups):
            for j in range(num_blocks):
                layers_.append(layers.blocks[block_type](
                    in_channels=out_channels,
                    out_channels=out_channels,
                    expansion_factor=expansion_factor,
                    num_layers=num_layers,
                    kernel_size=3,
                    stride=1,
                    norm_layer_type=norm_layer_type,
                    activation_type=activation_type,
                    conv_layer_type=conv_layer_type))

        if use_psp:
            layers_.append(PSP(
                levels=4,
                num_channels=out_channels * expansion_factor,
                conv_layer=layers.conv_layers[conv_layer_type],
                norm_layer=layers.norm_layers[norm_layer_type],
                activation=layers.activations[activation_type]))

        for i in reversed(range(num_groups)):
            in_channels = out_channels
            out_channels = min(num_channels * 2 ** max(i - 1, 0), max_channels)

            for j in range(num_blocks):
                layers_.append(layers.blocks[block_type](
                    in_channels=in_channels,
                    out_channels=in_channels if j < num_blocks - 1 else out_channels,
                    mid_channels=in_channels,
                    expansion_factor=expansion_factor,
                    num_layers=num_layers,
                    kernel_size=3,
                    stride=1,
                    norm_layer_type=norm_layer_type,
                    activation_type=activation_type,
                    conv_layer_type=conv_layer_type))

            if i > 0:
                layers_.append(nn.Upsample(scale_factor=2, mode='nearest'))

        layers_ += [
            layers.norm_layers[norm_layer_type](out_channels * expansion_factor, affine=True),
            layers.activations[activation_type](inplace=True),
            nn.Conv2d(out_channels * expansion_factor, output_channels, 1)]

        self.net = nn.Sequential(*layers_)

        if pretrained_model_path:
            state_dict_full = torch.load(pretrained_model_path, map_location='cpu')
            state_dict = OrderedDict()
            for k, v in state_dict_full.items():
                if pretrained_model_name in k:
                    state_dict[k.replace(f'{pretrained_model_name}.', '')] = v
            self.load_state_dict(state_dict)
            print('Loaded autoencoder state dict')

    def forward(self, x, no_grad=False):
        if no_grad:
            with torch.no_grad():
                return self.net(x)
        else:
            return self.net(x)


class PSP(nn.Module):
    def __init__(self, levels, num_channels, conv_layer, norm_layer, activation):
        super(PSP, self).__init__()
        self.blocks = nn.ModuleList()

        for i in range(1, levels + 1):
            self.blocks.append(nn.Sequential(
                nn.AvgPool2d(2 ** i),
                norm_layer(num_channels),
                activation(inplace=True),
                conv_layer(num_channels, num_channels // levels, 1),
                nn.Upsample(scale_factor=2 ** i, mode='bilinear')))

        self.squish = conv_layer(num_channels * 2, num_channels, 1)

    def forward(self, x):
        out = [x]
        for block in self.blocks:
            out.append(block(x))
        out = torch.cat(out, dim=1)

        return self.squish(out)

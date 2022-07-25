import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

from .common import layers


class UNet(nn.Module):
    def __init__(self, 
                 num_channels: int,
                 max_channels: int,
                 num_groups: int,
                 num_blocks: int,
                 num_layers: int,
                 block_type: str,
                 input_channels: int,
                 output_channels: int,
                 skip_connection_type: str,
                 norm_layer_type: str,
                 activation_type: str,
                 conv_layer_type: str,
                 downsampling_type: str,
                 upsampling_type: str,
                 multiscale_outputs: bool = False,
                 pretrained_model_path: str = '',
                 pretrained_model_name: str = 'unet'):
        super(UNet, self).__init__()
        self.skip_connection_type = skip_connection_type
        self.multiscale_outputs = multiscale_outputs
        expansion_factor = 4 if block_type == 'bottleneck' else 1

        if block_type != 'conv':
            self.from_inputs = nn.Conv2d(input_channels, num_channels * expansion_factor, 7, 1, 3, bias=False)

        out_channels = num_channels if block_type != 'conv' else input_channels

        if downsampling_type == 'maxpool':
            self.downsample = nn.MaxPool2d(kernel_size=2)
        elif downsampling_type == 'avgpool':
            self.downsample = nn.AvgPool2d(kernel_size=2)

        self.encoder = nn.ModuleList()

        for i in range(num_groups):
            layers_ = []

            in_channels = out_channels
            out_channels = min(num_channels * 2**(i+1), max_channels)

            for j in range(num_blocks):
                layers_.append(layers.blocks[block_type](
                    in_channels=in_channels, 
                    out_channels=in_channels if j < num_blocks - 1 else out_channels,
                    mid_channels=in_channels if in_channels != input_channels else num_channels,
                    expansion_factor=expansion_factor,
                    num_layers=num_layers,
                    kernel_size=3,
                    stride=1,
                    norm_layer_type=norm_layer_type,
                    activation_type=activation_type,
                    conv_layer_type=conv_layer_type))

            self.encoder.append(nn.Sequential(*layers_))

        if in_channels != out_channels:
            self.bottleneck = nn.Conv2d(out_channels, in_channels, 1, bias=False)
            out_channels = in_channels

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder = nn.ModuleList()

        for i in reversed(range(num_groups - 1)):
            in_channels = out_channels
            out_channels = min(num_channels * 2**i, max_channels)

            layers_ = []

            for j in range(num_blocks):
                layers_.append(layers.blocks[block_type](
                    in_channels=in_channels * 2 if j == 0 and skip_connection_type == 'cat' else in_channels, 
                    out_channels=in_channels if j < num_blocks - 1 else out_channels,
                    mid_channels=in_channels,
                    expansion_factor=expansion_factor,
                    num_layers=num_layers,
                    kernel_size=3,
                    stride=1,
                    norm_layer_type=norm_layer_type,
                    activation_type=activation_type,
                    conv_layer_type=conv_layer_type))

            self.decoder.append(nn.Sequential(*layers_))

        if block_type == 'conv':
            self.to_outputs = nn.Conv2d(out_channels * expansion_factor, output_channels, 1)

        else:
            self.to_outputs = nn.Sequential(
                layers.norm_layers[norm_layer_type](out_channels * expansion_factor, affine=True),
                layers.activations[activation_type](inplace=True),
                nn.Conv2d(out_channels * expansion_factor, output_channels, 1))

        self.upsample = nn.Upsample(scale_factor=2, mode=upsampling_type)

        if pretrained_model_path:
            state_dict_full = torch.load(pretrained_model_path, map_location='cpu')
            state_dict = OrderedDict()
            for k, v in state_dict_full.items():
                if pretrained_model_name in k:
                    state_dict[k.replace(f'{pretrained_model_name}.', '')] = v
            self.load_state_dict(state_dict, strict=False)
            print(f'Loaded {pretrained_model_name} state dict')

    def forward(self, x):
        if hasattr(self, 'from_inputs'):
            x = self.from_inputs(x)

        feats = []

        for i, block in enumerate(self.encoder):
            x = block(x)

            if i < len(self.encoder) - 1:
                feats.append(x)
                x = self.downsample(x)

        outputs = []

        if hasattr(self, 'bottleneck'):
            x = self.bottleneck(x)

        if self.multiscale_outputs:
            outputs.append(x)

        for j, block in zip(reversed(range(len(self.decoder))), self.decoder):
            x = self.upsample(x)

            if self.skip_connection_type == 'cat':
                x = torch.cat([x, feats[j]], dim=1)
            elif self.skip_connection_type == 'sum':
                x = x + feats[j]

            x = block(x)

            if self.multiscale_outputs:
                outputs.append(x)

        if self.multiscale_outputs:
            outputs[-1] = self.to_outputs(outputs[-1])
            return outputs

        else:
            return self.to_outputs(x)
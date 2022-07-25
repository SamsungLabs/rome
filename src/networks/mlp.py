import torch
import torch.nn as nn
from collections import OrderedDict

from .common import layers


class MLP(nn.Module):
    def __init__(self,
                 num_channels: int,
                 num_layers: int,
                 skip_layer: int,
                 input_channels: int,
                 output_channels: int,
                 activation_type: str,
                 pretrained_model_path: str = '',
                 pretrained_model_name: str = 'mlp',
                 last_bias=False):
        super(MLP, self).__init__()
        assert num_layers > 1
        layers_ = [
            nn.Linear(input_channels, num_channels),
            layers.activations[activation_type](inplace=True)]

        for i in range(skip_layer - 1):
            layers_ += [
                nn.Linear(num_channels, num_channels),
                layers.activations[activation_type](inplace=True)]

        self.block_1 = nn.Sequential(*layers_)

        layers_ = [
            nn.Linear(num_channels + input_channels, num_channels),
            layers.activations[activation_type](inplace=True)]

        for i in range(num_layers - skip_layer - 1):
            layers_ += [
                nn.Linear(num_channels, num_channels),
                layers.activations[activation_type](inplace=True)]

        layers_ += [
            nn.Linear(num_channels, output_channels, bias=last_bias)]

        self.block_2 = nn.Sequential(*layers_)

        if pretrained_model_path:
            state_dict_full = torch.load(pretrained_model_path, map_location='cpu')
            state_dict = OrderedDict()
            for k, v in state_dict_full.items():
                if pretrained_model_name in k:
                    state_dict[k.replace(f'{pretrained_model_name}.', '')] = v
            self.load_state_dict(state_dict)

    def forward(self, x):
        if len(x.shape) == 4:
            # Input is a 4D tensor
            b, c, h, w = x.shape
            x_ = x.permute(0, 2, 3, 1).reshape(-1, c)
        else:
            x_ = x

        y = self.block_1(x_)
        y = torch.cat([x_, y], dim=1)
        y = self.block_2(y)

        if len(x.shape) == 4:
            y = y.view(b, h, w, -1).permute(0, 3, 1, 2)

        return y
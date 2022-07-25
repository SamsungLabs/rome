import torch
from torch import nn
import torch.nn.functional as F
import functools
from typing import Union, Tuple, List

from . import norm_layers as norms
from . import conv_layers as convs


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, inputs):
        batch_size, channels, in_height, in_width = inputs.size()

        out_height = in_height // self.upscale_factor
        out_width = in_width // self.upscale_factor

        input_view = inputs.contiguous().view(
            batch_size, channels, out_height, self.upscale_factor,
            out_width, self.upscale_factor)

        channels *= self.upscale_factor ** 2
        unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
        return unshuffle_out.view(batch_size, channels, out_height, out_width)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        mid_channels: int = -1,
        spade_channels: int = -1,
        num_layers: int = 2,
        expansion_factor: int = None,
        kernel_size: Union[int, Tuple[int], List[int]] = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        conv_layer_type: str = 'conv',
        norm_layer_type: str = 'bn',
        activation_type: str = 'relu',
        skip_layer_type: str = 'conv',
        resize_layer_type: str = 'none'):
        super(BasicBlock, self).__init__()
        if stride > 1 and resize_layer_type in ['nearest', 'bilinear']:
            self.upsample = lambda inputs: F.interpolate(input=inputs, 
                                                         scale_factor=stride, 
                                                         mode=resize_layer_type, 
                                                         align_corners=None if resize_layer_type == 'nearest' else False)

        if mid_channels == -1:
            mid_channels = out_channels

        if norm_layer_type != 'none':
            norm_layer = norm_layers[norm_layer_type]
        activation = activations[activation_type]
        conv_layer = conv_layers[conv_layer_type]
        skip_layer = conv_layers[skip_layer_type]

        ### Initialize the layers of the first half of the block ###
        layers_ = []

        for i in range(num_layers):
            if norm_layer_type != 'none':
                if spade_channels != -1:
                    layers_ += [norm_layer(in_channels if i == 0 else mid_channels, spade_channels, affine=True)]
                else:
                    layers_ += [norm_layer(in_channels if i == 0 else mid_channels, affine=True)]

            layers_ += [
                activation(inplace=True),
                conv_layer(
                    in_channels=in_channels if i == 0 else mid_channels,
                    out_channels=out_channels if i == num_layers - 1 else mid_channels,
                    kernel_size=kernel_size, 
                    stride=stride if resize_layer_type == 'none' and i == num_layers - 1 else 1,
                    padding=padding, 
                    dilation=dilation, 
                    groups=groups,
                    bias=norm_layer_type == 'none')]

        self.main = nn.Sequential(*layers_)

        if in_channels != out_channels:
            self.skip = skip_layer(
                in_channels=in_channels,
                out_channels=out_channels, 
                kernel_size=1,
                bias=norm_layer_type == 'none')
        else:
            self.skip = nn.Identity()

        if stride > 1 and resize_layer_type in downsampling_layers:
            self.downsample = downsampling_layers[resize_layer_type](stride)

    def forward(self, x):
        if hasattr(self, 'upsample'):
            x = self.upsample(x)

        x = self.main(x) + self.skip(x)

        if hasattr(self, 'downsample'):
            x = self.downsample(x)

        return x


class BottleneckBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        mid_channels: int = -1,
        spade_channels: int = -1,
        num_layers: int = 3,
        expansion_factor: int = 4,
        kernel_size: Union[int, Tuple[int], List[int]] = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        conv_layer_type: str = 'conv',
        norm_layer_type: str = 'bn',
        activation_type: str = 'relu',
        skip_layer_type: str = 'conv',
        resize_layer_type: str = 'none'):
        """This is a base module for a residual bottleneck block"""
        super(BottleneckBlock, self).__init__()
        if stride > 1 and resize_layer_type in ['nearest', 'bilinear']:
            self.upsample = lambda inputs: F.interpolate(input=inputs, 
                                                         scale_factor=stride, 
                                                         mode=resize_layer_type, 
                                                         align_corners=None if resize_layer_type == 'nearest' else False)

        if mid_channels == -1:
            mid_channels = out_channels

        if norm_layer_type != 'none':
            norm_layer = norm_layers[norm_layer_type]
        activation = activations[activation_type]
        conv_layer = conv_layers[conv_layer_type]
        skip_layer = conv_layers[skip_layer_type]

        layers_ = []

        if norm_layer_type != 'none':
            if spade_channels != -1:
                layers_ += [norm_layer(in_channels * expansion_factor, spade_channels, affine=True)]
            else:
                layers_ += [norm_layer(in_channels * expansion_factor, affine=True)]

        layers_ += [
            activation(inplace=True),
            conv_layer(
                in_channels=in_channels * expansion_factor,
                out_channels=mid_channels,
                kernel_size=1,
                bias=norm_layer_type == 'none')]

        if norm_layer_type != 'none':
            if spade_channels != -1:
                layers_ += [norm_layer(mid_channels, spade_channels, affine=True)]
            else:
                layers_ += [norm_layer(mid_channels, affine=True)]
        layers_ += [activation(inplace=True)]

        assert num_layers > 2, 'Minimum number of layers is 3'
        for i in range(num_layers - 2):
            layers_ += [
                conv_layer(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size, 
                    stride=stride if resize_layer_type == 'none' and i == num_layers - 3 else 1,
                    padding=padding, 
                    dilation=dilation, 
                    groups=groups,
                    bias=norm_layer_type == 'none')]

            if norm_layer_type != 'none':
                if spade_channels != -1:
                    layers_ += [norm_layer(mid_channels, spade_channels, affine=True)]
                else:
                    layers_ += [norm_layer(mid_channels, affine=True)]
            layers_ += [activation(inplace=True)]

        layers_ += [
            skip_layer(
                in_channels=mid_channels,
                out_channels=out_channels * expansion_factor,
                kernel_size=1,
                bias=norm_layer_type == 'none')]

        self.main = nn.Sequential(*layers_)

        if in_channels != out_channels:
            self.skip = skip_layer(
                in_channels=in_channels * expansion_factor,
                out_channels=out_channels * expansion_factor, 
                kernel_size=1,
                bias=norm_layer_type == 'none')
        else:
            self.skip = nn.Identity()

        if stride > 1 and resize_layer_type in downsampling_layers:
            self.downsample = downsampling_layers[resize_layer_type](stride)

    def forward(self, x):
        if hasattr(self, 'upsample'):
            x = self.upsample(x)

        x = self.main(x) + self.skip(x)

        if hasattr(self, 'downsample'):
            x = self.downsample(x)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        mid_channels: int = -1,
        spade_channels: int = -1,
        num_layers: int = 1,
        expansion_factor: int = None,
        kernel_size: Union[int, Tuple[int], List[int]] = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        conv_layer_type: str = 'conv',
        norm_layer_type: str = 'none',
        activation_type: str = 'relu',
        skip_layer_type: str = 'conv',
        resize_layer_type: str = 'none'):
        """This is a base module for residual blocks"""
        super(ConvBlock, self).__init__()
        if stride > 1 and resize_layer_type in ['nearest', 'bilinear']:
            self.upsample = lambda inputs: F.interpolate(input=inputs, 
                                                         scale_factor=stride, 
                                                         mode=resize_layer_type, 
                                                         align_corners=None if resize_layer_type == 'nearest' else False)

        if mid_channels == -1:
            mid_channels = out_channels

        if norm_layer_type != 'none':
            norm_layer = norm_layers[norm_layer_type]
        activation = activations[activation_type]
        conv_layer = conv_layers[conv_layer_type]
        skip_layer = conv_layers[skip_layer_type]

        ### Initialize the layers of the first half of the block ###
        layers_ = []

        for i in range(num_layers):
            layers_ += [
                conv_layer(
                    in_channels=in_channels if i == 0 else mid_channels,
                    out_channels=out_channels if i == num_layers - 1 else mid_channels,
                    kernel_size=kernel_size,
                    stride=stride if resize_layer_type == 'none' and i == num_layers - 1 else 1,
                    padding=padding, 
                    dilation=dilation, 
                    groups=groups,
                    bias=norm_layer_type == 'none')]

            if norm_layer_type != 'none':
                if spade_channels != -1:
                    layers_ += [norm_layer(out_channels if i == num_layers - 1 else mid_channels, spade_channels, affine=True)]
                else:
                    layers_ += [norm_layer(out_channels if i == num_layers - 1 else mid_channels, affine=True)]
            layers_ += [activation(inplace=True)]

        self.main = nn.Sequential(*layers_)

        if stride > 1 and resize_layer_type in downsampling_layers:
            self.downsample = downsampling_layers[resize_layer_type](stride)

    def forward(self, x):
        if hasattr(self, 'upsample'):
            x = self.upsample(x)

        x = self.main(x)

        if hasattr(self, 'downsample'):
            x = self.downsample(x)
        
        return x


############################################################
#                Definitions for the layers                #
############################################################

# Supported blocks
blocks = {
    'basic': BasicBlock,
    'bottleneck': BottleneckBlock,
    'conv': ConvBlock
}

# Supported conv layers
conv_layers = {
    'conv': nn.Conv2d,
    'ws_conv': convs.WSConv2d,
    'conv_3d': nn.Conv3d,
    'ada_conv': convs.AdaptiveConv,
    'ada_conv_3d': convs.AdaptiveConv3d}

# Supported activations
activations = {
    'relu': nn.ReLU,
    'lrelu': functools.partial(nn.LeakyReLU, negative_slope=0.2)}

# Supported normalization layers
norm_layers = {
    'in': nn.InstanceNorm2d,
    'in_3d': nn.InstanceNorm3d,
    'bn': nn.BatchNorm2d,
    'gn': lambda num_features, affine=True: nn.GroupNorm(num_groups=min(32, num_features), num_channels=num_features, affine=affine),
    'ada_in': norms.AdaptiveInstanceNorm,
    'ada_gn': lambda num_features, affine=True: norms.AdaptiveGroupNorm(num_groups=min(32, num_features), num_features=num_features, affine=affine),
}

# Supported downsampling layers
downsampling_layers = {
    'avgpool': nn.AvgPool2d,
    'maxpool': nn.MaxPool2d,
    'avgpool_3d': nn.AvgPool3d,
    'maxpool_3d': nn.MaxPool3d,
    'pixelunshuffle': PixelUnShuffle}
import torch
from torch import nn
import torch.nn.functional as F


class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class AdaptiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = (3, 3), 
                 stride = 1, padding = 0, dilation = 1, groups = 1, bias = False):
        super(AdaptiveConv, self).__init__()
        # Set options
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        assert not bias, 'bias == True is not supported for AdaptiveConv'
        self.bias = None

        self.kernel_numel = kernel_size[0] * kernel_size[1]
        if len(kernel_size) == 3:
            self.kernel_numel *= kernel_size[2]

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.ada_weight = None # assigned externally

        if len(kernel_size) == 2:
            self.conv_func = F.conv2d
        elif len(kernel_size) == 3:
            self.conv_func = F.conv3d

    def forward(self, inputs):
        # Cast parameters into inputs.dtype
        if inputs.type() != self.ada_weight.type():
            weight = self.ada_weight.type(inputs.type())
        else:
            weight = self.ada_weight

        # Conv is applied to the inputs grouped by t frames
        B = weight.shape[0]
        T = inputs.shape[0] // B
        assert inputs.shape[0] == B*T, 'Wrong shape of weight'

        if self.kernel_numel > 1:
            if weight.shape[0] == 1:
                # No need to iterate through batch, can apply conv to the whole batch
                outputs = self.conv_func(inputs, weight[0], None, self.stride, self.padding, self.dilation, self.groups)

            else:
                outputs = []
                for b in range(B):
                    outputs += [self.conv_func(inputs[b*T:(b+1)*T], weight[b], None, self.stride, self.padding, self.dilation, self.groups)]
                outputs = torch.cat(outputs, 0)

        else:
            if weight.shape[0] == 1:
                if len(inputs.shape) == 5:
                    weight = weight[..., None, None, None]
                else:
                    weight = weight[..., None, None]

                outputs = self.conv_func(inputs, weight[0], None, self.stride, self.padding, self.dilation, self.groups)
            else:
                # 1x1(x1) adaptive convolution is a simple bmm
                if len(weight.shape) == 6:
                    weight = weight[..., 0, 0, 0]
                else:
                    weight = weight[..., 0, 0]

                outputs = torch.bmm(weight, inputs.view(B*T, inputs.shape[1], -1)).view(B, -1, *inputs.shape[2:])

        return outputs

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class AdaptiveConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, 
                 stride = 1, padding = 0, dilation = 1, groups = 1, bias = False):
        kernel_size = (kernel_size,) * 3
        super(AdaptiveConv, self).__init__(in_channels, out_channels, kernel_size,
                                           stride, padding, dilation, groups, bias)
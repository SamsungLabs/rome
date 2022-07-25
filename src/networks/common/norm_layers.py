import torch
from torch import nn


def init_parameters(self, num_features):
    self.weight = nn.Parameter(torch.ones(num_features))
    self.bias = nn.Parameter(torch.zeros(num_features))
    
    # These tensors are assigned externally
    self.ada_weight = None
    self.ada_bias = None

def init_spade_parameters(self, num_features, num_spade_features):
    self.conv_weight = nn.Conv2d(num_spade_features, num_features, 1, bias=False)
    self.conv_bias = nn.Conv2d(num_spade_features, num_features, 1, bias=False)

    nn.init.xavier_normal_(self.conv_weight.weight, gain=0.02)
    nn.init.xavier_normal_(self.conv_bias.weight, gain=0.02)

    # These tensors are assigned externally
    self.spade_features = None

def common_forward(x, weight, bias):
    B = weight.shape[0]
    T = x.shape[0] // B

    x = x.view(B, T, *x.shape[1:])

    if len(weight.shape) == 2:
        # Broadcast weight and bias accross T and spatial size of outputs
        if len(x.shape) == 5:
            x = x * weight[:, None, :, None, None] + bias[:, None, :, None, None]
        elif len(x.shape) == 6:
            x = x * weight[:, None, :, None, None, None] + bias[:, None, :, None, None, None]
    else:
        x = x * weight[:, None] + bias[:, None]

    x = x.view(B*T, *x.shape[2:])

    return x


class AdaptiveInstanceNorm(nn.modules.instancenorm._InstanceNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(AdaptiveInstanceNorm, self).__init__(
            num_features, eps, momentum, False, track_running_stats)
        init_parameters(self, num_features)
        
    def forward(self, x):
        x = super(AdaptiveInstanceNorm, self).forward(x)
        x = common_forward(x, self.ada_weight, self.ada_bias)

        return x

    def _check_input_dim(self, input):
        pass

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine=True, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class AdaptiveBatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(AdaptiveBatchNorm, self).__init__(
            num_features, eps, momentum, False, track_running_stats)
        init_parameters(self, num_features)
        
    def forward(self, x):
        x = super(AdaptiveBatchNorm, self).forward(x)
        x = common_forward(x, self.ada_weight, self.ada_bias)

        return x

    def _check_input_dim(self, input):
        pass

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine=True, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class AdaptiveGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_features, eps=1e-5, affine=True):
        super(AdaptiveGroupNorm, self).__init__(num_groups, num_features, eps, False)
        self.num_features = num_features
        init_parameters(self, num_features)
        
    def forward(self, x):
        x = super(AdaptiveGroupNorm, self).forward(x)
        x = common_forward(x, self.ada_weight, self.ada_bias)

        return x

    def _check_input_dim(self, input):
        pass

    def extra_repr(self) -> str:
        return '{num_groups}, {num_features}, eps={eps}, ' \
            'affine=True'.format(**self.__dict__)


class SPADEGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_features, num_spade_features, eps=1e-5, affine=True):
        super(SPADEGroupNorm, self).__init__(num_groups, num_features, eps, False)
        self.num_features = num_features
        self.num_spade_features = num_spade_features

        init_spade_parameters(self, num_features, num_spade_features)

    def forward(self, x):
        x = super(SPADEGroupNorm, self).forward(x)

        weight = self.conv_weight(self.spade_features) + 1.0
        bias = self.conv_bias(self.spade_features)
        
        x = common_forward(x, weight, bias)

        return x

    def _check_input_dim(self, input):
        pass

    def extra_repr(self) -> str:
        return '{num_groups}, {num_features}, eps={eps}, ' \
            'affine=True, spade_features={num_spade_features}'.format(**self.__dict__)


def assign_spade_features(net_or_nets, features):
    if isinstance(net_or_nets, list):
        modules = itertools.chain(*[net.modules() for net in net_or_nets])
    else:
        modules = net_or_nets.modules()

    for m in modules:
        m_name = m.__class__.__name__
        if (m_name == 'SPADEBatchNorm' or m_name == 'SPADESyncBatchNorm' or 
            m_name == 'SPADEInstanceNorm' or m_name == 'SPADEGroupNorm'):
            m.spade_features = features.pop()
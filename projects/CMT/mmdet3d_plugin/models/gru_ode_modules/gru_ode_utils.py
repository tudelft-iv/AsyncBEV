import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib

from collections import OrderedDict
from functools import partial
from timm.models.layers import DropPath

class ConvBlock(nn.Module):
    """2D convolution followed by
         - an optional normalisation (batch norm or instance norm)
         - an optional activation (ReLU, LeakyReLU, or tanh)
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, norm='bn', activation='lrelu',
                 bias=False, transpose=False):
        super().__init__()
        out_channels = out_channels or in_channels
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d if not transpose else partial(nn.ConvTranspose2d)
        self.conv = self.conv(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError('Invalid norm {}'.format(norm))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Invalid activation {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)

        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ResBlock(nn.Module):
    """Residual block:
       x -> Conv -> norm -> act. -> Conv -> norm -> act. -> ADD -> out
         |                                                   |
          ---------------------------------------------------
    """

    def __init__(self, in_channels, out_channels=None, norm='bn', activation='lrelu', bias=False):
        super().__init__()
        out_channels = out_channels or in_channels

        self.layers = nn.Sequential(OrderedDict([
            ('conv_1', ConvBlock(in_channels, in_channels, 3, stride=1, norm=norm, activation=activation, bias=bias)),
            ('conv_2', ConvBlock(in_channels, out_channels, 3, stride=1, norm=norm, activation=activation, bias=bias)),
            ('dropout', nn.Dropout2d(0.25)),
        ]))

        if out_channels != in_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.projection = None

    def forward(self, x):
        x_residual = self.layers(x)

        if self.projection:
            x = self.projection(x)
        return x + x_residual

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            ResBlock(in_c, out_c),
            SELayer(out_c),
            ResBlock(out_c, out_c),
            SELayer(out_c),
            ConvBlock(out_c, out_c, 3, stride=1, bias=True, norm='none'),
        )

    def forward(self, x):
        return self.model(x)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Bottleblock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(Bottleblock, self).__init__()

        bottleneck_channels = int(in_channels / 2)
        out_channels = out_channels or in_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=7, bias=False, padding=3),
            LayerNorm(bottleneck_channels, eps=1e-6, data_format='channels_first'),
            nn.GELU(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=1, bias=False),
            LayerNorm(bottleneck_channels, eps=1e-6, data_format='channels_first'),
            nn.GELU(),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=3, bias=False, padding=1),
            LayerNorm(out_channels, eps=1e-6, data_format='channels_first'),
            nn.GELU()
        )

        if out_channels == in_channels:
            self.projection = None
        else:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GELU()
            )

    def forward(self, *args):
        (x,) = args
        x_residual = self.layers(x)
        if self.projection is not None:
            return x_residual + self.projection(x)
        return x_residual + x


class SmallEncoder(nn.Module):
    def __init__(self,
                 nc,
                 nh,
                 nf):
        super(SmallEncoder, self).__init__()

        self.blocks = nn.ModuleList([
            ResBlock(nc, nf),
            ResBlock(nf, nf * 2),
            ResBlock(nf * 2, nf * 2),
            ResBlock(nf * 2, nf * 2),
            ResBlock(nf * 2, nf * 4)
        ])
        self.last_conv = nn.Sequential(
            ConvBlock(nf * 4, nh, 3, stride=1, activation='tanh')
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x, return_skip=False):
        h = x
        skips = []
        for i, layer in enumerate(self.blocks):
            if i in [1, 2]:
                h = self.maxpool(h)
            h = layer(h)
            skips.append(h)
        h = self.last_conv(h)
        if return_skip:
            return h, skips[::-1]
        return h


class SmallDecoder(nn.Module):
    def __init__(self, nc, nh, nf, skip):
        super(SmallDecoder, self).__init__()
        coef = 2 if skip else 1
        self.skip = skip

        self.first_upconv = ConvBlock(nc, nf * 4, stride=1, transpose=True)

        self.blocks = nn.ModuleList([
            ResBlock(nf * 4 * coef, nf * 2),
            ResBlock(nf * 2 * coef, nf * 2),
            ResBlock(nf * 2 * coef, nf * 2),
            ResBlock(nf * 2 * coef, nf),
            ResBlock(nf * coef, nf)
        ])
        self.last_conv = nn.Sequential(
            ConvBlock(nf * coef, nf, 3, stride=1),
            ConvBlock(nf, nh, 3, stride=1, transpose=True, bias=True, norm='none'),

        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z, skip=None, sigmoid=False):
        assert skip is None and not self.skip or self.skip and skip is not None
        h = self.first_upconv(z)
        for i, layer in enumerate(self.blocks):
            # print(i, h.shape, skip[i].shape)
            if skip is not None:
                h = torch.cat([h, skip[i]], 1)
            h = layer(h)
            if i in [2, 3]:
                h = self.upsample(h)
        x_ = self.last_conv(h)
        if sigmoid:
            x_ = torch.sigmoid(x_)
        return x_


def make_normal_from_raw_params(raw_params, scale_stddev=1, dim=2, eps=1e-8, max_log_sigma=-10000, min_log_sigma=10000):
    """
    Creates a normal distribution from the given parameters.

    Parameters
    ----------
    raw_params : torch.*.Tensor
        Tensor containing the Gaussian mean and a raw scale parameter on a given dimension.
    scale_stddev : float
        Multiplier of the final scale parameter of the Gaussian.
    dim : int
        Dimensions of raw_params so that the first half corresponds to the mean, and the second half to the scale.
    eps : float
        Minimum possible value of the final scale parameter.

    Returns
    -------
    torch.distributions.Normal
        Normal distribution with the input mean and eps + softplus(raw scale) * scale_stddev as standard deviation.
    """
    dim = 2 if len(raw_params.shape) == 5 else 1
    loc, raw_scale = torch.chunk(raw_params, 2, dim)
    assert loc.shape[dim] == raw_scale.shape[dim], f'{loc.shape[dim]}, {raw_scale.shape[dim]}'
    # raw_scale = torch.clamp(raw_scale, min_log_sigma, max_log_sigma)
    scale = F.softplus(raw_scale) + eps
    normal = distrib.Normal(loc, scale * scale_stddev)
    return normal

def rsample_normal(raw_params, scale_stddev=1, max_log_sigma=-10000, min_log_sigma=10000):
    """
    Samples from a normal distribution with given parameters.

    Parameters
    ----------
    raw_params : torch.*.Tensor
        Tensor containing a Gaussian mean and a raw scale parameter on its last dimension.
    scale_stddev : float
        Multiplier of the final scale parameter of the Gaussian.

    Returns
    -------
    torch.*.Tensor
        Sample from the normal distribution with the input mean and eps + softplus(raw scale) * scale_stddev as
        standard deviation.
    """

    normal = make_normal_from_raw_params(raw_params, scale_stddev=scale_stddev)
    sample = normal.rsample()
    return sample

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, hidden_channel=256):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36], hidden_channel),
            nn.Conv2d(hidden_channel, hidden_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(),
            nn.Conv2d(hidden_channel, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


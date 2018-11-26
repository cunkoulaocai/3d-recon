from collections import namedtuple

import torch as tr
from torch import nn
from torch.nn import functional as F


#     use only when h == w for an image
def get_padding(stride, in_dim, kernel_dim, out_dim=None, mode='SAME'):
    k = kernel_dim
    if out_dim == None:
        out_dim = (in_dim + stride - 1) // stride
    if mode.lower() == 'same':
        val = stride * (out_dim - 1) - in_dim + k
        if val % 2 == 0:
            p1, p2 = val // 2, val // 2
        else:
            p1, p2 = (val + 1) // 2, (val + 1) // 2
        return (p1, p2, p1, p2)


class Conv2DBlock(nn.Module):
    def __init__(self, in_filters, out_filters, batch_norm=False, layer_norm=False, kernel_size=3, stride=2, padding=None,
                 act=None, ln_shape=None):
        super(Conv2DBlock, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel = kernel_size

        layers = [nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride)]

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_filters, 0.8))

        if layer_norm:
            layers.append(nn.LayerNorm(ln_shape))

        if act == 'relu':
            layers.append(nn.ReLU())
        elif act == 'lrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif act == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif act is not None:  # general case - if act is a nn.Module class
            layers.append(act())

        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        padding = self.padding
        if isinstance(padding, str):
            padding = get_padding(self.stride, input.shape[2], self.kernel, mode=self.padding)
        elif padding is None:
            padding = 0
        input = F.pad(input, padding)
        input = self.conv_block(input)
        return input


class ConvT3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, layer_norm=False, kernel_size=3, stride=2,
                 padding=0, output_padding=0, act=None, ln_shape=None):
        super(ConvT3DBlock, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel = kernel_size

        layers = [nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                     padding=(2, 2, 2),output_padding=1)]

        if batch_norm:
            layers.append(nn.BatchNorm3d(out_channels, 0.8))

        if layer_norm:
            layers.append(nn.LayerNorm(ln_shape))

        if act == 'relu':
            layers.append(nn.ReLU())
        elif act == 'lrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif act == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif act is not None:  # general case - if act is a nn.Module class
            layers.append(act())

        self.convt_block = nn.Sequential(*layers)

    def forward(self, input):
        input = self.convt_block(input)
        return input


class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, batch_norm=False, layer_norm=False, act='relu'):
        super(DenseBlock, self).__init__()

        layers = [nn.Linear(in_features, out_features)]

        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features, 0.8))

        if layer_norm:
            layers.append(nn.LayerNorm(out_features))

        if act == 'relu':
            layers.append(nn.ReLU())
        elif act == 'lrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif act is not None:  # general case - if act is a nn.Module class
            layers.append(act())

        self.dense_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.dense_block(x)
        return x

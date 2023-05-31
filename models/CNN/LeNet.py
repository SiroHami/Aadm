"""
Implementation of LeNet, from paper
"Gradient-Based Learning Applied to Document Recognition"
original paper:
http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
"""

__all__ = ['LeNet']

import os
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from blocks import ConvBlock, LinearBlock


class LeNetConv(ConvBlock):
    """
    LeNet specific convolution block

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_sizes : tuple/list of 2 int
        Convolution window size.
    strides : tuple/list of 2 int
        Stride of the convolution.
    paddings : tuple/list of 2 int
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 strides,
                 paddings):
        super(LeNetConv, self).__init__(in_channels=in_channels)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes,
            stride=strides,
            padding=paddings)

    def forward(self, x):
        x = super(LeNetConv, self).forward(x)
        return x


class LeNetOuputBlock(nn.Module):
    """
    LeNet specific output block

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    classes : int
        Number of classification classes.
    """
    def __init__(self,
                 in_channels,
                 classes):
        super(LeNetOuputBlock, self).__init__()
        mid_channels = 120

        self.fc1 = LinearBlock(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.fc2 = LinearBlock(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.fc3 = nn.Linear(
            in_features=mid_channels,
            out_features=classes)
        
        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            return x


class LeNet(ConvBlock):
    def __init__(self,
                 channels,
                 kernel_sizes,
                    strides,
                    paddings,
                    in_size,
                    num_classes
                 ):
        super(LeNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        for i, channel_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channel_per_stage):
                stage.add_module("unit{}".format(j + 1), LeNetConv(
                    in_chnnels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes[i][j],
                    strides=strides[i][j],
                    paddings=paddings[i][j]
                ))
                in_channels = out_channels
            stage.add_module("pool{}".format(i + 1), nn.MaxPool2d(
                kernel_size=2, stride=2))
            self.features.add_module("stage{}".format(i + 1), stage)

            self.output = LeNetOuputBlock(
                in_channels=(in_channels),
                classes = num_classes)
            
            self._init_param()

    def _init_param(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  
        x = self.output(x)
        return x
        


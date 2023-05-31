"""
    Implementation of AlexNet, from paper
    "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.
    original paper: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

    url: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""

__all__ = ['AlexNet']

import os
import torch
import torch.nn as nn
import torch.nn.init as init


class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """

    def __init__(self, num_classes=1000, dropout=0.5):
        """
        Define and allocate layers for this neural net.

        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        self.features = nn.Sequential(
            #layer1 
            nn.Conv2d(input=3, output=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #layer2
            nn.Conv2d(input=96, output=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #layer3
            nn.Conv2d(input=256, output=384, kernel_size=3, padding=1),
            nn.ReLU(),
            #layer4
            nn.Conv2d(input=384, output=384, kernel_size=3, padding=1),
            nn.ReLU(),
            #layer5
            nn.Conv2d(input=384, output=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            #fc layer1
            nn.Dropout(p=dropout),
            nn.Linear(input=256 * 6 * 6, output=4096),
            nn.ReLU(),
            #fc layer2
            nn.Dropout(p=dropout),
            nn.Linear(input=4096, output=4096),
            nn.ReLU(),
            #fc layer3
            nn.Linear(input=4096, output=num_classes),
        )



    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
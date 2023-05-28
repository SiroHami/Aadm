"""
@article{
zhang2018mixup,
title={mixup: Beyond Empirical Risk Minimization},
author={Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz},
journal={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=r1Ddp1-Rb},
}
"""

import torch
import torch.nn as nn

class Mixup(nn.Module):
    def __init__(self, beta):
        super(Mixup, self).__init__()
        self.beta = beta

    def forward(self, x, l):
        batch_size = x.size(0)
        mix = torch.distributions.Beta(self.beta, self.beta).sample([batch_size, 1, 1, 1])
        mix = torch.max(mix, 1 - mix)
        x_mix = x * mix + x.flip(dims=[0]) * (1 - mix)
        l_mix = l * mix[:, :, 0, 0] + l.flip(dims=[0]) * (1 - mix[:, :, 0, 0])
        return x_mix, l_mix
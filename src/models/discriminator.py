"""
As described in https://arxiv.org/abs/2001.07466,
"""

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, patch_size, n=128):
        super().__init__()

        self.patch_size = patch_size
        self.n = n

        self.patch_factors = self._get_patch_factors()
        self.channels = self._get_channels()
        print(self.patch_factors)
        print(self.channels)

        modules = []
        for i, pf in enumerate(self.patch_factors):
            modules.append(nn.Conv2d(*self.channels[i], pf, pf, padding=1))
            if i > 0:
                modules.append(nn.InstanceNorm2d(self.channels[i][-1]))
            if i < len(self.patch_factors):
                modules.append(nn.LeakyReLU(0.2, inplace=True))

        modules.append(nn.Conv2d(self.channels[-1][-1], 1, 1, 1, padding=1))

        self.model = nn.Sequential(*modules)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)

        return x.mean((-2, -1)).squeeze()

    def _get_patch_factors(self):
        patch_factors = list(reversed([f for f in range(1, self.patch_size + 1)
                                       if self.patch_size % f == 0][1:-1]))
        if len(patch_factors) == 1:
            patch_factors *= 2

        return patch_factors

    def _get_channels(self):
        channels = [(self.n*2**i, self.n*2**(i+1))
                    for i, _ in enumerate(self.patch_factors)]
        channels[0] = (3, self.n*2)

        return channels

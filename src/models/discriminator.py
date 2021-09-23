"""
Discriminator module as described in https://arxiv.org/abs/2001.07466
"""

import torch.nn as nn


class Discriminator(nn.Module):
    """
    This module is subject to the rules:
    a) kernel_size = stride for each layer
    b) kernel_size is a factor of patch_size
    Parameters
    ----------
    patch_size : int
        The size of the patches that the kernels will process
    ch_mult : int
        The channel multiplier that will define the depth of the conv layers
    """
    def __init__(self, patch_size, ch_mult=128):
        super().__init__()

        self.patch_size = patch_size
        self.n = ch_mult
        self.patch_factors = self._get_patch_factors()

        channels = self._get_channels()

        modules = nn.ModuleList()
        for i, pf in enumerate(self.patch_factors):
            modules.append(nn.Conv2d(*channels[i], pf, pf))
            modules.append(nn.BatchNorm2d(channels[i][-1]))
            modules.append(nn.LeakyReLU(inplace=True))

        modules.append(nn.Conv2d(channels[-1][-1], 1, 1, 1))

        self.model = nn.Sequential(*modules)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)

        return x.mean((-2, -1)).squeeze()

    def _get_patch_factors(self):
        """
        Obtain all factors of the patch size except the number 1 and itself
        """
        patch_factors = list(reversed([f for f in range(1, self.patch_size + 1)
                                       if self.patch_size % f == 0][1:-1]))
        # if there is only one factor, it is the square root
        if len(patch_factors) == 1:
            patch_factors *= 2

        return patch_factors

    def _get_channels(self):
        """
        Obtain channel pairs for each of the convolutional layers of the network
        """
        channels = [(self.n * 2 ** i, self.n * 2 ** (i + 1))
                    for i, _ in enumerate(self.patch_factors)]
        # the first set of channels needs to be RGB for image inputs
        channels[0] = (3, self.n * 2)

        return channels

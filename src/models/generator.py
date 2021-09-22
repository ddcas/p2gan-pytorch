"""
As described in https://arxiv.org/abs/2001.07466,
"""

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        modules = []

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        x = self.model(x)
        
        return x

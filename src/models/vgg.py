"""
VGG16 model trained on ImageNet and used as described in
https://arxiv.org/pdf/2001.07466.pdf
"""

import torch
import torchvision.models as models
import torch.nn as nn


class VGG(nn.Module):
    """
    Download and slice the original VGG16 pre-trained model

    Parameters
    ----------
    device : str
        The model is pushed into the GPU memory if available
    batchnorm : int
        The option to use the normalized original or normalized version of VGG
    """
    def __init__(self, device: str, batchnorm: bool = True):
        super().__init__()

        if batchnorm:
            vgg = models.vgg16_bn(pretrained=True)
        else:
            vgg = models.vgg16(pretrained=True)

        # extract feature map from relu1_1 and push model to GPU if available
        self.vgg = vgg.features[:4].to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg(x)

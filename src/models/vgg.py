import torchvision.models as models
import torch.nn as nn


class VGG(nn.Module):
    """
    Slice of the VGG16 model trained on ImageNet. It outputs the feature
    map at the output of layer named relu1_1, as described in
    https://arxiv.org/pdf/2001.07466.pdf
    """
    def __init__(self, device, batchnorm=True):
        super().__init__()

        if batchnorm:
            vgg = models.vgg16_bn(pretrained=True)
        else:
            vgg = models.vgg16(pretrained=True)

        self.vgg = vgg.features[:4].to(device)

    def forward(self, x):
        return  self.vgg(x)

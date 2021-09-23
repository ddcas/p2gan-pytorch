"""
Generator module based on the problem described
in https://arxiv.org/abs/2001.07466,
"""

import torch
import torch.nn as nn


class GeneratorBlock(nn.Module):
    """
    Main component of the Generator module. Similar to the classic residual
    blocks, first developed in https://arxiv.org/abs/1512.03385

    Parameters
    ----------
    in_ch : int
        The number of input channels
    out_ch : int
        The number of output channels
    residual: bool
        The option to add the skip connection at the output
    """
    def __init__(self, in_ch, out_ch, residual=False):
        super().__init__()

        self.residual = residual
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.residual:
            return x + self.block(x)
        else:
            return self.block(x)


class Generator(nn.Module):
    """
    The Generator module. It resembles the U-Net architecture as an
    encoder-decoder module that is supposed to capture the selected style
    while preserving the content learned from the real images processed

    Parameters
    ----------
    enc_layers : int
        The number of encoder blocks
    dec_layers : int
        The number of decoder blocks
    res_layers: int
        The number of residual blocks
    ch_multi : int
        The channel multiplier that will define the depth of the conv layers
    """

    def __init__(self, enc_layers, dec_layers, res_layers, ch_multi=16):
        super().__init__()

        self.enc_layers = enc_layers
        self.res_layers = res_layers
        self.dec_layers = dec_layers
        self.n = ch_multi

        channels_enc = self._get_channels()
        channels_dec = [(c2+c1, c2)
                        for i, (c2, c1) in enumerate(reversed(channels_enc))]
        channels_dec[-1] = channels_enc[0][::-1]

        # build encoder blocks
        self.down, self.enc = nn.ModuleList(), nn.ModuleList()
        for layer in range(self.enc_layers):
            self.down.append(nn.MaxPool2d(2))
            self.enc.append(GeneratorBlock(*channels_enc[layer]))

        # build residual blocks
        self.res = nn.ModuleList()
        for layer in range(self.res_layers):
            self.res.append(GeneratorBlock(
                channels_enc[-1][-1],
                channels_enc[-1][-1],
                residual=True))

        # build decoder blocks
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for layer in range(self.dec_layers):
            self.up.append(nn.Upsample(scale_factor=2))
            self.dec.append(GeneratorBlock(*channels_dec[layer]))

        self.final = nn.Conv2d(channels_dec[-1][0], 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # going down the U
        skip_connections = []
        for layer in range(self.enc_layers):
            x = self.enc[layer](x)
            x = self.down[layer](x)
            skip_connections.append(x)

        # residual block/s at the bottom of the U
        for layer in range(self.res_layers):
            x = self.res[layer](x)

        # going up the U
        for layer in range(self.dec_layers - 1):
            x = self.up[layer](x)
            x = self.dec[layer](
                torch.cat([x, skip_connections[::-1][layer + 1]], dim=1))

        x = self.up[-1](x)
        x = self.final(x)
        x = self.tanh(x)

        return x

    def _get_channels(self) -> list:
        """
        Obtain channel pairs for each of the convolutional layers of the network
        """
        channels = [(self.n * 2 ** i, self.n * 2 ** (i + 1))
                    for i in range(self.enc_layers)]
        # the first set of channels needs to be RGB for image inputs
        channels[0] = (3, self.n * 2)

        return channels

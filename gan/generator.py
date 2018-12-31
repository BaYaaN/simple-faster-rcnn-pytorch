import torch.nn as nn


class Generator(nn.Module):
    # TODO REMOVE unsues inputs
    def __init__(self, in_channels=64, n_residual_blocks=6):
        super(Generator, self).__init__()

        # TODO FIX SIZES
        self.convBlock = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3),
            nn.Conv2d(128, 512, 1)
        ).cuda()

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock())
        self.res_blocks = nn.Sequential(*res_blocks)

    def forward(self, x):
        outFromConvBlock = self.convBlock(x)
        out = self.res_blocks(outFromConvBlock)
        return out


class ResidualBlock(nn.Module):
    def __init__(self,):
        super(ResidualBlock, self).__init__()
        conv_block = [nn.Conv2d(512, 512, 3, 1, 1),
                      nn.BatchNorm2d(512),
                      nn.ReLU(),
                      nn.Conv2d(512, 512, 3, 1, 1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

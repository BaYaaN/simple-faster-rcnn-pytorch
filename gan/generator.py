import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, firstConvLayer, in_channels=64, n_residual_blocks=6):
        super(Generator, self).__init__()

        # TODO FIX SIZES
        self.convBlock = nn.Sequential(
            firstConvLayer,
            nn.Conv2d(128, 64, 1),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(n_residual_blocks))
        self.res_blocks = nn.Sequential(*res_blocks)

    def forward(self, x):
        outFromConvBlock = self.convBlock(x)
        out = self.res_blocks(outFromConvBlock)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      nn.Conv2d(in_features, in_features, 3, 1, 1)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

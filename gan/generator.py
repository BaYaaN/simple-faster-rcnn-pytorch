import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 9, 1, 4),
        )

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 9, 1, 4),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(n_residual_blocks))
        self.res_blocks = nn.Sequential(*res_blocks)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out = self.res_blocks(out2)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

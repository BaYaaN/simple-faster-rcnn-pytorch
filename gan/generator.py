import torch.nn as nn


class Generator(nn.Module):
    # TODO REMOVE unsues inputs
    def __init__(self, rpn, roi, in_channels=64, n_residual_blocks=6):
        super(Generator, self).__init__()

        # TODO FIX SIZES!
        self.convBlock = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3),
            nn.Conv2d(128, 512, 1)
        )

        # TODO right now we have init weight from voc pascal, is it good or use xavier ?
        self.rpn = rpn
        self.roi = roi

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock())
        self.res_blocks = nn.Sequential(*res_blocks)

    def forward(self, x, imgSize, scale, bbox, label):
        featureMap = self.convBlock(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(featureMap, imgSize, scale)
        pool, gt_roi_loc, gt_roi_label = self.roi(featureMap, rois, bbox, label)

        return self.res_blocks(pool), gt_roi_loc, gt_roi_label


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

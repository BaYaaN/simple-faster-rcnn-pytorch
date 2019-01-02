import torch as torch
from torch import nn

from FasterRcnn.utils import array_tool as at


class Roi(nn.Module):
    def __init__(self, n_class, roi):
        super(Roi, self).__init__()
        self.n_class = n_class
        self.roi = roi

    def forward(self, x, rois, roi_indices):
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()
        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)

        return pool

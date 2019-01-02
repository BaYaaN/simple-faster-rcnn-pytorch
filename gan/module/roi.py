import torch as torch
from torch import nn

from FasterRcnn.model.utils.creator_tool import ProposalTargetCreator
from FasterRcnn.utils import array_tool as at


class Roi(nn.Module):
    def __init__(self, n_class, roi):
        super(Roi, self).__init__()
        self.n_class = n_class
        self.roi = roi
        self.proposal_target_creator = ProposalTargetCreator()
        self.loc_normalize_mean = (0., 0., 0., 0.),
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)

    def forward(self, x, roi, bbox, label):
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        roi_indices = torch.zeros(len(sample_roi))
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(sample_roi).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()
        pool = self.roi(x, indices_and_rois)
        # pool = pool.view(pool.size(0), -1)

        return pool, gt_roi_loc, gt_roi_label

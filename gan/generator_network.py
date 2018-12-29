import torch as torch
from FasterRcnn.utils import array_tool as at


class GeneratorNetwork(torch.nn.Module):
    def __init__(self, extractor, rpn, roi, classifier):
        super(GeneratorNetwork, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.roi = roi
        self.classifier = classifier

    def forward(self, x, scale=1.):
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)

        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(h, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        roi_cls_locs, roi_scores = self.classifier(pool)


        return roi_cls_locs, roi_scores, rois, roi_indices



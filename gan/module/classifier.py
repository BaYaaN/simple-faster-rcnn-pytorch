from torch import nn


class Classifier(nn.Module):
    def __init__(self, classifier, cls_loc, score):
        super(Classifier, self).__init__()
        self.classifier = classifier
        self.cls_loc = cls_loc
        self.score = score

    def forward(self, x):
        fc7 = self.classifier(x)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores

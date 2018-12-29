import torch

from FasterRcnn.model import FasterRCNNVGG16
from gan.generator_network import GeneratorNetwork
from gan.module.classifier import Classifier


def decomposeFasterRCNN():
    model = FasterRCNNVGG16().cuda()
    # state_dict = t.load('../' + opt.load_path)
    # model.load_state_dict(state_dict['model'])
    extractor = model.extractor
    fistConvBlock = list(model.extractor)[:5]
    rpn = model.rpn
    roi = model.head.roi
    classifier = model.head.classifier
    cls_loc = model.head.cls_loc
    score = model.head.score

    return extractor, torch.nn.Sequential(*fistConvBlock), rpn, roi, classifier, cls_loc, score


def generatorNetworkCreator():
    extractor, firstConvBlock, rpn, roi, classifier, cls_loc, score = decomposeFasterRCNN()
    classifier = Classifier(classifier, cls_loc, score)
    generatorNetwork = GeneratorNetwork(extractor, rpn, roi, classifier)

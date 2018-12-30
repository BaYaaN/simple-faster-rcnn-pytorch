import torch

from FasterRcnn.model import FasterRCNNVGG16
from FasterRcnn.utils.config import opt
from gan.generator import Generator
from gan.generator_network import GeneratorNetwork
from gan.module.classifier import Classifier
from gan.module.roi import Roi

# TODO move it co conf file
n_class_with_background = 21


def decomposeFasterRCNN():
    model = FasterRCNNVGG16().cuda()
    fillModeWithPretrainedWeights(model)
    fistConvBlock = list(model.extractor)[:5]
    lastFourConvBlocks = list(model.extractor)[5:]
    rpn = model.rpn
    roi = model.head.roi
    classifier = model.head.classifier
    cls_loc = model.head.cls_loc
    score = model.head.score

    return torch.nn.Sequential(*fistConvBlock), torch.nn.Sequential(
        *lastFourConvBlocks), rpn, roi, classifier, cls_loc, score


def generatorNetworkCreator(**kwargs):
    opt._parse(kwargs)
    firstConvBlock, lastFourConvBlocks, rpn, roi, classifier, cls_loc, score = decomposeFasterRCNN()
    classifier = Classifier(classifier, cls_loc, score)
    roi = Roi(n_class_with_background, roi)
    generator = Generator()

    return GeneratorNetwork(firstConvBlock, lastFourConvBlocks, rpn, roi, classifier, generator)

def fillModeWithPretrainedWeights(model):
    state_dict = torch.load('../' + opt.load_path)
    model.load_state_dict(state_dict['model'])


if __name__ == '__main__':
    import fire

    fire.Fire()

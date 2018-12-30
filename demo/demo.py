# %load_ext autoreload
# %autoreload 2
# import torch as t
#
# from FasterRcnn.data.util import read_image
# from FasterRcnn.model import FasterRCNNVGG16
# from FasterRcnn.utils import array_tool as at
# from FasterRcnn.utils.vis_tool import vis_bbox
# %matplotlib tk
#
# img = read_image('demo/demo.jpg')
# img = t.from_numpy(img)[None]
# faster_rcnn = FasterRCNNVGG16().cuda()
#
#
# state_dict = t.load('pretrained-model/fasterrcnn_12222105_0.712649824453_caffe_pretrain.pth')
# faster_rcnn.load_state_dict(state_dict['model'])
#
# _bboxes, _labels, _scores = faster_rcnn.predict(img,visualize=True)
# vis_bbox(at.tonumpy(img[0]),
#          at.tonumpy(_bboxes[0]),
#          at.tonumpy(_labels[0]).reshape(-1),
#          at.tonumpy(_scores[0]).reshape(-1))
#
#

from __future__ import division
import os
from PIL import Image
from wider import WIDER
from skimage import transform as sktsf
from data.dataset import Dataset, TestDataset,inverse_normalize
from data.dataset import pytorch_normalze
import numpy as np
import ipdb
import torch
import matplotlib
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.config import opt
from model import FasterRCNNVGG16
from torch.autograd import Variable
from torch.utils import data as data_
from trainer import FasterRCNNTrainer,VictimFasterRCNNTrainer
from utils import array_tool as at
import utils.vis_tool as vz
from utils.vis_tool import visdom_bbox
from utils.vis_tool import vis_bbox,visdom_bbox
from utils.eval_tool import eval_detection_voc
from data.util import  read_image
import pandas as pd
from PIL import Image
import attacks
import cv2
import torch.onnx
import torch
import caffe2.python.onnx.backend as backend
import os
import onnx
import numpy as np

def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.
         (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray:
        A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    try:
        img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect')
    except:
        ipdb.set_trace()
    # both the longer and shorter should be less than
    # max_size and min_size
    normalize = pytorch_normalze
    return normalize(img)

if __name__ == '__main__':
    attacker = attacks.Inference_DCGAN(train_adv=False)
    attacker.load('/home/joey/Desktop/simple-faster-rcnn-pytorch/checkpoints/max_min_attack_6.pth')
    attacker.cpu()
    attacker.train(False)
    img = read_image('/home/joey/Desktop/simple-faster-rcnn-pytorch/akira_img.jpeg')
    img = preprocess(img)
    img = torch.from_numpy(img)[None]
    img = Variable(img.float())
    adv_img  = attacker(img,epsilon=1)
    # Export ONNX model
    torch.onnx.export(attacker, img, "attacker.proto", export_params=True, verbose=True)
    # Load ONNX model
    model = onnx.load("attacker.proto")
    graph = model.graph
    # Check Formation
    onnx.checker.check_model(model)
    onnx.helper.printable_graph(model.graph)
    # Print Graph to get blob names
    onnx.helper.printable_graph(graph)
    # Check model output
    ipdb.set_trace()
    rep = backend.prepare(graph, device="CPU")
    output_onnx = rep.run(img.cpu().data.numpy().astype(np.float32))
    # Verify the numerical correctness upto 3 decimal places
    np.testing.assert_almost_equal(adv_img.data.cpu().numpy(),
            output_onnx[0], decimal=3)


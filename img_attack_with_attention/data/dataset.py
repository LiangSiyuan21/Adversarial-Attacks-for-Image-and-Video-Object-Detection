from __future__ import division
import torch as t
from .voc_dataset import VOCBboxDataset
from .wider_dataset import WIDERBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from . import util
import numpy as np
import pandas as pd
from utils.config import opt
from data.util import read_image
import torch
import ipdb

def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.5 + 0.5).clip(min=0, max=1) * 255
    # return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    # normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225])
    normalize = tvtsf.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=600, max_size=800):
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

    # img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect')
    img = sktsf.resize(img, (C, 300, 300), mode='reflect')
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform(object):

    def __init__(self, min_size=600, max_size=800):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        if opt.data == 'wider':
            self.db = WIDERBboxDataset(opt.wider_label_dir,\
                    opt.wider_data_dir,opt.wider_fname_mat)
        else:
            self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = '../ext_face/cropped_global_300w.csv'
        self.globalDF = pd.read_csv(self.csv_file)
        self.g_images = self.globalDF['imgPath']
        self.save_dir = '/media/drive/ibug/300W_cropped/frcnn/'
        self.save_dir_adv = '/media/drive/ibug/300W_cropped/frcnn_adv/'
        self.save_dir_comb = '/media/drive/ibug/300W_cropped/frcnn_comb/'

    def __len__(self):
        return len(self.g_images)

    def __getitem__(self, idx):
        img = read_image(self.g_images[idx])
        _, H, W = img.shape
        scale = H / H
        try:
            img = preprocess(img)
            img, params = util.random_flip(
                img, x_random=True, return_param=True)
        except:
            print("Exception")
        img = torch.from_numpy(img)[None]
        return img,self.g_images[idx],scale


# class TestDataset:
#     def __init__(self, opt, split='test', use_difficult=True):
#         self.opt = opt
#         self.db = WIDERBboxDataset(opt.wider_label_dir,\
#                 opt.wider_val_data_dir,opt.wider_val_fname_mat)
#
#     def __getitem__(self, idx):
#         ori_img, bbox, label, difficult = self.db.get_example(idx)
#         img = preprocess(ori_img)
#         return img, ori_img.shape[1:], bbox, label, difficult
#
#     def __len__(self):
#         return len(self.db)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True, preprocess_gt=False):
        self.opt = opt
        self.preprocess_gt = preprocess_gt
        self.db = VOCBboxDataset(opt.voc_test_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult, o_H, o_W = self.db.get_example(idx, preprocess_gt=self.preprocess_gt)
        img = preprocess(ori_img)
        H = img.shape[1]
        W = img.shape[2]
        bbox = util.resize_bbox(bbox, (o_H, o_W), (H, W))
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
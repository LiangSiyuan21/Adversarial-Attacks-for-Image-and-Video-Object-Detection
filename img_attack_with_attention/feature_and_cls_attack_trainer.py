# -*-coding:utf-8-*- # -*- coding: utf-8 -*-
import os
import ipdb
import torch
import torch.nn as nn
from utils.config import opt
from data.util import read_image
from data.dataset import Dataset
from data.dataset import inverse_normalize
from torch.utils import data as data_
from model import FasterRCNNVGG16
from functools import partial
from trainer import BRFasterRcnnTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from torch.autograd import Variable
from tqdm import tqdm
from data.dataset import preprocess, pytorch_normalze
import attacks

layer_idies = [15, 20]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def store(model):
    """
    make hook for feature map
    """

    def hook(module, input, output, key):
        for layer_idx in layer_idies:
            if key == layer_idx:
                model.feature_maps[key] = output[0]

    for idx, layer in enumerate(model._modules.get('extractor')):
        layer.register_forward_hook(partial(hook, key=idx))

def train(**kwargs):
    opt._parse(kwargs)
    # opt.caffe_pretrain = True
    TrainResume = False
    dataset = Dataset(opt)
    print('load dataset')
    dataloader = data_.DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=opt.num_workers)

    target_img_path = 'target_img.jpg'
    target_img = read_image(target_img_path) / 255
    target_img = torch.from_numpy(pytorch_normalze(target_img))
    target_img = torch.unsqueeze(target_img, 0).numpy()

    attacker = attacks.Blade_runner(train_BR=True)

    if TrainResume:
        attacker.load('checkpoints/attack_02152100_0.path')
    # attacker = attacks_no_target.Blade_runner(train_BR=True)
    faster_rcnn = FasterRCNNVGG16().eval()
    faster_rcnn.cuda()
    store(faster_rcnn)
    trainer = BRFasterRcnnTrainer(faster_rcnn, attacker, layer_idx=layer_idies, attack_mode=True).cuda()

    if opt.load_path:
        trainer.load(opt.load_path)
        print('from %s Load model parameters' % opt.load_path)

    trainer.vis.text(dataset.db.label_names, win='labels')
    target_features_list = list()

    img_feature = trainer.faster_rcnn(torch.from_numpy(target_img).cuda())
    del img_feature
    target_features = trainer.faster_rcnn.feature_maps
    for target_feature_idx in target_features:
        target_features_list.append(target_features[target_feature_idx].cpu().detach().numpy())
    del target_features

    for epoch in range(opt.epoch):
        trainer.reset_meters(BR=True)
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            img, bbox, label = Variable(img), Variable(bbox), Variable(label)
            rois, roi_scores = faster_rcnn(img, flag=True)
            if len(rois) != len(roi_scores):
                print('The generated ROI and ROI score lengths are inconsistent')
            trainer.train_step(img, bbox, label, scale, target_features_list, rois=rois, roi_scores=roi_scores)

            if (ii) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                # trainer.vis.plot_many(trainer.get_meter_data())
                trainer.vis.plot_many(trainer.get_meter_data(BR=True))

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicted bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)
                if trainer.attacker is not None:
                    adv_img = trainer.attacker.perturb(img, rois=rois, roi_scores=roi_scores)
                    adv_img_ = inverse_normalize(at.tonumpy(adv_img[0]))
                    _bboxes, _labels, _scores = trainer.faster_rcnn.predict([adv_img_], visualize=True)
                    adv_pred_img = visdom_bbox(adv_img_,
                                               at.tonumpy(_bboxes[0]),
                                               at.tonumpy(_labels[0]).reshape(-1),
                                               at.tonumpy(_scores[0]))
                    trainer.vis.img('adv_img', adv_pred_img)
                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())

            if ii % 500 == 0 and ii != 0:
                best_path = trainer.save(epochs=ii, save_rcnn=False)
                print('best path is %s' % best_path)
        if epoch % 2 == 0:
            best_path = trainer.save(epochs=epoch, save_rcnn=False)


if __name__ == '__main__':
    train()

# coding=utf-8
from collections import namedtuple
import time
import torch
from torch.nn import functional as F
from functools import partial
import pdb
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
import attacks
from torch import nn
import torch as t
from torch.autograd import Variable
from utils import array_tool as at
from utils.vis_tool import Visualizer

from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

LossTupleAdv = namedtuple('LossTupleAdv',
                          ['NLL_loss',
                           'Match_loss',
                           'pertubation_loss',
                           'total_adv_loss'
                           ])

LossTupleBR = namedtuple('LossTupleBR',
                         ['feature_loss',
                          'pertubation_loss',
                          'NLL_loss',
                          'loss_GAN',
                          'total_D_loss',
                          'total_BR_loss',
                          ])


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    flag = Variable(flag)
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, Variable(in_weight), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= (gt_label >= 0).sum()  # ignore gt_label==-1 for rpn_loss
    return loc_loss


class BRFasterRcnnTrainer(nn.Module):
    def __init__(self, faster_rcnn, attacker=None, layer_idx=None, attack_mode=False):
        super(BRFasterRcnnTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.attacker = attacker
        self.layer_idx = layer_idx
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma
        self.attack_mode = attack_mode

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()

        self.vis = Visualizer(env=opt.env)

        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}
        self.BR_meters = {k: AverageValueMeter() for k in LossTupleBR._fields}

    def forward(self, imgs, bboxes, labels, scale, attack=False):
        """Forward Faster R-CNN and calculate losses.

            Here are notations used.

            * :math:`N` is the batch size.
            * :math:`R` is the number of bounding boxes per image.

            Currently, only :math:`N=1` is supported.

            Args:
                imgs (~torch.autograd.Variable): A variable with a batch of images.
                bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                    Its shape is :math:`(N, R, 4)`.
                labels (~torch.autograd..Variable): A batch of labels.
                    Its shape is :math:`(N, R)`. The background is excluded from
                    the definition, which means that the range of the value
                    is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                    classes.
                scale (float): Amount of scaling applied to
                    the raw image during preprocessing.

            Returns:
                namedtuple of 5 losses
            """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        # 创造钩子函数,记录featureamp的值
        features = self.faster_rcnn.extractor(imgs)
        feature_maps = self.faster_rcnn.feature_maps

        if not features.sum()[0] == 0:
            rpn_locs, rpn_scores, rois, roi_indices, anchor = \
                self.faster_rcnn.rpn(features, img_size, scale)
            # Since batch size is one, convert variables to singular form
            bbox = bboxes[0]
            label = labels[0]
            rpn_score = rpn_scores[0]
            rpn_loc = rpn_locs[0]
            roi = rois

            # Sample RoIs and forward
            # it's fine to break the computation graph of rois,
            # consider them as constant input
            if rois.size == 0:
                print("Features are 0 for some reason")
                losses = [Variable(torch.zeros(1)).cuda(), Variable(torch.zeros(1)).cuda(), \
                          Variable(torch.zeros(1)).cuda(), Variable(torch.zeros(1)).cuda()]
                losses = losses + [sum(losses)]
                return losses, features

            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
                roi,
                at.tonumpy(bbox),
                at.tonumpy(label),
                self.loc_normalize_mean,
                self.loc_normalize_std)
            # NOTE it's all zero because now it only support for batch=1 now
            sample_roi_index = t.zeros(len(sample_roi))
            roi_cls_loc, roi_score = self.faster_rcnn.head(
                features,
                sample_roi,
                sample_roi_index)

            # ------------------ RPN losses -------------------#
            if not attack:
                if anchor.size != 0:
                    gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                        at.tonumpy(bbox),
                        anchor,
                        img_size)
                    gt_rpn_label = at.tovariable(gt_rpn_label).long()
                    gt_rpn_loc = at.tovariable(gt_rpn_loc)
                    rpn_loc_loss = _fast_rcnn_loc_loss(
                        rpn_loc,
                        gt_rpn_loc,
                        gt_rpn_label.data,
                        self.rpn_sigma)

                    # NOTE: default value of ignore_index is -100 ...
                    rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
                    _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
                    _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
                    self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())
                    # adv_losses = self.attacker.forward(imgs.detach(),gt_rpn_label.cuda(), img_size, scale, self)
                    # adv_losses = LossTupleAdv(*adv_losses)
                    # self.update_meters(adv_losses,adv=True)
                else:
                    rpn_cls_loss = 0
                    rpn_loc_loss = 0

            # ------------------ ROI losses (fast rcnn loss) -------------------#
            n_sample = roi_cls_loc.shape[0]
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                                  at.totensor(gt_roi_label).long()]
            gt_roi_label = at.tovariable(gt_roi_label).long()
            gt_roi_loc = at.tovariable(gt_roi_loc)
            if attack:
                return roi_score, gt_roi_label, feature_maps
            else:
                roi_loc_loss = _fast_rcnn_loc_loss(
                    roi_loc.contiguous(),
                    gt_roi_loc,
                    gt_roi_label.data,
                    self.roi_sigma)

                roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

                self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

                losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
                losses = losses + [sum(losses)]

            # if attack:
            #     del rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, losses, features
            #     return roi_score, gt_roi_label, feature_map
            # else:
                return LossTuple(*losses)
        else:
            print("Features are 0 for some reason")
            losses = [Variable(torch.zeros(1)).cuda(), Variable(torch.zeros(1)).cuda(), \
                      Variable(torch.zeros(1)).cuda(), Variable(torch.zeros(1)).cuda()]
            losses = losses + [sum(losses)]
            return losses

    def train_step(self, imgs, bboxes, labels, scale, target_feature=None, rois=None, roi_scores=None):
        if not self.attack_mode:
            print ('....')
        else:
            BR_losses = self.attacker.forward(imgs, self, labels, bboxes, scale, target_feature, rois, roi_scores)
            BR_losses = LossTupleBR(*BR_losses)
            self.update_meters(BR_losses, BR=True)

    # 将save_rcnn设置成False,因为我们在训练生成器过程中,不动rcnn的参数
    def save(self, save_optimizer=False, save_path=None, save_rcnn=False, **kwargs):
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/faterrcnn_full_%s' % timestr
            if not self.attack_mode:
                for k_, v_ in kwargs.items():
                    save_path += '%s' % v_
            if self.attacker is not None:
                self.attacker.save('checkpoints/attack_%s_%d.path' % (timestr, kwargs['epochs']))
        if save_rcnn:
            t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:
            self.faster_rcnn.load_state_dict(state_dict)
            return self

        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses, BR=False):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        if not BR:
            for key, meter in self.meters.items():
                meter.add(loss_d[key])
        else:
            for key, meter in self.BR_meters.items():
                meter.add(loss_d[key])

    def reset_meters(self, BR=False):
        for key, meter in self.meters.items():
            meter.reset()
        if BR:
            for key, meter in self.BR_meters.items():
                meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self, BR=False):
        if BR:
            return {k: v.value()[0] for k, v in self.BR_meters.items()}
        else:
            return {k: v.value()[0] for k, v in self.meters.items()}


def store(model):
    """
    make hook for feature map
    """
    def hook(module, input, output, key):
        model.feature_maps[key] = output[0]

    for idx, layer in enumerate(model._modules.get('extractor')):
        layer.register_forward_hook(partial(hook, key=idx))

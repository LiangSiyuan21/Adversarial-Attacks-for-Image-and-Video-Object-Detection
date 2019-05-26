# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from model import GAN
import sys

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def reduce_sum(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)

    return x


def calAttentionMask(rois, roi_scores, state=None):
    rois_num = len(roi_scores)
    ymins = []
    xmins = []
    ymaxs = []
    xmaxs = []

    if state == 0:
        mask = np.zeros((300, 300), dtype=np.float32)
    if state == 1:
        mask = np.zeros((75, 75), dtype=np.float32)
    if state == 2:
        mask = np.zeros((37, 37), dtype=np.float32)

    for i in range(rois_num):
        ymins.append(int(rois[i][0]))
        xmins.append(int(rois[i][1]))
        ymaxs.append(int(rois[i][2]))
        xmaxs.append(int(rois[i][3]))

    for i in range(rois_num):
        h = ymaxs[i] - ymins[i]
        w = xmaxs[i] - xmins[i]
        roi_weight = np.ones((h, w)) * roi_scores[i]
        if h == 0 or w == 0:
            mask = mask + 0
        else:
            mask[ymins[i]:ymaxs[i], xmins[i]:xmaxs[i]] = mask[ymins[i]:ymaxs[i], xmins[i]:xmaxs[i]] + roi_weight

    mask_min = np.min(mask)
    mask_max = np.max(mask)

    mask = (mask - mask_min) / (mask_max - mask_min)
    return mask

def L2_dist(x, y):
    return reduce_sum((x - y) ** 2)


def torch_arctanh(x, eps=1e-6):
    x = x * (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5


class Blade_runner(nn.Module):
    def __init__(self, num_channels=3, ngf=100, cg=0.05, lr=1e-4, train_BR=False):
        super(Blade_runner, self).__init__()
        self.discriminator = GAN.define_D(input_nc=3, ndf=64)

        self.generator = nn.Sequential(  # input is (nc) x 32 x 32
            nn.Conv2d(num_channels, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 3 x 32 x 32
            nn.Conv2d(ngf, num_channels, 1, 1, 0, bias=True),
            nn.Tanh()
        )

        self.cuda = torch.cuda.is_available()

        if self.cuda:
            self.generator.cuda()
            self.generator = torch.nn.DataParallel(self.generator, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        self.cg = cg
        self.criterionGAN = GAN.GANLoss()
        self.optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=0.0002, betas=(0.5, 0.999))
        self.train_BR = train_BR
        self.max_iter = 20
        # self.c_feature_weights = 0.002
        self.c_feature_weights = [0.00010, 0.00020]
        self.c_feature_confidence = 0
        self.c_misclassify = 1
        self.confidence = 0

    def forward(self, inputs, model, labels=None, bboxes=None, scale=None, target_features=None, rois=None, roi_scores=None):
        global losses
        num_objects = 0
        iter_count = 0
        loss_perturb = 20
        loss_feature = 100
        loss_misclassify = 10
        gt_labels_np = labels[0].cpu().detach().numpy()
        gt_labels_list = np.unique(gt_labels_np + 1)
        rois_feature1 = np.array(rois/4, dtype=np.int32)
        rois_feature2 = np.array(rois/8.1, dtype=np.int32)
        feature1_mask = calAttentionMask(rois_feature1, roi_scores, state=1)
        feature2_mask = calAttentionMask(rois_feature2, roi_scores, state=2)
        img_mask = calAttentionMask(rois, roi_scores, state=0)
        feature1_mask = torch.from_numpy(np.tile(feature1_mask, (256, 1, 1))).cuda()
        feature2_mask = torch.from_numpy(np.tile(feature2_mask, (512, 1, 1))).cuda()
        img_mask = torch.from_numpy(np.tile(img_mask, (3, 1, 1))).cuda()


        for i in range(1):
            perturbation = self.generator(inputs)
            adv_inputs = inputs + perturbation * img_mask
            adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)
            self.optimizer_D.zero_grad()

            pred_real = self.discriminator(inputs.detach())
            loss_D_real = self.criterionGAN(pred_real, True)
            # loss_D_real.backward(retain_graph=False)

            pred_fake = self.discriminator(adv_inputs.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # loss_D_fake.backward(retain_graph=False)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()
            self.optimizer_D.step()

            print('Discriminator loss:REAL %f, FAKE %f' % (loss_D_real, loss_D_fake))

        while loss_feature > 1 and loss_perturb > 1:
            perturbation = self.generator(inputs)
            adv_inputs = inputs + perturbation * img_mask
            adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)
            pred_fake = self.discriminator(adv_inputs.detach())
            loss_GAN = self.criterionGAN(pred_fake, True)
            # self.optimizer_D.zero_grad()
            #
            # pred_real = self.discriminator(inputs)
            # loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
            # loss_D_real.backward(retain_graph=False)
            #
            # pred_fake = self.discriminator(adv_inputs)
            # loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
            # loss_D_fake.backward(retain_graph=False)
            # loss_D_GAN = loss_D_fake + loss_D_real

            # self.optimizer_D.step()
            # print('Discriminator loss:REAL %f, FAKE %f' % (loss_D_real, loss_D_fake))
            # del loss_D_real, loss_D_fake

            scores, gt_labels, adv_features = model(adv_inputs, \
                                                   bboxes, labels, scale, attack=True)

            self.optimizer.zero_grad()

            adv_feature1 = adv_features[15] * feature1_mask
            adv_feature1 = torch.unsqueeze(adv_feature1, 0).cuda()
            h_feature = adv_feature1.shape[2]
            w_feature = adv_feature1.shape[3]
            if not isinstance(target_features[0], torch.Tensor):
                target_feature1 = target_features[0][:, 0:h_feature, 0:w_feature]
                target_feature1 = torch.unsqueeze(torch.from_numpy(target_feature1), 0).cuda()
                target_feature1 = target_feature1 * feature1_mask

            adv_feature2 = adv_features[20] * feature2_mask
            adv_feature2 = torch.unsqueeze(adv_feature2, 0).cuda()
            h_feature = adv_feature2.shape[2]
            w_feature = adv_feature2.shape[3]
            if not isinstance(target_features[1], torch.Tensor):
                target_feature2 = target_features[1][:, 0:h_feature, 0:w_feature]
                target_feature2 = torch.unsqueeze(torch.from_numpy(target_feature2), 0).cuda()
                target_feature2 = target_feature2 * feature2_mask

            probs = F.softmax(scores)
            suppress_labels, probs, mask = model.faster_rcnn._suppress(None, probs, gt_labels_list, attack=True)

            scores = scores[mask]
            gt_labels = gt_labels[mask]
            self.optimizer.zero_grad()
            # try:

            if mask.sum() != 0:
                idices = scores.max(1)[1]
                one_hot_labels = np.zeros((gt_labels.size()[0], 21))
                for i in range(0, gt_labels.size(0)):
                    one_hot_labels[i][idices[i]] = 1
                # one_hot_labels = torch.zeros(gt_labels.size() + (21,))
                one_hot_labels = torch.from_numpy(one_hot_labels).float()
                if self.cuda: one_hot_labels = one_hot_labels.cuda()
                labels_vars = Variable(one_hot_labels, requires_grad=False)

                real = (labels_vars * scores).sum(1)
                other = ((1. - labels_vars) * scores - labels_vars * 10000.).max(1)[0]
            else:
                real = torch.zeros(1).cuda()
                other = torch.zeros(1).cuda()
            loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
            loss_misclassify = self.c_misclassify * torch.sum(loss1)


            # loss_feature计算的是feature_map之间的L2损失
            # loss_feature = self.c_feature_weights * activation_L2_dist(adv_feature, target_feature)
            loss_feature1 = self.c_feature_weights[0] * L2_dist(adv_feature1, target_feature1)
            loss_feature2 = self.c_feature_weights[1] * L2_dist(adv_feature2, target_feature2)
            loss_feature = loss_feature1 + loss_feature2
            loss_perturb = self.cg * L2_dist(inputs, adv_inputs)
            loss_total = loss_feature + loss_perturb + loss_misclassify + loss_GAN
            loss_total.backward(retain_graph=False)
            self.optimizer.step()

            print ('Loss feature is %f, Loss perturb %f, Loss cls %f, Loss GAN %f, total loss %f' % (loss_feature, loss_perturb, loss_misclassify, loss_GAN, loss_total))
            # predictions = torch.max(F.log_softmax(scores), 1)[1].cpu().numpy()
            # num_objects = (predictions == gt_labels).sum()
            #
            # print ('There are %d objects in the picture' % num_objects)

            iter_count = iter_count + 1

            # loss_feature_v = Variable(loss_feature.data)
            # loss_perturb_v = Variable(loss_perturb.data)

            losses = [Variable(loss_feature.data), Variable(loss_perturb.data), Variable(loss_misclassify.data), Variable(loss_GAN.data), Variable(loss_D.data)]
            losses = losses + [sum(losses)]

            # losses_v = (sum(losses)

            if iter_count > self.max_iter:
                break

        del loss_feature, loss_perturb, loss_misclassify, loss_feature1, loss_feature2, loss_D
        return losses

    def perturb(self, inputs, epsilon=1, save_perturb=None, rois=None, roi_scores=None):
        img_mask = calAttentionMask(rois, roi_scores, state=0)
        img_mask = torch.from_numpy(np.tile(img_mask, (3, 1, 1))).cuda()
        perturbation = self.generator(inputs)
        adv_inputs = inputs + img_mask * perturbation
        adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)
        if save_perturb is not None:
            clamped = torch.clamp(perturbation, -1.0, 1.0)
            distance = self.cg * L2_dist(inputs, adv_inputs).cpu().detach().numpy().max()
            return adv_inputs, clamped, distance
        else:
            return adv_inputs

    def save(self, fn):
        torch.save(self.generator.state_dict(), fn)

    def load(self, fn):
        self.generator.load_state_dict(torch.load(fn))






import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
import pandas as pd
from torch.autograd import Variable
from ._base import Distiller
import numpy as np


def ce_loss(hid, target):
    target = target.reshape(hid.shape[0], 1)
    log_pro = -1.0 * F.log_softmax(hid, dim=1)
    one_hot = torch.zeros(hid.shape[0], hid.shape[1]).cuda()
    one_hot = one_hot.scatter_(1, target, 1)
    loss_our = torch.mul(log_pro, one_hot).sum(dim=1)
    L = loss_our.cpu().detach().numpy().tolist()
    return L


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def kd_loss(logits_student, logits_teacher, f_s, f_t, idx, epc, target, CL, T_MAX, T_MIN, Reduce, pt_s):
    T_max = 9
    T_min = 6
    alpha = 0
    # sm = [0, 0.04, 0.06, 0.08, 0.1]
    if epc > 25 and epc <= 50:
        alpha = 25 / 240
    elif epc > 50 and epc <= 75:
        alpha = 50 / 240
    elif epc > 75 and epc <= 100:
        alpha = 75 / 240
    elif epc > 100 and epc <= 240:
        alpha = 100 / 240
    T_min = (1 - alpha) * T_min
    T_max = (1 - alpha) * T_max
    tmp = []
    if T_min < 2:
        T_min = 2
    if T_max < 2:
        T_max = 2
    logprobs = F.log_softmax(logits_student, dim=1)

    target = F.one_hot(target, 100)  # 转换成one-hot
    pt = 1 - pt_s
    target = target.float()
    CL_S = []
    beta = []
    T = []

    for i in range(len(idx)):
        CL_S.append(CL[idx[i], -1])
        tmp = CL[idx[i], -1] * (T_max - T_min) + T_min
        logits_teacher[i] = torch.div(logits_teacher[i], tmp)
        T.append(tmp)

    CL_S = torch.tensor(CL_S).cuda(non_blocking=True)
    T = torch.tensor(T).cuda(non_blocking=True)
    log_pred_student = F.log_softmax(logits_student, dim=1)
    pred_teacher = F.softmax(logits_teacher, dim=1)
    loss_all = 0.0
    for fs, ft in zip(f_s, f_t):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="none")
        loss = loss.mean()
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            tmploss = F.mse_loss(tmpfs, tmpft, reduction='none')

            tmploss = tmploss.mean()
            cnt /= 2.0
            loss += tmploss * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss = loss_kd
    tmp = torch.tensor(tmp).cuda(non_blocking=True)
    loss_kd = torch.mul(loss_kd, T).mean() / 1.0
    loss_all = loss_all * (CL_S + 1)
    return loss_kd, loss, loss_all


class IAKD(Distiller):
    def __init__(self, student, teacher, cfg, dim):
        super(IAKD, self).__init__(student, teacher)
        self.shapes = dim[0]  # cfg.REVIEWKD.SHAPES
        self.out_shapes = dim[1]  # cfg.REVIEWKD.OUT_SHAPES
        in_channels = dim[2]  # cfg.REVIEWKD.IN_CHANNELS
        out_channels = dim[3]  # cfg.REVIEWKD.OUT_CHANNELS
        self.T_MAX = cfg.KD_.T_MAX
        self.T_MIN = cfg.KD_.T_MIN
        self.Reduce = cfg.KD_.Reduce
        self.ce_loss_weight = cfg.REVIEWKD.CE_WEIGHT
        self.reviewkd_loss_weight = cfg.REVIEWKD.REVIEWKD_WEIGHT
        self.warmup_epochs = cfg.REVIEWKD.WARMUP_EPOCHS
        self.stu_preact = cfg.REVIEWKD.STU_PREACT
        self.max_mid_channel = cfg.REVIEWKD.MAX_MID_CHANNEL
        abfs = nn.ModuleList()
        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                ABF(
                    in_channel,
                    mid_channel,
                    out_channels[idx],
                    idx < len(in_channels) - 1,
                )
            )
        self.abfs = abfs[::-1]

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.abfs.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.abfs.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)

        index = kwargs['index']
        epc = kwargs['epc']
        ten = kwargs['ten']
        ind_ = list(index)
        ind = []
        for i in ind_:
            ind.append(int(i))
        loss_student_ce = ce_loss(logits_student, target)

        a = 1
        b = 2 * min(kwargs["epc"] / 20, 2)

        logprobs = F.log_softmax(logits_student, dim=1)
        tmp_target = target.view(-1, 1)
        logpt = logprobs.gather(1, tmp_target)
        logpt = logpt.view(-1)
        pt_s = Variable(logpt.data.exp())

        if self.stu_preact:
            x = features_student["preact_feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        else:
            x = features_student["feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        x = x[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)
        features_teacher = features_teacher["preact_feats"][1:] + [
            features_teacher["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]

        # losses
        loss_kd, loss_student, loss_all = kd_loss(
            logits_student, logits_teacher, results, features_teacher, ind, epc, target, ten, self.T_MAX, self.T_MIN,
            self.Reduce, pt_s
        )
        if epc in [1, 25, 50, 75, 100, 125]:
            # l_kd_ = list(l_kd_each)
            for i in range(len(loss_student)):
                ten[ind[i], 0] = loss_student[i] + loss_student_ce[
                    i]  # self.ce_loss_weight*loss_student[i]+self.kd_loss_weight*float(l_kd_[i])
                ten[ind[i], 1] = 1 - pt_s[i]

        losses_dict = {
            "loss_ce": a * F.cross_entropy(logits_student, target),
            "loss_kd": b * loss_kd,
            "loss_rekd": 7
                         * min(kwargs["epoch"] / self.warmup_epochs, 2.0)
                         * loss_all
        }
        return logits_student, losses_dict


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (h, w), mode="nearest")  # (shape, shape)
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        # output
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x

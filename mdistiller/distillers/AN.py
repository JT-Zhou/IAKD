import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def an_loss(logits_student, logits_teacher, temperature,epoch,target):
    if epoch <=100:
        log_pred_student = F.log_softmax(logits_student, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        #loss_kd *= temperature**2
    else:

        loss_kd = F.cross_entropy(logits_student, target)
    return loss_kd


class AN(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(AN, self).__init__(student, teacher)

        self.temperature = cfg.AN.T
        self.ce_loss_weight = cfg.AN.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.AN.LOSS.KD_WEIGHT
        self.max_temperature = cfg.AN.T_MAX
    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        epoch = kwargs['epoch']
        self.temperature = kwargs['T']
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        if epoch <= 100:
            loss_kd = self.kd_loss_weight * an_loss(
                logits_student, logits_teacher, self.temperature,epoch,target
            )
            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_kd,
            }
        else:
            losses_dict = {
                "loss_ce": loss_ce,
                #"loss_kd": loss_kd,
            }
        return logits_student, losses_dict
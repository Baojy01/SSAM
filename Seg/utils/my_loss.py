import torch
import torch.nn as nn
import torch.nn.functional as F
from .cldice import Soft_clDice


def ce_loss(inputs, target, loss_weight=None, ignore_index: int = 255):
    loss = F.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
    return loss


def focal_loss(inputs, target, alpha=0.5, gamma=2, loss_weight=None, ignore_index: int = 255):
    log_pt = - F.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
    pt = torch.exp(log_pt)
    loss = - alpha * ((1 - pt) ** gamma) * log_pt
    return loss


def bce_loss(inputs, target, num_chasses: int = 2, loss_weight=None):
    target = F.one_hot(target, num_chasses).permute(0, 3, 1, 2).float()
    loss = F.binary_cross_entropy_with_logits(inputs, target, weight=loss_weight)
    return loss


def dice_loss(inputs, target, num_chasses: int = 2, smooth=1e-6):
    N = inputs.shape[0]
    temp_inputs = F.softmax(inputs, dim=1).reshape(N, -1)
    temp_target = F.one_hot(target, num_chasses).permute(0, 3, 1, 2).float().reshape(N, -1)

    tp = torch.sum(temp_target * temp_inputs, dim=1)
    fp = torch.sum(temp_inputs, dim=1) - tp
    fn = torch.sum(temp_target, dim=1) - tp

    score = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

    loss = 1 - score.sum() / N
    return loss


def criterion(inputs, target, num_chasses=2, flag: str = 'ce', use_dice=False):
    LOS = {'ce': ce_loss, 'bce': bce_loss, 'focal': focal_loss}
    loss_fn = LOS[flag]

    if flag == 'bce':
        loss = loss_fn(inputs, target, num_chasses)
    else:
        loss = loss_fn(inputs, target)

    if use_dice:
        loss = loss + dice_loss(inputs, target, num_chasses)

    return loss


class LOSS(nn.Module):
    def __init__(self, num_classes=2, loss_type='ce', use_dice=False,use_aux=False):
        super().__init__()
        self.num_classes = num_classes
        self.type = loss_type
        self.use_dice = use_dice
        self.use_aux = use_aux
        self.cl_dice = Soft_clDice()

    def forward(self, img, label):
        loss = criterion(img, label, self.num_classes, self.type, self.use_dice)
        if self.use_aux:
            loss = loss + 0.5 * self.cl_dice(img, label)
            
        return loss

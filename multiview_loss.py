# Copyright (c) 2020-present, Muzammil Behzad
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MV_LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(MV_LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MV_SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(MV_SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class MVloss(nn.Module):

    def __init__(self):
        super(MVloss, self).__init__()

    def forward(self, xRA0, xRA20, xRA_20, target, wRA0, wRA20, wRA_20):
        criterion_MV = torch.nn.CrossEntropyLoss()
        loss_multiview = criterion_MV(wRA0*xRA0 + wRA20*xRA20 + wRA_20*xRA_20, target)
        return loss_multiview


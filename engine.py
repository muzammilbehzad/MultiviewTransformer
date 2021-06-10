# Copyright (c) 2020-present, Muzammil Behzad
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from multiview_loss import MVloss


def train_one_epoch(modelRA0: torch.nn.Module, modelRA20: torch.nn.Module, modelRA_20: torch.nn.Module,
                    criterion: DistillationLoss,
                    data_loader: Iterable, optimizer_MV: torch.optim.Optimizer, 
                    optimizerRA0: torch.optim.Optimizer, optimizerRA20: torch.optim.Optimizer, optimizerRA_20: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_emaRA0: Optional[ModelEma] = None, model_emaRA20: Optional[ModelEma] = None, model_emaRA_20: Optional[ModelEma] = None, 
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):


    modelRA0.train(set_training_mode)
    modelRA20.train(set_training_mode)
    modelRA_20.train(set_training_mode)


    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if 'wRA0' not in locals() or 'wRA20' not in locals() or 'wRA_20' not in locals():
        wRA0 = 1/3
        wRA20 = 1/3
        wRA_20 = 1/3

    for i, ((samplesRA0, targetsRA0), (samplesRA20, targetsRA20), (samplesRA_20, targetsRA_20))  in enumerate(data_loader):
        samplesRA0 = samplesRA0.to(device, non_blocking=True)
        samplesRA20 = samplesRA20.to(device, non_blocking=True)
        samplesRA_20 = samplesRA_20.to(device, non_blocking=True)

        targetsRA0 = targetsRA0.to(device, non_blocking=True)
        targetsRA20 = targetsRA20.to(device, non_blocking=True)
        targetsRA_20 = targetsRA_20.to(device, non_blocking=True)

        assert torch.equal(targetsRA0, targetsRA20) and torch.equal(targetsRA0, targetsRA_20)
        targets = targetsRA0
        
        optimizerRA0.zero_grad()
        optimizerRA20.zero_grad()
        optimizerRA_20.zero_grad()

        mixup_fn_skip = False
        if not mixup_fn_skip:
            if mixup_fn is not None:
                samplesRA0, targetsRA0 = mixup_fn(samplesRA0, targetsRA0)
                samplesRA20, targetsRA20 = mixup_fn(samplesRA20, targetsRA20)
                samplesRA_20, targetsRA_20 = mixup_fn(samplesRA_20, targetsRA_20)

        with torch.cuda.amp.autocast():
            outputsRA0 = modelRA0(samplesRA0)
            outputsRA20 = modelRA20(samplesRA20)
            outputsRA_20 = modelRA_20(samplesRA_20)

            loss = MVloss(outputsRA0, outputsRA20, outputsRA_20, targets, wRA0, wRA20, wRA_20)
            loss.backward()
            
            optimizerRA0.step()
            optimizerRA20.step()
            optimizerRA_20.step()
        
            loss1 = torch.nn.CrossEntropyLoss(samplesRA0, outputsRA0, targetsRA0)
            loss2 = torch.nn.CrossEntropyLoss(samplesRA20, outputsRA20, targetsRA20)
            loss3 = torch.nn.CrossEntropyLoss(samplesRA_20, outputsRA_20, targetsRA_20)

        loss_value1 = loss1.item()
        loss_value2 = loss2.item()
        loss_value3 = loss3.item()

        use_trainable_weights = True
        if use_trainable_weights:
            w_function = torch.nn.Softmax(dim=0)
            learnt_weights = w_function(torch.tensor([loss_value1, loss_value2, loss_value3]))
            wRA0 = learnt_weights[0].item()
            wRA20 = learnt_weights[1].item()
            wRA_20 = learnt_weights[2].item()

        if (not math.isfinite(loss_value1)) or (not math.isfinite(loss_value2)) or (not math.isfinite(loss_value3)):
            print("Stopping training because RA0 loss: {}, RA20 loss: {}, RA_20 loss: {}, ".format(loss_value1, loss_value2, loss_value3))
            sys.exit(1)

        torch.cuda.synchronize()
        if model_emaRA0 is not None:
            model_emaRA0.update(modelRA0)
        if model_emaRA20 is not None:
            model_emaRA20.update(modelRA20)
        if model_emaRA_20 is not None:
            model_emaRA_20.update(modelRA_20)

        metric_logger.update(lossRA0=loss_value1)
        metric_logger.update(lossRA20=loss_value2)
        metric_logger.update(lossRA_20=loss_value3)

        
        metric_logger.update(wRA0=wRA0)
        metric_logger.update(wRA20=wRA20)        
        metric_logger.update(wRA_20=wRA_20)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_logger



@torch.no_grad()
def evaluate(data_loader, modelRA0, modelRA20, modelRA_20, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    modelRA0.eval()
    modelRA20.eval()
    modelRA_20.eval()
    
    Acc1 = []
    Acc2 = []
    Acc3 = []
    Acc4 = []
    Acc5 = []
    Acc6 = []

    for i, ((samplesRA0, targetsRA0), (samplesRA20, targetsRA20), (samplesRA_20, targetsRA_20))  in enumerate(data_loader):

        samplesRA0 = samplesRA0.to(device, non_blocking=True)
        samplesRA20 = samplesRA20.to(device, non_blocking=True)
        samplesRA_20 = samplesRA_20.to(device, non_blocking=True)

        targetsRA0 = targetsRA0.to(device, non_blocking=True)
        targetsRA20 = targetsRA20.to(device, non_blocking=True)
        targetsRA_20 = targetsRA_20.to(device, non_blocking=True)

        assert torch.equal(targetsRA0, targetsRA20) and torch.equal(targetsRA0, targetsRA_20)
        targets = targetsRA0

        # compute output
        with torch.cuda.amp.autocast():
            outputsRA0 = modelRA0(samplesRA0)
            outputsRA20 = modelRA20(samplesRA20)
            outputsRA_20 = modelRA_20(samplesRA_20)

            wRA0 = 1/3
            wRA20 = 1/3
            wRA_20 = 1/3
            output_multiview = wRA0*outputsRA0 + wRA20*outputsRA20 + wRA_20*outputsRA_20
            loss = criterion(output_multiview, targets)

        acc1, acc2, acc3, acc4, acc5, acc6 = accuracy(output_multiview, targets, topk=(1, 2, 3, 4, 5, 6))
        Acc1.append(acc1)
        Acc2.append(acc2)
        Acc3.append(acc3)
        Acc4.append(acc4)
        Acc5.append(acc5)
        Acc6.append(acc6)
    

        batch_size = output_multiview.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top2=metric_logger.acc2, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

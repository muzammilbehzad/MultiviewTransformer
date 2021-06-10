# Copyright (c) 2020-present, Muzammil Behzad
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
import models
import utils
import os

import copy
import torch.distributed
from multiview_loss import MV_LabelSmoothingCrossEntropy, MV_SoftTargetCrossEntropy


def get_args_parser(model_num, model_name):
    parser = argparse.ArgumentParser('MiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/path/to/data/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='BU4DFE', choices=['BU4DFE', 'CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='tcp://localhost:26543', help='url used to set up distributed training')

    opt = parser.parse_args()
    opt.output_dir = r"./results/out_" + str(model_name)

    opt.data_path_base = "/path/to/dataset"

    port = find_free_port()
    opt.dist_url = "tcp://localhost:{}".format(port)
    
    opt.model = model_name
    if '224' in opt.model: 
        opt.input_size = 224
    elif '384' in opt.model: 
        opt.input_size = 384
    else:
        ValueError('Invalid data input size')
    
    return opt

def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.



class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def main(args):
    if not torch.distributed.is_initialized():
        utils.init_distributed_mode(args)
    else:
        torch.distributed.destroy_process_group()
        utils.init_distributed_mode(args)
        
    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)        

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True


    # preparing datasets for multi-views
    RA_views = ['RA0', 'RA20', 'RA_20']

    args.data_path = os.path.join(args.data_path_base, RA_views[0])
    dataset_trainRA0, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_valRA0, _ = build_dataset(is_train=False, args=args)
    
    args.data_path = os.path.join(args.data_path_base, RA_views[1])
    dataset_trainRA20, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_valRA20, _ = build_dataset(is_train=False, args=args)
    
    args.data_path = os.path.join(args.data_path_base, RA_views[2])
    dataset_trainRA_20, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_valRA_20, _ = build_dataset(is_train=False, args=args)

    args.data_path = '' # removing path to avoid logical errors



    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # else:
        # sampler_train = torch.utils.data.RandomSampler(dataset_train)
        # sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        ConcatDataset(
                 dataset_trainRA0,
                 dataset_trainRA20,
                 dataset_trainRA_20
                 ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        ConcatDataset(
                 dataset_valRA0,
                 dataset_valRA20,
                 dataset_valRA_20
                 ),
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    dataset_list = ['Bosphorus','BU3DFE','BU4DFE','BP4D']
    if args.data_set in dataset_list: # to load model with original parameters and then adjust the last layer
        pretrained_model = True
        args.nb_classes = 1000

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=pretrained_model,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    if args.data_set in dataset_list: # to load model with original parameters and then adjust the last layer
        args.nb_classes = 6
        if args.model in ['deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224', 'deit_base_patch16_384']:
            model.head = torch.nn.Linear(model.head.in_features, args.nb_classes, bias=True)
        elif args.model in ['deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224', 'deit_base_distilled_patch16_224' , 'deit_base_distilled_patch16_384']:
            model.head = torch.nn.Linear(model.head.in_features, args.nb_classes, bias=True)
            model.head_dist = torch.nn.Linear(model.head_dist.in_features, args.nb_classes, bias=True)


    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)


    modelRA0 = copy.deepcopy(model)
    modelRA20 = copy.deepcopy(model)
    modelRA_20 = copy.deepcopy(model)
    del model

    modelRA0.to(device)
    modelRA20.to(device)
    modelRA_20.to(device)

    for RA_iter in range(3):        
        if RA_iter == 0:
            model_emaRA0 = None
            if args.model_ema:
                # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
                model_emaRA0 = ModelEma(
                    modelRA0,
                    decay=args.model_ema_decay,
                    device='cpu' if args.model_ema_force_cpu else '',
                    resume='')
            model_without_ddpRA0 = modelRA0

        elif RA_iter == 1:
            model_emaRA20 = None
            if args.model_ema:
                # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
                model_emaRA20 = ModelEma(
                    modelRA20,
                    decay=args.model_ema_decay,
                    device='cpu' if args.model_ema_force_cpu else '',
                    resume='')
            model_without_ddpRA20 = modelRA20
        elif RA_iter == 2:
            model_emaRA_20 = None
            if args.model_ema:
                # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
                model_emaRA_20 = ModelEma(
                    modelRA_20,
                    decay=args.model_ema_decay,
                    device='cpu' if args.model_ema_force_cpu else '',
                    resume='')
            model_without_ddpRA_20 = modelRA_20
        else:
            ValueError('Invalid indexing')    
    

    if args.distributed:
        modelRA0 = torch.nn.parallel.DistributedDataParallel(modelRA0, device_ids=[args.gpu])
        modelRA20 = torch.nn.parallel.DistributedDataParallel(modelRA20, device_ids=[args.gpu])
        modelRA_20 = torch.nn.parallel.DistributedDataParallel(modelRA_20, device_ids=[args.gpu])

        model_without_ddpRA0 = modelRA0.module
        model_without_ddpRA20 = modelRA20.module
        model_without_ddpRA_20 = modelRA_20.module


    n_parametersRA0 = sum(p.numel() for p in modelRA0.parameters() if p.requires_grad)
    n_parametersRA20 = sum(p.numel() for p in modelRA20.parameters() if p.requires_grad)
    n_parametersRA_20 = sum(p.numel() for p in modelRA_20.parameters() if p.requires_grad)
    assert n_parametersRA0 == n_parametersRA20 == n_parametersRA_20
    n_parameters = n_parametersRA0

    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizerRA0 = create_optimizer(args, model_without_ddpRA0)
    optimizerRA20 = create_optimizer(args, model_without_ddpRA20)
    optimizerRA_20 = create_optimizer(args, model_without_ddpRA_20)

    optimizer_MV = torch.optim.SGD(list(modelRA0.parameters()) + list(modelRA20.parameters()) + list(modelRA_20.parameters()), lr=args.lr, momentum=args.momentum)

    loss_scaler = NativeScaler()

    lr_schedulerRA0, _ = create_scheduler(args, optimizerRA0)
    lr_schedulerRA20, _ = create_scheduler(args, optimizerRA20)
    lr_schedulerRA_20, _ = create_scheduler(args, optimizerRA_20)
    use_multiview_loss = True
    if use_multiview_loss:
        criterion = MV_LabelSmoothingCrossEntropy()
    else:
        criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        if use_multiview_loss:
            criterion = MV_SoftTargetCrossEntropy()
        else:
            criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        if use_multiview_loss:
            criterion = MV_LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        NotImplementedError('Loss function not implemented')
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        test_stat = evaluate(data_loader_val, modelRA0, modelRA20, modelRA_20, device)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
           data_loader_train.sampler.set_epoch(epoch)

        train_stats, metric_logger = train_one_epoch(
            modelRA0, modelRA20, modelRA_20, criterion, 
            data_loader_train,optimizer_MV,
            optimizerRA0, optimizerRA20, optimizerRA_20,  
            device, epoch, loss_scaler, args.clip_grad, 
            model_emaRA0, model_emaRA20, model_emaRA_20, mixup_fn,
            set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
        )
        localtime = time.asctime(time.localtime(time.time()))
        print("\n[{}] Epoch: {}, Averaged stats: {}".format(localtime, epoch, metric_logger))
     
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'modelRA0': model_without_ddpRA0.state_dict(),
                    'modelRA20': model_without_ddpRA20.state_dict(),
                    'modelRA_20': model_without_ddpRA_20.state_dict(),
                    'optimizerMV': optimizer_MV.state_dict(),
                    'optimizerRA0': optimizerRA0.state_dict(),
                    'optimizerRA20': optimizerRA20.state_dict(),
                    'optimizerRA_20': optimizerRA_20.state_dict(),
                    'lr_schedulerRA0': lr_schedulerRA0.state_dict(),
                    'lr_schedulerRA20': lr_schedulerRA20.state_dict(),
                    'lr_schedulerRA_20': lr_schedulerRA_20.state_dict(),
                    'epoch': epoch,
                    'model_emaRA0': get_state_dict(model_emaRA0),
                    'model_emaRA20': get_state_dict(model_emaRA20),
                    'model_emaRA_20': get_state_dict(model_emaRA_20),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, modelRA0, modelRA20, modelRA_20, device)

        print(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')


        localtime = time.asctime(time.localtime(time.time()))
        log_stats = {'time/date': f'[{localtime}]',
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'time': time.time() - start_time}


        if 'first_print' not in locals():
            import datetime
            log_filename = "logs_" + datetime.datetime.now().strftime("%d%B%Y") + ".txt"
            with (output_dir / log_filename).open("a") as f:
                f.write(str(args) + "\n\n")
            f.close()
            first_print = None


        if args.output_dir and utils.is_main_process():
            with (output_dir / log_filename).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    model_names = ['deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224']
    my_model_names = ['MiT_tiny_patch16_224', 'MiT_small_patch16_224', 'MiT_base_patch16_224']
    
    for model_num, model_name in enumerate(model_names):
        if model_num != 2:
            continue
        args = get_args_parser(model_num, model_name)
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        print("\n\n=======>>> Experiment started for {} <<<=======".format(my_model_names[model_num]))
        main(args)
        print("=======>>> Experiment completed for {} <<<=======\n\n".format(my_model_names[model_num]))
            

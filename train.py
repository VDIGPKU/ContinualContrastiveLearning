#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

import moco.loader
from moco.loader import split_images_labels
from moco.loader import merge_images_labels
from moco.loader import ImageFolder_with_id
import moco.builder
from moco.builder import concat_all_gather
from tqdm import tqdm

import numpy as np
import random
from sklearn.cluster import KMeans

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# incremental setting
parser.add_argument('--method', default='CCL', type=str,
                    help='choice of method')
parser.add_argument('--n-tasks', default=10, type=int,
                    help='number of tasks')
parser.add_argument('--n-save', default=20, type=int,
                    help='number of saved images for each class')
parser.add_argument('--imagenetsub', default=False, action='store_true',
                    help='use imagenet-sub')

# original MoCo setting
parser.add_argument('--data', metavar='DIR', default='/data/public_data/ImageNet/imagenet/',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--ccl-teacher-m', default=0.996, type=float,
                    help='momentum of updating teacher (default: 0.996)')
parser.add_argument('--ccl-k', default=256, type=int,
                    help='extra sample queue size; number of negative keys (default: 256)')


# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # ngpus_per_node = torch.cuda.device_count()
    ngpus_per_node = 8
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    assert args.method in ['CCL', 'Finetuning', 'SimpleReplay']

    print("=> creating model '{}', Method: {}".format(args.arch, args.method))
    if args.method == 'CCL':
        model = moco.builder.MoCoCCL(
            models.__dict__[args.arch],
            dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t, mlp=args.mlp,
            extra_sample_K=args.ccl_k, teacher_m=args.ccl_teacher_m)
    
    else:
        model = moco.builder.MoCo(
            models.__dict__[args.arch],
            dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t, mlp=args.mlp)
    # print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    if not os.path.isdir('checkpoints/{}'.format(args.method)):
        os.mkdir('checkpoints/{}'.format(args.method))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # MoCo v2's aug
    augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    base_augmentation = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]

    train_dataset_current = ImageFolder_with_id(
        traindir,
        moco.loader.TwoCropsTransform(transforms.Compose(augmentation), transforms.Compose(base_augmentation), is_old_sample=False))

    train_dataset_old = ImageFolder_with_id(
        traindir,
        moco.loader.TwoCropsTransform(transforms.Compose(augmentation), transforms.Compose(base_augmentation), is_old_sample=True))
    
    train_dataset_current_multi_view = ImageFolder_with_id(
        traindir,
        moco.loader.MultiViewTransform(transforms.Compose(augmentation), transforms.Compose(base_augmentation)))

    if args.imagenetsub:
        # Get the first 100 categories for simplicity
        order = np.arange(100) # imagenet-sub
        nb_cl = int(100/args.n_tasks)
    else:
        order = np.arange(1000) # imagenet-full
        nb_cl = int(1000/args.n_tasks)
    seed = 1
    np.random.seed(seed)
    np.random.shuffle(order)

    X_train_total, Y_train_total = split_images_labels(train_dataset_current.imgs)
    X_train_saved, Y_train_saved = [], []

    for t in range(args.n_tasks):
        actual_cl        = order[range(t*nb_cl, (t+1)*nb_cl)]
        indices_train = np.array([i in order[range(t*nb_cl, (t+1)*nb_cl)] for i in Y_train_total])
        X_train          = X_train_total[indices_train]
        Y_train          = Y_train_total[indices_train]
        current_train_imgs = merge_images_labels(X_train, Y_train)
        train_dataset_current.imgs = train_dataset_current.samples = current_train_imgs
        train_dataset_current_multi_view.imgs = train_dataset_current_multi_view.samples = current_train_imgs
        if t>0 and args.method != 'Finetuning':
            X_protoset = np.concatenate(X_train_saved, axis=0)
            Y_protoset = np.concatenate(Y_train_saved)

            old_train_imgs = merge_images_labels(X_protoset, Y_protoset)
            train_dataset_old.imgs = train_dataset_old.samples = old_train_imgs

            train_dataset_ensemble = ConcatDataset([train_dataset_current, train_dataset_old])
        else:
            train_dataset_ensemble = train_dataset_current

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_ensemble)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset_ensemble, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args, t)
            if args.method == 'CCL':
                model.module.update_teacher()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'saved_X': X_train_saved,
                    'saved_Y': Y_train_saved
                }, is_best=False, filename='./checkpoints/{}_ntask_{}/moco_checkpoint_{}.pth.tar'.format(args.method,args.n_tasks,t))
        
        if args.method != 'Finetuning':
            print('Image Saving ...')
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_current_multi_view)
            else:
                train_sampler = None

            train_loader_current_multi = torch.utils.data.DataLoader(
                train_dataset_current_multi_view, batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)
            if args.distributed:
                train_sampler.set_epoch(epoch)
            if args.method == 'CCL':
                X_saved, Y_saved = save_replay_image(train_loader_current_multi, [X_train,Y_train], model, args, indicator='min_var')
            elif args.method == 'SimpleReplay':
                X_saved, Y_saved = save_replay_image(train_loader_current_multi, [X_train,Y_train], model, args, indicator='random')
            X_train_saved.append(X_saved)
            Y_train_saved.append(Y_saved)



def save_replay_image(val_loader, img_set, model, args, indicator='random'):
    if args.imagenetsub:
        n_cls = 100//args.n_tasks
    else:
        n_cls = 1000//args.n_tasks

    if indicator=='random':
        X_train,Y_train = img_set
        idx = np.random.randint(X_train.shape[0], size=args.n_save*n_cls)
        return X_train[idx], Y_train[idx]
    else:
        assert indicator=='min_var'
        model.eval()
        X_train,Y_train = img_set
        feature_bank = []
        idx = []

        with torch.no_grad():
            for (images, _, im_id) in val_loader:
                if args.gpu is not None:
                    im_id = im_id.cuda(args.gpu, non_blocking=True)
                feature = []
                for i in range(len(images)):
                    if args.gpu is not None:
                        images[i] = images[i].cuda(args.gpu, non_blocking=True)
                    f = model(images[i], mode='feature')
                    feature.append(f.unsqueeze(dim=-1))
                feature = torch.cat(feature, dim=-1)
                feature_bank.append(feature)
                idx.append(im_id)

            feature_bank = torch.cat(feature_bank, dim=0)
            idx = torch.cat(idx, dim=0)

            feature_bank = concat_all_gather(feature_bank)
            idx = concat_all_gather(idx)

            feature_bank = feature_bank.cpu().numpy()
            idx = idx.cpu().numpy()
            idx = np.squeeze(idx).astype('int')
            idx, indices = np.unique(idx, return_index=True)
            feature_bank = feature_bank[indices]

            idx_sort = np.argsort(idx)
            feature_bank = feature_bank[idx_sort]
            feature_bank = np.squeeze(feature_bank)

            if feature_bank.shape[0]>X_train.shape[0]:
                feature_bank = feature_bank[:X_train.shape[0]]
                idx = idx[:X_train.shape[0]]
            if feature_bank.shape[0]<X_train.shape[0]:
                X_train = X_train[idx]
                Y_train = Y_train[idx]

            # t1 = time.time()
            kmeans=KMeans(n_clusters=n_cls)
            kmeans.fit(feature_bank[:,:,-1])
            # t2 = time.time()
            # print("time = ",t2-t1)

            prototypes = torch.from_numpy(kmeans.cluster_centers_)
            kmeans_label = torch.from_numpy(kmeans.labels_)
            feature_bank = torch.from_numpy(feature_bank)

            saved_X = []
            saved_Y = []
            for i in range(torch.min(kmeans_label), torch.max(kmeans_label)+1):
                index = kmeans_label==i
                f = feature_bank[index]

                m = f.mean(dim=-1, keepdim=True)
                x = X_train[index]
                y = Y_train[index]

                m = F.normalize(m, dim=1)
                std = torch.pow(f - m, 2).sum(dim=-1, keepdim=False).sum(-1, keepdim=False)
                ind = std.argsort(dim=-1, descending=False)[:args.n_save]

                saved_X.append(x[ind])
                saved_Y.append(y[ind])
            saved_X = np.concatenate(saved_X, axis=0)
            saved_Y = np.concatenate(saved_Y)

            return saved_X, saved_Y


def train(train_loader, model, criterion, optimizer, epoch, args, t=0):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Task {},Epoch: [{}]".format(t+1, epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            images[2] = images[2].cuda(args.gpu, non_blocking=True)
            is_from_old = images[3].cuda(args.gpu, non_blocking=True)

        # compute output
        if args.method == 'CCL':
            loss, output, target = model(im_q=images[0], im_k=images[1], im_raw=images[2], is_from_old=is_from_old, loss_fun=criterion, t=t)

            loss += criterion(output, target)
        else:
            output, target = model(im_q=images[0], im_k=images[1])
            loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    main()

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random

import argparse
import os
import shutil
import time
import numpy as np
import torch
import torchvision.datasets as datasets

class ImageFolder_with_id(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, torch.ones(1)*index

#split trainset.imgs
def split_images_labels(imgs):
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

#merge into trainset.imgs
def merge_images_labels(images, labels):
    images = list(images)
    labels = list(labels)
    assert(len(images)==len(labels))
    imgs = []
    for i in range(len(images)):
        item = (images[i], labels[i])
        imgs.append(item)
    
    return imgs

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, view_transform, base_transform=None, is_old_sample=False):
        self.view_transform = view_transform
        self.base_transform = base_transform
        self.is_old_sample = is_old_sample

    def __call__(self, x):
        q = self.view_transform(x)
        k = self.view_transform(x)
        if self.base_transform is not None:
            anchor = self.base_transform(x)
            if self.is_old_sample:
                return [q, k, anchor, torch.ones(1)]
            else:
                return [q, k, anchor, torch.zeros(1)]
        return [q, k]


class MultiViewTransform:
    
    def __init__(self, view_transform, base_transform=None, num=6):
        self.view_transform = view_transform
        self.base_transform = base_transform
        self.num = num

    def __call__(self, x):
        out = []
        for i in range(self.num):
            out.append(self.view_transform(x))
        out.append(self.base_transform(x))
        return out


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

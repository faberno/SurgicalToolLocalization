"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets. Also
    includes some transformation functions.
"""
from abc import ABC, abstractmethod
import cv2
import numpy as np
import torch
import torch.utils.data as data
from transforms.RandomMasking import RandomMasking
import torchvision.transforms as T

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    """

    def __init__(self, configuration):
        """Initialize the class; save the configuration in the class.
        """
        self.configuration = configuration

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point (usually data and labels in
            a supervised setting).
        """
        pass

    def pre_epoch_callback(self, epoch):
        """Callback to be called before every epoch.
        """
        pass

    def post_epoch_callback(self, epoch):
        """Callback to be called after every epoch.
        """
        pass

def channelswitcher(x):
    """
    Shuffles the color channels of an image.
    """
    if len(x.shape) == 4:
        return x[:, torch.randperm(3)]
    else:
        return x[torch.randperm(3)]

def get_transform(opt):
    """
    Builds the transform list from the configuration.
    """
    transform_list = []
    if 'transforms' in opt:
        print(opt)
        if 'toTensor' in opt['transforms'] and opt['transforms']['toTensor']:
            transform_list.append(T.ToTensor())
        if 'resize' in opt['transforms']:
            size = opt['transforms']['resize']
            transform_list.append(T.Resize(size))
        if 'flip' in opt['transforms']:
            transform_list.append(T.RandomHorizontalFlip(opt['transforms']['flip']))
        if 'masking' in opt['transforms']:
            masking = opt['transforms']['masking']
            transform_list.append(RandomMasking(*masking, value=opt['mean']))
        if 'rotation' in opt['transforms']:
            transform_list.append(T.RandomRotation(opt['transforms']['rotation'],
                                                   fill=opt['mean']))
        if 'normalize' in opt['transforms'] and opt['transforms']['normalize']:
            transform_list.append(T.Normalize(opt['mean'], opt['std']))
        if 'channelswitcher' in opt['transforms'] and opt['transforms']['channelswitcher']:
            transform_list.append(channelswitcher)

    return T.Compose(transform_list)
from typing import Any
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.resnet import ResNet, _resnet, BasicBlock, Bottleneck
from models.backbones.vgg import _vggnet
from models.backbones.alexnet import _alexnet

import torch
from torch.nn.functional import interpolate
from utils.peak_stimulation import peak_stimulation


def _median_filter(input):
    batch_size, num_channels, h, w = input.size()
    threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1, 1)


def _mean_filter(input):
    batch_size, num_channels, h, w = input.size()
    threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1, 1)


def _max_filter(input):
    batch_size, num_channels, h, w = input.size()
    threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1, 1)


def find_peaks(crm, upsample_size=None, aggregation=False, win_size=11,
               peak_filter=_median_filter, threshold=0.8):
    """
    Find the peaks in the class response maps.
    Arguments:
        crm: torch.tensor - Class Response Maps
        upsample_size: tuple/list - size of the original image. If specified the peaks position is
                                    translated to this size.
        aggregation: bool - wether or not to return the class scores of the peak response pooling
        win_size: int - Size of the sliding window
        peak_filter: function - Filter to distinguish peaks
    """
    out = dict()
    if aggregation:
        peak_list, aggregation = peak_stimulation(crm, return_aggregation=aggregation,
                                                  win_size=win_size, peak_filter=peak_filter)
        out['class_scores'] = aggregation
    else:
        peak_list = peak_stimulation(crm, return_aggregation=aggregation, win_size=win_size,
                                     peak_filter=peak_filter)
    peak_values = crm[peak_list[:, 0], peak_list[:, 1], peak_list[:, 2], peak_list[:, 3]]
    if upsample_size is not None:
        upsample = torch.tensor([upsample_size[0] / crm.shape[2], upsample_size[1] / crm.shape[3]],
                                device=peak_list.device)
        peak_list_upsample = peak_list.float()
        peak_list_upsample[:, 2:] += 0.5
        peak_list_upsample[:, 2:] *= upsample[None, :]
        peak_list = peak_list_upsample.round().int()

    out['peak_list'] = peak_list
    out['peak_values'] = peak_values
    return out


def minmaxpooling(crm, upsample_size, inference=True):
    """
    Wildcat pooling from the paper "WILDCAT: Weakly Supervised Learning of Deep ConvNets for Image
    Classification, Pointwise Localization and Segmentation"
    Arguments:
        crm: torch.tensor - Class Response Maps
        upsample_size: list/tuple - size of the original image
        inference: bool - Are we looking for peaks
    """
    maxpool = F.max_pool2d(crm, crm.shape[2:]).squeeze()
    minpool = F.max_pool2d(-crm, crm.shape[2:]).squeeze()
    if inference and crm.shape[-2:] != torch.Size(upsample_size):
        crm = interpolate(crm, upsample_size, mode='bilinear', align_corners=True)
    output = {'class_scores': maxpool - 0.6 * minpool,
              'crm': crm}
    return pooling_postprocess(output, inference)


def peakresponsepooling(crm, upsample_size, inference=True):
    """
    Peak Response from the paper "Weakly Supervised Instance Segmentation using Class Peak
    Response"
    Arguments:
        crm: torch.tensor - Class Response Maps
        upsample_size: list/tuple - size of the original image
        inference: bool - Are we looking for peaks. Not important for this function, but for other
                        poolings
    """
    if crm.shape[-2:] != torch.Size(upsample_size):
        crm = interpolate(crm, upsample_size, mode='bilinear', align_corners=True)
    out = find_peaks(crm, aggregation=True)
    out['crm'] = crm
    return pooling_postprocess(out, inference)


def pooling_postprocess(inputs, inference):
    """
    If peaks are not found yet, do it here. Filter out CRMs where no classes are found and peaks
    that are negative. Also normalize each map by their biggest peak.
    """
    if not inference:
        return inputs
    class_found = torch.sigmoid(inputs['class_scores']) > 0.5
    inputs['crm'][~class_found] = 0
    inputs['crm'][inputs['crm'] < 0] = 0
    inputs['crm'] /= torch.amax(inputs['crm'], dim=(2, 3), keepdim=True)
    inputs['crm'] = torch.nan_to_num(inputs['crm'])
    peaks = find_peaks(inputs['crm'], aggregation=False, threshold=0.5)
    inputs.update(peaks)
    return inputs


def locmap(num_features, num_classes):
    """
    2D-Convolutional Layer that acts as classifier and produces the class response maps.
    """
    classifier = nn.Sequential(
        nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
    return classifier


def fcresnet(model):
    """
    Fully Convolutional version of a ResNet
    """
    module = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4
    )
    return module


def resnet18(pretrained: bool = False, strides=(2, 2), progress: bool = True,
             **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, strides, progress, **kwargs)
    return fcresnet(net)


def resnet34(pretrained: bool = False, strides=(2, 2), progress: bool = True,
             **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, strides, progress, **kwargs)
    return fcresnet(net)


def resnet50(pretrained: bool = False, strides=(2, 2), progress: bool = True,
             **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, strides, progress, **kwargs)
    return fcresnet(net)


def resnet101(pretrained: bool = False, strides=(2, 2), progress: bool = True,
              **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, strides, progress, **kwargs)
    return fcresnet(net)


def fcvgg(model):
    """
    Fully Convolutional Version if the VGGNet.
    """
    return model.features


def vgg11(pretrained: bool = False, strides=(2, 2), progress: bool = True, **kwargs: Any):
    r"""VGG-11 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _vggnet("vgg11", False, pretrained, strides, progress, **kwargs)
    return fcvgg(net)


def vgg11_bn(pretrained: bool = False, strides=(2, 2), progress: bool = True, **kwargs: Any):
    r"""VGG-11BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _vggnet("vgg11", True, pretrained, strides, progress, **kwargs)
    return fcvgg(net)


def vgg16(pretrained: bool = False, strides=(2, 2), progress: bool = True, **kwargs: Any):
    r"""VGG-16 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _vggnet("vgg16", False, pretrained, strides, progress, **kwargs)
    return fcvgg(net)


def vgg16_bn(pretrained: bool = False, strides=(2, 2), progress: bool = True, **kwargs: Any):
    r"""VGG-16BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _vggnet("vgg16", True, pretrained, strides, progress, **kwargs)
    return fcvgg(net)


def fcalexnet(model):
    """
    Fully Convolutional Version of the AlexNet
    """
    return model.features


def alexnet(pretrained: bool = False, strides=(2, 2), progress: bool = True, **kwargs: Any):
    r"""AlexNet model architecture from `One weird trick for parallelizing convolutional neural networks <https://arxiv.org/abs/1404.5997>`__.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _alexnet(pretrained, strides, progress, **kwargs)
    return fcalexnet(net)

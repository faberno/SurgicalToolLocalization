from typing import Any
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.resnet import ResNet,_resnet, BasicBlock, Bottleneck


def minmax_pooling(input):
    maxpool = F.max_pool2d(input, input.shape[2:]).squeeze()
    minpool = F.max_pool2d(-input, input.shape[2:]).squeeze()
    return maxpool - 0.6 * minpool

def max_pooling(input):
    maxpool = F.max_pool2d(input, input.shape[2:]).squeeze()
    return maxpool


def locmap(num_features, num_classes):
    classifier = nn.Sequential(
        nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
    return classifier

def fcresnet(model):
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

def resnet18(pretrained: bool = False, strides = (2, 2), progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, strides, progress, **kwargs)
    return fcresnet(net)

def resnet34(pretrained: bool = False, strides = (2, 2), progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, strides, progress, **kwargs)
    return fcresnet(net)

def resnet50(pretrained: bool = False, strides = (2, 2), progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, strides, progress, **kwargs)
    return fcresnet(net)

def resnet101(pretrained: bool = False, strides = (2, 2), progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, strides, progress, **kwargs)
    return fcresnet(net)
from typing import Any
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.resnet import ResNet,_resnet, BasicBlock, Bottleneck
from models.backbones.vgg import _vggnet
from models.backbones.alexnet import _alexnet

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


def fcvgg(model):
    return model.features

def vgg11(pretrained: bool = False, strides = (2, 2), progress: bool = True, **kwargs: Any):
    r"""VGG-11 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _vggnet("vgg11", False, pretrained, strides, progress, **kwargs)
    return fcvgg(net)

def vgg11_bn(pretrained: bool = False, strides = (2, 2), progress: bool = True, **kwargs: Any):
    r"""VGG-11BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _vggnet("vgg11", True, pretrained, strides, progress, **kwargs)
    return fcvgg(net)

def vgg16(pretrained: bool = False, strides = (2, 2), progress: bool = True, **kwargs: Any):
    r"""VGG-16 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _vggnet("vgg16", False, pretrained, strides, progress, **kwargs)
    return fcvgg(net)

def vgg16_bn(pretrained: bool = False, strides = (2, 2), progress: bool = True, **kwargs: Any):
    r"""VGG-16BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _vggnet("vgg16", True, pretrained, strides, progress, **kwargs)
    return fcvgg(net)

def fcalexnet(model):
    return model.features

def alexnet(pretrained: bool = False, strides = (2, 2), progress: bool = True, **kwargs: Any):
    r"""AlexNet model architecture from `One weird trick for parallelizing convolutional neural networks <https://arxiv.org/abs/1404.5997>`__.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    net = _alexnet(pretrained, strides, progress, **kwargs)
    return fcalexnet(net)

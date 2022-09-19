# modified https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py

from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision._internally_replaced_utils import load_state_dict_from_url

__all__ = [
    "VGG",
    "VGG11_Weights",
    "VGG11_BN_Weights",
    "VGG16_Weights",
    "VGG16_BN_Weights"
]


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, strides=(2, 2)) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    length = len(cfg)
    for i,  v in enumerate(cfg):
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            stride = (1, 1)
            if i == (length - 2):
                stride = strides[0]
            if i == (length - 1):
                stride = strides[1]
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512],
}


# def _vgg(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> VGG:
#     if weights is not None:
#         kwargs["init_weights"] = False
#         if weights.meta["categories"] is not None:
#             _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
#     model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
#     if weights is not None:
#         model.load_state_dict(weights.get_state_dict(progress=progress))
#     return model

def _vggnet(cfg: str, batch_norm: bool, pretrained: bool, strides: tuple, progress: bool, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, strides=strides), **kwargs)
    if pretrained:
        if batch_norm:
            cfg += "_bn"
        state_dict = load_state_dict_from_url(model_urls[cfg], progress=progress)
        model.load_state_dict(state_dict)
    return model


_COMMON_META = {
    "min_size": (32, 32),
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
    "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
}




class VGG11_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg11-8a719046.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 132863336,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.020,
                    "acc@5": 88.628,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_V1




class VGG11_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 132868840,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 70.370,
                    "acc@5": 89.810,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_V1





class VGG16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg16-397923af.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 138357544,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.592,
                    "acc@5": 90.382,
                }
            },
        },
    )
    IMAGENET1K_FEATURES = Weights(
        # Weights ported from https://github.com/amdegroot/ssd.pytorch/
        url="https://download.pytorch.org/models/vgg16_features-amdegroot-88682ab5.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            mean=(0.48235, 0.45882, 0.40784),
            std=(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0),
        ),
        meta={
            **_COMMON_META,
            "num_params": 138357544,
            "categories": None,
            "recipe": "https://github.com/amdegroot/ssd.pytorch#training-ssd",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": float("nan"),
                    "acc@5": float("nan"),
                }
            },
            "_docs": """
                These weights can't be used for classification because they are missing values in the `classifier`
                module. Only the `features` module has valid values and can be used for feature extraction. The weights
                were trained using the original input standardization method as described in the paper.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1




class VGG16_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 138365992,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 73.360,
                    "acc@5": 91.516,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_V1

# The dictionary below is internal implementation detail and will be removed in v0.15
from torchvision.models._utils import _ModelURLs


model_urls = _ModelURLs(
    {
        "vgg11": VGG11_Weights.IMAGENET1K_V1.url,
        "vgg16": VGG16_Weights.IMAGENET1K_V1.url,
        "vgg11_bn": VGG11_BN_Weights.IMAGENET1K_V1.url,
        "vgg16_bn": VGG16_BN_Weights.IMAGENET1K_V1.url,
    }
)
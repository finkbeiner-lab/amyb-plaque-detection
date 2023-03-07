import os
import pdb
import sys
__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from collections import OrderedDict
from dataclasses import dataclass, field, asdict, is_dataclass, replace, InitVar
from functools import partial

import torch
from torch import nn, Tensor

import torchvision

import models
from models import rcnn_conf
from models.modules.retinanet import RetinaNet
from models.modules.retinanet_heads import RetinaNetHeads


def replace_keys(
    self: Any,
    field: str,
    keys: List[str],
) -> object:
    assert is_dataclass(self)
    assert field not in keys
    assert is_dataclass(self.__getattribute__(field))
    self.__setattr__(field, replace(self.__getattribute__(field), **dict([(k, self.__getattribute__(k)) for k in keys])))

def load_submodule_params(
    src_dict: Mapping[str, Tensor], # state_dict which contains weights to be copied
    dest_dict: Mapping[str, Tensor], # state_dict into which weights are copied
    submodules: List[str], # list of fully qualified submodule names which whose weights will be copied
) -> nn.Module:
    submodules = [submodule.split('.') for submodule in submodules]
    is_submodule_param = lambda param_name: (lambda names: len([i for i in range(len(names)) if names[:i + 1] in submodules]) > 0)(param_name.split('.'))
    return OrderedDict(list(dest_dict.items()) + [item for item in src_dict.items() if is_submodule_param(item[0])])


@dataclass
class anchor_conf(rcnn_conf.anchor_conf):
    scales: List[float] = field(default_factory=lambda: [2 ** (i / 3) for i in range(0, 2 + 1)])

# @dataclass
# class classification_head_conf:
#     in_channels: int
#     num_anchors: int
#     num_classes: int
#     prior_probability: float = 1e-2
#     norm_layer: Optional[Callable[..., nn.Module]] = None
#
#     def module(self) -> nn.Module:
#         return torchvision.models.detection.retinanet.RetinaNetClassificationHead(
#             self.in_channels,
#             self.num_anchors,
#             self.num_classes,
#             prior_probability=self.prior_probability,
#             norm_layer=self.norm_layer,
#         )
#
# @dataclass
# class regression_head_conf:
#     in_channels: int
#     num_anchors: int
#     norm_layer: Optional[Callable[..., nn.Module]] = None
#
#     def module(self) -> nn.Module:
#         return torchvision.models.detection.retinanet.RetinaNetRegressionHead(
#             in_channels,
#             num_anchors,
#             norm_layer=norm_layer,
#         )


@dataclass
class heads_conf:
    num_classes: int = 91
    num_channels: int = 256
    num_anchors: int = 9

    fg_iou_thresh: float = 0.5
    bg_iou_thresh: float = 0.4
    batch_size_per_image: int = 1000
    bbox_reg_weights: Tuple[float, float, float, float] = field(default_factory=lambda: (1., 1., 1., 1.,))
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    detections_per_image: int = 300
    prior_probability: float = 1e-2
    norm_layer: Optional[Callable[..., nn.Module]] = None

    def module(self) -> nn.Module:
        names = 'fg_iou_thresh bg_iou_thresh batch_size_per_image bbox_reg_weights score_thresh nms_thresh detections_per_image prior_probability norm_layer'
        kwargs = dict([(k, v) for k, v in asdict(self).items() if k in names])
        return RetinaNetHeads(
            self.num_channels,
            self.num_anchors,
            self.num_classes,
            **kwargs,
        )

@dataclass
class backbone_conf:
    return_layers: List[int] = field(default_factory=lambda: list(range(1, 4)))
    extra_blocks: nn.Module = field(default_factory=lambda: torchvision.ops.feature_pyramid_network.LastLevelP6P7(256, 256))

@dataclass
class retinanet_conf:
    num_classes: int = 91
    num_channels: int = 256
    pretrained: bool = False

    backbone: backbone_conf = field(default_factory=backbone_conf)
    anchor_generator: anchor_conf = field(default_factory=anchor_conf)
    heads: heads_conf = field(default_factory=heads_conf)
    transform: rcnn_conf.transform_conf = field(default_factory=rcnn_conf.transform_conf)

    weights: object = torchvision.models.detection.retinanet.RetinaNet_ResNet50_FPN_Weights

    def __post_init__(self):
        if self.pretrained:
            self.backbone.backbone_norm_layer = torchvision.ops.misc.FrozenBatchNorm2d

        keys = 'num_classes num_channels num_anchors'.split()
        replace_keys(self, 'backbone', keys[1:2])
        replace_keys(self, 'anchor_generator', keys[2:])
        replace_keys(self, 'heads', keys)

    def _module(self) -> nn.Module:
        return RetinaNet(
            self.backbone.module(),
            self.anchor_generator.module(),
            self.heads.module(),
            self.transform.module(),
        )

    def module(self, freeze_submodules=None, skip_submodules=None) -> nn.Module:
        model = self._module()
        if self.pretrained:
            state_dict = self.weights.get_state_dict(progress=True)
            if skip_submodules is not None:
                state_dict = load_submodule_params(model.state_dict(), state_dict, skip_submodules)
            model.load_state_dict(state_dict)
        if freeze_submodules is not None:
            for submodule in map(model.get_submodule, freeze_submodules):
                for param in submodule.parameters():
                    param.requires_grad_(False)
        return model

@dataclass
class retinanet_v2_conf:
    backbone: backbone_conf = field(default_factory=backbone_conf(extra_blocks=torchvision.ops.feature_pyramid_network.LastLevelP6P7(2048, 256)))
    heads: heads_conf = field(default_factory=heads_conf(
        norm_layer=partial(nn.GroupNorm, 32),
        loss_type='giou',
    ))

    def __post_init__(self):
        super().__post_init__()
        self.backbone.backbone_norm_layer = None

import os
import pdb
import sys
__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from collections import OrderedDict
from dataclasses import dataclass, field, asdict, is_dataclass, replace, InitVar

import torch
from torch import nn, Tensor

import torchvision

import models
from models import rcnn_conf
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
    in_channels: int
    num_anchors: int
    num_classes: int
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
        return RetinaNetHeads(
            self.in_channels,
            self.num_anchors,
            self.num_classes,
            fg_iou_thresh=self.fg_iou_thresh,
            bg_iou_thresh=self.bg_iou_thresh,
            batch_size_per_image=self.batch_size_per_image,
            bbox_reg_weights=self.bbox_reg_weights,
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_thresh,
            detections_per_image=self.detections_per_image,
            prior_probability=self.prior_probability,
            norm_layer=self.norm_layer,
        )

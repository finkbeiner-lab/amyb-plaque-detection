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


class RetinaNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        anchor_generator: nn.Module,
        heads: nn.Module,
        transform: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.anchor_generator = anchor_generator
        self.heads = heads
        self.transform = transform

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Mapping[str, Tensor]]] = None,
        sizes: Optional[Tuple[int, int]] = None,
    ) -> Union[Mapping[str, Tensor], List[Mapping[str, Tensor]]]:
        assert (not self.training) or (targets is not None), 'Targets  should not be "None" in training mode'

        original_sizes = [image.size()[-2:] for image in images]
        images, targets = self.transform(images, targets, sizes=sizes)

        features = self.backbone(images.tensors)
        anchors = self.anchor_generator(images, features)
        detections, detection_losses = self.heads(features, anchors, images.image_sizes, targets)

        if self.training:
            return detection_losses
        else:
            return self.transform.postprocess(detections, images.image_sizes, original_sizes)

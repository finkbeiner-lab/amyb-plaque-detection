from collections import OrderedDict
from typing import List, Mapping, Optional, Tuple

import torch
from torch import nn, Tensor

import torchvision
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import paste_masks_in_image
from torchvision.transforms import functional as F


class RCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        rpn: nn.Module,
        roi_heads: nn.Module,
        transform: nn.Module = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Mapping[str, Tensor]]] = None,
        sizes: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Mapping[str, Tensor], List[Mapping[str, Tensor]]]:
        assert (not self.training) or (targets is not None), 'Targets  should not be "None" in training mode'

        original_sizes = [image.size()[-2:] for image in images]
        images, targets = self.transform(images, targets, sizes=sizes)

        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detection_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        if self.training:
            return dict(**proposal_losses, **detection_losses)
        else:
            return self.transform.postprocess(detections, images.image_sizes, original_sizes)

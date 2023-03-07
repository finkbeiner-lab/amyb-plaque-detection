from collections import OrderedDict
from typing import List, Mapping, Optional, Tuple, Union
import pdb

import torch
from torch import nn, Tensor

import torchvision
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import paste_mask_in_image, paste_masks_in_image

"""
TODO:
  - Unit tests of RCNNTransform subroutines
  - Implement RetinaNet-compatible RCNN
"""


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
    ) -> Union[Mapping[str, Tensor], List[Mapping[str, Tensor]]]:
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



class RCNNTransform(nn.Module):
    def __init__(
        self,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        divisor: int = 32,
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.divisor = divisor

    def normalize(
        self,
        image: Tensor,
    ) -> Tensor:
        return (image - torch.tensor(self.mean).to(image)[..., None, None]) / torch.tensor(self.std).to(image)[..., None, None]

    def resize(
        self,
        image: Tensor,
        target: Optional[Mapping[str, Tensor]] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Optional[Mapping[str, Tensor]]]:
        if size is None:
            return image, target
        if target is not None:
            if 'boxes' in target.keys():
                target['boxes'] *= torch.tensor([*size[::-1]] * 2).to(target['boxes']) / torch.tensor([*tuple(image.size())[-2:][::-1]] * 2).to(target['boxes'])
                if 'masks' in target.keys():
                    target['masks'] = nn.functional.interpolate(target['masks'][:, None, ...].to(torch.uint8), size=size, mode='nearest')[:, 0, ...].to(torch.bool)
        return nn.functional.interpolate(image[None, ...], size=size, mode='bilinear')[0, ...], target

    def batch(
        self,
        images: List[Tensor],
    ) -> ImageList:
        sizes = [tuple(image.size())[-2:] for image in images]
        max_size = tuple([i + ((-i) % self.divisor) for i in map(max, zip(*sizes))])
        images = [nn.functional.pad(image, pad=padding, mode='constant', value=0) for (image, padding) in zip(images, [(0, max_size[1] - size[1], 0, max_size[0] - size[0]) for size in sizes])]
        return ImageList(torch.stack(images, dim=0), sizes)

    def postprocess(
        self,
        targets: List[Mapping[str, Tensor]],
        sizes: List[Tuple[int, int]],
        original_sizes: List[Tuple[int, int]],
    ) -> List[Mapping[str, Tensor]]:
        for i, (size, original_size) in enumerate(zip(sizes, original_sizes)):
            if 'boxes' in targets[i].keys():
                targets[i]['boxes'] *= torch.tensor([*original_size[::-1]] * 2).to(targets[i]['boxes']) / torch.tensor([*size[::-1]] * 2).to(targets[i]['boxes'])
                if 'masks' in targets[i].keys():
                    # masks = [paste_mask_in_image(mask[0], box, *original_size) for box, mask in zip(targets[i]['boxes'].to(torch.int), targets[i]['masks'])]
                    # targets[i]['masks'] = (torch.stack(masks, dim=0) if len(masks) > 0 else targets[i]['masks'].new_empty((0, *original_size)))[:, None]
                    targets[i]['masks'] = paste_masks_in_image(targets[i]['masks'], targets[i]['boxes'], original_size)
        return targets

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Mapping[str, Tensor]]] = None,
        sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> ImageList:
        images = [self.normalize(image) for image in images]
        images, targets = (lambda _: _ if targets is not None else ([*_][0], None))(map(list, zip(*[self.resize(image, target, size) for image, target, size in zip(images, [None] * len(images) if targets is None else targets, [None] * len(images) if sizes is None else sizes)])))
        return self.batch(images), targets

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std}, divisor={self.divisor})'

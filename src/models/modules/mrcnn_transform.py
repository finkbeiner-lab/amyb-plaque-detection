from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor

import torchvision
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import paste_mask_in_image


# TODO:
#   Consider moving the mask resizing to roi_heads? This would allow completely decoupling the model from the composed transforms


class RCNNTransform(nn.Module):
    def __init__(
        self,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        divisor: int = 32,
    ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.divisor = divisor


    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
        assert self.training == (targets is not None)
        images = [self.normalize(image) for image in images]
        images = self.batch_images(images)
        return images, targets
    

    # equivalent to torchvision.transforms.Normalize
    def normalize(
        self,
        image: Tensor,
    ) -> Tensor:
        assert len(image.size()) == 3 and image.size()[0] == len(self.mean) == len(self.std)
        mean = torch.tensor(self.mean).to(image)
        std = torch.tensor(self.std).to(image)
        return (image - mean[..., None, None]) / std[..., None, None]
    

    # equivalent to torchvision.transforms.Pad (here using torch.nn.functional.pad instead due to its consistent axis ordering)
    def batch_images(
        self,
        images: List[Tensor],
    ) -> Tensor:
        sizes = [tuple(image.size())[-2:] for image in images]
        max_size = tuple([i + ((-i) % self.divisor) for i in map(max, zip(*sizes))])
        paddings = [(0, max_size[1] - size[1], 0, max_size[0] - size[0]) for size in sizes]
        images = [torch.nn.functional.pad(image, pad=padding, mode='constant', value=0) for (image, padding) in zip(images, paddings)]
        return ImageList(torch.stack(images, dim=0), sizes)

    
    # expects image_sizes == original_image_sizes; keeping the latter to maintain compatibility with GeneralizedRCNN
    def postprocess(
        self,
        results: List[Dict[str, Tensor]],
        image_sizes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        if self.training:
            return results
        for i, (result, size, original_size) in enumerate(zip(results, image_sizes, original_image_sizes)):
            assert size == original_size
            if 'masks' in result.keys():
                masks = [paste_mask_in_image(mask[0], box, *size) for box, mask in zip(result['boxes'].to(torch.int), result['masks'])]
                results[i]['masks'] = (torch.stack(masks, dim=0) if len(masks) > 0 else result['masks'].new_empty((0, *size)))[:, None]
                
        return results

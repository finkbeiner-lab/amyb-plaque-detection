from typing import Dict, List, Tuple, Optional

import torch
from torch import nn, Tensor


class ImageList(object):
    def __init__(
        self,
        tensors: Tensor,
        image_sizes: List[Tuple[int, int]],
    ) -> None:
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(
        self,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        self.tensors = self.tensors.to(device=device, dtype=dtype)
        return self


class GeneralizedRCNNTransform(nn.Module):

    def __init__(
        self,
        image_size: Tuple[int, int]
    ) -> None:
        super().__init__()
        self.image_size = image_size

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
        # assert False not in [tuple(image.size())[:-2] == self.image_size for image in images]
        # assert targets is None or len(images) == len(targets)
        # tensors = torch.cat([image.unsqueeze(0) for image in images], dim=0)

        assert tuple(images.size())[-2:] == self.image_size
        tensors = images
        images = ImageList(tensors, [self.image_size for _ in images])
        return images, targets




class GeneralizedRCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        proposal: nn.Module,
        heads: nn.Module,
        transform: nn.Module,
    ):
        self.backbone = backbone
        self.proposal = proposal
        self.heads = heads
        self.transform = transform

    def forward(
        self,
        images,
        targets=None
    ):
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.proposal(images, features, targets)
        detections, detection_losses = self.heads(features, proposals, images.image_sizes, targets)
        losses = dict(proposal=proposal_losses, detection=detection_losses)
        return losses, detections
#
# n, h, w = 1, 512, 512
# images = torch.rand(torch.Size([1, 3, h, w]))

# inputs = dict([(str(i), torch.rand(torch.Size([1,256,1024 // (2 ** i),1024 // (2 ** i)]))) for i in range(5)])
#
# targets = [dict(
#     labels=torch.ones(torch.Size([n,])).to(dtype=torch.int64),
#     boxes=torch.cat([f(t, dim=1, keepdim=True).values for f in [torch.min, torch.max] for t in [torch.randint(0, x, torch.Size([n, 2])) for x in [h, w]]], dim=-1),
#     masks=torch.zeros(torch.Size([n, 1, h, w])),
# )]
# images = transform((h, w))(images, targets)
# # images = ImageList(torch.rand(torch.Size([1, 3, h, w])), [(h, w)])
#
# bh = torch.randint(0, h, torch.Size([n, 2]))
# bw = torch.randint(0, w, torch.Size([n, 2]))
# torch.cat([f(t, dim=1, keepdim=True).values for f in [torch.min, torch.max] for t in [torch.randint(0, x, torch.Size([n, 2])) for x in [h, w]]], dim=-1)

"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches



import pdb


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections
     
    # custom visualizaton function
    def visualize_feature_maps(self, features, layer_type='pool'):

        feature = features['pool']
        feature = feature[0]
        feature = feature.detach().cpu().numpy()
        feature = feature.transpose(1, 2, 0)
        plt.imshow(feature[:,:,0])
        # plt.show()
    
    
    def visualize_rpn_proposals(self, images, proposals):
        img = images.tensors[0]
        img = img.detach().cpu().numpy()
        img = img.transpose(1, 2, 0)
        fig, ax = plt.subplots(1, figsize=(10, 10))
        
        ax.imshow(img)
        proposal = proposals[0].detach().cpu().numpy()
        pdb.set_trace()

        for i, prop in enumerate(proposal):
            x = prop[0]
            y = prop[1]
            w = prop[2]
            h = prop[3]

            if i == 100:
                break

            p = patches.Rectangle((x, y), w, h, linewidth=2, facecolor='none',
                              edgecolor=(1.0, 0, 0))
            ax.add_patch(p)
        plt.show()



    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        #type hint
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # Image is passed through backbone model
        features = self.backbone(images.tensors)
        self.visualize_feature_maps(features)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # Features - odict_keys(['0', '1', '2', '3', 'pool'])
        # targets - dict_keys(['boxes', 'labels', 'masks', 'image_id', 'area'])
        # images - torch.Size([3, 3, 1024, 1024])
        # proposals - torch.Size([2000, 4])
        proposals, proposal_losses = self.rpn(images, features, targets)
        self.visualize_rpn_proposals(images, proposals)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

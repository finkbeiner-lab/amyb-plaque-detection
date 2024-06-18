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


# Color Coding
#id2label = {'1':'Core', '2':'Diffuse',
#             '3':'Neuritic', '4':'CAA','Unknown':'0'}

id2label = {"1":"Cored","2":"Diffuse",
             "3":"Coarse-Grained","4": "CAA"}


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
    def visualize_feature_maps(self, images, features, show=False):

        if show:
            img = images.tensors[0]
            img = img.detach().cpu().numpy()
            img = img.transpose(1, 2, 0)

            feature_img_list = [img]

            for key, val in features.items():
                feature = features[key]
                feature = feature[-1]
                feature = feature.detach().cpu().numpy()
                feature = feature.transpose(1, 2, 0)

                feature_img_list.append(feature[:,:,0])

            plt.figure(figsize=(10,10)) # specifying the overall grid size
            plt.suptitle('Feature maps at different scale')
            plt.subplot(1,6, 1)
            plt.title('Input Image')
            scale = ['', '1/2', '1/4', '1/8', '1/16', '1/32']
            for i in range(6):
                
                plt.subplot(1,6,i+1)    # the number of images in the grid is 5*5 (25)
                # title_name = 'Feature maps at {i}th scale}'
                if i == 0:
                    plt.imshow(feature_img_list[i])
                    continue
                else:
                    plt.title('Feature maps at {i} scale'.format(i=scale[i]))
                    plt.imshow(feature_img_list[i])
            save_name = "../../reports/figures/feature_maps.png"
            plt.savefig(save_name)
            plt.show()
    
    def visualize_roi_detections(self, images, detections, no_of_top_detections=20, show=False):
        if show:
            img = images.tensors[0]
            img = img.detach().cpu().numpy()
            img = img.transpose(1, 2, 0)

            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(img)

            for detection in detections:
                boxes = detection['boxes']
                labels = detection['labels']
                scores = detection['scores']
                masks = detection['masks']
                for i in range(no_of_top_detections):
                    box = boxes[i].detach().cpu().numpy()
                    label = labels[i].detach().cpu().numpy()
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]
                    p = patches.Rectangle((x, y), w, h, linewidth=2, facecolor='none', label=id2label[str(label)],
                                    edgecolor=(1.0, 0, 0))
                    ax.add_patch(p)
            save_name = "../../reports/figures/roi_detections.png"
            plt.savefig(save_name)
            plt.show()
  
    def visualize_rpn_proposals(self, images, proposals, show=False):
        img = images.tensors[0]
        img = img.detach().cpu().numpy()
        img = img.transpose(1, 2, 0)
       

        if show:
            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(img)
            proposal = proposals[0].detach().cpu().numpy()

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
            save_name = "../../reports/figures/rpn_proposals.png"
            plt.savefig(save_name)
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

        #TODO Why Another Transform Here?
        images, targets = self.transform(images, targets)
        

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    print(target_idx)
                    print(target["boxes"])
                    pdb.set_trace()
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # Image is passed through backbone model
        features = self.backbone(images.tensors)
        self.visualize_feature_maps(images, features, show=False)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # Features - odict_keys(['0', '1', '2', '3', 'pool'])
        # targets - dict_keys(['boxes', 'labels', 'masks', 'image_id', 'area'])
        # images - torch.Size([3, 3, 1024, 1024])
        # proposals - torch.Size([2000, 4])
        proposals, proposal_losses = self.rpn(images, features, targets)
        self.visualize_rpn_proposals(images, proposals, False)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if len(detections)!= 0:
            self.visualize_roi_detections(images, detections, 20,False)

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

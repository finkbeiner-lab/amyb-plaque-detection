import os
import pdb
import sys
__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from collections import OrderedDict
from dataclasses import dataclass, field, asdict, is_dataclass, replace, InitVar

import torch
from torch import nn, Tensor

import torchvision


"""
TODO:
  - option for batched/non-batched nms
  - option for loss type (i.e. giou loss)
  - option for iou type (i.e. giou)
"""


class RetinaNetHeads(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        fg_iou_thresh: float = 0.5,
        bg_iou_thresh: float = 0.4,
        batch_size_per_image: int = 1000,
        bbox_reg_weights: Tuple[float, float, float, float] = (1., 1., 1., 1.,),
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_image: int = 300,
        prior_probability: float = 1e-2,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        loss_type: str = 'l1',
        iou_type: str = None,
    ) -> None:
        super().__init__()

        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh

        self.batch_size_per_image = batch_size_per_image
        self.box_coder = torchvision.models.detection._utils.BoxCoder(weights=bbox_reg_weights)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_image = detections_per_image

        self.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
            in_channels,
            num_anchors,
            num_classes,
            prior_probability=prior_probability,
            norm_layer=norm_layer,
        )

        self.regression_head = torchvision.models.detection.retinanet.RetinaNetRegressionHead(
            in_channels,
            num_anchors,
            norm_layer=norm_layer,
        )
        self.regression_head._loss_type = loss_type
        self.iou_type = iou_type

    def select_training_samples(
        self,
        all_targets: List[Mapping[str, Tensor]],
        all_anchors: List[Tensor],
    ) -> List[Tensor]:
        all_matched_idxs = []
        for targets, anchors in zip(all_targets, all_anchors):
            if targets['boxes'].numel() > 0:
                # matrix = torchvision.ops.boxes.box_iou(targets['boxes'], anchors)
                matrix = dict(
                    iou=torchvision.ops.boxes.box_iou,
                    giou=torchvision.ops.boxes.generalized_box_iou,
                    ciou=torchvision.ops.boxes.complete_box_iou,
                    diou=torchvision.ops.boxes.distance_box_iou,
                )[self.iou_type](targets['boxes'], anchors)
                matched_vals, matched_idxs = matrix.max(dim=0)

                bg_idxs = matched_vals < self.bg_iou_thresh
                ignore_idxs = (matched_vals < self.fg_iou_thresh) & (self.bg_iou_thresh <= matched_vals)
                keep_idxs = torch.where(matrix == matrix.max(dim=1)[0][:, None])[1]

                matched_idxs[bg_idxs] = -1
                matched_idxs[ignore_idxs] = -2
                matched_idxs[keep_idxs] = matrix.max(dim=0)[1][keep_idxs]
            else:
                matched_idxs = torch.full(anchors.size()[0], -1, dtype=torch.long, device=anchors.device)

            all_matched_idxs.append(matched_idxs)
        return all_matched_idxs

    def postprocess_detections(
        self,
        all_head_outputs: Mapping[str, List[Tensor]],
        all_anchors: List[List[Tensor]],
        image_shapes: List[Tuple[int, int]],
    ) -> List[Mapping[str, Tensor]]:
        all_logits = all_head_outputs['cls_logits']
        all_regression = all_head_outputs['bbox_regression']
        num_images = len(image_shapes)

        detections = list()

        for idx in range(num_images):
            image_shape = image_shapes[idx]
            image_scores, image_labels, image_boxes = list(), list(), list()
            for logits, regression, anchors in zip([l[idx] for l in all_logits], [r[idx] for r in all_regression], all_anchors[idx]):
                num_classes = logits.size()[-1]

                scores = torch.sigmoid(logits).flatten()
                keep_idxs = torch.where(scores > self.score_thresh)[0]

                scores, idxs = scores[keep_idxs].topk(min(keep_idxs.size()[0], self.batch_size_per_image))
                keep_idxs = keep_idxs[idxs]

                labels = keep_idxs % num_classes

                anchor_idxs = torch.div(keep_idxs, num_classes, rounding_mode='floor')
                boxes = self.box_coder.decode_single(regression[anchor_idxs], anchors[anchor_idxs])
                boxes = torchvision.ops.boxes.clip_boxes_to_image(boxes, image_shape)

                image_scores.append(scores)
                image_labels.append(labels)
                image_boxes.append(boxes)

            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_boxes = torch.cat(image_boxes, dim=0)

            keep_idxs = torchvision.ops.boxes.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)[:self.detections_per_image]

            detections.append(dict(
                scores=image_scores[keep_idxs],
                labels=image_labels[keep_idxs],
                boxes=image_boxes[keep_idxs],
            ))

        return detections

    def forward(
        self,
        features: List[Tensor],
        anchors: List[Tensor],
        image_sizes: List[Tuple[int, int]],
        targets: List[Mapping[str, Tensor]] = None,
    ) -> Tuple[List[Mapping[str, Tensor]], Mapping[str, Tensor]]:
        head_outputs = dict(
            cls_logits=self.classification_head.forward(features),
            bbox_regression=self.regression_head.forward(features),
        )

        detections = list()
        losses = dict()

        if self.training:
            matched_idxs = self.select_training_samples(targets, anchors)
            losses.update(
                classification=self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
                bbox_regression=self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
            )
        else:
            num_anchors_per_level = [feature.size(2) * feature.size(3) for feature in features]
            num_anchors_per_location = head_outputs['cls_logits'].size(1) // sum(num_anchors_per_level)
            num_anchors_per_level = [num * num_anchors_per_location for num in num_anchors_per_level]

            head_outputs_per_level = dict()
            for k in head_outputs:
                head_outputs_per_level[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            anchors_per_level = [list(a.split(num_anchors_per_level)) for a in anchors]

            detections.extend(self.postprocess_detections(head_outputs_per_level, anchors_per_level, image_sizes))

        return detections, losses

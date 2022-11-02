from typing import List, Tuple

import torch
from torch import nn, Tensor

import torchvision
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, maskrcnn_loss, maskrcnn_inference


class RCNNHeads(RoIHeads):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fg_iou_thresh = self.proposal_matcher.high_threshold
        self.bg_iou_thresh = self.proposal_matcher.low_threshold

        self.batch_size_per_image = self.fg_bg_sampler.batch_size_per_image
        self.positive_fraction = self.fg_bg_sampler.positive_fraction
        self.bbox_reg_weights = self.box_coder.weights
    
    # def __init__(
    #     self,
    #     box_roi_pool,
    #     box_head,
    #     box_predictor,
    #     fg_iou_thresh,
    #     bg_iou_thresh,
    #     batch_size_per_image,
    #     positive_fraction,
    #     bbox_reg_weights,
    #     score_thresh,
    #     nms_thresh,
    #     detections_per_img,
    #     mask_roi_pool=None,
    #     mask_head=None,
    #     mask_predictor=None,
    # ) -> None:
    #     super(RoIHeads, self).__init__()

    #     self.fg_iou_thresh = fg_iou_thresh
    #     self.bg_iou_thresh = bg_iou_thresh

    #     self.batch_size_per_image = batch_size_per_image
    #     self.positive_fraction = positive_fraction
    #     self.bbox_reg_weights = bbox_reg_weights
    #     self.box_coder = torchvision.models.detection._utils.BoxCoder(self.bbox_reg_weights)

    #     self.score_thresh = score_thresh
    #     self.nms_thresh = nms_thresh
    #     self.detections_per_img = detections_per_img

    #     self.box_roi_pool = box_roi_pool
    #     self.box_head = box_head
    #     self.box_predictor = box_predictor

    #     self.mask_roi_pool = mask_roi_pool
    #     self.mask_head = mask_head
    #     self.mask_predictor = mask_predictor

    #     self._has_mask = None not in [self.mask_roi_pool, self.mask_head, self.mask_predictor]
    
    def postprocess_detections(
        self,
        class_logits: Tensor,
        box_regression: Tensor,
        proposals: List[Tensor],
        image_sizes: List[Tuple[int, int]],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        # store class count and per-image proposal count
        num_classes = class_logits.size()[-1]
        num_proposals = [proposal.size()[0] for proposal in proposals]

        # decode boxes and reshape boxes and scores according to proposal count
        pred_boxes = self.box_coder.decode(box_regression, proposals).split(num_proposals, dim=0)
        pred_scores = nn.functional.softmax(class_logits, -1).split(num_proposals, dim=0)

        all_boxes, all_scores, all_labels = list(), list(), list()
        for boxes, scores, image_size in zip(pred_boxes, pred_scores, image_sizes):
            # set up boxes and labels
            boxes = torchvision.ops.boxes.clip_boxes_to_image(boxes, image_size)
            labels = torch.arange(num_classes, device=scores.device).view(1, -1).expand_as(scores)

            # clip off background class and reshape instance-wise
            boxes = boxes[:, 1:].reshape(-1, 4)
            scores = scores[:, 1:].reshape(-1)
            labels = labels[:, 1:].reshape(-1)

            # clip off boxes with score below score threshold
            keep = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # clip off boxes below area threshold
            keep = torchvision.ops.boxes.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # clip off non-maximum boxes with overlap above nms threshold
            keep = torchvision.ops.boxes.batched_nms(boxes, scores, labels, self.nms_thresh)[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        return all_boxes, all_scores, all_labels
    

    def select_training_samples(
        self,
        original_proposals: List[Tensor],
        original_targets: Optional[List[Dict[str, Tensor]]],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        # maximum number of positive/negative proposals
        max_pos = int(self.batch_size_per_image * self.positive_fraction)
        max_neg = self.batch_size_per_image - max_pos

        all_proposals, all_matched_idxs, all_proposal_labels, all_proposal_targets = list(), list(), list(), list()
        for proposals, targets in zip(original_proposals, original_targets):
            boxes, labels = targets['boxes'], targets['labels']

            # include ground truth boxes in case proposals are poor
            proposals = torch.cat((proposals, boxes))

            # set zero tensors by default unless ground truth boxes provided
            matched_idxs = torch.zeros((proposals.size()[0],), device=proposals.device, dtype=torch.long)
            proposal_labels = torch.zeros((proposals.size()[0],), device=proposals.device, dtype=torch.long)
            proposal_boxes = torch.zeros((proposals.size()[0], 4,), device=proposals.device)
            
            if boxes.numel() > 0:
                # matched_idxs maps proposals to the indices of their ground truth match (maximizing IOU)
                matched_vals, matched_idxs = torchvision.ops.boxes.box_iou(boxes, proposals).max(dim=0)
                # matched_vals, matched_idxs = torchvision.ops.boxes.generalized_box_iou(boxes, proposals).max(dim=0)
                bg_idxs = matched_vals < self.bg_iou_thresh
                ignore_idxs = (matched_vals < self.fg_iou_thresh) & (self.bg_iou_thresh <= matched_vals)

                # collect labels and boxes per proposal
                proposal_labels = labels[matched_idxs]
                proposal_boxes = boxes[matched_idxs]
                
                # mark background and ignored proposals
                proposal_labels[bg_idxs] = 0
                proposal_labels[ignore_idxs] = -1
            
            # select (at random) a balanced sample of proposal indices
            pos_idxs = torch.where(proposal_labels >= 1)[0]
            neg_idxs = torch.where(proposal_labels == 0)[0]
            pos_idxs = pos_idxs[torch.randperm(pos_idxs.numel(), device=pos_idxs.device)[:max_pos]]
            neg_idxs = neg_idxs[torch.randperm(neg_idxs.numel(), device=neg_idxs.device)[:max_neg]]
            
            sampled_idxs_mask = torch.zeros_like(proposal_labels, dtype=torch.uint8)
            sampled_idxs_mask[pos_idxs] = 1
            sampled_idxs_mask[neg_idxs] = 1
            sampled_idxs = torch.where(sampled_idxs_mask)[0]

            # sample proposals, ground truth indices, labels, and boxes
            proposals = proposals[sampled_idxs]
            matched_idxs = matched_idxs[sampled_idxs]
            proposal_labels = proposal_labels[sampled_idxs]
            proposal_boxes = proposal_boxes[sampled_idxs]

            # get regression targets of ground truth boxes with respect to proposals
            proposal_targets = self.box_coder.encode_single(proposal_boxes, proposals)

            all_proposals.append(proposals)
            all_matched_idxs.append(matched_idxs)
            all_proposal_labels.append(proposal_labels)
            all_proposal_targets.append(proposal_targets)
        
        return all_proposals, all_matched_idxs, all_proposal_labels, all_proposal_targets
    


    def _forward(
        self,
        features,
        proposals,
        image_sizes
    ) -> Tensor:
        features = self.box_roi_pool(features, proposals, image_sizes)
        features = self.box_head(features)
        class_logits, box_regression = self.box_predictor(features)
        return class_logits, box_regression
    
    def _forward_mask(
        self,
        features,
        proposals,
        image_sizes,
    ) -> Tensor:
        features = self.mask_roi_pool(features, proposals, image_sizes)
        features = self.mask_head(features)
        mask_logits = self.mask_predictor(features)
        return mask_logits

    # def forward(
    #     self,
    #     features: Dict[str, Tensor],
    #     proposals: List[Tensor],
    #     image_sizes: List[Tuple[int, int]],
    #     targets: Optional[List[Dict[str, Tensor]]] = None,
    # ) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]:
    #     result = dict()
    #     losses = dict()

    #     if self.training:
    #         proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
    #         class_logits, box_regression = self._forward(features, proposals, image_sizes)
            
    #         loss_classifier, loss_box_reg = _fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    #         losses.update(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
    #     else:
    #         class_logits, box_regression = self._forward(features, proposals, image_sizes)
            
    #         boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_sizes)
    #         result.update(boxes=boxes, scores=scores, labels=labels)
        
    #     if self._has_mask:
    #         if self.training:
    #             pos_idxs = [torch.where(l > 0)[0] for l in labels]
    #             mask_proposals = [p[idxs] for p, idxs in zip(proposals, pos_idxs)]
    #             pos_matched_idxs = [m[idxs] for m, idxs in zip(matched_idxs, pos_idxs)]
    #             mask_logits = self._forward_mask(features, mask_proposals, image_sizes)

    #             gt_masks = [target['masks'] for target in targets]
    #             gt_labels = [target['labels'] for target in targets]
    #             loss_mask = _maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
    #             losses.update(loss_mask=loss_mask)
    #         else:
    #             mask_proposals = result['boxes']
    #             mask_logits = self._forward_mask(features, mask_proposals, image_sizes)

    #             masks = _maskrcnn_inference(mask_logits, result['labels'])
    #             result.update(masks=masks)
        
    #     if not self.training:
    #         keys, values = tuple(zip(*result.items()))
    #         result = [dict(zip(keys, item)) for item in zip(*values)]

    #     return result, losses

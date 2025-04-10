"""
Customized mask RCNN function for pytorch ligtning which is get called in train_pl.py
"""

import lightning as L
import numpy as np
from torchvision import transforms

from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
#from helper_functions import evaluate_metrics, get_outputs


class LitMaskRCNN(L.LightningModule):
    def __init__(self, optim_config,backbone,rpn,roi_heads,transform):
        super().__init__()
        #self.model_config = _default_mrcnn_config(num_classes=1 + train_config['num_classes']).config
        #self.model = model
        self.optim_config = optim_config
        #self.model = build_default(model_config, im_size=1024)
        self.loss_names = 'objectness rpn_box_reg classifier box_reg mask'.split()
        self.loss_weights = [1., 4., 1., 4., 1.,]
        self.loss_weights = OrderedDict([(f'loss_{name}', weight) for name, weight in zip(self.loss_names, self.loss_weights)])
        self.backbone=backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform
        # used only on torchscript mode
        self._has_warned = False
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.avg_segmentation_overlap = 0.0
        self.val_acc = 0.0

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

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
        #self.visualize_feature_maps(images, features, show=False)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        proposals, proposal_losses = self.rpn(images, features, targets)
        #self.visualize_rpn_proposals(images, proposals, False)
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
        
       
    def get_loss_fn(self, weights, default=0.):
        def compute_loss_fn(losses):
            item = lambda k: (k, losses[k].item())
            metrics = OrderedDict(list(map(item, [k for k in weights.keys() if k in losses.keys()] + [k for k in losses.keys() if k not in weights.keys()])))
            loss = sum(map(lambda k: losses[k] * (weights[k] if weights is not None and k in weights.keys() else default), losses.keys()))
            return loss, metrics
        return compute_loss_fn
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        opt = self.optimizers()
        images, targets = batch 
        opt.zero_grad()
        loss_fn = self.get_loss_fn(self.loss_weights)
        loss, metrics = loss_fn(self.forward(images, targets))
        self.manual_backward(loss)
        opt.step()
        print_logs = "batch no : {batch_no}, total loss : {loss},  classifier :{classifier}, mask: {mask} ==================="
        print(print_logs.format( batch_no=batch_idx, loss=loss.item(),  classifier=metrics['loss_classifier'], mask=metrics['loss_mask']))
        self.log("loss", loss.item())
        self.log("metrics-loss_classifier", metrics['loss_classifier'])
        self.log("metrics-loss_mask", metrics['loss_mask'])
      

    def configure_optimizers(self):
        # set optimizer based on config
        optimizer = self.optim_config['cls']([dict(params=list(self.parameters()))], **self.optim_config['defaults'])
        return optimizer

    
    def get_outputs(self, outputs, threshold):
        """
        Extracts masks, labels, and bounding boxes from model outputs that have a score above a given threshold.

        Args:
            outputs (list): List of model outputs for each image, containing 'scores', 'masks', 'boxes', and 'labels'.
            threshold (float): The threshold for filtering predictions based on score.

        Returns:
            mask_list (list): List of masks for the predictions above the threshold.
            label_list (list): List of labels for the predictions above the threshold.
        """
        mask_list = []
        label_list = []
        class_names = ['Cored', 'Diffuse', 'Coarse-Grained', 'CAA']
        for j in range(len(outputs)):
            scores = outputs[j]['scores'].tolist()
            thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
            scores = [scores[x] for x in thresholded_preds_inidices]
            # get the masks
            masks = (outputs[j]['masks']>0.5).squeeze()
            # discard masks for objects which are below threshold
            masks = [masks[x] for x in thresholded_preds_inidices]
            # get the bounding boxes, in (x1, y1), (x2, y2) format
            boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[j]['boxes'].tolist()]
            # discard bounding boxes below threshold value
            boxes = [boxes[x] for x in thresholded_preds_inidices]
            # get the classes labels
            labels = outputs[j]['labels'].tolist()
            labels = [labels[x] for x in thresholded_preds_inidices]
            mask_list.append(masks)
            label_list.append(labels)
        return mask_list, label_list

    def match_label(self, pred_label, gt_label):
        """
        Compares predicted label with ground truth label.

        Args:
            pred_label (int): The predicted label.
            gt_label (int): The ground truth label.

        Returns:
            int: 1 if the labels match, else 0.
        """
        if pred_label==gt_label:
            return 1
        else:
            return 0
    
    def actual_label_target(self, gt_label):
        """
        Converts the ground truth label to a NumPy array.

        Args:
            gt_label (Tensor): The ground truth label as a tensor.

        Returns:
            np.ndarray: The ground truth label as a NumPy array.
        """
        return gt_label
    
    
    def compute_iou(self, mask1, mask2):
        """
        Computes Intersection over Union (IoU) for two binary masks.

        Args:
            mask1 (Tensor): The first binary mask.
            mask2 (Tensor): The second binary mask.

        Returns:
            float: The IoU value, ranging from 0 to 1.
        """
        intersection = torch.logical_and(mask1, mask2).sum().item()
        union = torch.logical_or(mask1, mask2).sum().item()
        iou = (2*intersection) / union if union != 0 else 0
        return iou
    
    
    def evaluate_metrics(self, target,masks, labels):
        """
        Evaluates the performance of a model by computing the mean F1 score and matched labels across all targets.

        Args:
            target (list): List of ground truth annotations containing labels and masks.
            masks (list): List of predicted masks.
            labels (list): List of predicted labels.

        Returns:
            tuple: The mean F1 score and mean matched labels across all predictions.
        """
        f1_score_list=[]
        matched_label_list=[]
        mean_f1_score = -1
        mean_matched_label=-1
        actual_label_list = []
        pred_label_list = []
        for i in range(len(target)):
            target_labels = self.actual_label_target(target[i]['labels'])
            for l in range(len(target_labels)):
                for j in range(len(masks)):
                    for k in range(len(masks[j])):
                        target_mask = target[i]['masks'][l]
                        target_mask = torch.where(target_mask > 0, torch.tensor(1), torch.tensor(0))
                        if target_mask.shape==masks[j][k].shape:
                            f1_score = self.compute_iou(masks[j][k],target_mask)
                            if f1_score>0:
                                f1_score_list.append(f1_score)
                                matched_label = self.match_label(labels[j][k],target_labels[l])
                                matched_label_list.append(matched_label)
                            #else:
                            #    matched_label_list.append(0)
            if len(f1_score_list)>0:
                mean_f1_score=np.nansum(f1_score_list)/len(f1_score_list)
            if len(matched_label_list)>0:
                mean_matched_label = sum(matched_label_list)/len(matched_label_list)
            #print(f1_score_list, matched_label_list)
            return mean_f1_score, mean_matched_label

    
    def validation_step(self, batch, batch_idx):
        """
        this is the validation loop on validation batch, compute metrics and log
        """
        images, targets = batch 
        outputs = self.forward(images, targets)
        masks, labels = self.get_outputs(outputs, 0.50)
        f1_mean, labels_matched =  self.evaluate_metrics(targets, masks, labels)
        self.avg_segmentation_overlap = f1_mean
        self.val_acc = labels_matched
        if (f1_mean>=0) or (labels_matched>=0):
            print(" Validation f1 mean score:", f1_mean, " perc labels matched", labels_matched)
        self.log('avg_seg_overlap',f1_mean)
        self.log('val_acc', labels_matched)
        return f1_mean, labels_matched
        
    
    def test_step(self, batch, batch_idx):
        # this is the test loop, compute test metrics
        images, targets = batch 
        outputs = self.forward(images)
        masks, labels = self.get_outputs(outputs, 0.50)
        f1_mean, labels_matched =  self.evaluate_metrics(targets, masks, labels)
        return f1_mean, labels_matched
    
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
from helper_functions import evaluate_metrics, get_outputs




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
        
        #self.run = run

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return losses, detections

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
                    #pdb.set_trace()
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

        # Features - odict_keys(['0', '1', '2', '3', 'pool'])
        # targets - dict_keys(['boxes', 'labels', 'masks', 'image_id', 'area'])
        # images - torch.Size([3, 3, 1024, 1024])
        # proposals - torch.Size([2000, 4])
        proposals, proposal_losses = self.rpn(images, features, targets)
        #self.visualize_rpn_proposals(images, proposals, False)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        #if len(detections)!= 0:
            #self.visualize_roi_detections(images, detections, 20,False)

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
        #print(batch)
        opt = self.optimizers()
        images, targets = batch 
        #images = [image for image in images]
        #targets = [dict([(k, v) for k, v in target.items()]) for target in targets]
        opt.zero_grad()
        loss_fn = self.get_loss_fn(self.loss_weights)
        loss, metrics = loss_fn(self.forward(images, targets))
        
        #loss.backward()
        self.manual_backward(loss)
        opt.step()
        #log_metrics.append(dict(epoch=epoch, loss=loss.item(), metrics=metrics))
        print_logs = "batch no : {batch_no}, total loss : {loss},  classifier :{classifier}, mask: {mask} ==================="
        print(print_logs.format( batch_no=batch_idx, loss=loss.item(),  classifier=metrics['loss_classifier'], mask=metrics['loss_mask']))
        self.log("loss", loss.item(),logger=True)
        self.log("metrics-loss_classifier", metrics['loss_classifier'],logger=True)
        self.log("metrics-loss_mask", metrics['loss_mask'],logger=True)
      
        #yield log_metrics
    
    #def backward(self, loss):
    #    loss.backward()
    
    def configure_optimizers(self):
        optimizer = self.optim_config['cls']([dict(params=list(self.parameters()))], **self.optim_config['defaults'])
        return optimizer

    
    def get_outputs(self, outputs, threshold):
        mask_list = []
        label_list = []
        class_names = ['Cored', 'Diffuse', 'Coarse-Grained', 'CAA']
        for j in range(len(outputs)):
            scores = outputs[j]['scores'].tolist()
            thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
            scores = [scores[x] for x in thresholded_preds_inidices]
            # get the masks
            masks = (outputs[j]['masks']>0.5).squeeze()
            # print("masks", masks)
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
        if pred_label==gt_label:
            return 1
        else:
            return 0
    
    def actual_label_target(self, gt_label):
        return gt_label
    
    
    def compute_iou(self, mask1, mask2):
        intersection = torch.logical_and(mask1, mask2).sum().item()
        union = torch.logical_or(mask1, mask2).sum().item()
        iou = (2*intersection) / union if union != 0 else 0
        return iou
    
    
    def evaluate_metrics(self, target,masks, labels, iou_threshold):
        f1_score_list=[]
        matched_label_list=[]
        mean_f1_score = 0
        mean_matched_label=0
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
                            if f1_score>iou_threshold:
                                f1_score_list.append(f1_score)
                                matched_label = self.match_label(labels[j][k],target_labels[l])
                                matched_label_list.append(matched_label)
                            #else:
                            #    matched_label_list.append(0)
            if len(f1_score_list)>0:
                mean_f1_score=sum(f1_score_list)/len(f1_score_list)
            if len(matched_label_list)>0:
                mean_matched_label = sum(matched_label_list)/len(matched_label_list)
            #print(f1_score_list, matched_label_list)
            return mean_f1_score, mean_matched_label

    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        images, targets = batch 
        #images = [image for image in batch[0]]
        #targets = [dict([(k, v) for k, v in target.items()]) for target in batch[1]]
        #loss_fn = self.get_loss_fn(self.loss_weights)
        val_loss_fn = self.get_loss_fn(self.loss_weights)
        
        losses, outputs = self.forward(images, targets)
        print(losses)
        val_loss, val_metrics = val_loss_fn(losses)
        #print(val_loss, val_metrics)
        masks, labels = self.get_outputs(outputs, 0.5)
        f1_mean, labels_matched =  self.evaluate_metrics(targets, masks, labels, 0.5)
        self.avg_segmentation_overlap = f1_mean
        self.val_acc = labels_matched
        if (f1_mean>=0) or (labels_matched>=0):
            print(" Validation f1 mean score:", f1_mean, " perc labels matched", labels_matched)
        self.log('avg_seg_overlap',f1_mean,logger=True)
        self.log('val_acc', labels_matched,logger=True)
        self.log("val_loss", val_loss.item(),logger=True)
        self.log("val_metrics-loss_classifier", val_metrics['loss_classifier'],logger=True)
        self.log("val_metrics-loss_mask", val_metrics['loss_mask'],logger=True)

        
        return f1_mean, labels_matched
        #outputs1 = [x for x in outputs if len(x["labels"])!=0]
        #if len(outputs1)>0:
        #    print(outputs1[0].keys())
        #loss, metrics = loss_fn(outputs1[0])
        #loss, metrics = loss_fn(self.forward(images, targets))
        #print_logs = "batch no : {batch_no}, total loss : {loss},  classifier :{classifier}, mask: {mask} ==================="
        #print(print_logs.format( batch_no=batch_idx, loss=loss.item(),  classifier=metrics['loss_classifier'], mask=metrics['loss_mask']))
        
    
    def test_step(self, batch, batch_idx):
        # this is the validation loop
        images, targets = batch 
        #images = [image for image in batch[0]]
        #targets = [dict([(k, v) for k, v in target.items()]) for target in batch[1]]
        #loss_fn = self.get_loss_fn(self.loss_weights)
        #loss, metrics = loss_fn(self.forward(images, targets))
        outputs = self.forward(images)
        masks, labels = self.get_outputs(outputs, 0.25)
        f1_mean, labels_matched =  self.evaluate_metrics(targets, masks, labels)
        return f1_mean, labels_matched
    
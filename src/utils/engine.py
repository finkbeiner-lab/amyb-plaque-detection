import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
from utils import utils
from utils.coco_eval import CocoEvaluator
from utils.coco_utils import get_coco_api_from_dataset
import pdb
import matplotlib.pyplot as plt
import numpy as np
import wandb
from visualization.explain import ExplainPredictions


def train_one_epoch(model, optimizer, data_loader, device, epoch, wandb, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    return metric_logger


def _get_iou_types(model):
    iou_types = ["bbox", "segm"]
    return iou_types


@torch.inference_mode()
def evaluate(run, model, data_loader, device, epoch):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)

    #TODO: HardCode Exp name # later replace by run.name
    exp_name = "runtest"
    coco_evaluator = CocoEvaluator(coco, iou_types, epoch, exp_name)
    explain = ExplainPredictions(model, model_input_path = "", test_input_path="", detection_threshold=0.75, 
                                wandb=wandb, save_result=True, ablation_cam=True, save_thresholds=False)

    for images, targets in metric_logger.log_every(data_loader, 100, header):

       
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        for i in range(len(images)):
            log_results = []

            img = images[i].detach().cpu().numpy()
            img = img.transpose(1, 2, 0)
        
            masks, boxes, labels, scores = explain.get_outputs(images, model, 0.75)
            result_img, result_masks = explain.draw_segmentation_map(img, masks, boxes, labels)

            log_results.append(result_img)
            log_results.append(result_masks)
          
            # run.log({"Evaluation": [wandb.Image(image) for image in log_results]})


        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        # print(len(outputs))
        # print("Outputs", outputs[0]['labels'])
        
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    # torch.set_num_threads(1)
    coco_evaluator.accumulate()
    coco_evaluator.summarize(run)
    torch.set_num_threads(n_threads)
    return coco_evaluator

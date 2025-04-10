
import numpy as np
from torchvision import transforms
from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union
from torchmetrics.classification import MulticlassConfusionMatrix
import torchvision.ops.boxes as bops
from torchmetrics.classification import Dice
import pandas as pd


class_names = ['Cored', 'Diffuse', 'Coarse-Grained', 'CAA']
def get_outputs(outputs, threshold):
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
    for j in range(len(outputs)):
        scores = list(outputs[j]['scores'].detach().cpu().numpy())
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
        scores = [scores[x] for x in thresholded_preds_inidices]
        masks = (outputs[j]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        masks = [masks[x] for x in thresholded_preds_inidices]
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in outputs[j]['boxes'].detach().cpu()]
        boxes = [boxes[x] for x in thresholded_preds_inidices]
        labels = list(outputs[j]['labels'].detach().cpu().numpy())
        labels = [labels[x] for x in thresholded_preds_inidices]
        mask_list.append(masks)
        label_list.append(labels)
    return mask_list, label_list



def match_mask(masked_image, binary_array):
    """
    Calculates evaluation metrics (precision, recall, F1-score, IoU) for the predicted and ground truth masks.

    Args:
        masked_image (list): List of predicted binary masks.
        binary_array (list): List of ground truth binary masks.

    Returns:
        float: F1-score for the False class based on confusion matrix.
    """
    device = torch.device('cuda', 0)
    num_classes = 2
    metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    conf_final = torch.tensor(np.zeros((num_classes, num_classes))).to(device).to(torch.int64)
    for i in range(len(masked_image)):
        preds = torch.tensor(masked_image[i]).to(device).to(torch.int64)
        target = torch.tensor(binary_array[i]).to(device).to(torch.int64)
        a1 = metric(preds, target)
        conf_final = conf_final + a1
    conf_final_np = conf_final.cpu().numpy()
    total_predicted = np.sum(conf_final_np, axis=0)
    diag_elements = np.diag(conf_final_np)
    precision = diag_elements / total_predicted
    total_actual = np.sum(conf_final_np, axis=1)
    recall = diag_elements / total_actual
    f1_score = (2 * precision * recall) / (recall + precision)
    iou_coeff = diag_elements / (total_predicted + total_actual - diag_elements)
    eval_metrics = pd.DataFrame({"Class": ["True", "False"], "Precision": precision, "Recall": recall, "F1_score": f1_score, "IoU": iou_coeff})
    return eval_metrics[eval_metrics["Class"] == "False"]["f1_score"].values[0]



def match_label(pred_label, gt_label):
    """
    Compares predicted label with ground truth label.

    Args:
        pred_label (int): The predicted label.
        gt_label (int): The ground truth label.

    Returns:
        int: 1 if the labels match, else 0.
    """
    if pred_label == gt_label:
        return 1
    else:
        return 0
    
    
def actual_label_target(gt_label):
    """
    Converts the ground truth label to a NumPy array.

    Args:
        gt_label (Tensor): The ground truth label as a tensor.

    Returns:
        np.ndarray: The ground truth label as a NumPy array.
    """
    return gt_label.cpu().numpy()



def evaluate_metrics(target, masks, labels):
    """
    Evaluates the performance of a model by computing the mean F1 score and matched labels across all targets.

    Args:
        target (list): List of ground truth annotations containing labels and masks.
        masks (list): List of predicted masks.
        labels (list): List of predicted labels.

    Returns:
        tuple: The mean F1 score and mean matched labels across all predictions.
    """
    f1_score_list = []
    matched_label_list = []
    mean_f1_score = -1
    mean_matched_label = -1
    for i in range(len(target)):
        target_label = actual_label_target(target[i]['labels'])
        for l in range(len(target_label)):
            for j in range(len(masks)):
                for k in range(len(masks[j])):
                    target_mask = target[i]['masks'][l].cpu().numpy()
                    target_mask = np.where(target_mask > 0, 1, 0)
                    if target_mask.shape == masks[j][k].shape:
                        f1_score = match_mask(masks[j][k], target_mask)
                        f1_score_list.append(f1_score)
                        if f1_score > 0:
                            matched_label = match_label(labels[j][k], target_label[l])
                            matched_label_list.append(matched_label)
                        else:
                            matched_label_list.append(0)
        if len(f1_score_list) > 0:
            mean_f1_score = np.nansum(f1_score_list) / len(f1_score_list)
        if len(matched_label_list) > 0:
            mean_matched_label = sum(matched_label_list) / len(matched_label_list)
    return mean_f1_score, mean_matched_label


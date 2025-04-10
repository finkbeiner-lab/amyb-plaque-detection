
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
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class_names = ['Cored', 'Diffuse', 'Coarse-Grained', 'CAA']

def get_outputs(outputs, threshold):
    """
    Extracts masks, labels, scores, and bounding boxes from model outputs that have a score above a given threshold.

    Args:
        outputs (list): List of model outputs for each image, containing 'scores', 'masks', 'boxes', and 'labels'.
        threshold (float): The threshold for filtering predictions based on score.

    Returns:
        tuple: 
            - mask_list (list): List of masks for predictions above the threshold.
            - label_list (list): List of labels for predictions above the threshold.
            - score_list (list): List of scores for predictions above the threshold.
            - box_list (list): List of bounding boxes for predictions above the threshold.
    """
    mask_list = []
    label_list = []
    score_list = []
    box_list = []
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
        score_list.append(scores)
        box_list.append(boxes)
    return mask_list, label_list, score_list, box_list



def match_mask(masked_image, binary_array):
    """
    Computes evaluation metrics (precision, recall, F1-score, IoU) for the predicted and ground truth masks.

    Args:
        masked_image (list): List of predicted binary masks.
        binary_array (list): List of ground truth binary masks.

    Returns:
        float: F1-score for the "False" class based on the confusion matrix.
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



def evaluate_metrics(target, masks, labels, scores, iou_threshold):
    """
    Evaluates model performance by computing F1 score, matched labels, and scores for each target above the IoU threshold.

    Args:
        target (list): List of ground truth annotations containing labels and masks.
        masks (list): List of predicted masks.
        labels (list): List of predicted labels.
        scores (list): List of predicted scores.
        iou_threshold (float): The IoU threshold for considering a match.

    Returns:
        tuple:
            - f1_score_list (list): List of F1 scores for each valid mask match.
            - matched_label_list (list): List of matched labels (1 for match, 0 for no match).
            - actual_label_list (list): List of actual labels from ground truth.
            - pred_label_list (list): List of predicted labels.
            - score_list (list): List of scores corresponding to each predicted mask.
    """
    f1_score_list = []
    matched_label_list = []
    actual_label_list = []
    pred_label_list = []
    score_list = []
    for i in range(len(target)):
        target_labels = actual_label_target(target[i]['labels'])
        for l in range(len(target_labels)):
            for j in range(len(masks)):
                for k in range(len(masks[j])):
                    target_mask = target[i]['masks'][l].cpu().numpy()
                    target_mask = np.where(target_mask > 0, 1, 0)
                    if target_mask.shape == masks[j][k].shape:
                        f1_score = match_mask(masks[j][k], target_mask)
                        if f1_score > iou_threshold:
                            f1_score_list.append(f1_score)
                            matched_label = match_label(labels[j][k], target_labels[l])
                            matched_label_list.append(matched_label)
                            actual_label_list.append(target_labels[l])
                            pred_label_list.append(labels[j][k])
                            score_list.append(scores[j][k])
    return f1_score_list, matched_label_list, actual_label_list, pred_label_list, score_list


def compute_iou(mask1, mask2):
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
    iou = intersection / union if union != 0 else 0
    return iou


def evaluate_mask_rcnn(predictions, ground_truths, iou_threshold=0.5):
    """
    Evaluates the Mask R-CNN model's performance using precision, recall, and F1 score for mask matching.

    Args:
        predictions (list): List of predicted outputs containing 'masks' and 'labels'.
        ground_truths (list): List of ground truth annotations containing 'masks' and 'labels'.
        iou_threshold (float): The threshold for determining whether a predicted mask matches a ground truth mask.

    Returns:
        tuple: The mean precision, mean recall, and mean F1 score for all predictions and ground truths.
    """
    all_precisions = []
    all_recalls = []
    all_f1_scores = []

    for pred, gt in zip(predictions, ground_truths):
        pred_masks = pred['masks']
        gt_masks = gt['masks']
        pred_labels = pred['labels']
        gt_labels = gt['labels']

        iou_matrix = torch.zeros((len(pred_masks), len(gt_masks)))
        for i, pred_mask in enumerate(pred_masks):
            for j, gt_mask in enumerate(gt_masks):
                iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)

        matches = iou_matrix > iou_threshold

        precision, recall, f1, _ = precision_recall_fscore_support(
            matched_gt_labels.cpu().numpy(), matched_pred_labels.cpu().numpy(), average='weighted', zero_division=0)

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)

    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    mean_f1 = np.mean(all_f1_scores)

    return mean_precision, mean_recall, mean_f1



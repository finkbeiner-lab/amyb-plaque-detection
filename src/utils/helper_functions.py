
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
    mask_list = []
    label_list = []
    score_list =[]
    for j in range(len(outputs)):
        scores = list(outputs[j]['scores'].detach().cpu().numpy())
        # print("\n scores", max(scores))
        # index of those scores which are above a certain threshold
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
        #print(thresholded_preds_inidices)
        scores = [scores[x] for x in thresholded_preds_inidices]
        # get the masks
        masks = (outputs[j]['masks']>0.5).squeeze().detach().cpu().numpy()
        # print("masks", masks)
        # discard masks for objects which are below threshold
        masks = [masks[x] for x in thresholded_preds_inidices]
        # get the bounding boxes, in (x1, y1), (x2, y2) format
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[j]['boxes'].detach().cpu()]
        # discard bounding boxes below threshold value
        boxes = [boxes[x] for x in thresholded_preds_inidices]
        # get the classes labels
        # print('labels', outputs[0]['labels'])
        #print(outputs[0]['labels'])
        #print(thresholded_preds_count)
        #print(outputs[0]['labels'])
        labels = list(outputs[j]['labels'].detach().cpu().numpy())
        #labels = [i for i in outputs[j]['labels'].detach().cpu().numpy()]
        #labels = [i for i in outputs[0]['labels']]
        #print(labels)
        labels = [labels[x] for x in thresholded_preds_inidices]
        mask_list.append(masks)
        label_list.append(labels)
        score_list.append(scores)
    return mask_list, label_list, score_list


def match_mask(masked_image,binary_array):
    device = torch.device('cuda', 0)
    #num_classes = len(set(list(np.unique(masked_image)) +  list(np.unique(binary_array))))
    num_classes = 2
    metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    conf_final=torch.tensor(np.zeros((num_classes,num_classes))).to(device).to(torch.int64)
    for i in range(len(masked_image)):
        preds = torch.tensor(masked_image[i]).to(device).to(torch.int64)
        target = torch.tensor(binary_array[i]).to(device).to(torch.int64)
        a1 = metric(preds,target)
        conf_final = conf_final + a1     
    conf_final_np = conf_final.cpu().numpy()
    #print(conf_final_np)
    total_predicted = np.sum(conf_final_np, axis=0) 
    diag_elements = np.diag(conf_final_np)
    precision = diag_elements/total_predicted
    total_actual = np.sum(conf_final_np, axis=1) 
    recall = diag_elements/total_actual
    f1_score = (2*precision*recall)/(recall+precision)
    iou_coeff = (diag_elements)/(total_predicted+total_actual-diag_elements)
    #csv_filename_tosave = "Eval_Metric_"+ geofile.split(".")[0] + ".csv"
    eval_metrics = pd.DataFrame({"Class":["True","False"],"Precision":precision, "recall":recall,"f1_score":f1_score,"iou_coeff":iou_coeff})
    #print(eval_metrics)
    #eval_metrics.to_csv(os.path.join(eval_dir,csv_filename_tosave))
    return eval_metrics[eval_metrics["Class"]=="False"]["f1_score"].values[0]


def match_label(pred_label, gt_label):
    #print(pred_label)
    #print(gt_label)
    if pred_label==gt_label:
        return 1
    else:
        return 0

def actual_label_target(gt_label):
    return gt_label.cpu().numpy()
    #idx = gt_label.cpu().numpy()
    #if idx>0:
    #    return class_names[idx-1]
    #return idx.astype(int)


def evaluate_metrics(target,masks, labels, scores, iou_threshold):
    f1_score_list=[]
    matched_label_list=[]
    #mean_f1_score = -1
    #mean_matched_label=-1
    actual_label_list = []
    pred_label_list = []
    score_list = []
    for i in range(len(target)):
        target_labels = actual_label_target(target[i]['labels'])
        #print(target[i]['masks'][0].shape, masks[0].shape)
        for l in range(len(target_labels)):
            for j in range(len(masks)):
                for k in range(len(masks[j])):
                    target_mask = target[i]['masks'][l].cpu().numpy()
                    target_mask= np.where(target_mask > 0, 1, 0)
                    if target_mask.shape==masks[j][k].shape:
                        f1_score = match_mask(masks[j][k],target_mask)
                        if f1_score>iou_threshold:
                            f1_score_list.append(f1_score)
                            matched_label = match_label(labels[j][k],target_labels[l])
                            matched_label_list.append(matched_label)
                            actual_label_list.append(target_labels[l])
                            pred_label_list.append(labels[j][k])
                            score_list.append(scores[j][k])
    return f1_score_list, matched_label_list, actual_label_list,pred_label_list, score_list



def compute_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2).sum().item()
    union = torch.logical_or(mask1, mask2).sum().item()
    iou = intersection / union if union != 0 else 0
    return iou

def evaluate_mask_rcnn(predictions, ground_truths, iou_threshold=0.5):
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


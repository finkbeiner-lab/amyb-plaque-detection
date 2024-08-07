import os
os.environ['OMP_NUM_THREADS'] = '1'
import glob
import sys
sys.path.append(os.path.join(os.getcwd(), *tuple(['..'])))
import argparse

from typing import Callable, Dict, List, Optional, Set
from collections import OrderedDict
import pdb
import numpy as np
import torch
from model_mrcnn_config import _default_mrcnn_config, build_default
from features import build_features
from generalized_mask_rcnn_pl import LitMaskRCNN
from utils.helper_functions import evaluate_metrics, get_outputs, compute_iou, evaluate_mask_rcnn


from features import transforms as T
#from utils.engine import evaluate
import torchvision
from utils.helper_functions import evaluate_metrics, get_outputs, compute_iou, evaluate_mask_rcnn
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#import visualize_output


def plot_pr_curve(precisions, recalls,class_names):
    fig = go.Figure()
    for i,class_name in enumerate(class_names):
        fig.add_trace(go.Scatter(x=recalls[:,i], y=precisions[:,i], name=class_name, mode='lines',line=dict(color=colors[class_name], width=2)))
        
    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=700
    )
    fig.update_layout( plot_bgcolor='white',title="Precision-Recall Curve")
    fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black',gridcolor='lightgrey',range=[0, 1])
    fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black',gridcolor='lightgrey',range=[0, 1])
    fig.write_html("/workspace/Projects/Amyb_plaque_detection/reports/"+model_name.split("/")[-3]+"_"+model_name.split("/")[-1]+"_test_pr_curve.html")

def plot_acc_curve(thresholds, acc_scores,class_names):
    fig = go.Figure()
    #for i,class_name in enumerate(class_names):
    fig.add_trace(go.Scatter(x=thresholds, y=acc_scores, name=class_name, mode='lines'))
        
    fig.update_layout(
        xaxis_title='Threshold',
        yaxis_title='Accuracy',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=700
    )
    fig.update_layout( plot_bgcolor='white',title="Precision-Recall Curve")
    fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black',gridcolor='lightgrey')
    fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black',gridcolor='lightgrey')
    fig.write_html("/workspace/Projects/Amyb_plaque_detection/reports/"+model_name.split("/")[-3]+"_"+model_name.split("/")[-1]+"_test_pr_curve.html")


def plot_f1_score(thresholds,f1_scores,class_names):
    fig = go.Figure()
    for i,class_name in enumerate(class_names):
        fig.add_trace(go.Scatter(x=thresholds, y=f1_scores[:,i], name=class_name, mode='lines',line=dict(color=colors[class_name], width=2)))
        
    fig.update_layout(
        xaxis_title='Threshold',
        yaxis_title='F1-score',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=700
    )
    fig.update_layout( plot_bgcolor='white',title="Precision-Recall Curve")
    fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black',gridcolor='lightgrey')
    fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black',gridcolor='lightgrey')
    fig.write_html("/workspace/Projects/Amyb_plaque_detection/reports/"+model_name.split("/")[-3]+"_"+model_name.split("/")[-1]+"_test_f1score_curve.html")


def plot_confusion_matrix(conf_mat, save_path):
    
    # Set formatting and styling options for the confusion matrices
    title_size = 16
    plt.rcParams.update({'font.size':16})
    display_labels = class_names  # Customize labels of the classes
    colorbar = False
    cmap = "Blues"  # Try "Greens". Change the color of the confusion matrix.
    ## Please see other alternatives at https://matplotlib.org/stable/tutorials/colors/colormaps.html
    values_format = ".3f"  # Determine the number of decimal places to be displayed.

    f, ax = plt.subplots(1, 1, figsize=(10, 16))

    # Plot the confusion matrix
    #ax.set_title("Model 1", size=title_size)
    ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=display_labels).plot(
        include_values=True, cmap=cmap, ax=ax, colorbar=colorbar, values_format=values_format)

    # Remove x-axis labels and ticks
    #ax.xaxis.set_ticklabels(['', '', '', ''])
    #ax.set_xlabel('')
    #ax.tick_params(axis='x', which='both')

    # Set the overall title and show the plot
    f.suptitle("Multiple Confusion Matrices", size=title_size, y=0.93)
    # Save the figure as a PDF file
    f.savefig(save_path, bbox_inches='tight')
    
    plt.show()




    

def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)
    #print(masks1.shape, masks2.shape)
    #print(area1,area2)
    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.75, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    #print(pred_masks.shape)
    #print(gt_masks.shape)
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]
    #gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    #print(pred_masks.shape)
    #print(gt_masks.shape)
    overlap_matrix = np.zeros((len(pred_masks),len(gt_masks)))
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)
    #print(overlaps)
    """
    # Compute IoU overlaps [pred_masks, gt_masks]
    for k in range(len(pred_masks)):
        for l in range(len(gt_masks)):
            print(pred_masks[k].shape)
            print(gt_masks[l].shape)
            overlaps = compute_overlaps_masks(pred_masks[k], gt_masks[l])
            print(overlaps)
            overlap_matrix[k,l]=overlaps
    """
    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    class_match = -1 * np.ones([pred_boxes.shape[0]])
    pred_labels=[]
    gt_labels=[]
    overlap_mapped = []
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        #print ("overlaps[i, sorted_ixs]",overlaps[i, sorted_ixs])
        low_score_idx = np.where(overlaps[i, sorted_ixs] < 0)[0]
        low_score_idx = pred_scores[pred_scores< score_threshold]
        #print(low_score_idx)
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            
            pred_labels.append(pred_class_ids[i])
            gt_labels.append(gt_class_ids[j])
            if pred_class_ids[i] == gt_class_ids[j]:
                overlap_mapped.append(iou)
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                #class_match[i]= gt_class_ids[j]
                break
            
    return pred_scores, gt_match, pred_match, overlap_mapped, class_match, gt_class_ids,pred_class_ids,pred_labels,gt_labels

def compute_accuracy(confusion_matrix):
    """
    Compute accuracy from the confusion matrix for multi-class classification.
    
    Args:
        confusion_matrix (np.ndarray): A square matrix of shape (n_classes, n_classes) where
                                       confusion_matrix[i, j] is the number of times class i
                                       was predicted as class j.
    
    Returns:
        float: Accuracy of the model.
    """
    # Ensure the confusion matrix is a numpy array
    confusion_matrix = np.array(confusion_matrix)
    
    # The number of correctly predicted instances is the sum of the diagonal elements
    correct_predictions = np.trace(confusion_matrix)
    
    # The total number of instances is the sum of all elements in the confusion matrix
    total_instances = np.sum(confusion_matrix)
    
    # Accuracy is the ratio of correctly predicted instances to the total number of instances
    accuracy = correct_predictions / total_instances
    
    return accuracy

colors={"Cored":"royalblue", "Diffuse":"firebrick","Coarse-Grained":"orange","CAA":"green"}
class_names = ["Cored","Diffuse","Coarse-Grained","CAA"]

test_config = dict(
    batch_size = 2,
    num_classes=4,
    device_id =0
)
#model_name = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/270ij5uq_epoch_66_step_800.ckpt"
#model_name = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/1i4pfmil-epoch=59-step=720.ckpt"
#model_name = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/2ufah260-epoch=49-step=600.ckpt"
#model_name = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/csvy8yix_epoch=31-step=384.ckpt"
#model_name = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/ha0c9pja_epoch=7-step=88.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/3pvj7tcu/checkpoints/epoch=47-step=528.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/1qmp7uqp/checkpoints/epoch=39-step=640.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/3kw68a8y/checkpoints/epoch=52-step=848.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/1g1wcai5/checkpoints/epoch=78-step=632.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/16oitvjh/checkpoints/epoch=71-step=576.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/16oitvjh/checkpoints/epoch=63-step=512.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/3nf68u55/checkpoints/epoch=138-step=1112.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/3ntey7wc/checkpoints/epoch=91-step=736.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/2d4cukpu/checkpoints/epoch=111-step=448.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/1e1iyjlz/checkpoints/epoch=188-step=756.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/2d4cukpu/checkpoints/epoch=143-step=576.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/yp2mf3i8/checkpoints/epoch=60-step=488.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning//nps-ad-nature/1w57qrp3/checkpoints/epoch=108-step=436.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/xjcpsx01/checkpoints/epoch=13-step=112.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/3jxx7il4/checkpoints/epoch=146-step=588.ckpt"
model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/yp2mf3i8/checkpoints/epoch=108-step=872.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/3fw50aml/checkpoints/epoch=35-step=576.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/3crpdzk6/checkpoints/epoch=18-step=304.ckpt"

model_config = _default_mrcnn_config(num_classes=1 + test_config['num_classes']).config
backbone, rpn, roi_heads, transform1 = build_default(model_config, im_size=1024)

optim_config = dict(
    cls=torch.optim.Adam,
    defaults=dict(lr=0.00001,weight_decay=1e-6) 
)

model = LitMaskRCNN.load_from_checkpoint(model_name)
device = torch.device('cuda', test_config['device_id'])
model = model.to(device)
model.eval()


precisions = np.zeros([1, 4])
recalls = np.zeros([1, 4])
f1_scores =np.zeros([1, 4])
thresholds =  np.linspace(0.01, 0.95,50) # 0.01 step size, 100 points
thresholds = [ 0.6]
acc_scores = np.zeros(1)

max_f1_score=0.0

for idx, threshold in enumerate(thresholds):
    print("--------------threshold---------------",threshold )
    predictions = []
    matches = []
    gt_all = []
    cored_gt, diffuse_gt,cg_gt, caa_gt  = 0,0,0,0
    cored_pred, diffuse_pred,cg_pred, caa_pred  = 0,0,0,0
    cored_matched, diffuse_matched,cg_matched, caa_matched  = 0,0,0,0
    #gt = {"1":0, "2":0, "3":0,"4":0}
    gt = {1:0, 2:0, 3:0,4:0}
    pred =  {1:0, 2:0, 3:0,4:0}
    matched = {1:0, 2:0, 3:0,4:0}
    precision_dict = {1:0, 2:0, 3:0,4:0}
    recall_dict = {1:0, 2:0, 3:0,4:0}
    pred_labels_all=[]
    gt_labels_all = []
    overlaps_all= []
    #iou_overlap = []
    #pred_scores = []
    
    collate_fn=lambda x: tuple(zip(*x))
        #exp_name = run.name
    dataset_test_location = '/workspace/Projects/Amyb_plaque_detection/Datasets/test'
    dataset_val_location = '/workspace/Projects/Amyb_plaque_detection/Datasets/val'
    #dataset_test_location = '/workspace/Projects/Amyb_plaque_detection/Datasets/test'
    #test_folders = glob.glob(os.path.join(dataset_test_location, "*"))
    #for test_folder in test_folders:
    #    test_dataset = build_features.AmyBDataset(os.path.join(dataset_test_location,test_folder), T.Compose([T.ToTensor()]))
    test_dataset = build_features.AmyBDataset(dataset_val_location, T.Compose([T.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_config['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)
        #print(test_folder)
    for j, (images, targets) in enumerate(test_loader):
        images = [image.to(device) for image in images]
        targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]
        _, outputs = model.forward(images, targets)
            
            
        for i in range(len(images)):
            gt_boxes = targets[i]["boxes"].cpu().detach().numpy() 
            gt_class_ids = targets[i]["labels"].cpu().detach().numpy()
            gt_masks = targets[i]["masks"].permute(2, 1, 0)
            gt_masks = gt_masks.cpu().detach().numpy()

            

            pred_boxes =outputs[i]["boxes"].cpu().detach().numpy()
            pred_class_ids = outputs[i]["labels"].cpu().detach().numpy()
            pred_scores = outputs[i]["scores"].cpu().detach().numpy()
            pred_masks = outputs[i]["masks"][pred_scores>=threshold]
            
            pred_masks=pred_masks.squeeze(1).permute(2, 1, 0)
            pred_masks = pred_masks.cpu().detach().numpy()

            pred_boxes =pred_boxes[pred_scores>=threshold]
            pred_class_ids = pred_class_ids[pred_scores>=threshold]
            pred_scores = pred_scores[pred_scores>=threshold]

            pred_scores, gt_match, pred_match, overlaps, class_match, gt_class_ids,pred_class_ids,pred_labels,gt_labels = compute_matches(gt_boxes, 
                                gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks,
                                iou_threshold=0.75, score_threshold=0.0)

            pred_labels_all.extend(pred_labels)
            gt_labels_all.extend(gt_labels)
            predictions.append(np.sum(pred_scores>=threshold))
            matches.append(np.sum(pred_match!=-1))
            gt_all.append(len(gt_class_ids))
            overlaps_all.extend(overlaps)
            for x in gt_class_ids:
            #    gt[str(int(x))]=gt[str(x)]+1
                gt[x]=gt[x]+1
                
            for x in pred_class_ids:
                pred[x]=pred[x]+1
                
                
            class_match = gt_class_ids[pred_match[pred_match!=-1].astype(int)]
            #print(class_match)
            for x in class_match:
                matched[int(x)]=matched[int(x)]+1
                
            if len(class_match)>len(pred_class_ids):
                #print(pred_score, gt_match, pred_match, overlaps, class_match, gt_class_ids,pred_class_ids)
                print("class match", class_match)
                print("pred_class", pred_class_id2)
                print("gt_class", gt_class_ids)
                print("pred match", pred_match)
                
    print("gt",gt)
    print("pred",pred)
    print("matched",matched)
    for k in range(4):         
        if pred[k+1]!=0:
            precisions[idx, k] = matched[k+1]/pred[k+1]
        else:
            precisions[idx, k] = 0
        if gt[k+1]!=0:
            recalls[idx, k] = matched[k+1]/gt[k+1]
        else:
            recalls[idx, k] = 0
        if (precisions[idx, k]>0) and (recalls[idx, k]>0):
            f1_scores[idx,k] = (2*precisions[idx, k]*recalls[idx, k])/(precisions[idx, k]+recalls[idx, k])
        else:
            f1_scores[idx,k] = 0
            
    #if np.mean(f1_scores[idx,:])>max_f1_score:
    #    max_f1_score= np.mean(f1_scores[idx,:])
    #else:
    #if idx==50:
    conf_mat = confusion_matrix(gt_labels_all, pred_labels_all)
    print(conf_mat)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                                  display_labels=class_names)
    accuracy = compute_accuracy(conf_mat)
    print("Model Accuracy", accuracy)
    plot_confusion_matrix(conf_mat, "/workspace/Projects/Amyb_plaque_detection/reports/"+model_name.split("/")[-3]+"_"+model_name.split("/")[-1]+"_conf_mat"+str(idx)+".png")
    #plot_confusion_matrix(conf_mat, "/workspace/Projects/Amyb_plaque_detection/reports/"+model_name.split("/")[-3]+"_"+model_name.split("/")[-1]+"_conf_mat_val"+str(idx)+".png")
        
    
    acc_scores[idx] = (np.sum([matched[k+1] for k in range(4)]))/(np.sum([gt[k+1] for k in range(4)])+np.sum([pred[k+1] for k in range(4)]) -np.sum([matched[k+1] for k in range(4)]))
        
    print("precision", np.sum(matches)/np.sum(predictions))
    print("recall", np.sum(matches)/np.sum(gt_all))

#print(precisions)
#print(recalls)
#print(f1_scores)
print(overlaps_all)
print(np.mean(overlaps_all))

#plot_pr_curve(precisions, recalls,class_names)
#plot_f1_score(thresholds,f1_scores,class_names)


"""
plt.figure(2)
plt.plot(recalls[:,0], precisions[:,0], marker='.', label="cored")
plt.plot(recalls[:,1], precisions[:,1], marker='.', label="diffuse")
plt.plot(recalls[:,2], precisions[:,2], marker='.', label="cg")
plt.plot(recalls[:,3], precisions[:,3], marker='.', label="caa")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.figlegend()
plt.title('Precision-Recall Curve')
plt.savefig("/workspace/Projects/Amyb_plaque_detection/reports/"+model_name.split("/")[-3]+"_"+model_name.split("/")[-1]+"_pr_curve.png")
plt.show()

plt.figure(3)
plt.plot(thresholds, acc_scores, marker='.')
plt.xlabel('thresholds')
plt.ylabel('acc_scores')
plt.figlegend()
plt.title('Accuracy v/s threshold')
plt.savefig("/workspace/Projects/Amyb_plaque_detection/reports/"+model_name.split("/")[-3]+"_"+model_name.split("/")[-1]+"_acc_curve.png")
plt.show()

plt.figure(4)
plt.plot(thresholds, f1_scores[:,0], marker='.', label="cored")
plt.plot(thresholds, f1_scores[:,1], marker='.', label="diffuse")
plt.plot(thresholds, f1_scores[:,2], marker='.', label="cg")
plt.plot(thresholds, f1_scores[:,3], marker='.', label="caa")
plt.xlabel('Threshold')
plt.ylabel('F1-score')
plt.title('F1score')
plt.figlegend()
plt.savefig("/workspace/Projects/Amyb_plaque_detection/reports/"+model_name.split("/")[-3]+"_"+model_name.split("/")[-1]+"_f1_score.png")
plt.show()
"""



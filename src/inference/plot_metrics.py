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
from models_pytorch_lightning.model_mrcnn_config import _default_mrcnn_config, build_default
from features import build_features
from models_pytorch_lightning.generalized_mask_rcnn_pl import LitMaskRCNN
from utils.helper_functions import evaluate_metrics, get_outputs, compute_iou, evaluate_mask_rcnn
from features import build_features
from features import build_features
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
import pdb


#OUTPUT_DIR = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/reports/interrater-test-metrics"
OUTPUT_DIR = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/reports/test-metrics"

model_name= "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/yp2mf3i8_epoch=108-step=872.ckpt"
#dataset_test_location = '/home/mahirwar/Desktop/Monika/npsad_data/vivek/interrater-study/tiles'
dataset_test_location = '/home/mahirwar/Desktop/Monika/npsad_data/vivek/Datasets/amyb_wsi_v2/test

colors={"Cored":"royalblue", "Diffuse":"firebrick","Coarse-Grained":"orange","CAA":"green"}
class_names = ["Cored","Diffuse","Coarse-Grained","CAA"]
def plot_roc_curve(all_df, save_path):
    
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    for class_name in class_names:
        fpr, tpr, _ = roc_curve(all_df[class_name], all_df["pred_score"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=class_name, mode='lines',line=dict(color=colors[class_name], width=2)))
    
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=700
    )
    fig.update_layout( plot_bgcolor='white', title="Receiver operating characteristic Curve")
    fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black',gridcolor='lightgrey')
    fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black',gridcolor='lightgrey')
    fig.write_html(save_path)


def plot_pr_curve(all_df, save_path) :
    fig = go.Figure()
    for class_name in class_names:
        pr, rc, _ = precision_recall_curve(all_df[class_name], all_df["pred_score"])
        rp = (all_df[class_name]).sum()/len(all_df)
        fig.add_trace(go.Scatter(x=rc, y=pr, name=class_name, mode='lines',line=dict(color=colors[class_name], width=2)))
        
    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=700
    )
    fig.update_layout( plot_bgcolor='white',title="Precision-Recall Curve")
    fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black',gridcolor='lightgrey')
    fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black',gridcolor='lightgrey')
    fig.write_html(save_path)



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
    plt.show()

    # Save the figure as a PDF file
    f.savefig(save_path, bbox_inches='tight')




test_config = dict(
    batch_size = 1,
    num_classes=4,
    device_id =0
)



#model_name = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/270ij5uq_epoch_66_step_800.ckpt"
#model_name = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/1i4pfmil-epoch=59-step=720.ckpt"
#model_name = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/2ufah260-epoch=49-step=600.ckpt"
#model_name=  "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/csvy8yix_epoch=31-step=384.ckpt"
#model_name = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/3ul8krzf_epoch=47-step=576.ckpt"
#model_name = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/cnf1sro4_epoch=68-step=690.ckpt"
#model_name = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/3kw68a8y_epoch=52-step=848.ckpt"

model_config = _default_mrcnn_config(num_classes=1 + test_config['num_classes']).config
backbone, rpn, roi_heads, transform1 = build_default(model_config, im_size=1024)

optim_config = dict(
    cls=torch.optim.Adam,
    defaults=dict(lr=0.00001,weight_decay=1e-6) 
)

model = LitMaskRCNN.load_from_checkpoint(model_name)


f1_list =[]
label_matched_list =[]
actual_labels_list = []
pred_labels_list = []
score_list = []

device = torch.device('cuda', test_config['device_id'])
model = model.to(device)
model.eval()


output_path = os.path.join(OUTPUT_DIR,model_name.split("/")[-1])
if not os.path.exists(output_path):
    os.mkdir(output_path)





f1_list =[]
label_matched_list =[]
actual_labels_list = []
pred_labels_list = []
score_list = []

collate_fn=lambda x: tuple(zip(*x))
    #exp_name = run.name

test_folders = glob.glob(os.path.join(dataset_test_location, "*"))
print("test_folders",test_folders)

df = pd.DataFrame()
for test_folder in test_folders:
    test_dataset = build_features.AmyBDataset(os.path.join(dataset_test_location, test_folder), T.Compose([T.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_config['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)
    for i, (images, targets) in enumerate(test_loader):
        images = [image.to(device) for image in images]
        targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]
        #print(images)
        #print(targets)
        #break
        outputs = model.forward(images, targets)
        #print(outputs)

        masks, labels, scores ,_ = get_outputs(outputs, 0.5)
        f1_mean, labels_matched,actual_labels,pred_labels, scores =  evaluate_metrics(targets, masks, labels,scores,0.5 )
        f1_list.extend(f1_mean)
        label_matched_list.extend(labels_matched)
        actual_labels_list.extend(actual_labels)
        pred_labels_list.extend(pred_labels)
        score_list.extend(scores)

        
    
""" 
    for threshold in [0.25,0.5,0.75,0.9,0.95]:
        for i, (images, targets) in enumerate(test_loader):
            images = [image.to(device) for image in images]
            targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]
            outputs = model.forward(images, targets)
            #print(outputs)
            masks, labels, scores = get_outputs(outputs, threshold)
            f1_mean, labels_matched,actual_labels,pred_labels, scores =  evaluate_metrics(targets, masks, labels,scores )
            print(f1_mean, labels_matched)
            f1_list.extend(f1_mean)
            label_matched_list.extend(labels_matched)
            actual_labels_list.extend(actual_labels)
            pred_labels_list.extend(pred_labels)
            score_list.extend(scores)
        precision, recall, fscore, support = precision_recall_fscore_support(actual_labels_list,pred_labels_list)
        temp = pd.DataFrame({"class":["Cored", "Diffuse","Coarse-Grained","CAA"], "precision":list(precision), "recall":list(recall), "fscore":fscore, "support":support} )
        temp["threshold"]= threshold
        if len(df)==0:
            df = temp
        else:
            df = pd.concat([df,temp], ignore_index=True)
    
df.to_csv("/home/mahirwar/Desktop/Monika/npsad_data/vivek/reports/test-metrics"+model_name.split("/")[-3]+"_eval_metrics.csv")   
"""   
    
all_df = pd.DataFrame({"f1_score":f1_list, "matched_labels":label_matched_list, "actual_labels":actual_labels_list, "pred_labels":pred_labels_list,
"pred_score":score_list})

print(np.mean(all_df["f1_score"].values), np.mean(all_df["matched_labels"].values))

print(np.unique(all_df["actual_labels"].values))
print(np.unique(all_df["pred_labels"].values))

all_df["Cored"] = np.where(all_df["actual_labels"]==1, 1, 0)
all_df["Diffuse"] = np.where(all_df["actual_labels"]==2, 1, 0)
all_df["Coarse-Grained"] = np.where(all_df["actual_labels"]==3, 1, 0)
all_df["CAA"] = np.where(all_df["actual_labels"]==4, 1, 0)


#all_df["Cored"]= all_df["actual_labels"].apply(lambda l: 1 if l==0 else 0)
#print(np.unique(all_df["Cored"].values))

#all_df["Diffuse"]= all_df["actual_labels"].apply(lambda l: 1 if l==1 else 0)
#all_df["Coarse-Grained"]= all_df["actual_labels"].apply(lambda l: 1 if l==2 else 0)
#all_df["CAA"]= all_df["actual_labels"].apply(lambda l: 1 if l==3 else 0)
roc_save_path = os.path.join(output_path,"ROC_Curve.html" )
#plot_roc_curve(all_df, roc_save_path)

pr_save_path = os.path.join(output_path,"PR_Curve.html" )
#plot_pr_curve(all_df,pr_save_path)

conf_mat_save_path = os.path.join(output_path,"conf_mat.png" )
conf_mat = confusion_matrix(all_df["actual_labels"], all_df["pred_labels"])
print(conf_mat)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                              display_labels=class_names)
disp.plot()
plot_confusion_matrix(conf_mat, conf_mat_save_path)

pd.DataFrame(conf_mat ).to_csv((os.path.join(output_path,"confusion_matrix.csv")))
#precision, recall, fscore, support = precision_recall_fscore_support(all_df["actual_labels"], all_df["pred_labels"])
#temp = pd.DataFrame({"class":["Cored", "Diffuse","Coarse-Grained","CAA"], "precision":list(precision), "recall":list(recall), "fscore":fscore, "support":support} )
#temp.to_csv(os.path.join(output_path,"prec_recall.csv"))
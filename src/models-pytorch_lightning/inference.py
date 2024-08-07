import os
os.environ['OMP_NUM_THREADS'] = '1'

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

colors={"Cored":"royalblue", "Diffuse":"firebrick","Coarse-Grained":"orange","CAA":"green"}
class_names = ["Cored","Diffuse","Coarse-Grained","CAA"]
def plot_roc_curve(all_df):
    
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
    fig.write_html("/workspace/Projects/Amyb_plaque_detection/reports/"+model_name.split("/")[-3]+"_ROC_Curve.html")


def plot_pr_curve(all_df):
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
    fig.write_html("/workspace/Projects/Amyb_plaque_detection/reports/"+model_name.split("/")[-3]+"_PR_Curve.html")







test_config = dict(
    batch_size = 8,
    num_classes=4,
    device_id =0
)

dataset_val_location = '/workspace/Projects/Amyb_plaque_detection/Datasets/val'
val_dataset = build_features.AmyBDataset(dataset_val_location, T.Compose([T.ToTensor()]))

model_config = _default_mrcnn_config(num_classes=1 + test_config['num_classes']).config
backbone, rpn, roi_heads, transform1 = build_default(model_config, im_size=1024)

optim_config = dict(
        cls=torch.optim.Adam,
        defaults=dict(lr=0.0001,weight_decay=1e-5) 
    )

model  = LitMaskRCNN(optim_config,backbone,rpn,roi_heads,transform1)

dataset_base_dir = '/workspace/Projects/Amyb_plaque_detection/'

#ckpt_path= os.path.join(dataset_base_dir, "pytorch_lightning_model_output/35dc6j8j")

#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/270ij5uq/checkpoints/epoch=66-step=800.ckpt"

#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/2r3vmu7j/checkpoints/epoch=33-step=400.ckpt"

#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/25i4i1ck/checkpoints/epoch=66-step=800.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/2ufah260/checkpoints/epoch=49-step=600.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/2mshh693/checkpoints/epoch=59-step=360.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/3ul8krzf/checkpoints/epoch=47-step=576.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/csvy8yix/checkpoints/epoch=31-step=384.ckpt"
#model_name ="/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/3t64yp20/checkpoints/epoch=22-step=253.ckpt"
#model_name ="/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/3t64yp20/checkpoints/epoch=22-step=253.ckpt"
#model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/1qmp7uqp/checkpoints/epoch=32-step=528.ckpt"
model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/3kw68a8y/checkpoints/epoch=45-step=736.ckpt"
model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/3kw68a8y/checkpoints/epoch=45-step=736.ckpt"
model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/1g1wcai5/checkpoints/epoch=78-step=632.ckpt"
model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/16oitvjh/checkpoints/epoch=71-step=576.ckpt"
model_name = "/workspace/Projects/Amyb_plaque_detection/amyb-plaque-detection/src/models-pytorch_lightning/nps-ad-nature/1e1iyjlz/checkpoints/epoch=188-step=756.ckpt"

chkpt = ModelCheckpoint(monitor="val_acc", mode="max")

checkpoint = torch.load(model_name)

model.load_state_dict(checkpoint["state_dict"])
#optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#print(model)


collate_fn=lambda x: tuple(zip(*x))
    #exp_name = run.name
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_config['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)
f1_list =[]
label_matched_list =[]
actual_labels_list = []
pred_labels_list = []
score_list = []

device = torch.device('cuda', test_config['device_id'])
model = model.to(device)
model.eval()
"""
df = pd.DataFrame()
for threshold in [0.25,0.5,0.75,0.9,0.95]:
    for i, (images, targets) in enumerate(val_loader):
        images = [image.to(device) for image in images]
        targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]
        outputs = model.forward(images, targets)
        #print(outputs)
        masks, labels, scores = get_outputs(outputs, 0.50)
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



df.to_csv("/workspace/Projects/Amyb_plaque_detection/reports/"+model_name.split("/")[-3]+"_eval_metrics.csv")
"""

f1_list =[]
label_matched_list =[]
actual_labels_list = []
pred_labels_list = []
score_list = []

for i, (images, targets) in enumerate(val_loader):
    images = [image.to(device) for image in images]
    targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]
    losses, outputs = model.forward(images, targets)
    #print(outputs)
    masks, labels, scores = get_outputs(outputs, 0.5)
    f1_mean, labels_matched,actual_labels,pred_labels, scores =  evaluate_metrics(targets, masks, labels,scores,0.5 )
    f1_list.extend(f1_mean)
    label_matched_list.extend(labels_matched)
    actual_labels_list.extend(actual_labels)
    pred_labels_list.extend(pred_labels)
    score_list.extend(scores)
    
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

plot_roc_curve(all_df)
plot_pr_curve(all_df)

conf_mat = confusion_matrix(all_df["actual_labels"], all_df["pred_labels"])
print(conf_mat)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                              display_labels=class_names)
disp.plot()
pd.DataFrame(conf_mat ).to_csv("/workspace/Projects/Amyb_plaque_detection/reports/"+model_name.split("/")[-3]+"_confusion_matrix.csv")
precision, recall, fscore, support = precision_recall_fscore_support(all_df["actual_labels"], all_df["pred_labels"])
temp = pd.DataFrame({"class":["Cored", "Diffuse","Coarse-Grained","CAA"], "precision":list(precision), "recall":list(recall), "fscore":fscore, "support":support} )
temp.to_csv("/workspace/Projects/Amyb_plaque_detection/reports/"+model_name.split("/")[-3]+"_prec_recall.csv")

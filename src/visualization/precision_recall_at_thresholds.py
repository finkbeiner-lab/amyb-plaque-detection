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
model_name = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/csvy8yix_epoch=31-step=384.ckpt"
#model_name = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/3ul8krzf_epoch=47-step=576.ckpt"

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


output_path = os.path.join("/home/mahirwar/Desktop/Monika/npsad_data/vivek/reports/test-metrics",model_name.split("/")[-1])
if not os.path.exists(output_path):
    os.mkdir(output_path)




collate_fn=lambda x: tuple(zip(*x))
    #exp_name = run.name
dataset_test_location = '/home/mahirwar/Desktop/Monika/npsad_data/vivek/Datasets/amyb_wsi/test'
test_folders = glob.glob(os.path.join(dataset_test_location, "*"))
df = pd.DataFrame()
for threshold in [0.3, 0.4, 0.5,0.6, 0.7]:
    f1_list =[]
    label_matched_list =[]
    actual_labels_list = []
    pred_labels_list = []
    score_list = []
    for test_folder in test_folders:
        test_dataset = build_features.AmyBDataset(os.path.join(dataset_test_location,test_folder), T.Compose([T.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_config['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)
        for i, (images, targets) in enumerate(test_loader):
            images = [image.to(device) for image in images]
            targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]
            outputs = model.forward(images, targets)
            #print(outputs)
            masks, labels, scores,_ = get_outputs(outputs, threshold)
            f1_mean, labels_matched,actual_labels,pred_labels, scores =  evaluate_metrics(targets, masks, labels,scores, 0.5)
            f1_list.extend(f1_mean)
            label_matched_list.extend(labels_matched)
            actual_labels_list.extend(actual_labels)
            pred_labels_list.extend(pred_labels)
            score_list.extend(scores)
            
    assert(len(f1_list)==len(label_matched_list))
    assert(len(actual_labels_list)==len(pred_labels_list))
    precision, recall, fscore, support = precision_recall_fscore_support(actual_labels_list,pred_labels_list)
    print("---------threshold----------",threshold)
    print(precision, recall, fscore, support)
    if len(precision)==4:
        temp = pd.DataFrame({"class":["Cored", "Diffuse","Coarse-Grained","CAA"], "precision":list(precision), "recall":list(recall), "fscore":fscore, "support":support} )
        temp["threshold"] = threshold
        if len(df)==0:
            df = temp
        else:
            df = pd.concat([df,temp], ignore_index=True)

df.to_csv(os.path.join(output_path,"eval_metrics.csv"))
#print(precision, recall, fscore, support)

#temp =pd.DataFrame({"pred_labels":pred_labels_list, "actual_labels":actual_labels_list, "scores":score_list, "f1-score":f1_list, ""


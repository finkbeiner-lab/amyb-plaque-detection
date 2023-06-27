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
from torch import nn, Tensor
import torch.optim
import wandb
from model_mrcnn import _default_mrcnn_config, build_default
from features import build_features
from features import transforms as T
from utils.engine import evaluate
import torchvision
import matplotlib.pyplot as plt
from visualization.explain import ExplainPredictions
import pandas as pd
import plotly.graph_objects as go
import pdb


# def testmodel():
def plotPRcurve(eval2, epoch, run):

    df = pd.DataFrame(columns=["class","tpr","fpr","recall","precision"])
    len_classes = 3 ## Assuming 3 classes

    

    len_classes = len(eval2['bbox'])

   
    for c in range(len_classes): # running for all 3 classes
        ## parameters
        area_index = 0 # area - all (areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]  -> areaRngLbl = ['all', 'small', 'medium', 'large'])
        threshold_index= 0 # threshold 0.5  (iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)) -- can vary threshold from here 
        iou_type="bbox"
        maxDet = 100 
        # Selecting from 
        eval_table = eval2[iou_type][c][area_index]

        dt_score_list = np.concatenate([eval_table[i]['dtScores'][0:maxDet] for i in range(len(eval_table)) if eval_table[i]!=None])
        inds = np.argsort(-dt_score_list, kind='mergesort') 
        dtScoresSorted = dt_score_list[inds]
        dtm  = np.concatenate([eval_table[i]['dtMatches'][threshold_index][0:maxDet]  for i in range(len(eval_table)) if eval_table[i]!=None]) [inds]
        dtIg  = np.concatenate([eval_table[i]['dtIgnore'][threshold_index][0:maxDet]  for i in range(len(eval_table)) if eval_table[i]!=None]) [inds]
        gtIg = np.concatenate([eval_table[i]['gtIgnore'] for i in range(len(eval_table)) if eval_table[i]!=None])
        npig = np.count_nonzero(gtIg==0)
        tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )
        tp_sum = np.cumsum(tps, axis=0, dtype=float)
        

        if len(tp_sum) == 0:
            continue
        
        # tp_sum=tp_sum/tp_sum[-1]
        fp_sum = np.cumsum(fps, axis=0, dtype=float)
        # fp_sum=fp_sum/fp_sum[-1]
        rc_list =[]
        pr_list =[]
        for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
            rc = tp / npig
            pr = tp / (fp+tp+np.spacing(1))
            rc_list.append(rc)
            pr_list.append(pr)
        tmp = pd.DataFrame({"tpr":tp_sum,"fpr":fp_sum,"recall":rc_list,"precision":pr_list})
        tmp["class"] = c
        df = pd.concat([df, tmp])
    
    # plotly plot
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    classes = ['cored', 'diffuse', 'caa']
    for i in range(3):
        fig.add_trace(go.Scatter(x=df[df["class"]==i]["recall"], y=df[df["class"]==i]["precision"], name=classes[i], mode='lines'))


    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=1000, height=500,
        title='Precision-Recall Curve'
    )
    
    save_name = "prcurve_{epoch}.html"
    fig_name = save_name.format(epoch=epoch)
    
    fig.write_html(fig_name)
    run.log({"Precision-Recall": wandb.Html(open(fig_name))})

    # plt.plot(rc_list,pr_list)
    # plt.show()



if __name__ == '__main__':
    dataset_test_location = "/mnt/new-nas/work/data/npsad_data/vivek/Datasets/amyb_wsi/val"

    model_path = "/mnt/new-nas/work/data/npsad_data/vivek/models/eager-frog-489_mrcnn_model_100.pth"
    # model_path = "/home/vivek/Projects/amyb-plaque-detection/models/dry-disco-560_mrcnn_model_50.pth"

    epoch = 0

    ## CONFIGS ##
    collate_fn = lambda _: tuple(zip(*_)) # one-liner, no need to import

    test_config = dict(
        epochs = 32,
        batch_size = 10,
        num_classes = 3,
        device_id = 0,
        ckpt_freq =500,
        eval_freq = 30,
    )

    device = torch.device('cuda', test_config['device_id'])
    

    # Mapping
    fn_relabel = lambda i: [1, 2, 1, 3][i - 1]
    test_dataset = build_features.AmyBDataset(dataset_test_location, T.Compose([T.ToTensor()]))
    test_dataset = build_features.DatasetRelabeled(test_dataset, fn_relabel) 


    test_data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_config['batch_size'], shuffle=False, num_workers=4,
            collate_fn=collate_fn)
    
    # Buid Model
    model_config = _default_mrcnn_config(num_classes=1 + test_config['num_classes']).config
    model = build_default(model_config, im_size=1024)
    model = model.to(device)

    # Load Weights
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    run = wandb.init(project='nps-ad-vivek',
        entity='hellovivek', mode="online")

    # Evaluate
    eval_res = evaluate(run, model, test_data_loader, device=device, epoch=epoch)
    plotPRcurve(eval_res, epoch, run)

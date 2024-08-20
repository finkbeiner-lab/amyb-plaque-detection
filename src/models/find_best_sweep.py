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
#import matplotlib.pyplot as plt
from visualization.explain import ExplainPredictions
import pandas as pd
import plotly.graph_objects as go
from utils.helper_functions import evaluate_metrics, get_outputs, compute_iou, evaluate_mask_rcnn
import pdb
from torch.optim.lr_scheduler import ExponentialLR


# Define sweep config
sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        #"batch_size": {"values": [16, 24]},
        "epochs": {"values": [80,100,150]},
        "lr": {"max": 0.00001, "min": 0.0000001},
        "weight_decay":{"max": 0.000001, "min": 0.00000001}
        
        
    },
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="nps-ad-nature")
dataset_base_dir = '/workspace/Projects/Amyb_plaque_detection/Datasets'
dataset_train_location = '/workspace/Projects/Amyb_plaque_detection/Datasets/train' 
dataset_test_location = '/workspace/Projects/Amyb_plaque_detection/Datasets/val'


collate_fn = lambda _: tuple(zip(*_))

#---------------------------functions-------------------------------------------------

#def get_loss_fn(weights, default=0.):
def compute_loss_fn(weights, losses):
    item = lambda k: (k, losses[k].item())
    metrics = OrderedDict(list(map(item, [k for k in weights.keys() if k in losses.keys()] + [k for k in losses.keys() if k not in weights.keys()])))
    loss = sum(map(lambda k: losses[k] * (weights[k] if weights is not None and k in weights.keys() else default), losses.keys()))
    return loss, metrics
#    return compute_loss_fn


def get_resp(prompt, prompt_fn=None, resps='n y'.split()):
    resp = input(prompt)
    while resp not in resps:
        resp = input(prompt if prompt_fn is None else propt_fn(resp))
    return resps.index(resp)


def train_one_epoch(
    model: torch.nn.Module,
    #loss_fn: Callable[[Dict[str, Tensor]], Tensor],
    loss_weights,
    optimizer: torch.optim.Optimizer,
    scheduler:torch.optim.lr_scheduler,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int = 1,
    log_freq: int = 10,) -> None:
    
    model.train(True)
    assert model.training
    model_params = set(model.parameters())
    model_devices = set([p.device for p in model_params])
    assert model_devices == set([device]) # validate model params device
    for g in optimizer.param_groups: # validate optimizer params
        assert set(g['params']).issubset(model_params)

    log_metrics = list()
    average_loss = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]
        # visualize_augmentations(images , targets)
        #pdb.set_trace()
        optimizer.zero_grad()
        loss, metrics = compute_loss_fn(loss_weights, model.forward(images, targets))
        loss.backward()
        optimizer.step()

        #log_metrics.append(dict(epoch=epoch, loss=loss.item(), metrics=metrics))
        # print(dict(epoch=epoch, loss=loss.item(), metrics=metrics))
        print_logs = "epoch no : {epoch}, batch no : {batch_no}, total loss : {loss},  classifier :{classifier}, mask: {mask} ==================="
        print(print_logs.format(epoch=epoch, batch_no=i, loss=loss.item(),  classifier=metrics['loss_classifier'], mask=metrics['loss_mask']))
        
        #print(list(loss.item())[0])
        average_loss = average_loss+loss.item()
        print(average_loss)
        #if (i % log_freq) == 0:
        #    yield log_metrics
        #    log_metrics = list()
    #scheduler.step()
    #yield log_metrics
    #print(list(loss.item()))
    return average_loss


def evaluate_one_epoch(
    model: torch.nn.Module,
    dataset_test_location,
    test_patient_ids,
    test_config,
    device
):
    f1_mean_list = []
    labels_matched_list = []
    #for t in range(len(test_patient_ids)):
    model.eval()
    #    if len(os.listdir(os.path.join(dataset_test_location,test_patient_ids[t],"images")))==0:
    #       continue
    test_ds = build_features.AmyBDataset(os.path.join(dataset_test_location,test_patient_ids[t]),T.Compose([T.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_config['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)
    for i, (images, targets) in enumerate(val_loader):
        images = [image.to(device) for image in images]
        targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]
        outputs = model.forward(images, targets)
        masks, labels = get_outputs(outputs, 0.50)
        f1_mean, labels_matched, _, _ =  evaluate_metrics(targets, masks, labels)
        if len(f1_mean)>0 or len(labels_matched)>0:
            #print(" Validation f1 mean score:", f1_mean, " perc labels matched", labels_matched)
            f1_mean_list.extend(f1_mean)
            labels_matched_list.extend(labels_matched)
    return np.nansum(f1_mean_list)/len(f1_mean_list), np.sum(labels_matched_list)/len(labels_matched_list)

    
def main():
    collate_fn = lambda _: tuple(zip(*_)) # one-liner, no need to import


    wandb_config = dict(
        project='nps-ad-nature',
        entity='monika-ahirwar',
        )
    
    run = wandb.init(**wandb_config)
    
    """
    wandb_config = dict(
        project='nps-ad-nature',
        entity='monika-ahirwar',
        # mode = 'offline',
        config=dict(
            train_config=train_config,
            model_config=model_config,
            optim_config=optim_config,
        ),
        save_code=False,
        group='runs',
        job_type='train',
    )
    """
    train_config = dict(
        epochs = wandb.config.epochs,
        batch_size = wandb.config.batch_size,
        num_classes = 4,
        device_id = 0,
        ckpt_freq =500,
        eval_freq = 10,
)

    test_config = dict(
        batch_size = 1
    )
    model_config = _default_mrcnn_config(num_classes=1 + train_config['num_classes']).config
    optim_config = dict(
        # cls=grad_optim.GradSGD,
        cls=torch.optim.Adam,
        defaults=dict(lr=wandb.config.lr,weight_decay=wandb.config.weight_decay)  #-4 is too slow,
        #cls=torch.optim.SGD,
        #defaults=dict(lr=1. * (10. ** (-3)))  #-4 is too slow 
    )
    
    ## Dataset loading
    train_dataset = build_features.AmyBDataset(dataset_train_location, T.Compose([T.ToTensor()]))

    train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=4,
            collate_fn=collate_fn)
    
    test_patient_ids = os.listdir(dataset_test_location)
    
    if '.DS_Store' in test_patient_ids:
        test_patient_ids.remove('.DS_Store')
    
    # Model Building
    model = build_default(model_config, im_size=1024)
    device = torch.device('cpu')
    # if torch.cuda.is_available():
        # assert train_config['device_id'] >= 0 and train_config['device_id'] < torch.cuda.device_count()
    device = torch.device('cuda', train_config['device_id'])
   
    model = model.to(device)
    model.train(True)

    loss_names = 'objectness rpn_box_reg classifier box_reg mask'.split()
    loss_weights = [1., 4., 1., 4., 1.,]
    loss_weights = OrderedDict([(f'loss_{name}', weight) for name, weight in zip(loss_names, loss_weights)])

    #loss_fn = get_loss_fn(loss_weights)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer = optim_config['cls']([dict(params=list(model.parameters()))], **optim_config['defaults'])
    
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    assert run is wandb.run # run was successfully initialized, is not None
    run_id, run_dir = run.id, run.dir
    print("run Id", run_id)

    # #TODO: replace this with run.name
    exp_name = run.name
    print("*****RUN Name******", exp_name)
    # exp_name = "runtest"

    artifact_name = f'{run_id}-logs'

    # # Train Data
    for epoch in range(train_config['epochs']):
        train_loss = train_one_epoch(model, loss_weights, optimizer, scheduler, train_data_loader, device, epoch=epoch, log_freq=1)
        mask_acc, val_acc=evaluate_one_epoch(model, dataset_test_location,test_patient_ids,test_config,device)
        print(train_loss, mask_acc, val_acc)
        wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "mask_acc": mask_acc,
                    "val_acc": val_acc,
                }
            )


    #run.finish()

wandb.agent(sweep_id, function=main, count=4)
#run.finish()





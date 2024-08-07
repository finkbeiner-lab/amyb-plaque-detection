import lightning as L

import os
import sys
sys.path.append(os.path.join(os.getcwd(), *tuple(['..'])))
import argparse

from typing import Callable, Dict, List, Optional, Set
from collections import OrderedDict
import pdb
import torch
from torch import nn, Tensor
import torch.optim
#import wandb
from model_mrcnn_config import _default_mrcnn_config, build_default
from features import build_features
from generalized_mask_rcnn_pl import LitMaskRCNN
#from features import transforms as T
from utils.engine import evaluate
import torchvision
import matplotlib.pyplot as plt
from visualization.explain import ExplainPredictions
import pandas as pd
import plotly.graph_objects as go
import pdb
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
from features import transforms as T

from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union
from lightning.pytorch.callbacks import ModelCheckpoint
#from pytorch_lightning.loggers import WandbLogger
#import wandb
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import datetime


def train_model(wandb_logger, train_config,model_config,optim_config, dataset_base_dir, dataset_train_location, dataset_val_location  ):
    #run = wandb.init(**wandb_config)
    #run_id, run_dir = run.id, run.dir
    #wandb_logger = WandbLogger()
    collate_fn=lambda x: tuple(zip(*x))
    #exp_name = str(datetime.datetime.now())
    train_dataset = build_features.AmyBDataset(dataset_train_location, T.Compose([T.ToTensor()]))
    val_dataset = build_features.AmyBDataset(dataset_val_location, T.Compose([T.ToTensor()]))

    train_data_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=18, persistent_workers=False,
                collate_fn=collate_fn)

    val_data_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=18, persistent_workers=False,
                collate_fn=collate_fn)

    backbone, rpn, roi_heads, transform1 = build_default(model_config, im_size=1024)

    model  = LitMaskRCNN(optim_config,backbone,rpn,roi_heads,transform1)

    #ckpt_path= os.path.join(dataset_base_dir, "pytorch_lightning_model_output/"+str(exp_name))
    #os.makedirs(ckpt_path)
    #model_ckpt = 
    #chkpt = ModelCheckpoint(monitor="val_metrics-loss_classifier", mode="min", save_top_k=5,save_last=True)
    chkpt = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=7,save_last=True)

    #tensorboard = pl_loggers.TensorBoardLogger(save_dir="/gladstone/finkbeiner/steve/work/data/npsad_data/monika/models/")
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.01, patience=5, verbose=False, mode="max")
    
    trainer = L.Trainer(limit_train_batches=train_config["batch_size"], max_epochs=train_config['epochs'],devices=4, accelerator="gpu", num_sanity_val_steps=0,enable_checkpointing=True, check_val_every_n_epoch=train_config["eval_freq"],callbacks=[chkpt], logger =wandb_logger )
    #,early_stop_callback
    train_loader = train_data_loader
    valid_loader = val_data_loader
    trainer.fit(model, train_loader, valid_loader) 
    #exp_name = run.name
    
    #model_save_name = "/workspace/Projects/Amyb_plaque_detection/" + "models/{name}_mrcnn_model_{epoch}.pth"
    #torch.save(model.state_dict(), model_save_name.format(name=exp_name, epoch=train_config['epochs']))
    #run.finish()
    #return model.load_from_checkpoint(chkpt.best_model_path)
    return chkpt.best_model_path


if __name__ == '__main__':
    # TODO:
    #   - add functionality for calling backward with create_graph, i.e. for higher-order derivatives
    #   - switch to support for standard torchvision-bundled transforms (i.e. instead of `features.transforms as T` try `torchvision.transforms.transforms` or `torchvision.transforms.functional`)
    #   - complete feature: add grad_optimizer support transparently (so that usage is the same for users and train_one_epoch interface whether torch.optim or grad_optim is selected, i.e. log grads automatically)
    #   - do ^^ via closures
    #   - experimental: add an API to collect params and bufs by on module and/or name; generate on-the-fly state_dicts, gradient_dicts, higher-order gradient_dicts, etc.
    """
    parser = argparse.ArgumentParser(description='Maskrcnn training')

    parser.add_argument('base_dir', help="Enter the base dir (NAS)")
    parser.add_argument('dataset_train_location',
                        help='Enter the path train dataset resides')
    parser.add_argument('dataset_test_location',
                        help='Enter the path where test dataset resides')
    
    args = parser.parse_args()

    ## CONFIGS ##
    collate_fn = lambda _: tuple(zip(*_)) # one-liner, no need to import

    dataset_base_dir = args.base_dir
    dataset_train_location = args.dataset_train_location
    dataset_test_location = args.dataset_test_location
    """
    dataset_base_dir = '/workspace/Projects/Amyb_plaque_detection/'
    dataset_train_location = '/workspace/Projects/Amyb_plaque_detection/Datasets/train'
    dataset_val_location = '/workspace/Projects/Amyb_plaque_detection/Datasets/val'
    
    train_config = dict(
        epochs = 75,
        batch_size = 16,
        num_classes = 4,
        device_id = 0,
        ckpt_freq =500,
        eval_freq = 1,
    )
    model_config = _default_mrcnn_config(num_classes=1 + train_config['num_classes']).config
    optim_config = dict(
            cls=torch.optim.Adam,
            #cls = torch.optim.SGD,

            defaults=dict(lr=0.001,weight_decay=0.00001) 
            #defaults=dict(lr=1. * (10. ** (-3)))
        )

wandb_logger = WandbLogger(log_model="True",project='nps-ad-nature',
        entity='monika-ahirwar')

best_model_path = train_model(wandb_logger, train_config,model_config,optim_config, dataset_base_dir, dataset_train_location, dataset_val_location  )
print(best_model_path)
#backbone, rpn, roi_heads, transform1 = build_default(model_config, im_size=1024)
#model  = LitMaskRCNN(optim_config,backbone,rpn,roi_heads,transform1)

#model = model.load_from_checkpoint(chkpt.best_model_path)

#print(model)







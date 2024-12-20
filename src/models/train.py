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


# Sets the behavior of calls such as
#   - model.to(device=torch.device(type='cpu', index=None))
#   - model.cpu()
#   - model.cuda(), model.cuda(0)
#   - model.apply(fn)
# where isinstance(model, torch.nn.Module);
# True implies model.param = model.param.data.to(device),
# False implies model.param.data = model.param.data.to(device).
#
# Selecting True emulates upcoming change in semantics of torch.nn.Parameter;
# if torch.nn.Parameter inherits directly from torch.nn.Tensor,
# then as per torch docs for these methods, their invocation yields a new Tensor (equivalently Parameter) object.
# Alternatively, the recursive calls at present function by setting the
# torch.nn.Parameter.data field for a particular parameter object,
# as well as its torch.nn.Parameter.data.grad field if applicable.
# The Parameter class here is simply a wrapper for its data Tensor,
# and any previous reference to the Parameter instance will yield its
# now-altered data Tensor. However, this may not always be possible;
# if the semantics of Parameter change, an in-place modification of the
# underlying Tensor object itself would be required (i.e. a _to()), but this is not universally supported.
torch.__future__.set_overwrite_module_params_on_conversion(True)

def visualize_augmentations(images, targets):

    plt.figure(figsize=(10,10)) # specifying the overall grid size
    plt.suptitle('Data Augmentations')
    plt.subplot(1,2, 1)
    

    for i in range(len(images)):
        display_list = []
        img = images[i].detach().cpu().numpy()
        display_list.append(img)
        mask = targets[i]['masks'].detach().cpu().numpy()
        mask = mask.transpose(1, 2, 0)
        display_list.append(mask)

        for j in range(2):
            plt.subplot(1,2,j+1)
            plt.imshow(display_list[j])
        
        save_name = "../../../reports/figures/augmentation_{img_no}.png"

        plt.savefig(save_name.format(img_no=i))


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: Callable[[Dict[str, Tensor]], Tensor],
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int = 1,
    log_freq: int = 10,) -> None:

    assert model.training
    model_params = set(model.parameters())
    model_devices = set([p.device for p in model_params])
    assert model_devices == set([device]) # validate model params device
    for g in optimizer.param_groups: # validate optimizer params
        assert set(g['params']).issubset(model_params)

    log_metrics = list()

    for i, (images, targets) in enumerate(train_data_loader):
        images = [image.to(device) for image in images]
        targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]
        # visualize_augmentations(images , targets)
        # pdb.set_trace()
        optimizer.zero_grad()
        loss, metrics = loss_fn(model.forward(images, targets))
        loss.backward()
        optimizer.step()

        log_metrics.append(dict(epoch=epoch, loss=loss.item(), metrics=metrics))
        # print(dict(epoch=epoch, loss=loss.item(), metrics=metrics))
        print_logs = "epoch no : {epoch}, batch no : {batch_no}, total loss : {loss},  classifier :{classifier}, mask: {mask} ==================="
        print(print_logs.format(epoch=epoch, batch_no=i, loss=loss.item(),  classifier=metrics['loss_classifier'], mask=metrics['loss_mask']))
        if (i % log_freq) == 0:
            yield log_metrics
            log_metrics = list()

    yield log_metrics


def get_loss_fn(weights, default=0.):
    
    def compute_loss_fn(losses):
        item = lambda k: (k, losses[k].item())
        metrics = OrderedDict(list(map(item, [k for k in weights.keys() if k in losses.keys()] + [k for k in losses.keys() if k not in weights.keys()])))

        loss = sum(map(lambda k: losses[k] * (weights[k] if weights is not None and k in weights.keys() else default), losses.keys()))
        return loss, metrics
    return compute_loss_fn


def get_resp(prompt, prompt_fn=None, resps='n y'.split()):
    resp = input(prompt)
    while resp not in resps:
        resp = input(prompt if prompt_fn is None else propt_fn(resp))
    return resps.index(resp)

def plotPRcurve(eval2, epoch, run):

    df = pd.DataFrame(columns=["class","tpr","fpr","recall","precision"])
    len_classes = 3 ## Assuming 3 classes
   
    for c in range(len_classes): # running for all 3 classes
        ## parameters
        area_index = 0 # area - all (areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]  -> areaRngLbl = ['all', 'small', 'medium', 'large'])
        threshold_index= 0 # threshold 0.5  (iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)) -- can vary threshold from here 
        iou_type="bbox"
        maxDet = 100 
        # Selecting from 
        pdb.set_trace()
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
        
        tp_sum=tp_sum/tp_sum[-1]
        fp_sum = np.cumsum(fps, axis=0, dtype=float)
        fp_sum=fp_sum/fp_sum[-1]
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
    # TODO:
    #   - add functionality for calling backward with create_graph, i.e. for higher-order derivatives
    #   - switch to support for standard torchvision-bundled transforms (i.e. instead of `features.transforms as T` try `torchvision.transforms.transforms` or `torchvision.transforms.functional`)
    #   - complete feature: add grad_optimizer support transparently (so that usage is the same for users and train_one_epoch interface whether torch.optim or grad_optim is selected, i.e. log grads automatically)
    #   - do ^^ via closures
    #   - experimental: add an API to collect params and bufs by on module and/or name; generate on-the-fly state_dicts, gradient_dicts, higher-order gradient_dicts, etc.

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

    train_config = dict(
        epochs = 1,
        batch_size = 10,
        num_classes = 3,
        device_id = 0,
        ckpt_freq =500,
        eval_freq = 1,
    )

    test_config = dict(
        batch_size = 6
    )

    model_config = _default_mrcnn_config(num_classes=1 + train_config['num_classes']).config
    optim_config = dict(
        # cls=grad_optim.GradSGD,
        cls=torch.optim.SGD,
       
        defaults=dict(lr=1. * (10. ** (-2)))  #-4 is too slow 
    )
    wandb_config = dict(
        project='nps-ad-vivek',
        entity='hellovivek',
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

    
    ## Dataset loading
    train_dataset = build_features.AmyBDataset(dataset_train_location, T.Compose([T.ToTensor()]))
    # val_dataset = build_features.AmyBDataset(val_dataset, T.Compose([T.ToTensor()]))
    test_dataset = build_features.AmyBDataset(dataset_test_location, T.Compose([T.ToTensor()]))

    # Mapping
    # fn_relabel = lambda i: [1, 2, 1, 3][i - 1]
    def remap_label(orig_label):
        # print("\nOrig", orig_label)
        if (orig_label==1) or (orig_label==3):
            # print("mapped 1/3", 1)
            return 1
        if orig_label==4:
            # print("mapped 4", 3)
            return 3
        return orig_label
    
    train_dataset, test_dataset = [build_features.DatasetRelabeled(dataset, remap_label) for dataset in (train_dataset, test_dataset)]

    train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=4,
            collate_fn=collate_fn)
    
        
    # val_data_loader = torch.utils.data.DataLoader(
    #         test_dataset, batch_size=test_config['batch_size'], shuffle=False, num_workers=4,
    #         collate_fn=collate_fn)

    test_data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_config['batch_size'], shuffle=False, num_workers=4,
            collate_fn=collate_fn)

    
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

    loss_fn = get_loss_fn(loss_weights)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer = optim_config['cls']([dict(params=list(model.parameters()))], **optim_config['defaults'])

    run = wandb.init(**wandb_config)
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
        # print(f'Epoch {epoch}=======================================>.')

        for logs in train_one_epoch(model, loss_fn, optimizer, train_data_loader, device, epoch=epoch, log_freq=1):
            for log in logs:
                run.log(log)

        # if epoch + 1 == train_config['epochs'] or epoch % train_config['ckpt_freq'] == 0:

        #     artifact = wandb.Artifact(artifact_name, type='files')
        #     with artifact.new_file(f'ckpt/{epoch}.pt', 'wb') as f:
        #         torch.save(model.state_dict(), f)
            # run.log_artifact(artifact)

        if epoch % train_config['eval_freq'] == 0:
            eval_res = evaluate(run, model, test_data_loader, device=device, epoch=epoch)
            # pdb.set_trace()
            # plotPRcurve(eval_res, epoch)

        
        model.train(True)
    
    # TODO change the directory if running on desktop
    model_save_name = "/home/vivek/Projects/amyb-plaque-detection/" + "models/{name}_mrcnn_model_{epoch}.pth"
    torch.save(model.state_dict(), model_save_name.format(name=exp_name, epoch=train_config['epochs']))


    # print("\n =================The Model is Trained!====================")
    # print("-----------------Visualizing Model predictions----------------")

    # # # TODO Testing is done on Individual WSI Folders
    # input_path = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/test-patients'

    # model = build_default(model_config, im_size=1024)
   
    # explain = ExplainPredictions(model, model_input_path = model_save_name.format(name=exp_name, epoch=train_config['epochs']), test_input_path=input_path, 
    #                             detection_threshold=0.75, wandb=run, save_result=True, ablation_cam=True, save_thresholds=False)
    # explain.generate_results()

    # run.finish()

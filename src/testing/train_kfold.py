import os
import sys
sys.path.append(os.path.join(os.getcwd(), *tuple(['..'])))
import argparse
import pandas as pd
from typing import Callable, Dict, List, Optional, Set
from collections import OrderedDict
import pdb
import torch
from torch import nn, Tensor
import torch.optim
import wandb
from models.model_mrcnn import _default_mrcnn_config, build_default
from features import build_features
from features import transforms as T
from utils.engine import evaluate
import torchvision
import matplotlib.pyplot as plt
from visualization.explain import ExplainPredictions
from sklearn.model_selection import KFold
import random
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
import numpy as np
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

    for i, (images, targets) in enumerate(data_loader):
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
    parser.add_argument('k',
                        help='Number of folds for k-fold cross-validation')
    
    args = parser.parse_args()

    ## CONFIGS ##
    collate_fn = lambda _: tuple(zip(*_)) # one-liner, no need to import

    dataset_base_dir = args.base_dir
    dataset_train_location = args.dataset_train_location
    dataset_test_location = args.dataset_test_location
    k = int(args.k)
    isKfold_eval = True    

    train_config = dict(
        epochs = 50,
        batch_size = 6,
        num_classes = 4,
        device_id = 0,
        ckpt_freq =500,
        eval_freq = 50,
    )

    test_config = dict(
        batch_size = 1
    )

    model_config = _default_mrcnn_config(num_classes=1 + train_config['num_classes']).config
    optim_config = dict(
        # cls=grad_optim.GradSGD,
        cls=torch.optim.SGD,
        defaults=dict(lr=1. * (10. ** (-2)))  #-4 is too slow 
    )
    #wandb.init(project="amyloid_beta_runs", entity="monika-ahirwar")
    wandb_config =dict(
        project='amyloid_beta_runs',
        entity='ml_gladstone', config = dict(
            train_config=train_config,
            model_config=model_config,
            optim_config=optim_config,), save_code=False,
        group='runs',
        job_type='train',
    )

    
    splits=KFold(n_splits=k,shuffle=True,random_state=42)

    ## Get list of patient-ids for train and test
    train_patient_ids = os.listdir(dataset_train_location)
    if '.DS_Store' in train_patient_ids:
        train_patient_ids.remove('.DS_Store')

    test_patient_ids = os.listdir(dataset_test_location)
    if '.DS_Store' in test_patient_ids:
        test_patient_ids.remove('.DS_Store')

    ## Initialize dataframes to store k-fold evaluation metrics
    kfold_perf_df = pd.DataFrame(columns=["iteration","iou_type","metric_name","metric_value","epoch","patient_id"])
    test_kfold_perf_df = pd.DataFrame(columns=["iteration","iou_type","metric_name","metric_value","epoch","patient_id"])

    ## train data for k-folds
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_patient_ids)))):
        print('Fold {}'.format(fold + 1))
        # Split data into train and val
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        # Load train dataset
        train_dataset = build_features.AmyBDataset(os.path.join(dataset_train_location,train_patient_ids[0]), T.Compose([T.ToTensor()]))
        for s in list(train_sampler)[1:]:
            ds = build_features.AmyBDataset(os.path.join(dataset_train_location,train_patient_ids[s]), T.Compose([T.ToTensor()]))
            train_dataset = ConcatDataset([train_dataset, ds])
        
        train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True,
                                  num_workers=4, collate_fn=collate_fn)

        # Model Building
        model = build_default(model_config, im_size=1024)
        device = torch.device('cpu')
        if torch.cuda.is_available():
            assert train_config['device_id'] >= 0 and train_config['device_id'] < torch.cuda.device_count()
            device = torch.device('cuda', train_config['device_id'])
        model = model.to(device)
        model.train(True)

        loss_names = 'objectness rpn_box_reg classifier box_reg mask'.split()
        loss_weights = [1., 4., 1., 4., 1.,]
        loss_weights = OrderedDict([(f'loss_{name}', weight) for name, weight in zip(loss_names, loss_weights)])

        loss_fn = get_loss_fn(loss_weights)

        optimizer = optim_config['cls']([dict(params=list(model.parameters()))], **optim_config['defaults'])

        run = wandb.init(**wandb_config)
        assert run is wandb.run # run was successfully initialized, is not None
        run_id, run_dir = run.id, run.dir
        exp_name = run.name

        artifact_name = f'{run_id}-logs'
        eval_metric_full_training = pd.DataFrame()
        # Train Data
        for epoch in range(train_config['epochs']):
            # print(f'Epoch {epoch}=======================================>.')

            for logs in train_one_epoch(model, loss_fn, optimizer, train_loader, device, epoch=epoch, log_freq=1):
                for log in logs:
                    run.log(log)

            if epoch + 1 == train_config['epochs'] or epoch % train_config['ckpt_freq'] == 0:

                artifact = wandb.Artifact(artifact_name, type='files')
                with artifact.new_file(f'ckpt/{epoch}.pt', 'wb') as f:
                    torch.save(model.state_dict(), f)
                run.log_artifact(artifact)

            if (epoch + 1) % train_config['eval_freq'] == 0:
                for v in list(val_sampler):
                    print("validation set",v, train_patient_ids[v])
                    val_ds = build_features.AmyBDataset(os.path.join(dataset_train_location,train_patient_ids[v]), T.Compose([T.ToTensor()]))
                    val_loader = DataLoader(val_ds, batch_size=train_config['batch_size'], shuffle=False,
                                  num_workers=4, collate_fn=collate_fn)
                    eval_res, eval_res_df = evaluate(run, model, val_loader,isKfold_eval,device=device)
            #if (epoch + 1) == train_config['epochs']:
                # Using evaluation metric of last epoch for each k-fold 
                    eval_metric_full_training = eval_res_df
                    eval_metric_full_training["iteration"] = fold+1
                    eval_metric_full_training["epoch"] = epoch
                    eval_metric_full_training["patient_id"] = train_patient_ids[v]
                kfold_perf_df = pd.concat([kfold_perf_df,eval_metric_full_training])
                for t in range(len(test_patient_ids)):
                    test_ds = build_features.AmyBDataset(os.path.join(dataset_test_location,test_patient_ids[t]), T.Compose([T.ToTensor()]))
                    #test_ds = torchvision.datasets.ImageFolder(os.path.join(dataset_test_location,test_patient_ids[t],'images'), T.Compose([T.ToTensor()]))
                    test_loader = DataLoader(test_ds, batch_size=test_config['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)
                    test_eval_res, test_eval_res_df = evaluate(run, model, test_loader, isKfold_eval, device=device)
                    test_eval_res_df["iteration"] = fold+1
                    test_eval_res_df["epoch"] = epoch
                    test_eval_res_df["patient_id"] = test_patient_ids[t]
                    test_kfold_perf_df = pd.concat([test_kfold_perf_df,test_eval_res_df])  
            model.train(True)
        model_save_name = dataset_base_dir + "models/{name}_mrcnn_model_{epoch}_{fold_number}.pth"
        torch.save(model.state_dict(), model_save_name.format(name=exp_name, epoch=train_config['epochs'], fold_number=k))
    
    print("\n =================The Model is Trained!====================")
    
    print("-------------Evaluation metric results of all k-fold training models------------------")
    print(kfold_perf_df)
    print("-------------Average Evaluation metric summary of all k-fold training models------------------")
    avg_kfold_perf_df = kfold_perf_df.groupby(['iou_type','metric_name','epoch'])['metric_value'].mean().reset_index()
    avg_kfold_perf_df = avg_kfold_perf_df.rename(columns = {'metric_value':'avg_metric_value'}) 
    kfold_perf_df_by_patientid = kfold_perf_df.groupby(['patient_id','iou_type','metric_name','epoch'])['metric_value'].mean().reset_index()

    kfold_perf_df = pd.merge(kfold_perf_df, avg_kfold_perf_df, on = ['iou_type','metric_name','epoch'], how = 'left')
    kfold_perf_df = kfold_perf_df.groupby(['iteration','iou_type','metric_name','epoch'])['metric_value'].mean().reset_index()
    kfold_perf_df['percentage_change']=(kfold_perf_df['metric_value']-kfold_perf_df['avg_metric_value'])/kfold_perf_df['avg_metric_value']

    tbl = wandb.Table(data=avg_kfold_perf_df)
    run.log({"Average K-fold Evaluation Metric": tbl})

    tbl1 = wandb.Table(data=kfold_perf_df_by_patientid)
    run.log({"Average K-fold Evaluation Metric By Patient ID": tbl1})

    tbl2 = wandb.Table(data=kfold_perf_df)
    run.log({"All K-fold Evaluation Metric": tbl2})

    max_val = kfold_perf_df["epoch"].max()
    final_epoch_kfold_perf_bbox = kfold_perf_df[(kfold_perf_df["epoch"]==max_val) & (kfold_perf_df["metric_name"]==' Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]') & (kfold_perf_df["iou_type"]=="bbox")]
    final_epoch_kfold_perf_segm = kfold_perf_df[(kfold_perf_df["epoch"]==max_val) & (kfold_perf_df["metric_name"]==' Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]') & (kfold_perf_df["iou_type"]=="segm")]

    table_bbox = wandb.Table(data=final_epoch_kfold_perf_bbox, columns=["iteration","metric_value", "percentage_change"])
    table_segm = wandb.Table(data=final_epoch_kfold_perf_segm, columns=["iteration", "metric_value","percentage_change"])
    
    print("\n =================Test Output====================")
    test_kfold_by_patientid = test_kfold_perf_df.groupby(['patient_id','iou_type','metric_name','epoch'])['metric_value'].mean().reset_index()
    tbl3 = wandb.Table(data=test_kfold_by_patientid)
    run.log({"Test Evaluation Metric": tbl3})

    # Plots with percentage change for each iteration
    run.log({"my_custom_plot_id" : wandb.plot.line(table_bbox, "iteration", "percentage_change",
           title="k-th fold vs percent change - bbox")}) 
    run.log({"my_custom_plot_id2" : wandb.plot.line(table_bbox, "iteration", "metric_value",
           title="k-th fold vs AP@[ IoU=0.50:0.95| area=all|maxDets=100] - bbox")})
    run.log({"my_custom_plot_id3" : wandb.plot.line(table_segm, "iteration", "percentage_change",
           title="k-th fold vs percent change - segm")})
    run.log({"my_custom_plot_id4" : wandb.plot.line(table_segm, "iteration", "metric_value",
           title="k-th fold vs AP@[ IoU=0.50:0.95| area=all|maxDets=100] - segm")})
    run.finish()

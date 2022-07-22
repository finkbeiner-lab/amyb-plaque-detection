import os
import sys
sys.path.append(os.path.join(os.getcwd(), *tuple(['..'] * 2)))

from typing import Callable, Dict, List, Optional, Set
from collections import OrderedDict

import pdb

import torch
from torch import nn, Tensor
import torch.optim

import wandb

from model_mrcnn import _default_mrcnn_configs, build_default
from features import build_features
from features import transforms as T


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





def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: Callable[[Dict[str, Tensor]], Tensor],
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int = 1,
    log_freq: int = 10,
) -> None:
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

        optimizer.zero_grad()
        loss, metrics = loss_fn(model.forward(images, targets))
        loss.backward()
        optimizer.step()

        log_metrics.append(dict(epoch=epoch, loss=loss.item(), metrics=metrics))
        if (i % log_freq) == 0:
            yield log_metrics
            log_metrics = list()


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
    #   - switch to support for standard torchvision-bundled transforms
    #   - complete feature: add grad_optimizer support transparently (so that usage is the same for users and train_one_epoch interface whether torch.optim or grad_optim is selected, i.e. log grads automatically)
    #   - do ^^ via closures

    collate_fn = lambda _: tuple(zip(*_)) # one-liner, no need to import

    dataset_location = '/gladstone/finkbeiner/steve/work/data/npsad_data/gennadi/amy-def/'
    train_config = dict(
        epochs=20,
        batch_size=1,
        num_classes=3,
        device_id=0,
        checkpoints='/home/projects/amyb-plaque-detection/models/checkpoints/'
    )
    model_config = _default_mrcnn_configs(num_classes=1 + train_config['num_classes']).config_dict
    optim_config = dict(
        # cls=grad_optim.GradSGD,
        cls=torch.optim.SGD,
        defaults=dict(lr=5 * (10. ** (-3)))
    )
    wandb_config = dict(
        project='mrcnn_train',
        entity='gladstone-npsad',
        config=dict(train_config=train_config, model_config=model_config, optim_config=optim_config),
        save_code=False,
        group='warmup_runs',
        job_type='train',
        tags='train'.split(),
        name=None,
    )


    dataset = build_features.AmyBDataset(dataset_location, T.Compose([T.ToTensor()]))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=1, collate_fn=collate_fn)


    model = build_default(model_config, im_size=1024)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        assert train_config['device_id'] >= 0 and train_config['device_id'] < torch.cuda.device_count()
        device = torch.device('cuda', train_config['device_id'])
    model = model.to(device)
    model.train(True)

    loss_weights = OrderedDict([(f'loss_{name}', 1.) for name in 'objectness rpn_box_reg classifier box_reg mask'.split()])
    loss_fn = get_loss_fn(loss_weights)

    optimizer = optim_config['cls']([dict(params=list(model.parameters()))], **optim_config['defaults'])

    wandb.init(**wandb_config)
    for epoch in range(train_config['epochs']):
        print(f'Epoch {epoch} started.')
        for logs in train_one_epoch(model, loss_fn, optimizer, data_loader, device, epoch=epoch, log_freq=1):
            for log in logs:
                wandb.log(log)
        print(f'Epoch {epoch} ended.')



    # train_one_epoch(model, , )

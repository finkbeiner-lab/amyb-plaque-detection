import os
import sys
sys.path.append(os.path.join(os.getcwd(), *tuple(['..'] * 2)))

from typing import Callable, Dict, List, Optional, Set
from collections import OrderedDict

import pdb

import torch
from torch import nn, Tensor
import torch.optim

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
    log_freq: int = 5,
) -> None:
    model_params = set(model.parameters())
    model_devices = set([p.device for p in model_params])
    assert model_devices == set([device]) # validate model params device
    for g in optimizer.param_groups: # validate optimizer params
        assert set(g['params']).issubset(model_params)

    # metric_names = 'loss_classifier loss_box_reg loss_mask loss_objectness loss_rpn_box_reg'.split()
    log_metrics = list()

    for i, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]

        optimizer.zero_grad()
        loss, metrics = loss_fn(model.forward(images, targets))
        loss.backward()
        optimizer.step()

        log_metrics.append(dict(loss=loss.item(), metrics=metrics))

        if (i % log_freq) == 0:
            print(f'Step {i} loss: {loss}')
            print('\n'.join(f'  {k[5:]}: {v}' for k, v in metrics.items()))
            print()

            yield log_metrics
            log_metrics = list()





def get_resp(prompt, prompt_fn=None, resps='n y'.split()):
    resp = input(prompt)
    while resp not in resps:
        resp = input(prompt if prompt_fn is None else propt_fn(resp))
    return resps.index(resp)


if __name__ == '__main__':
    # TODO:
    #   - add functionality for calling backward with create_graph, i.e. for higher-order derivatives
    #   - switch to support for standard torchvision-bundled transforms

    collate_fn = lambda _: tuple(zip(*_)) # one-liner, no need to import

    batch_size = 2
    dataset_path = '/gladstone/finkbeiner/steve/work/data/npsad_data/gennadi/amy-def/'
    optim_config = dict(
        cls=torch.optim.SGD,
        defaults=dict(lr=5 * (10. ** (-3)))
    )

    dataset = build_features.AmyBDataset(dataset_path, T.Compose([T.ToTensor()]))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

        print('CUDA Device Availability: ')
        print('\n'.join(list(map(lambda t: f'{t[0] + 1}: {t[1]}', enumerate(device_names))) + [str()]))

        device_id = 0
        if len(device_names) > 1:
            device_id = get_resp(f'Device (1-{len(device_names)}): ', resps=list(map(str, range(1, len(device_names) + 1))))
        device = torch.device('cuda', device_id)




    config = _default_mrcnn_configs().config_dict
    model = build_default(config, im_size=1024)

    def get_loss_fn(weights=None, default=None):
        def compute_loss_fn(losses):
            metrics = OrderedDict([(k, v.item()) for k, v in losses.items()])
            loss = sum(map(lambda k: losses[k] * (weights[k] if weights is not None and k in weights.keys() else default), losses.keys()))
            return loss, metrics
        return compute_loss_fn

    model = model.to(device)
    loss_fn = get_loss_fn(default=1.)
    optimizer = optim_config['cls']([dict(params=list(model.parameters()))], **optim_config['defaults'])

    if get_resp('Start training (y/n): '):
        for logs in train_one_epoch(model, loss_fn, optimizer, data_loader, device):
            pass



    # train_one_epoch(model, , )

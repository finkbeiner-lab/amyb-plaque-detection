import os
import sys
__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

from typing import Any, Callable, List, Mapping, Optional, Tuple
from collections import OrderedDict
from functools import partial
import time

import torch
from torch import nn, Tensor

import torchvision

import data.datasets as datasets
from data.process_tiles import read_metrics
from features.torch_transforms import _Compose, _RandomHorizontalFlip, _RandomVerticalFlip
import models.rcnn_conf as rcnn_conf
import models.retinanet_conf as retinanet_conf
from models.modules.rcnn import RCNN, RCNNTransform
from models.model_utils import train, eval, evaluate, show


def get_areas(boxes):
    areas = boxes[:, 2:] - boxes[:, :2]
    sizes = (areas[:, 0] * areas[:, 1]).sqrt()
    minimum, mean, std, maximum = sizes.min().item(), sizes.mean().item(), sizes.std().item(), sizes.max().item()
    bounds = [min(minimum, mean - std), mean - std, mean + std, max(mean + std, maximum)]
    bounds = [float(bound ** 2) for bound in bounds]
    return dict(
        all=(bounds[0], bounds[3]),
        small=(bounds[0], bounds[1]),
        medium=(bounds[1], bounds[2]),
        large=(bounds[2], bounds[3]),
    )

def get_norm(path, slides):
    return read_metrics(path, list(map(lambda s: f'XE{s}_1_AmyB_1', slides)))[1][1:]


if __name__ == '__main__':
    # num_classes = 4
    num_classes = 3
    model_conf = retinanet_conf.retinanet_v2_conf(
        pretrained=True,
        num_classes=num_classes + 1,
        heads=retinanet_conf.heads_conf(
            norm_layer=partial(nn.GroupNorm, 32),
            loss_type='giou',
            iou_type='giou',
            allow_low_quality_matches=True,
            batched_nms=False,
        ),
    )
    optim_conf = dict(
        lr=1e-2,
        momentum=9e-1,
        weight_decay=1e-4,
    )
    train_conf = dict(
        run_id=str(int(time.time())),
        epochs=30,
        ckpt_freq=1,
        batch_size=2,
        device=0,
    )
    data_conf = (lambda tile_size: dict(
        labels=dict(
            # names='Core Diffuse Neuritic CAA'.split(),
            names='Core Diffuse CAA'.split(),
            # colors='red green blue black'.split(),
            colors='red green blue'.split(),
        ),
        slides=dict(
            train='09-028 10-033'.split(),
            # train='09-028'.split(),
            test='09-063'.split(),
        ),
        size=dict(
            train=tuple([tile_size] * 2),
            test=tuple([tile_size] * 2),
        ),
        step=dict(
            train=tuple([tile_size // 2] * 2),
            test=tuple([tile_size] * 2),
        ),
    ))(1024)

    slide_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg-test/'
    slide_name_fn = lambda name: os.path.join(slide_dir, f'XE{name}_1_AmyB_1.mrxs')
    json_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/gennadi/jsons/amyb/'
    json_name_fn = lambda name: os.path.join(json_dir, f'{name}.json')
    out_dir = os.path.join('/wynton/home/finkbeiner/gryan/runs/', train_conf['run_id'])
    norms_path = '/gladstone/finkbeiner/steve/work/data/npsad_data/gennadi/slide_metrics.txt'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # core, neuritic -> core; diffuse -> diffuse; caa -> caa
    old_label_names = 'Core Diffuse Neuritic CAA'.split()
    fn_relabel = lambda i: [1, 2, 1, 3][i - 1]
    dsets_train, dsets_test = [[datasets.VipsJsonDataset(
        slide_name_fn(slide),
        json_name_fn(slide),
        old_label_names,
        size=data_conf['size'][train],
        step=data_conf['step'][train],
    ) for slide in data_conf['slides'][train]] for train in 'train test'.split()]
    dset_train, dset_test = [torch.utils.data.ConcatDataset(dsets) for dsets in (dsets_train, dsets_test)]
    dset_train, dset_test = [datasets.DatasetRelabeled(dset, fn_relabel) for dset in (dset_train, dset_test)]
    dset_train = datasets.DatasetTransformed(dset_train, _Compose([_RandomHorizontalFlip(), _RandomVerticalFlip()]))
    loader = torch.utils.data.DataLoader(dset_train, batch_size=train_conf['batch_size'], shuffle=True, collate_fn=lambda _: tuple(zip(*_)))
    area_ranges = get_areas(torch.cat([it[1]['boxes'] for it in dset_test]))

    model_conf.transform.mean, model_conf.transform.std = get_norm(norms_path, data_conf['slides']['train'] + data_conf['slides']['test'])

    device = torch.device('cuda', train_conf['device'])
    # model = rcnn_conf.rcnn_v2_conf(**model_conf).module(
    model = model_conf.module(
        # freeze_submodules=['backbone.body.conv1', 'backbone.body.bn1', 'backbone.body.layer1'],
        # skip_submodules=['roi_heads.box_predictor', 'roi_heads.mask_predictor.mask_fcn_logits']
        skip_submodules=['heads.classification_head.cls_logits'],
        # skip_submodules=['heads'],
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), **optim_conf)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_conf['epochs'])

    for epoch in range(1, train_conf['epochs'] + 1):
    # for epoch in range(train_conf['epochs']):
        losses = train(model, optimizer, device, loader, epoch=epoch, scheduler=None)

        if epoch == train_conf['epochs'] or epoch % train_conf['ckpt_freq'] == 0:
            metrics, visualizations = evaluate(
                model,
                device,
                dset_test,
                thresh=0.4,
                label_names=data_conf['labels']['names'],
                label_colors=data_conf['labels']['colors'],
                viz=list(range(18)),
                iou_type='bbox',
                area_ranges=area_ranges,
            )

            log = dict(
                losses=losses,
                metrics=metrics,
                weights=model.state_dict(),
            )
            viz = torchvision.utils.make_grid([torchvision.utils.make_grid(_, nrow=1, padding=0) for _ in visualizations], nrow=6, padding=4)

            torch.save(log, os.path.join(out_dir, f'{epoch}.pt'))
            torchvision.transforms.ToPILImage()(viz).save(os.path.join(out_dir, f'{epoch}.png'))

import os
import sys
__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

from typing import Any, Callable, List, Mapping, Optional, Tuple
from collections import OrderedDict
import time

import torch
from torch import nn, Tensor

import torchvision

import data.datasets as datasets
import models.rcnn_conf as rcnn_conf
from models.modules.rcnn import RCNN, RCNNTransform
from models.model_utils import train, eval, evaluate, show


class DatasetRelabeled(torch.utils.data.Dataset):
    def __init__(self,
        dataset: torch.utils.data.Dataset,
        fn: Callable[[int,], int],
    ) -> None:
        self.dataset = dataset
        self.fn = fn

    def __len__(self) -> int:
        return self.dataset.__len__()

    def __getitem__(self,
        idx: int,
    ) -> Tuple[Tensor, Mapping[str, Tensor]]:
        image, target = self.dataset.__getitem__(idx)
        for i, v in enumerate(target['labels']):
            target['labels'][i] = self.fn(v)


if __name__ == '__main__':
    num_classes = 4
    model_conf = dict(
        pretrained=True,
        num_classes=num_classes + 1
    )
    optim_conf = dict(
        lr=1e-3,
        momentum=9e-1,
        weight_decay=5e-4
    )
    train_conf = dict(
        run_id=str(int(time.time())),
        epochs=0,
        ckpt_freq=1,
        batch_size=2,
        device=0,
    )
    data_conf = (lambda tile_size: dict(
        labels=dict(
            names='Core Diffuse Neuritic CAA'.split(),
            colors='red green blue black'.split(),
        ),
        slides=dict(
            train='09-028 10-033'.split(),
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

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    dsets_train, dsets_test = [[datasets.VipsJsonDataset(
        slide_name_fn(slide),
        json_name_fn(slide),
        data_conf['labels']['names'],
        size=data_conf['size'][train],
        step=data_conf['step'][train],
    ) for slide in data_conf['slides'][train]] for train in 'train test'.split()]
    dset_train, dset_test = [torch.utils.data.ConcatDataset(dsets) for dsets in (dsets_train, dsets_test)]
    loader = torch.utils.data.DataLoader(dset_train, batch_size=train_conf['batch_size'], shuffle=True, collate_fn=lambda _: tuple(zip(*_)))

    device = torch.device('cuda', train_conf['device'])
    model = rcnn_conf.rcnn_v2_conf(**model_conf).module(
        # freeze_submodules=['backbone.body.conv1', 'backbone.body.bn1', 'backbone.body.layer1'],
        skip_submodules=['roi_heads.box_predictor', 'roi_heads.mask_predictor.mask_fcn_logits']
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), **optim_conf)

    for epoch in range(1, train_conf['epochs'] + 1):
        losses = train(model, optimizer, device, loader, epoch=epoch)

        if epoch == train_conf['epochs'] or epoch % train_conf['ckpt_freq'] == 0:
            metrics, visualizations = evaluate(
                model,
                device,
                dset_test,
                label_names=data_conf['labels']['names'],
                label_colors=data_conf['labels']['colors'],
                viz=list(range(18)),
            )

            log = dict(
                losses=losses,
                metrics=metrics,
                weights=model.state_dict(),
            )
            viz = torchvision.utils.make_grid([torchvision.utils.make_grid(_, nrow=1, padding=0) for _ in visualizations], nrow=6, padding=4)

            torch.save(log, os.path.join(out_dir, f'{epoch}.pt'))
            torchvision.transforms.ToPILImage()(viz).save(os.path.join(out_dir, f'{epoch}.png'))

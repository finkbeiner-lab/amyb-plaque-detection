import os
import sys
__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

import numpy as np

import torch
from torch import nn, Tensor

import torchvision
from torchvision.transforms import ToPILImage

import data
from data.datasets import VipsJsonDataset, get_tile, tiles_per_box

import features
from features.torch_transforms import _ToTensor, _Compose, _RandomHorizontalFlip, _RandomVerticalFlip

import models
from models.rcnn_conf import rcnn_v2_conf
from models.model_utils import train, eval, show


if __name__ == '__main__':
    label_names = 'Core Diffuse Neuritic CAA'.split()
    label_colors = 'red green blue black'.split()

    vips_img_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg'
    json_dir = '/home/gryan/projects/qupath/annotations/amyloid'
    vips_img_names = ['09-063', '10-033']

    vips_img_fnames = [os.path.join(vips_img_dir, f'XE{vips_img_name}_1_AmyB_1.mrxs') for vips_img_name in vips_img_names]
    json_fnames = [os.path.join(json_dir, f'{vips_img_name}.json') for vips_img_name in vips_img_names]


    tile_size = 1024
    ds_train = VipsJsonDataset(vips_img_fnames[0], json_fnames[0], label_names, step=(tile_size // 2, tile_size // 2), size=(tile_size, tile_size))
    ds_test = VipsJsonDataset(vips_img_fnames[0], json_fnames[0], label_names, step=(tile_size, tile_size), size=(tile_size, tile_size))

    # ds_test_tiles = np.array(ds_test.tiles)
    # ds_test_tiles = list(map(tuple, ds_test_tiles[np.random.permutation(np.arange(ds_test_tiles.shape[0]))]))
    # test_tiles = ds_test_tiles[:16]
    test_tiles = [(60, 70), (65, 66), (64, 29), (61, 32), (63, 70), (61, 69), (65, 28), (66, 66), (72, 66), (57, 12), (66, 36), (62, 21), (65, 29), (74, 68), (71, 66), (62, 69)]

    test_boxes = [get_tile(tile, ds_test.step, ds_test.size, ds_test.offset) for tile in test_tiles]
    train_tiles = list(set(sum([tiles_per_box(box, ds_train.step, ds_train.size, ds_train.offset) for box in test_boxes], start=list())))

    test_idxs = [i for i, tile in enumerate(ds_test.tiles) if tile in test_tiles]
    train_idxs = [i for i, tile in enumerate(ds_train.tiles) if tile not in train_tiles]

    ds_train = torch.utils.data.dataset.Subset(ds_train, train_idxs)
    ds_test = torch.utils.data.dataset.Subset(ds_test, test_idxs)
    # print(len(ds_train), len(ds_test), test_tiles)
    # exit()

    device = torch.device('cuda', 0)
    epochs = 4
    freq = 1

    model_conf = rcnn_v2_conf(pretrained=True, num_classes=4)
    model = model_conf.module(
        # freeze_submodules=['backbone.body.conv1', 'backbone.body.bn1', 'backbone.body.layer1'],
        skip_submodules=['roi_heads.box_predictor', 'roi_heads.mask_predictor.mask_fcn_logits']
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), **dict(lr=2e-4, momentum=9e-2, weight_decay=1e-5,))
    loader = torch.utils.data.DataLoader(ds_train, batch_size=2, shuffle=True, collate_fn=lambda _: tuple(zip(*_)))

    for epoch in range(1, epochs + 1):
        train(model, optimizer, device, loader, epoch=epoch, progress=False,)

        grid = torchvision.utils.make_grid([show(image, eval(model, device, image, thresh=0.5, mask_thresh=0.5,), label_names=label_names, label_colors=label_colors,) for image, _ in ds_test], nrow=4)
        if epoch % freq == 0 or epoch == epochs:
            ToPILImage()(grid).save(f'/home/gryan/projects/amyb-plaque-detection/reports/eval/{epoch}.png')
            torch.save(model.state_dict(), f'/home/gryan/projects/amyb-plaque-detection/reports/eval/{epoch}.pt')

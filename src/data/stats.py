import os
import sys
__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

from functools import reduce
import numpy as np
import pyvips

import torch
from torch import nn, Tensor

import torchvision
from torchvision.transforms import ToPILImage

import data
from data.datasets import VipsJsonDataset, VipsDataset, get_tile, tiles_per_box
from data.test_data.generate_contours import get_slide_tiles

import features
from features.torch_transforms import _ToTensor, _Compose, _RandomHorizontalFlip, _RandomVerticalFlip


def norm(x: Tensor):
    dims = tuple(range(1, len(x.size())))
    n = reduce(lambda a, b: a * b, x.size()[1:], 1)
    s1, s2 = [(x ** (i + 1)).sum(dim=tuple(range(1, len(x.size())))) for i in range(2)]
    return torch.as_tensor(n), s1 / n, s2 / n

def summarize(norms):
    n, s1, s2 = map(torch.stack, zip(*norms))
    factors = (n / n.sum()).unsqueeze(1)
    mean = (s1 * factors).sum(dim=0)
    std = ((s2 * factors).sum(dim=0) - (mean ** 2)).sqrt()
    return mean, std


def get_cropped(slide, level):
    x, y = [int(slide.get(f'openslide.bounds-{_}')) // (2 ** level) for _ in 'xy']
    return slide.crop(x, y, slide.width - x, slide.height - y)

def get_tiles(h, w, tile_size, level):
    return [(y, x) for x in range(((w * (2 ** level)) // tile_size) + 1) for y in range(((h * (2 ** level)) // tile_size) + 1)]


if __name__ == '__main__':
    label_names = 'Core Diffuse Neuritic CAA'.split()
    label_colors = 'red green blue black'.split()

    vips_img_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg-test'
    json_dir = '/home/gryan/projects/qupath/annotations/amyloid'
    viz_dir = '/home/gryan/Pictures'
    vips_img_names = ['07-056', '09-063', '10-033']

    vips_img_fnames = [os.path.join(vips_img_dir, f'XE{vips_img_name}_1_AmyB_1.mrxs') for vips_img_name in vips_img_names]
    json_fnames = [os.path.join(json_dir, f'{vips_img_name}.json') for vips_img_name in vips_img_names]
    viz_fnames = [os.path.join(viz_dir, f'{name}.png') for name in vips_img_names]

    tile_size = 128
    level = 3

    for fname, viz_fname in zip(vips_img_fnames, viz_fnames):
        slide = pyvips.Image.new_from_file(fname, level=level)

        tiles, _, thresh, mask, _ = get_slide_tiles(slide, 0, tile_size, use_contours=True, top_n=1)

        ht, wt = np.array(mask.shape) // tile_size
        ts = np.array([(y, x) for x in range(wt + 1) for y in range(ht + 1)])
        ts = np.concatenate([ts + i for i in range(2)], axis=1) * tile_size
        ts_neg = [(y1, x1, y2, x2) for y1, x1, y2, x2 in ts if mask[y1:y2, x1:x2].sum() == 0]

        im = slide.numpy()[..., :3]
        for y1, x1, y2, x2 in ts_neg:
            im[y1:y2, x1:x2, ...] = 0

        im.save(ToPILImage()(viz_fname))





    # for x, y in invalid_tiles:
    #     x *= tile_size
    #     y *= tile_size
    #     x //= (2 ** level)
    #     y //= (2 ** level)
    #     d = tile_size
    #     d //= (2 ** level)
    #
    #     im[y:y + d, x:x + d, ...] = 0



    # tile_size = 1024
    # ds_train = VipsJsonDataset(vips_img_fnames[0], json_fnames[0], label_names, step=(tile_size // 2, tile_size // 2), size=(tile_size, tile_size))
    # ds_test = VipsJsonDataset(vips_img_fnames[0], json_fnames[0], label_names, step=(tile_size, tile_size), size=(tile_size, tile_size))
    #
    # # ds_test_tiles = np.array(ds_test.tiles)
    # # ds_test_tiles = list(map(tuple, ds_test_tiles[np.random.permutation(np.arange(ds_test_tiles.shape[0]))]))
    # # test_tiles = ds_test_tiles[:16]
    # test_tiles = [(60, 70), (65, 66), (64, 29), (61, 32), (63, 70), (61, 69), (65, 28), (66, 66), (72, 66), (57, 12), (66, 36), (62, 21), (65, 29), (74, 68), (71, 66), (62, 69)]
    #
    # test_boxes = [get_tile(tile, ds_test.step, ds_test.size, ds_test.offset) for tile in test_tiles]
    # train_tiles = list(set(sum([tiles_per_box(box, ds_train.step, ds_train.size, ds_train.offset) for box in test_boxes], start=list())))
    #
    # test_idxs = [i for i, tile in enumerate(ds_test.tiles) if tile in test_tiles]
    # train_idxs = [i for i, tile in enumerate(ds_train.tiles) if tile not in train_tiles]
    #
    # ds_train = torch.utils.data.dataset.Subset(ds_train, train_idxs)
    # ds_test = torch.utils.data.dataset.Subset(ds_test, test_idxs)

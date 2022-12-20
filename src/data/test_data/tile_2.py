from typing import Any, List, Optional, Tuple
import os
import sys
__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

from functools import reduce

import cv2
import numpy as np
import pyvips

import torch
from torch import nn, Tensor

import torchvision
from torchvision.transforms import ToPILImage


def norm(x: Tensor):
    dims = tuple(range(1, len(x.size())))
    n = reduce(lambda a, b: a * b, x.size()[1:], 1)
    s1, s2 = [(x ** (i + 1)).sum(dim=tuple(range(1, len(x.size())))) for i in range(2)]
    return torch.as_tensor(n), s1 / n, s2 / n

def norm_merge(norms: List[Tuple[Tensor, Tensor, Tensor]]):
    n, s1, s2 = map(torch.stack, zip(*norms))
    n_sum = n.sum()
    factors = (n / n_sum).unsqueeze(1)
    s1 *= factors
    s2 *= factors
    return n_sum, s1.sum(dim=0), s2.sum(dim=0)

def norm_summarize(norm: Tuple[Tensor, Tensor, Tensor]):
    _, s1, s2 = norm
    mean = s1
    std = (s2 - (mean ** 2)).sqrt()
    return mean, std

def tile_as_tensor(slide, tile):
    return torch.as_tensor(slide.crop(*tile).numpy()).permute(2, 0, 1) / 255

def norm_lambda(slide, tiles):
    norms = [norm(tile_as_tensor(slide, tile)) for tile in tiles]
    n = len(norms)
    merged = norm_merge(norms)
    return n, merged


def crop_tile(slide, out_dir, tile, ext='png'):
    out_file = os.path.join(out_dir, '_'.join(map(lambda _: '_'.join(_), zip('xy', map(str, tile[:2])))) + f'.{ext}')
    crop = slide.crop(*tile).numpy()
    ToPILImage()(crop).save(out_file)
    return True

def crop_lambda(out_dir, slide_name, ext='png'):
    out_dir = os.path.join(out_dir, slide_name)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    return lambda slide, tiles: len([crop_tile(slide, out_dir, tile, ext=ext) for tile in tiles])


def slide_tile_map(slide_name, slide_dir, tile_dir, size, f=None):
    if f is None:
        f = lambda slide, tiles: (slide, tiles)

    slide_path = os.path.join(slide_dir, f'{slide_name}.mrxs')
    tile_path = os.path.join(tile_dir, f'{slide_name}.tiles.npy')

    slide = pyvips.Image.new_from_file(slide_path, level=0)[:3]
    tiles = np.load(tile_path, allow_pickle=False)
    tiles = np.concatenate([tiles + i for i in range(2)], axis=1)
    tiles *= np.concatenate([size] * 2)
    tiles[:, 2:] -= tiles[:, :2]

    return f(slide, tiles)



if __name__ == '__main__':
    slide_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg-test'
    tile_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/slide_masks'
    out_dir = '/home/gryan/Pictures/stats/tile_dump'
    # out_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/test/images'
    tile_size = (1024, 1024)

    slide_names = ['.'.join(file.split('.')[:-1]) for file in next(os.walk(slide_dir))[2]]

    # slide, tiles = slide_tile_map(slide_names[0], slide_dir, tile_dir, tile_size)
    out = slide_tile_map(slide_names[0], slide_dir, tile_dir, tile_size, f=norm_lambda)

    # for slide_name in slide_names:
    #     num_tiles = slide_tile_map(slide_name, slide_dir, tile_dir, tile_size, crop_lambda(out_dir, slide_name))
    #     print(f'{slide_name}: {num_tiles} tiles')

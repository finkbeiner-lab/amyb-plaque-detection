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
    return torch.as_tensor(n).to(x.device), s1 / n, s2 / n

def norm_merge(norms: List[Tuple[Tensor, Tensor, Tensor]]):
    n, s1, s2 = map(torch.stack, zip(*norms))
    n_sum = n.sum()
    factors = (n / n_sum).unsqueeze(1)
    s1 *= factors
    s2 *= factors
    return n_sum, s1.sum(dim=0), s2.sum(dim=0)

def norm_summarize(norm_values: Tuple[Tensor, Tensor, Tensor]):
    _, s1, s2 = norm_values
    mean = s1
    std = (s2 - (mean ** 2)).sqrt()
    return mean, std

def tile_as_tensor(slide, tile):
    device = torch.device('cuda', 0)
    return (torch.as_tensor(slide.crop(*tile).numpy()).permute(2, 0, 1) / 255).to(device)

def norm_lambda(slide, tiles):
    norms = [norm(tile_as_tensor(slide, tile)) for tile in tiles]
    return len(norms), norm_merge(norms)

def norm_to_str(norm_values):
    n, s1, s2 = norm_values
    n, s1, s2 = (n.item(), *[list(map(lambda _: _.item(), s)) for s in (s1, s2)])
    n, s1, s2 = (str(n), *map(lambda s: ' '.join(map(str, s)), (s1, s2)))
    return ','.join((n, s1, s2))


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
    tiles = np.concatenate([tiles + i for i in range(2)], axis=1) * np.array(size * 2)
    tiles = np.minimum(np.maximum(tiles, 0), np.array([slide.width, slide.height] * 2))
    tiles[:, 2:] -= tiles[:, :2]
    assert (tiles[:, :2] >= 0).all() and (tiles[:, 2:] > 0).all()

    return f(slide, tiles)


def read_metrics(metrics_file):
    with open(metrics_file, 'r') as f:
        lines = f.read().split('\n')[1:-1]
    lines = [line.split(',') for line in lines]

    slides = [line[0] for line in lines]
    norms_each = [line[2:] for line in lines]
    norms_each = [(torch.as_tensor(int(n)), *[torch.as_tensor(list(map(float, s.split(' ')))) for s in (s1, s2)]) for n, s1, s2 in norms_each]
    norms = norm_merge(norms_each)

    summary_each = [(norm_values[0], *norm_summarize(norm_values)) for norm_values in norms_each]
    summary_each = [(n.item(), *[list(map(lambda _: _.item(), m)) for m in (mean, std)]) for n, mean, std in summary_each]
    summary_each = sorted(list(zip(slides, summary_each)), key=lambda _: _[0])

    summary = [(n.item(), *[list(map(lambda _: _.item(), m)) for m in (mean, std)]) for n, mean, std in [(norms[0], *norm_summarize(norms))]][0]

    return summary_each, summary

def format_metrics(metrics):
    slide_name, (num_pixels, mean, std) = metrics
    return f'{slide_name} ({num_pixels} px):\n  mean: {mean}\n  std: {std}\n'

def write_metrics(slide_name, slide_dir, tile_dir, metrics_file, tile_size):
    num_tiles, norm_values = slide_tile_map(slide_name, slide_dir, tile_dir, tile_size, f=norm_lambda)
    with open(os.path.join(metrics_file, 'a')) as f:
        f.write(f'{slide_name},{num_tiles},{norm_to_str(norm_values)}\n')





if __name__ == '__main__':
    slide_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg-test'
    tile_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/slide_masks'
    out_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/test/images'
    tile_size = (1024, 1024)

    slide_names = ['.'.join(file.split('.')[:-1]) for file in next(os.walk(slide_dir))[2]]

    for slide_name in slide_names:
        num_tiles = slide_tile_map(slide_name, slide_dir, tile_dir, tile_size, f=crop_lambda(out_dir, slide_name))
        print(f'{slide_name}: {num_tiles} tiles')


    # # Metrics
    # out_dir = '/home/gryan/Pictures/stats'
    # metrics_file = os.path.join(out_dir, f'metrics.txt')
    #
    # with open(metrics_file, 'w') as f:
    #     f.write('slide_name,num_tiles,num_pixels,mean,std\n')
    # for slide_name in slide_names:
    #     write_metrics(slide_name, slide_dir, tile_dir, metrics_file, tile_size)
    #
    # metrics, metrics_total = read_metrics(metrics_file)
    # out = '\n'.join(map(format_metrics, metrics))
    # out += '\n' * 2
    # out += format_metrics(('Total', metrics_total))
    # out += '\n'



    # slide, tiles = slide_tile_map(slide_names[0], slide_dir, tile_dir, tile_size)

    # for slide_name in slide_names:
    #     num_tiles = slide_tile_map(slide_name, slide_dir, tile_dir, tile_size, crop_lambda(out_dir, slide_name))
    #     print(f'{slide_name}: {num_tiles} tiles')

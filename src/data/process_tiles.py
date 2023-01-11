from typing import Any, List, Optional, Tuple
import argparse
import os
import sys
__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

from functools import reduce

import cv2
import numpy as np
import pyvips
import tqdm

import torch
from torch import nn, Tensor

import torchvision
from torchvision.transforms import ToPILImage


"""
process_tiles.py: map over slide positive tiles; write crops and compute metrics

Tile computation (where tile_size = (xs, ys)):
    Tiles are represented as (xi, yi), corresponding to the box
    (xi * xs, yi * ys, ((xi + 1) * xs) - 1, ((yi + 1) * ys) - 1) in (x1, y1, x2, y2) format, or
    (xi * xs, yi * ys, xs, ys) in (x, y, w, h) format.
    The tiles listed for a given slide are cropped from it and passed as input to the downstream task.

Metrics computation:
    tile_as_tensor: Each tile is cropped using pyvips, output as an ndarray, converted to a CUDA tensor, and normalized in the range (0, 1).
    norm: Considering each color channel in turn, the number of pixels, and the sum of a) pixel values and b) pixel values squared is recorded, normalized by the number of pixels.
    norm_merge: Given the pixel count and per-channel sum and square sum for all tiles in a slide, the metrics are aggregated into an aggregate pixel count and per-channel sum and square sum, normalized by the aggregate number of pixels.
    norm_summarize: The mean is given by the sum, while the std is related to the square sum by taking the square root of the difference of the square sum and the square of the mean.
"""


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
    with open(os.path.join(metrics_file), 'a') as f:
        f.write(f'{slide_name},{num_tiles},{norm_to_str(norm_values)}\n')


def save_slide_crops(slide_names, slide_dir, tile_dir, out_dir, tile_size):
    for slide_name in tqdm.tqdm(slide_names):
        slide_tile_map(slide_name, slide_dir, tile_dir, tuple([tile_size] * 2), f=crop_lambda(out_dir, slide_name))

def save_slide_metrics(slide_names, slide_dir, tile_dir, out_dir, tile_size):
    out_file = os.path.join(out_dir, 'metrics.csv')
    with open(out_file, 'w') as f:
        f.write('slide_name,num_tiles,num_pixels,sum_1,sum_2\n')
    for slide_name in tqdm.tqdm(slide_names):
        write_metrics(slide_name, slide_dir, tile_dir, out_file, tuple([tile_size] * 2))

def read_slide_metrics(out_dir):
    out_file = os.path.join(out_dir, 'metrics.txt')
    metrics, metrics_total = read_metrics(out_file)
    out = '\n' + '\n'.join(map(format_metrics, metrics)) + '\n'
    out += '\n' + format_metrics(('Total', metrics_total)) + '\n'
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('command', choices='crops metrics'.split(), help='whether to output tile crops or slide metrics')
    parser.add_argument('slide_dir', help='directory to read *.mrxs slides from')
    parser.add_argument('input_dir', help='directory to read tiles from')
    parser.add_argument('output_dir', help='directory to write output(s) to')
    parser.add_argument('tile_size', type=int, help='size of tiles referenced in the input')
    parser.add_argument('--slide_names', nargs='+', help='filenames (without extension) of slides to process (defaults to all)')

    args = parser.parse_args()
    command, slide_dir, input_dir, output_dir, tile_size, slide_names = map(args.__getattribute__, 'command slide_dir input_dir output_dir tile_size slide_names'.split())
    if slide_names is None:
        slide_names = sorted(['.'.join(fname.split('.')[:-1]) for fname in next(os.walk(slide_dir))[2] if fname.split('.')[-1] == 'mrxs'])

    if command == 'crops':
        save_slide_crops(slide_names, slide_dir, input_dir, output_dir, tile_size)
    if command == 'metrics':
        save_slide_metrics(slide_names, slide_dir, input_dir, output_dir, tile_size)
        read_slide_metrics(out_dir)


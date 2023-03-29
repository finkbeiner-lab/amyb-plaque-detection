from typing import Any, Callable, List, Mapping, Optional, Tuple
from dataclasses import dataclass, field, asdict

import os
import sys

import numpy as np
import pyvips

import torch
from torch import nn, Tensor

import torchvision
from torchvision.transforms import ToTensor, ToPILImage


def get_vips_tile(vips_src, coords, target, label_colors=None):
    """
    vips_img: a pyvips image
    coords: [x1, y1, x2, y2] bbox coordinates
    target: a dict(labels=[...], boxes=[...], masks=[...])
    label_colors: optional list of colors to use for masks, indexed by label

    returns: a pyvips tile of size [x2 - x1 + 1, y2 - y1 + 1] with overlaid masks
    """

    if label_colors is None:
        label_colors = ['red'] * len(target['labels'])
    else:
        label_colors = [label_colors[i - 1] for i in target['labels']]

    tile = vips_src.crop(*coords[:2], *(coords[2:] - coords[:2] + 1)).numpy()
    tile = torchvision.utils.draw_segmentation_masks(torch.tensor(tile).permute(2, 0, 1), target['masks'], alpha=0.5, colors=label_colors)
    tile = tile.permute(1, 2, 0).numpy()
    return pyvips.Image.new_from_array(tile)

def paste_vips_tile(vips_src, vips_dst, coords, target, label_colors=None):
    return vips_dst.insert(get_vips_tile(vips_src, coords, target, label_colors=label_colors), *coords[:2])


def paste_vips_instances(vips_img, instances, label_colors=None):
    """
    Replaces each instance with result of get_vips_tile

    vips_img: a pyvips image
    instances: a [(label, (x1, y1, x2, y2), mask), ...]
    label_colors: optional list of colors to use for masks, indexed by label
    """

    vips_img = vips_img.copy()

    for label, box, mask in instances:
        target = dict(zip('labels boxes masks'.split(), [it.unsqueeze(0) for it in (label, box, mask)]))
        vips_img = paste_vips_tile(vips_img, vips_img, box, target, label_colors=label_colors)

    return vips_img

def paste_vips_tiles(vips_img, tiles, label_colors=None):
    """
    Replaces each tile with the result of get_vips_tile

    vips_img: a pyvips image
    tiles: a [((x1, y1, x2, y2), dict(labels=[...], boxes=[...], masks=[...])), ...]
    label_colors: optional list of colors to use for masks, indexed by label
    """

    vips_img = vips_img.copy()

    for coords, target in tiles:
        vips_img = paste_vips_tile(vips_img, vips_img, coords, target, label_colors=label_colors)

    return vips_img


# ### Code below here is specific to a local machine ###
#
# __pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
# if __pkg not in sys.path:
#     sys.path.append(__pkg)
#
# from data.datasets import VipsJsonDataset, get_tile
#
# def get_instances(dataset, idxs=None):
#     """
#     Utility to format the labels, boxes, masks of the dataset for use with paste_vips_masks
#     """
#
#     return [(
#         torch.as_tensor(label).to(torch.long),
#         torch.as_tensor(box + np.array(dataset.vips_offset * 2)).to(torch.float),
#         torch.as_tensor(mask).to(torch.bool),
#     ) for i, (label, box, mask) in enumerate(zip(dataset.labels, dataset.boxes, dataset.masks)) if idxs is None or i in idxs]
#
# def get_tiles(dataset, idxs=None):
#     """
#     Utility to format the tiles of the dataset for use with paste_vips_tiles
#     """
#
#     return [(
#         torch.as_tensor(get_tile(dataset.tiles[i], dataset.step, dataset.size, tuple(map(sum, zip(dataset.vips_offset, dataset.offset))))).to(torch.float),
#         dataset[i][1],
#     ) for i in (idxs if idxs is not None else list(range(len(dataset))))]
#
# if __name__ == '__main__':
#     slide_dir = '/Volumes/STORAGE/slides/amyb/'
#     json_dir = '/Users/gennadiryan/Documents/gladstone/projects/wynton/'
#
#     slide_name = '09-028'
#     slide_fname = os.path.join(slide_dir, f'XE{slide_name}_1_AmyB_1.mrxs')
#     json_fname = os.path.join(json_dir, f'{slide_name}.json')
#
#     label_names='Core Diffuse Neuritic CAA'.split()
#     label_colors='red green blue black'.split()
#
#     dataset = VipsJsonDataset(slide_fname, json_fname, label_names, step=(1024, 1024), size=(1024, 1024))
#
#     out_slide_1 = paste_vips_instances(dataset.vips_img, get_instances(dataset), label_colors=label_colors)
#     out_slide_2 = paste_vips_tiles(dataset.vips_img, get_tiles(dataset), label_colors=label_colors)
